"""Evaluate note-level MoE technique classification on the viotech test set.

Prints per-class accuracy, confusion matrix, precision/recall/F1 for each head:
  - Tonal technique  (4-class)
  - Articulation      (4-class)
  - Legato            (binary)

Usage:
    python evaluate_moe_technique.py \
        --checkpoint_path /path/to/checkpoint.pth \
        --hdf5s_dir /path/to/hdf5s/viotech \
        --device 0
"""

import os
import sys

sys.path.insert(1, os.path.join(os.path.dirname(__file__), '..', 'utils'))

import argparse
import numpy as np
import torch
import logging

from models_contrast import Regress_onset_offset_frame_velocity_CRNN
from pytorch_utils import move_data_to_device
from data_generator import CustomDataset, CustomTestSampler, collate_fn
import config

TONAL_NAMES = {0: 'none', 1: 'pizzicato', 2: 'harmonics', 3: 'openstring'}
ARTIC_NAMES = {0: 'none', 1: 'release', 2: 'staccato', 3: 'spiccato'}
LEGATO_NAMES = {0: 'bow_change', 1: 'sustained'}


def evaluate(args):
    device = (torch.device(f'cuda:{args.device}')
              if args.device >= 0 and torch.cuda.is_available()
              else torch.device('cpu'))

    fps = config.frames_per_second
    segment_seconds = config.segment_seconds

    # ---- Model ----
    model = Regress_onset_offset_frame_velocity_CRNN(
        frames_per_second=fps,
        classes_num=config.classes_num,
        output_features=True,
        predict_technique=False,
        predict_technique_moe=True,
    )

    checkpoint = torch.load(args.checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.to(device)
    model.eval()
    print(f'Model loaded from {args.checkpoint_path}')

    # ---- Dataset + sampler ----
    hdf5s_dir = args.hdf5s_dir

    dataset = CustomDataset(
        hdf5s_dir=hdf5s_dir,
        segment_seconds=segment_seconds,
        frames_per_second=fps,
        max_note_shift=0,
        augmentor=None,
        include_technique_label=True,
    )

    test_sampler = CustomTestSampler(
        hdf5s_dir=hdf5s_dir,
        split=args.split,
        segment_seconds=segment_seconds,
        hop_seconds=config.hop_seconds,
        batch_size=args.batch_size,
        mini_data=False,
    )
    max_possible = len(test_sampler.segment_list) // args.batch_size
    test_sampler.max_evaluate_iteration = min(args.max_iterations, max_possible)
    print(f'Test segments: {len(test_sampler.segment_list)}, '
          f'max batches: {test_sampler.max_evaluate_iteration}')

    test_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_sampler=test_sampler,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ---- Collect predictions ----
    all_tonal_pred, all_tonal_gt = [], []
    all_artic_pred, all_artic_gt = [], []
    all_legato_pred, all_legato_gt = [], []
    total_notes = 0

    # Gate usage statistics — global + per technique class
    gate_usage_sum = None
    # {tonal_class_id: (sum_of_gate_probs_vector, count)}
    gate_by_tonal = {}
    gate_by_artic = {}
    gate_by_legato = {}

    print(f'Evaluating on {args.split} set from {hdf5s_dir}')
    print(f'Device: {device}  |  batch_size: {args.batch_size}  '
          f'|  max_iterations: {args.max_iterations}')
    print()

    with torch.no_grad():
        for batch_idx, batch_data_dict in enumerate(test_loader):
            for key in batch_data_dict.keys():
                batch_data_dict[key] = move_data_to_device(batch_data_dict[key], device)

            if 'note_onset_frames' not in batch_data_dict:
                continue

            num_notes = batch_data_dict['num_notes']
            if num_notes.sum() == 0:
                continue

            note_info = {
                'onset_frames': batch_data_dict['note_onset_frames'].long(),
                'offset_frames': batch_data_dict['note_offset_frames'].long(),
                'num_notes': num_notes.long(),
            }

            output_dict = model(batch_data_dict['waveform'], note_info=note_info)

            if 'note_tonal_logits' not in output_dict:
                continue

            B = num_notes.shape[0]

            tonal_pred = output_dict['note_tonal_logits'].argmax(dim=-1)
            artic_pred = output_dict['note_artic_logits'].argmax(dim=-1)
            legato_pred = (output_dict['note_legato_prob'].squeeze(-1) > 0.5).long()

            tonal_gt = batch_data_dict['note_tonal_technique']
            artic_gt = batch_data_dict['note_articulation']
            legato_gt = batch_data_dict['note_legato']

            # Accumulate gate statistics
            gate_probs = output_dict['moe_gate_probs']  # (B, N, E)

            for b in range(B):
                n = num_notes[b].item()
                if n == 0:
                    continue
                t_gt = tonal_gt[b, :n].cpu().numpy()
                a_gt = artic_gt[b, :n].cpu().numpy()
                l_gt = legato_gt[b, :n].cpu().numpy()

                all_tonal_pred.extend(tonal_pred[b, :n].cpu().numpy())
                all_tonal_gt.extend(t_gt)
                all_artic_pred.extend(artic_pred[b, :n].cpu().numpy())
                all_artic_gt.extend(a_gt)
                all_legato_pred.extend(legato_pred[b, :n].cpu().numpy())
                all_legato_gt.extend(l_gt)
                total_notes += n

                g = gate_probs[b, :n].cpu().numpy()  # (n, E)
                if gate_usage_sum is None:
                    gate_usage_sum = g.sum(axis=0)
                else:
                    gate_usage_sum += g.sum(axis=0)

                # Per-class gate accumulation
                for cls_id in np.unique(t_gt):
                    mask = t_gt == cls_id
                    s, c = gate_by_tonal.get(int(cls_id), (np.zeros(g.shape[1]), 0))
                    gate_by_tonal[int(cls_id)] = (s + g[mask].sum(axis=0), c + int(mask.sum()))
                for cls_id in np.unique(a_gt):
                    mask = a_gt == cls_id
                    s, c = gate_by_artic.get(int(cls_id), (np.zeros(g.shape[1]), 0))
                    gate_by_artic[int(cls_id)] = (s + g[mask].sum(axis=0), c + int(mask.sum()))
                for cls_id in np.unique(l_gt):
                    mask = l_gt == cls_id
                    s, c = gate_by_legato.get(int(cls_id), (np.zeros(g.shape[1]), 0))
                    gate_by_legato[int(cls_id)] = (s + g[mask].sum(axis=0), c + int(mask.sum()))

            if (batch_idx + 1) % 20 == 0:
                print(f'  batch {batch_idx + 1}: {total_notes} notes accumulated')

    if total_notes == 0:
        print('No notes found in the test set. Check your --hdf5s_dir and --split.')
        return

    all_tonal_pred = np.array(all_tonal_pred, dtype=int)
    all_tonal_gt = np.array(all_tonal_gt, dtype=int)
    all_artic_pred = np.array(all_artic_pred, dtype=int)
    all_artic_gt = np.array(all_artic_gt, dtype=int)
    all_legato_pred = np.array(all_legato_pred, dtype=int)
    all_legato_gt = np.array(all_legato_gt, dtype=int)

    print()
    print('=' * 70)
    print(f'  Total notes evaluated: {total_notes}')
    print('=' * 70)
    print()

    _print_multiclass_report('Tonal Technique', all_tonal_gt, all_tonal_pred, TONAL_NAMES)
    _print_multiclass_report('Articulation', all_artic_gt, all_artic_pred, ARTIC_NAMES)
    _print_multiclass_report('Legato', all_legato_gt, all_legato_pred, LEGATO_NAMES)

    # Gate usage
    if gate_usage_sum is not None:
        num_experts = len(gate_usage_sum)
        gate_pct = gate_usage_sum / gate_usage_sum.sum() * 100
        print('-' * 70)
        print('  MoE Gate Usage — Global (avg routing weight per expert)')
        print('-' * 70)
        for i, pct in enumerate(gate_pct):
            bar = '█' * int(pct / 2)
            print(f'  Expert {i}: {pct:5.1f}%  {bar}')
        print()

        _print_gate_by_class('Tonal Technique', gate_by_tonal, TONAL_NAMES, num_experts)
        _print_gate_by_class('Articulation', gate_by_artic, ARTIC_NAMES, num_experts)
        _print_gate_by_class('Legato', gate_by_legato, LEGATO_NAMES, num_experts)


def _print_multiclass_report(title, gt, pred, class_names):
    from sklearn.metrics import (classification_report, confusion_matrix,
                                 accuracy_score, f1_score)

    present_classes = sorted(set(gt) | set(pred))
    target_names = [class_names.get(c, f'class_{c}') for c in present_classes]

    acc = accuracy_score(gt, pred)
    macro_f1 = f1_score(gt, pred, labels=present_classes, average='macro', zero_division=0)
    weighted_f1 = f1_score(gt, pred, labels=present_classes, average='weighted', zero_division=0)

    print('-' * 70)
    print(f'  {title}')
    print('-' * 70)
    print(f'  Overall Accuracy : {acc:.4f}')
    print(f'  Macro F1         : {macro_f1:.4f}')
    print(f'  Weighted F1      : {weighted_f1:.4f}')
    print()
    print(classification_report(
        gt, pred,
        labels=present_classes,
        target_names=target_names,
        digits=4,
        zero_division=0,
    ))

    cm = confusion_matrix(gt, pred, labels=present_classes)
    col_w = max(12, max(len(n) for n in target_names) + 2)
    header = ''.ljust(col_w) + ''.join(n.ljust(col_w) for n in target_names)
    print('  Confusion Matrix (rows = GT, cols = Predicted):')
    print(f'  {header}')
    for i, row in enumerate(cm):
        cells = ''.join(str(v).ljust(col_w) for v in row)
        print(f'  {target_names[i].ljust(col_w)}{cells}')
    print()

    # Per-class accuracy
    print('  Per-class Accuracy:')
    for i, cls in enumerate(present_classes):
        mask = gt == cls
        n_total = mask.sum()
        if n_total > 0:
            n_correct = (pred[mask] == cls).sum()
            cls_acc = n_correct / n_total
        else:
            n_correct = 0
            cls_acc = 0.0
        name = class_names.get(cls, f'class_{cls}')
        print(f'    {name:15s}  {n_correct:5d} / {n_total:5d}  = {cls_acc:.4f}')
    print()


def _print_gate_by_class(title, gate_by_cls, class_names, num_experts):
    """Print per-class expert routing distribution as a table + bar chart."""
    if not gate_by_cls:
        return

    print('-' * 70)
    print(f'  MoE Gate Usage — Per {title} Class')
    print('-' * 70)

    # Table header
    expert_hdr = ''.join(f'  Expert {i}' for i in range(num_experts))
    print(f'  {"Class":15s}  {"Notes":>6s}{expert_hdr}')

    for cls_id in sorted(gate_by_cls.keys()):
        s, c = gate_by_cls[cls_id]
        name = class_names.get(cls_id, f'class_{cls_id}')
        if c == 0:
            continue
        avg = s / c  # avg gate weight per note for this class
        pcts = avg / avg.sum() * 100
        cells = ''.join(f'  {p:6.1f}%' for p in pcts)
        print(f'  {name:15s}  {c:6d}{cells}')

    print()

    # Visual bar chart per class
    for cls_id in sorted(gate_by_cls.keys()):
        s, c = gate_by_cls[cls_id]
        name = class_names.get(cls_id, f'class_{cls_id}')
        if c == 0:
            continue
        avg = s / c
        pcts = avg / avg.sum() * 100
        print(f'  {name}:')
        for i, pct in enumerate(pcts):
            bar = '█' * int(pct / 2) + '░' * (50 - int(pct / 2))
            print(f'    Expert {i}: {bar} {pct:5.1f}%')
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate MoE technique classification on viotech test set')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to trained model checkpoint (.pth)')
    parser.add_argument('--hdf5s_dir', type=str, required=True,
                        help='Directory containing viotech H5 files')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'validation', 'test'])
    parser.add_argument('--device', type=int, default=0,
                        help='-1 for CPU, 0+ for GPU')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_iterations', type=int, default=500,
                        help='Max batches to evaluate (set large for full test set)')
    args = parser.parse_args()
    evaluate(args)
