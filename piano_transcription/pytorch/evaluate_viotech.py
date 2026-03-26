"""Evaluate base VioTech frame-level technique classification.

Uses the parallel technique head (FrameLevelTechniqueHead) attached to
Regress_onset_offset_frame_velocity_CRNN with predict_technique=True.

Output keys:
  tonal_technique_output  (B, T, 4) logits
  articulation_output     (B, T, 4) logits
  legato_output           (B, T, 1) sigmoid probability

Reports per-task:
  - Overall Accuracy, Macro F1, Weighted F1
  - sklearn classification_report (precision / recall / f1 / support)
  - Confusion matrix
  - Per-class accuracy

Supports:
  1. Viotech  (--hdf5s_dir)   — default
  2. RWC      (--rwc_h5_path) — optional cross-dataset
"""

import os
import sys

sys.path.insert(1, os.path.join(os.path.dirname(__file__), '..', 'utils'))

import argparse
import numpy as np
import torch

from models_contrast import Regress_onset_offset_frame_velocity_CRNN
from pytorch_utils import move_data_to_device
from data_generator import CustomDataset, CustomTestSampler, collate_fn
from technique_label_utils import get_technique_labels
import config

_label_cfg = get_technique_labels()
TONAL_NAMES = _label_cfg.tonal_names
ARTIC_NAMES = _label_cfg.artic_names
LEGATO_NAMES = _label_cfg.legato_names


# ── Evaluation loop ───────────────────────────────────────────────────────────

def _run_eval_loop(model, test_loader, device, dataset_label='dataset'):
    """Frame-level evaluation loop for base VioTech model."""

    all_tonal_pred, all_tonal_gt = [], []
    all_artic_pred, all_artic_gt = [], []
    all_legato_pred, all_legato_gt = [], []
    total_frames = 0

    with torch.no_grad():
        for batch_idx, batch_data_dict in enumerate(test_loader):
            for key in batch_data_dict.keys():
                batch_data_dict[key] = move_data_to_device(
                    batch_data_dict[key], device
                )

            output_dict = model(batch_data_dict['waveform'])

            if 'tonal_technique_output' not in output_dict:
                continue

            # Active-frame mask: frames where GT has note activity
            if 'frame_roll' in batch_data_dict:
                active = batch_data_dict['frame_roll'].sum(dim=-1) > 0  # (B, T)
            else:
                B, T = output_dict['tonal_technique_output'].shape[:2]
                active = torch.ones(B, T, dtype=torch.bool, device=device)

            tonal_logits = output_dict['tonal_technique_output']    # (B, T, 4)
            artic_logits = output_dict['articulation_output']       # (B, T, 4)
            legato_prob = output_dict['legato_output'].squeeze(-1)  # (B, T)

            T_pred = tonal_logits.shape[1]
            T_active = active.shape[1]
            T = min(T_pred, T_active)

            tonal_pred = tonal_logits[:, :T].argmax(dim=-1)
            artic_pred = artic_logits[:, :T].argmax(dim=-1)
            legato_pred = (legato_prob[:, :T] > 0.5).long()
            active = active[:, :T]

            # GT
            tonal_gt = batch_data_dict.get('tonal_technique')
            artic_gt = batch_data_dict.get('articulation')
            legato_gt = batch_data_dict.get('legato')

            if tonal_gt is None:
                continue

            tonal_gt = tonal_gt[:, :T]
            artic_gt = artic_gt[:, :T]
            legato_gt = legato_gt[:, :T]

            B = active.shape[0]
            for b in range(B):
                m = active[b].cpu().numpy().astype(bool)
                n_active = m.sum()
                if n_active == 0:
                    continue

                all_tonal_pred.extend(tonal_pred[b].cpu().numpy()[m])
                all_tonal_gt.extend(tonal_gt[b].cpu().numpy()[m])
                all_artic_pred.extend(artic_pred[b].cpu().numpy()[m])
                all_artic_gt.extend(artic_gt[b].cpu().numpy()[m])
                all_legato_pred.extend(legato_pred[b].cpu().numpy()[m])
                all_legato_gt.extend(legato_gt[b].cpu().numpy()[m])
                total_frames += n_active

            if (batch_idx + 1) % 20 == 0:
                print(
                    f'  [{dataset_label}] batch {batch_idx + 1}: '
                    f'{total_frames} active frames accumulated'
                )

    return (
        np.array(all_tonal_gt, dtype=int),
        np.array(all_tonal_pred, dtype=int),
        np.array(all_artic_gt, dtype=int),
        np.array(all_artic_pred, dtype=int),
        np.array(all_legato_gt, dtype=int),
        np.array(all_legato_pred, dtype=int),
        total_frames,
    )


# ── Reporting ─────────────────────────────────────────────────────────────────

def _print_results(banner, tonal_gt, tonal_pred, artic_gt, artic_pred,
                   legato_gt, legato_pred, total_frames):
    print()
    print('=' * 70)
    print(f'  {banner} — Total active frames evaluated: {total_frames}')
    print('=' * 70)
    print()

    _print_multiclass_report('Tonal Technique', tonal_gt, tonal_pred, TONAL_NAMES)
    _print_multiclass_report('Articulation', artic_gt, artic_pred, ARTIC_NAMES)
    _print_multiclass_report('Legato', legato_gt, legato_pred, LEGATO_NAMES)


def _print_multiclass_report(title, gt, pred, class_names):
    from sklearn.metrics import (classification_report, confusion_matrix,
                                 accuracy_score, f1_score)

    if len(gt) == 0:
        print(f'  {title}: no samples')
        return

    present_classes = sorted(set(gt) | set(pred))
    target_names = [class_names.get(c, f'class_{c}') for c in present_classes]

    acc = accuracy_score(gt, pred)
    macro_f1 = f1_score(gt, pred, labels=present_classes,
                        average='macro', zero_division=0)
    weighted_f1 = f1_score(gt, pred, labels=present_classes,
                           average='weighted', zero_division=0)

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


# ── Main ──────────────────────────────────────────────────────────────────────

def evaluate(args):
    device = (
        torch.device(f'cuda:{args.device}')
        if args.device >= 0 and torch.cuda.is_available()
        else torch.device('cpu')
    )

    fps = config.frames_per_second
    segment_seconds = config.segment_seconds

    # Build model — base VioTech: predict_technique=True
    model = Regress_onset_offset_frame_velocity_CRNN(
        frames_per_second=fps,
        classes_num=config.classes_num,
        output_features=False,
        predict_technique=True,
    )

    checkpoint = torch.load(args.checkpoint_path, map_location='cpu',
                            weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.to(device)
    model.eval()
    print(f'Model loaded from {args.checkpoint_path}')

    # ── Viotech evaluation ────────────────────────────────────────────────

    dataset = CustomDataset(
        hdf5s_dir=args.hdf5s_dir,
        segment_seconds=segment_seconds,
        frames_per_second=fps,
        max_note_shift=0,
        augmentor=None,
        include_technique_label=True,
    )

    test_sampler = CustomTestSampler(
        hdf5s_dir=args.hdf5s_dir,
        split=args.split,
        segment_seconds=segment_seconds,
        hop_seconds=config.hop_seconds,
        batch_size=args.batch_size,
        mini_data=False,
    )
    max_possible = len(test_sampler.segment_list) // max(args.batch_size, 1)
    test_sampler.max_evaluate_iteration = min(args.max_iterations, max_possible)

    print(f'[Viotech] Test segments: {len(test_sampler.segment_list)}, '
          f'max batches: {test_sampler.max_evaluate_iteration}')

    test_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_sampler=test_sampler,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f'Evaluating on {args.split} set from {args.hdf5s_dir}')
    print(f'Device: {device}  |  batch_size: {args.batch_size}')
    print()

    results = _run_eval_loop(model, test_loader, device, dataset_label='Viotech')
    if results[-1] > 0:
        _print_results('Viotech Base Technique (end-to-end)', *results)
    else:
        print('No active frames found in the viotech test set.')

    # ── RWC evaluation (optional) ─────────────────────────────────────────

    rwc_path = getattr(args, 'rwc_h5_path', None)
    if rwc_path:
        rwc_path = os.path.expanduser(rwc_path)
        if not os.path.isfile(rwc_path):
            print(f'\n[RWC] H5 file not found: {rwc_path}, skipping.')
            return

        from rwc_moe_utils import RWCMoEDataset, RWCMoETestSampler

        print(f'\n{"=" * 70}')
        print(f'  RWC Evaluation (cross-dataset, base VioTech)')
        print(f'{"=" * 70}')

        rwc_dataset = RWCMoEDataset(
            rwc_h5_path=rwc_path,
            segment_seconds=segment_seconds,
            frames_per_second=fps,
        )
        rwc_sampler = RWCMoETestSampler(
            rwc_h5_path=rwc_path,
            split=args.rwc_split,
            segment_seconds=segment_seconds,
            hop_seconds=config.hop_seconds,
            batch_size=args.batch_size,
            fold_id=args.rwc_fold,
        )
        max_rwc = len(rwc_sampler.segment_list) // max(args.batch_size, 1)
        rwc_sampler.max_evaluate_iteration = min(args.max_iterations, max_rwc)

        print(f'[RWC] Split: {args.rwc_split}, fold: {args.rwc_fold}, '
              f'segments: {len(rwc_sampler.segment_list)}, '
              f'max batches: {rwc_sampler.max_evaluate_iteration}')

        rwc_loader = torch.utils.data.DataLoader(
            dataset=rwc_dataset,
            batch_sampler=rwc_sampler,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        rwc_results = _run_eval_loop(model, rwc_loader, device, dataset_label='RWC')
        if rwc_results[-1] > 0:
            _print_results('RWC Base Technique (cross-dataset)', *rwc_results)
        else:
            print('[RWC] No active frames found.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate base VioTech frame-level technique classification')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--hdf5s_dir', type=str, required=True,
                        help='Path to viotech HDF5s directory')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'validation', 'test'])
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_iterations', type=int, default=500)
    parser.add_argument('--rwc_h5_path', type=str, default=None,
                        help='Path to RWC processed H5 (optional)')
    parser.add_argument('--rwc_split', type=str, default='test',
                        choices=['train', 'test'])
    parser.add_argument('--rwc_fold', type=int, default=0,
                        help='RWC 3-fold CV fold index (0/1/2)')
    args = parser.parse_args()
    evaluate(args)
