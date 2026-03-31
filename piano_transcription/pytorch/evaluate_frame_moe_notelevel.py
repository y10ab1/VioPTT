"""Note-level evaluation of Frame-level Multi-Scale MoE technique classification.

End-to-end: uses the model's OWN transcription branch (onset/offset/frame
regression) to detect note boundaries — NO ground-truth note boundaries at
inference.  Frame-level technique predictions are then aggregated per detected
note via **average-logits-then-argmax**.

GT technique labels are also aggregated per detected note span via majority
vote over the frame-level GT arrays.

Supports:
  1. Viotech  (--hdf5s_dir)  — default
  2. RWC      (--rwc_h5_path) — optional
"""

import os
import sys

sys.path.insert(1, os.path.join(os.path.dirname(__file__), '..', 'utils'))

import argparse
import numpy as np
import torch
from collections import Counter

from models_contrast import Regress_onset_offset_frame_velocity_CRNN
from moe_frame_multiscale import FrameMultiScaleMoEConfig
from pytorch_utils import move_data_to_device
from data_generator import CustomDataset, CustomTestSampler, collate_fn
from utilities import RegressionPostProcessor
from technique_label_utils import get_technique_labels
import config

_label_cfg = get_technique_labels()
TONAL_NAMES = _label_cfg.tonal_names
ARTIC_NAMES = _label_cfg.artic_names
# Config class IDs: 0=bow_change, 1=sustained.  Note-level GT is flipped
# (1 - class_id) so arrays use: 1=bow_change, 0=sustained — matching
# the sigmoid output convention.
LEGATO_NAMES = {0: 'sustained', 1: 'bow_change'}
EXPERT_NAMES = {0: 'Onset', 1: 'Note', 2: 'Phrase', 3: 'Spectral'}

ONSET_THRESHOLD = 0.3
OFFSET_THRESHOLD = 0.3
FRAME_THRESHOLD = 0.1
PEDAL_OFFSET_THRESHOLD = 0.2


def _detect_notes_from_output(output_dict_np, fps, classes_num):
    """Run RegressionPostProcessor on a single sample's transcription outputs.

    Args:
        output_dict_np: dict with numpy arrays (T, C) for reg_onset_output,
                        reg_offset_output, frame_output, velocity_output.
        fps: frames per second
        classes_num: number of pitch classes (88)

    Returns:
        list of (onset_frame, offset_frame, midi_note) tuples
    """
    post_processor = RegressionPostProcessor(
        frames_per_second=fps,
        classes_num=classes_num,
        onset_threshold=ONSET_THRESHOLD,
        offset_threshold=OFFSET_THRESHOLD,
        frame_threshold=FRAME_THRESHOLD,
        pedal_offset_threshold=PEDAL_OFFSET_THRESHOLD,
    )

    est_on_off_note_vels = post_processor.output_dict_to_note_pedal_arrays(
        output_dict_np)[0]

    if est_on_off_note_vels is None or len(est_on_off_note_vels) == 0:
        return []

    notes = []
    for onset_t, offset_t, midi_note, vel in est_on_off_note_vels:
        f0 = max(0, int(round(onset_t * fps)))
        f1 = max(f0 + 1, int(round(offset_t * fps)))
        notes.append((f0, f1, int(midi_note)))
    return notes


def _majority_vote(arr):
    """Return the most common element in a 1-D array."""
    if len(arr) == 0:
        return 0
    counts = Counter(arr.tolist())
    return counts.most_common(1)[0][0]


def _match_note_to_gt(det_onset, det_pitch, gt_onsets, gt_pitches, tolerance=5):
    """Match a detected note to the closest GT note by pitch + onset proximity.

    Returns the index into gt_onsets/gt_pitches, or None if no match within
    *tolerance* frames.
    """
    pitch_mask = gt_pitches == det_pitch
    if not pitch_mask.any():
        return None
    dists = np.abs(gt_onsets.astype(np.int64) - int(det_onset))
    dists[~pitch_mask] = 999999
    best = int(dists.argmin())
    if dists[best] <= tolerance:
        return best
    return None


def _run_note_eval_loop(model, test_loader, device, fps, classes_num,
                        dataset_label='dataset'):
    """Note-level evaluation loop using model-predicted note boundaries."""
    all_tonal_pred, all_tonal_gt = [], []
    all_artic_pred, all_artic_gt = [], []
    all_legato_pred, all_legato_gt = [], []
    total_notes = 0
    legato_matched = 0
    legato_unmatched = 0

    gate_global = {'tonal': None, 'artic': None, 'legato': None}
    gate_by_class = {'tonal': {}, 'artic': {}, 'legato': {}}

    with torch.no_grad():
        for batch_idx, batch_data_dict in enumerate(test_loader):
            for key in batch_data_dict.keys():
                batch_data_dict[key] = move_data_to_device(
                    batch_data_dict[key], device)

            output_dict = model(batch_data_dict['waveform'])

            if 'fmoe_tonal_logits' not in output_dict:
                continue

            tonal_logits = output_dict['fmoe_tonal_logits']
            artic_logits = output_dict['fmoe_artic_logits']
            legato_prob = output_dict['fmoe_legato_prob'].squeeze(-1)
            T_fmoe = tonal_logits.shape[1]

            tonal_gt_frames = batch_data_dict.get('tonal_technique')
            artic_gt_frames = batch_data_dict.get('articulation')
            if tonal_gt_frames is None:
                continue

            # Note-level GT for legato (directly from annotation)
            gt_note_onsets = batch_data_dict.get('note_onset_frames')
            gt_note_pitches = batch_data_dict.get('note_pitches')
            gt_note_legato = batch_data_dict.get('note_legato')
            gt_num_notes = batch_data_dict.get('num_notes')
            has_note_gt = (gt_note_onsets is not None
                           and gt_note_legato is not None
                           and gt_num_notes is not None)

            gp_tonal = output_dict.get('fmoe_gate_tonal')
            gp_artic = output_dict.get('fmoe_gate_artic')
            gp_legato = output_dict.get('fmoe_gate_legato')

            B = tonal_logits.shape[0]
            for b in range(B):
                sample_output = {
                    'reg_onset_output':
                        output_dict['reg_onset_output'][b].cpu().numpy(),
                    'reg_offset_output':
                        output_dict['reg_offset_output'][b].cpu().numpy(),
                    'frame_output':
                        output_dict['frame_output'][b].cpu().numpy(),
                    'velocity_output':
                        output_dict['velocity_output'][b].cpu().numpy(),
                }

                detected_notes = _detect_notes_from_output(
                    sample_output, fps, classes_num)

                # Prepare per-sample GT note arrays for legato matching
                if has_note_gt:
                    n_gt = int(gt_num_notes[b].item())
                    b_gt_onsets = gt_note_onsets[b, :n_gt].cpu().numpy()
                    b_gt_pitches = gt_note_pitches[b, :n_gt].cpu().numpy()
                    # Original HDF5 label: 0=bow_change, 1=sustained → flip
                    b_gt_legato = 1 - gt_note_legato[b, :n_gt].cpu().numpy()
                else:
                    n_gt = 0

                T_gt = tonal_gt_frames.shape[1]
                T = min(T_fmoe, T_gt)

                for f0, f1, midi_note in detected_notes:
                    if f0 >= T or f1 <= 0:
                        continue
                    f0 = max(f0, 0)
                    f1 = min(f1, T)
                    if f1 <= f0:
                        continue

                    t_avg = tonal_logits[b, f0:f1, :].mean(dim=0)
                    a_avg = artic_logits[b, f0:f1, :].mean(dim=0)

                    # Bow_change prediction: peak near onset
                    onset_window = min(f0 + 5, f1)
                    l_peak = legato_prob[b, f0:onset_window].max().item()
                    l_pred = 1 if l_peak > 0.5 else 0

                    t_gt_maj = _majority_vote(
                        tonal_gt_frames[b, f0:f1].cpu().numpy())
                    a_gt_maj = _majority_vote(
                        artic_gt_frames[b, f0:f1].cpu().numpy())

                    # Legato GT: match to GT note and use note-level label
                    l_gt = None
                    if n_gt > 0:
                        idx = _match_note_to_gt(
                            f0, midi_note, b_gt_onsets, b_gt_pitches)
                        if idx is not None:
                            l_gt = int(b_gt_legato[idx])

                    all_tonal_pred.append(t_avg.argmax().item())
                    all_tonal_gt.append(t_gt_maj)
                    all_artic_pred.append(a_avg.argmax().item())
                    all_artic_gt.append(a_gt_maj)

                    if l_gt is not None:
                        all_legato_pred.append(l_pred)
                        all_legato_gt.append(l_gt)
                        legato_matched += 1
                    else:
                        legato_unmatched += 1
                    total_notes += 1

                    l_gt_for_gate = l_gt if l_gt is not None else l_pred
                    for task, gp, gt_val in [
                        ('tonal', gp_tonal, t_gt_maj),
                        ('artic', gp_artic, a_gt_maj),
                        ('legato', gp_legato, l_gt_for_gate),
                    ]:
                        if gp is None:
                            continue
                        g_avg = gp[b, f0:f1].mean(dim=0).cpu().numpy()
                        if gate_global[task] is None:
                            gate_global[task] = g_avg.copy()
                        else:
                            gate_global[task] += g_avg
                        s, c = gate_by_class[task].get(
                            gt_val, (np.zeros_like(g_avg), 0))
                        gate_by_class[task][gt_val] = (s + g_avg, c + 1)

            if (batch_idx + 1) % 20 == 0:
                print(f'  [{dataset_label}] batch {batch_idx + 1}: '
                      f'{total_notes} notes accumulated')

    print(f'  [{dataset_label}] Legato note matching: '
          f'{legato_matched} matched, {legato_unmatched} unmatched '
          f'({legato_matched / max(legato_matched + legato_unmatched, 1) * 100:.1f}%)')

    return (np.array(all_tonal_gt, dtype=int), np.array(all_tonal_pred, dtype=int),
            np.array(all_artic_gt, dtype=int), np.array(all_artic_pred, dtype=int),
            np.array(all_legato_gt, dtype=int), np.array(all_legato_pred, dtype=int),
            gate_global, gate_by_class, total_notes)


def _print_results(banner, tonal_gt, tonal_pred, artic_gt, artic_pred,
                   legato_gt, legato_pred, gate_global, gate_by_class,
                   total_notes):
    print()
    print('=' * 70)
    print(f'  {banner} — Total notes evaluated: {total_notes}')
    print('=' * 70)
    print()

    _print_multiclass_report('Tonal Technique', tonal_gt, tonal_pred,
                             TONAL_NAMES)
    _print_multiclass_report('Articulation', artic_gt, artic_pred,
                             ARTIC_NAMES)
    _print_multiclass_report('Legato', legato_gt, legato_pred,
                             LEGATO_NAMES)

    _print_pertask_gate('Tonal', gate_global.get('tonal'),
                        gate_by_class.get('tonal', {}), TONAL_NAMES)
    _print_pertask_gate('Articulation', gate_global.get('artic'),
                        gate_by_class.get('artic', {}), ARTIC_NAMES)
    _print_pertask_gate('Legato', gate_global.get('legato'),
                        gate_by_class.get('legato', {}), LEGATO_NAMES)


def _print_pertask_gate(task_title, global_sum, by_class, class_names):
    if global_sum is None:
        return
    num_experts = len(global_sum)
    pct = global_sum / global_sum.sum() * 100

    print('-' * 70)
    print(f'  Per-Task Gate — {task_title} (note-level avg routing weight)')
    print('-' * 70)
    for i, p in enumerate(pct):
        role = EXPERT_NAMES.get(i, f'Expert {i}')
        bar = '\u2588' * int(p / 2)
        print(f'  Expert {i} ({role:8s}): {p:5.1f}%  {bar}')
    print()

    if not by_class:
        return

    expert_hdr = ''.join(f'  E{i}({EXPERT_NAMES.get(i,"?"):>5s})'
                         for i in range(num_experts))
    print(f'  {"Class":15s}  {"Notes":>8s}{expert_hdr}')
    for cls_id in sorted(by_class.keys()):
        s, c = by_class[cls_id]
        name = class_names.get(cls_id, f'class_{cls_id}')
        if c == 0:
            continue
        avg = s / c
        pcts = avg / avg.sum() * 100
        cells = ''.join(f'  {p:6.1f}%' for p in pcts)
        print(f'  {name:15s}  {c:8d}{cells}')
    print()


def _print_multiclass_report(title, gt, pred, class_names):
    from sklearn.metrics import (classification_report, confusion_matrix,
                                 accuracy_score, f1_score)

    if len(gt) == 0:
        print(f'  {title}: no samples')
        return

    present_classes = sorted(set(gt) | set(pred))
    target_names = [class_names.get(c, f'class_{c}') for c in present_classes]

    acc = accuracy_score(gt, pred)
    macro_f1 = f1_score(gt, pred, labels=present_classes, average='macro',
                        zero_division=0)
    weighted_f1 = f1_score(gt, pred, labels=present_classes, average='weighted',
                           zero_division=0)

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


def evaluate(args):
    device = (torch.device(f'cuda:{args.device}')
              if args.device >= 0 and torch.cuda.is_available()
              else torch.device('cpu'))

    fps = config.frames_per_second
    classes_num = config.classes_num
    segment_seconds = config.segment_seconds

    fmoe_spectral = bool(getattr(args, 'fmoe_spectral_expert', 1))
    frame_moe_cfg = FrameMultiScaleMoEConfig(use_spectral_expert=fmoe_spectral)
    print(f'Frame-MoE spectral expert: {fmoe_spectral}')
    print(f'Note detection thresholds: onset={ONSET_THRESHOLD}, '
          f'offset={OFFSET_THRESHOLD}, frame={FRAME_THRESHOLD}')

    model = Regress_onset_offset_frame_velocity_CRNN(
        frames_per_second=fps,
        classes_num=classes_num,
        output_features=True,
        predict_technique=False,
        predict_technique_moe=False,
        predict_technique_moe_zone=False,
        predict_technique_moe_zone_pt=False,
        predict_technique_frame_moe=True,
        frame_moe_config=frame_moe_cfg,
    )

    checkpoint = torch.load(args.checkpoint_path, map_location='cpu',
                            weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.to(device)
    model.eval()
    print(f'Model loaded from {args.checkpoint_path}')

    # ---- Viotech evaluation ----
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
    print(f'[Viotech] Test segments: {len(test_sampler.segment_list)}, '
          f'max batches: {test_sampler.max_evaluate_iteration}')

    test_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_sampler=test_sampler,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f'Evaluating (NOTE-LEVEL, predicted boundaries) on '
          f'{args.split} set from {hdf5s_dir}')
    print(f'Aggregation: avg logits over predicted note span → argmax')
    print(f'GT per note: majority vote over frame-level GT in same span')
    print(f'Device: {device}  |  batch_size: {args.batch_size}')
    print()

    results = _run_note_eval_loop(model, test_loader, device, fps, classes_num,
                                  dataset_label='Viotech')
    if results[-1] > 0:
        _print_results('Viotech Frame-MoE NOTE-LEVEL (predicted boundaries)',
                       *results)
    else:
        print('No notes detected in the viotech test set.')

    # ---- RWC evaluation (optional) ----
    rwc_path = getattr(args, 'rwc_h5_path', None)
    if rwc_path:
        rwc_path = os.path.expanduser(rwc_path)
        if not os.path.isfile(rwc_path):
            print(f'\n[RWC] H5 file not found: {rwc_path}, skipping.')
            return

        from rwc_moe_utils import RWCMoEDataset, RWCMoETestSampler

        print(f'\n{"=" * 70}')
        print(f'  RWC Note-Level Evaluation (predicted boundaries)')
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

        rwc_loader = torch.utils.data.DataLoader(
            dataset=rwc_dataset,
            batch_sampler=rwc_sampler,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        rwc_results = _run_note_eval_loop(model, rwc_loader, device, fps,
                                          classes_num, dataset_label='RWC')
        if rwc_results[-1] > 0:
            _print_results('RWC Frame-MoE NOTE-LEVEL (cross-dataset)',
                           *rwc_results)
        else:
            print('[RWC] No notes detected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Note-level evaluation of Frame MoE technique '
                    '(predicted note boundaries, no GT boundaries)')
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--hdf5s_dir', type=str, required=True)
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'validation', 'test'])
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_iterations', type=int, default=500)
    parser.add_argument('--rwc_h5_path', type=str, default=None)
    parser.add_argument('--rwc_split', type=str, default='test',
                        choices=['train', 'test'])
    parser.add_argument('--rwc_fold', type=int, default=0)
    parser.add_argument('--fmoe_spectral_expert', type=int, default=1,
                        choices=[0, 1],
                        help='Must match training config: 1=spectral, 0=no')
    args = parser.parse_args()
    evaluate(args)
