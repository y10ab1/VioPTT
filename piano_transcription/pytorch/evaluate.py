import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import torch
import h5py
import time
import mir_eval
import librosa
import logging
from sklearn import metrics

from pytorch_utils import forward_dataloader
from losses import tonal_technique_loss, articulation_loss, legato_loss
from moe_frame_multiscale import frame_moe_technique_losses

def mae(target, output, mask):
    if mask is None:
        return np.mean(np.abs(target - output))
    else:
        target *= mask
        output *= mask
        return np.sum(np.abs(target - output)) / np.clip(np.sum(mask), 1e-8, np.inf)


class SegmentEvaluator(object):
    def __init__(self, model, batch_size):
        """Evaluate segment-wise metrics.

        Args:
          model: object
          batch_size: int
        """
        self.model = model
        self.batch_size = batch_size

    def evaluate(self, dataloader):
        """Evaluate over a few mini-batches.

        Args:
          dataloader: object, used to generate mini-batches for evaluation.

        Returns:
          statistics: dict, e.g. {
            'frame_f1': 0.800, 
            (if exist) 'onset_f1': 0.500, 
            (if exist) 'offset_f1': 0.300, 
            ...}
        """

        statistics = {}
        output_dict = forward_dataloader(self.model, dataloader, self.batch_size)
        
        # Helper to align (N, T, C) along T
        def _align_time(a, b):
            T = min(a.shape[1], b.shape[1])
            return a[:, :T, ...], b[:, :T, ...]

        # Frame and onset evaluation (only if both prediction and target exist)
        if ('frame_output' in output_dict.keys()) and ('frame_roll' in output_dict.keys()):
            y_true, y_score = _align_time(output_dict['frame_roll'], output_dict['frame_output'])
            statistics['frame_ap'] = metrics.average_precision_score(
                y_true.flatten(), y_score.flatten(), average='macro')
        
        if ('onset_output' in output_dict.keys()) and ('onset_roll' in output_dict.keys()):
            y_true, y_score = _align_time(output_dict['onset_roll'], output_dict['onset_output'])
            statistics['onset_macro_ap'] = metrics.average_precision_score(
                y_true.flatten(), y_score.flatten(), average='macro')

        if ('offset_output' in output_dict.keys()) and ('offset_roll' in output_dict.keys()):
            y_true, y_score = _align_time(output_dict['offset_roll'], output_dict['offset_output'])
            statistics['offset_ap'] = metrics.average_precision_score(
                y_true.flatten(), y_score.flatten(), average='macro')

        if ('reg_onset_output' in output_dict.keys()) and ('reg_onset_roll' in output_dict.keys()):
            y_pred, y_true = _align_time(output_dict['reg_onset_output'], output_dict['reg_onset_roll'])
            mask = (np.sign(y_pred + y_true - 0.01) + 1) / 2
            statistics['reg_onset_mae'] = mae(y_pred, y_true, mask)

        if ('reg_offset_output' in output_dict.keys()) and ('reg_offset_roll' in output_dict.keys()):
            y_pred, y_true = _align_time(output_dict['reg_offset_output'], output_dict['reg_offset_roll'])
            mask = (np.sign(y_pred + y_true - 0.01) + 1) / 2
            statistics['reg_offset_mae'] = mae(y_pred, y_true, mask)

        if ('velocity_output' in output_dict.keys()) and ('velocity_roll' in output_dict.keys()) and ('onset_roll' in output_dict.keys()):
            # Align all to min time
            v_pred, v_true = _align_time(output_dict['velocity_output'], output_dict['velocity_roll'])
            _, on_mask = _align_time(output_dict['velocity_output'], output_dict['onset_roll'])
            statistics['velocity_mae'] = mae(v_pred, v_true / 128, on_mask)

        if 'reg_pedal_onset_output' in output_dict.keys():
            statistics['reg_pedal_onset_mae'] = mae(
                output_dict['reg_pedal_onset_roll'].flatten(), 
                output_dict['reg_pedal_onset_output'].flatten(), 
                mask=None)

        if 'reg_pedal_offset_output' in output_dict.keys():
            statistics['reg_pedal_offset_mae'] = mae(
                output_dict['reg_pedal_offset_output'].flatten(), 
                output_dict['reg_pedal_offset_roll'].flatten(), 
                mask=None)

        if 'pedal_frame_output' in output_dict.keys():
            statistics['pedal_frame_mae'] = mae(
                output_dict['pedal_frame_output'].flatten(), 
                output_dict['pedal_frame_roll'].flatten(), 
                mask=None)

        # Technique losses (viotech 3-head)
        _tech_keys_present = (
            'tonal_technique_output' in output_dict
            and 'tonal_technique' in output_dict
        )
        if _tech_keys_present:
            _to = lambda a: torch.from_numpy(a)
            _od = {k: _to(output_dict[k]) for k in output_dict if k.endswith('_output')}
            _td = {k: _to(output_dict[k]) for k in ('tonal_technique', 'articulation',
                    'legato', 'frame_roll') if k in output_dict}
            statistics['loss_tonal_technique'] = tonal_technique_loss(_od, _td).item()
            statistics['loss_articulation'] = articulation_loss(_od, _td).item()
            statistics['loss_legato'] = legato_loss(_od, _td).item()

        # Frame-level Multi-Scale MoE technique losses + per-class accuracy
        _fmoe_present = (
            'fmoe_tonal_logits' in output_dict
            and 'tonal_technique' in output_dict
        )
        if _fmoe_present:
            _to = lambda a: torch.from_numpy(a)
            _od = {k: _to(output_dict[k]) for k in output_dict
                   if isinstance(output_dict[k], np.ndarray)}
            _td = {}
            for k in ('tonal_technique', 'articulation', 'legato', 'frame_roll'):
                if k in output_dict:
                    _td[k] = _to(output_dict[k])

            fmoe_t, fmoe_a, fmoe_l, fmoe_bal = frame_moe_technique_losses(
                _od, _td, device=None)
            statistics['fmoe_loss_tonal'] = fmoe_t.item()
            statistics['fmoe_loss_artic'] = fmoe_a.item()
            statistics['fmoe_loss_legato'] = fmoe_l.item()
            statistics['fmoe_loss_balance'] = fmoe_bal.item()
            statistics['fmoe_loss_technique'] = (fmoe_t + fmoe_a + fmoe_l).item()

            # Per-class accuracy for tonal technique & articulation
            active_mask = None
            if 'frame_roll' in output_dict:
                active_mask = output_dict['frame_roll'].sum(axis=-1) > 0  # (N, T)

            TONAL_NAMES = {0: 'none', 1: 'pizzicato', 2: 'harmonics', 3: 'openstring'}
            ARTIC_NAMES = {0: 'none', 1: 'release', 2: 'staccato', 3: 'spiccato'}

            for logits_key, target_key, class_names, prefix in [
                ('fmoe_tonal_logits', 'tonal_technique', TONAL_NAMES, 'fmoe_tonal'),
                ('fmoe_artic_logits', 'articulation', ARTIC_NAMES, 'fmoe_artic'),
            ]:
                if logits_key not in output_dict or target_key not in output_dict:
                    continue
                logits = output_dict[logits_key]
                targets = output_dict[target_key].astype(np.int64)
                T = min(logits.shape[1], targets.shape[1])
                logits = logits[:, :T, :]
                targets = targets[:, :T]
                preds = logits.argmax(axis=-1)

                if active_mask is not None:
                    m = active_mask[:, :T]
                    preds_flat = preds[m]
                    tgts_flat = targets[m]
                else:
                    preds_flat = preds.flatten()
                    tgts_flat = targets.flatten()

                if len(tgts_flat) > 0:
                    statistics[f'{prefix}_acc'] = (preds_flat == tgts_flat).mean()
                    for cls_id, cls_name in class_names.items():
                        cls_mask = tgts_flat == cls_id
                        n_cls = cls_mask.sum()
                        if n_cls > 0:
                            statistics[f'{prefix}_acc_{cls_name}'] = (
                                preds_flat[cls_mask] == cls_id).mean()
                            statistics[f'{prefix}_n_{cls_name}'] = int(n_cls)

            # Legato accuracy
            if 'fmoe_legato_prob' in output_dict and 'legato' in output_dict:
                pred_l = (output_dict['fmoe_legato_prob'].squeeze(-1) > 0.5).astype(np.int64)
                tgt_l = output_dict['legato'].astype(np.int64)
                T = min(pred_l.shape[1], tgt_l.shape[1])
                pred_l = pred_l[:, :T]
                tgt_l = tgt_l[:, :T]
                if active_mask is not None:
                    m = active_mask[:, :T]
                    pred_l = pred_l[m]
                    tgt_l = tgt_l[m]
                else:
                    pred_l = pred_l.flatten()
                    tgt_l = tgt_l.flatten()
                if len(tgt_l) > 0:
                    statistics['fmoe_legato_acc'] = (pred_l == tgt_l).mean()

        for key in statistics.keys():
            statistics[key] = np.around(statistics[key], decimals=4)

        return statistics