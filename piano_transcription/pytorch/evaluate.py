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


# Default technique class names aligned with dataset one-hot order
_DEFAULT_TECHNIQUE_NAMES = ['flageolet', 'normal', 'pizzicato', 'spiccato', 'no_technique']

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

        # Technique prediction accuracy (requires dataloader to include 'technique')
        if ('technique_output' in output_dict.keys()) and ('technique' in output_dict.keys()):
            try:
                tech_probs = output_dict['technique_output']  # (N, T_pred, C)
                tech_targets = output_dict['technique']       # (N, T_tgt, C) one-hot

                # Align time dimension first
                min_T = min(tech_probs.shape[1], tech_targets.shape[1])
                tech_probs = tech_probs[:, :min_T, :]
                tech_targets = tech_targets[:, :min_T, :]

                # Predicted and target class indices
                tech_pred_full = np.argmax(tech_probs, axis=-1)           # (N, T)
                tech_true_full = np.argmax(tech_targets, axis=-1)         # (N, T)

                # If available, select only labeled positions from predictions; labels already contain only labeled rows
                if 'technique_available' in output_dict.keys():
                    avail = output_dict['technique_available'].astype(bool)
                    # Align avail time dimension
                    if avail.shape[1] != min_T:
                        avail = avail[:, :min_T]
                    # Apply mask to both prediction and target
                    tech_pred = tech_pred_full[avail]
                    tech_true = tech_true_full[avail]
                else:
                    # No availability mask, use all positions
                    tech_pred = tech_pred_full.reshape(-1)
                    tech_true = tech_true_full.reshape(-1)

                # Overall accuracy across all frames
                if tech_true.size > 0:
                    statistics['technique_acc_overall'] = (tech_pred == tech_true).mean()

                # Accuracy on active frames only (exclude silent frames)
                if 'frame_roll' in output_dict.keys():
                    frame_activity_full = (output_dict['frame_roll'].sum(axis=2) > 0)  # (N, T_fr)
                    # Align time
                    if frame_activity_full.shape[1] != min_T:
                        frame_activity_full = frame_activity_full[:, :min_T]
                    if 'technique_available' in output_dict.keys():
                        # activity where labels available
                        activity_mask = frame_activity_full & avail
                        activity_flat = activity_mask.reshape(-1)
                        if activity_flat.any():
                            pred_all = tech_pred_full.reshape(-1)
                            true_all = tech_true_full.reshape(-1)
                            statistics['technique_acc_active_frame'] = (pred_all[activity_flat] == true_all[activity_flat]).mean()
                    else:
                        activity_flat = frame_activity_full.reshape(-1)
                        if activity_flat.any():
                            statistics['technique_acc_active_frame'] = (tech_pred[activity_flat] == tech_true[activity_flat]).mean()

                # Per-class accuracy (overall and active-only) with human-friendly names
                num_classes = tech_targets.shape[-1]
                if num_classes == len(_DEFAULT_TECHNIQUE_NAMES):
                    class_names = _DEFAULT_TECHNIQUE_NAMES
                else:
                    class_names = [f'c{c}' for c in range(num_classes)]

                for c in range(num_classes):
                    name = class_names[c]
                    cls_mask = (tech_true == c)
                    if cls_mask.any():
                        statistics[f'technique_acc_overall_{name}'] = (tech_pred[cls_mask] == tech_true[cls_mask]).mean()
                    else:
                        statistics[f'technique_acc_overall_{name}'] = 0.0

                    if 'frame_roll' in output_dict.keys():
                        if 'activity_labeled' in locals():
                            cls_active_mask = cls_mask & activity_labeled
                            if cls_active_mask.any():
                                statistics[f'technique_acc_active_frame_{name}'] = (tech_pred[cls_active_mask] == tech_true[cls_active_mask]).mean()
                            else:
                                statistics[f'technique_acc_active_frame_{name}'] = 0.0
            except Exception as e:
                # Be robust to shape issues without breaking evaluation
                logging.warning(f"Technique accuracy computation failed: {e}")

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

        for key in statistics.keys():
            statistics[key] = np.around(statistics[key], decimals=4)

        return statistics