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

        for key in statistics.keys():
            statistics[key] = np.around(statistics[key], decimals=4)

        return statistics