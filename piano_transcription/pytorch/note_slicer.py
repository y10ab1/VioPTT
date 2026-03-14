"""Note region extraction and mean-pooling for note-level technique classification.

Slices encoder output H: (B, T, D) into five zones per note:
  z_onset, z_body, z_offset, z_ctx_prev, z_ctx_next
each pooled to (B, N, D) via weighted mean-pooling.

All operations are fully vectorized (no Python loops over notes) and
differentiable w.r.t. H.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field


@dataclass
class NoteSlicerConfig:
    """Hyperparameters for note region slicing.

    All values are in frames.  At 100 fps (hop 10 ms):
      1 frame  = 10 ms
      5 frames = 50 ms
    """
    pre_onset: int = 1
    post_onset: int = 4
    body_margin_left: int = 3
    body_margin_right: int = 3
    pre_offset: int = 4
    post_offset: int = 1
    context_frames: int = 5
    short_note_thresh: int = 8
    min_body_frames: int = 1


class NoteSlicer(nn.Module):
    """Vectorized note region extractor with mean-pooling.

    Given encoder features H and note boundaries, produces fixed-size
    zone embeddings for each note without any Python loops over batch or
    note dimensions.
    """

    def __init__(self, config: NoteSlicerConfig = None):
        super().__init__()
        self.cfg = config or NoteSlicerConfig()

    def forward(self, H, note_onset_frames, note_offset_frames, note_mask):
        """Extract and pool zone features for every note.

        Args:
            H:                   (B, T, D)  encoder features
            note_onset_frames:   (B, N)     onset frame index per note
            note_offset_frames:  (B, N)     offset frame index (exclusive)
            note_mask:           (B, N)     True for real notes, False for padding

        Returns:
            dict  {zone_name: (B, N, D)}  for zone_name in
                  z_onset, z_body, z_offset, z_ctx_prev, z_ctx_next
        """
        B, T, D = H.shape
        device = H.device
        cfg = self.cfg

        s = note_onset_frames.long()
        e = note_offset_frames.long()
        mask_f = note_mask.float()

        frame_idx = torch.arange(T, device=device).view(1, 1, T)  # (1,1,T)
        L = (e - s).clamp(min=1)
        is_short = L < cfg.short_note_thresh  # (B, N)

        def _raw_weight(left, right):
            """Unnormalized indicator weight: 1 where left <= t < right."""
            l = left.clamp(0, T).unsqueeze(-1)   # (B, N, 1)
            r = right.clamp(0, T).unsqueeze(-1)   # (B, N, 1)
            w = ((frame_idx >= l) & (frame_idx < r)).float()  # (B, N, T)
            return w * mask_f.unsqueeze(-1)

        def _normalize(w):
            return w / w.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        def _pool(w):
            """(B, N, T) x (B, T, D) -> (B, N, D)"""
            return torch.bmm(_normalize(w), H)

        # ---- onset zone ----
        on_l = s - cfg.pre_onset
        on_r = torch.where(is_short,
                           torch.min(e, s + cfg.post_onset),
                           s + cfg.post_onset)
        w_onset = _raw_weight(on_l, on_r)
        # fallback: if onset window is empty, use single onset frame
        empty = w_onset.sum(dim=-1, keepdim=True) < 0.5
        w_onset_fb = _raw_weight(s, s + 1)
        w_onset = torch.where(empty, w_onset_fb, w_onset)

        # ---- body zone ----
        body_l_n = torch.min(e, s + cfg.body_margin_left)
        body_r_n = torch.max(body_l_n, e - cfg.body_margin_right)
        mid = (s + e) // 2
        body_l_s = mid
        body_r_s = mid + cfg.min_body_frames
        body_l = torch.where(is_short, body_l_s, body_l_n)
        body_r = torch.where(is_short, body_r_s, body_r_n)
        w_body = _raw_weight(body_l, body_r)
        body_empty = w_body.sum(dim=-1, keepdim=True) < 0.5
        w_body = torch.where(body_empty, w_onset, w_body)

        # ---- offset zone ----
        off_l = torch.where(is_short,
                            torch.max(s, e - cfg.pre_offset),
                            e - cfg.pre_offset)
        off_r = e + cfg.post_offset
        w_offset = _raw_weight(off_l, off_r)
        off_empty = w_offset.sum(dim=-1, keepdim=True) < 0.5
        w_offset = torch.where(off_empty, w_onset, w_offset)

        # ---- context prev (before note) ----
        w_ctx_prev = _raw_weight(s - cfg.context_frames, s)

        # ---- context next (after note) ----
        w_ctx_next = _raw_weight(e, e + cfg.context_frames)

        return {
            'z_onset':    _pool(w_onset),
            'z_body':     _pool(w_body),
            'z_offset':   _pool(w_offset),
            'z_ctx_prev': _pool(w_ctx_prev),
            'z_ctx_next': _pool(w_ctx_next),
        }
