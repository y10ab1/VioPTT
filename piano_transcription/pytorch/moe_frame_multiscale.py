"""Frame-level Multi-Scale MoE for technique classification.

Strictly end-to-end: no GT note boundaries at inference.
Uses the transcription branch's onset / frame-activity probabilities as
structural cues, combined with multi-scale temporal convolutions.

Expert 0 — Onset specialist (short-range ~110ms):
    Small 1D conv + onset probability injection.
    Captures attack transients (spiccato, pizzicato, bow-change).

Expert 1 — Note specialist (medium-range ~510ms):
    Medium 1D conv + frame-activity injection.
    Captures sustained note characteristics (vibrato, sustain).

Expert 2 — Phrase specialist (long-range ~1.5s):
    Large 1D conv (dilated for efficiency) + all cues.
    Captures legato transitions, phrase-level structure.

Expert 3 — Spectral specialist (frequency domain):
    2D CNN on logmel spectrogram + frame-activity cue.
    Captures harmonic energy distribution, spectral envelope differences
    (e.g. harmonics vs normal have different overtone patterns at same pitch).

Three independent per-task gates (tonal / articulation / legato), so each
task can learn its own routing pattern.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class FrameMultiScaleMoEConfig:
    feat_dim: int = 512
    proj_dim: int = 128
    num_experts: int = 4
    short_kernel: int = 11      # ~110ms at 100fps
    note_kernel: int = 51       # ~510ms
    phrase_kernel: int = 51     # dilated×3 → effective ~1.5s
    phrase_dilation: int = 3
    expert_hidden: int = 256
    shared_dim: int = 128
    num_tonal_classes: int = 4
    num_artic_classes: int = 4
    cue_dim: int = 16           # projected dim for each transcription cue
    mel_bins: int = 229         # logmel frequency bins
    spectral_channels: int = 32 # 2D CNN intermediate channels
    spectral_proj_dim: int = 128  # spectral expert projected dim (→ P)
    gate_dropout: float = 0.0
    expert_dropout: float = 0.2
    balance_loss_coeff: float = 0.001
    top_k: int = 2


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class _TemporalCtx(nn.Module):
    """1D depthwise-separable conv for temporal context extraction."""

    def __init__(self, in_dim: int, out_dim: int, kernel_size: int,
                 dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.dw = nn.Conv1d(in_dim, in_dim, kernel_size, padding=padding,
                            dilation=dilation, groups=in_dim, bias=False)
        self.pw = nn.Conv1d(in_dim, out_dim, 1, bias=True)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        # x: (B, T, D)
        h = self.dw(x.transpose(1, 2))  # (B, D, T)
        h = self.pw(h).transpose(1, 2)  # (B, T, out_dim)
        return F.gelu(self.norm(h))


class _SpectralCtx(nn.Module):
    """2D CNN on logmel spectrogram for frequency-domain feature extraction.

    Input : logmel (B, 1, T, mel_bins)  — raw log-mel spectrogram
    Output: (B, T, out_dim)             — per-frame spectral descriptor

    Architecture: two small 2D conv layers that preserve the time axis,
    then adaptive average-pool across frequency → per-frame vector.
    """

    def __init__(self, mel_bins: int, channels: int, out_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=(3, 7), padding=(1, 3)),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=(3, 7), padding=(1, 3)),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.freq_pool = nn.AdaptiveAvgPool2d((None, 1))  # pool freq → 1
        self.proj = nn.Linear(channels, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, logmel):
        # logmel: (B, 1, T, mel_bins)
        h = self.conv(logmel)          # (B, C, T, mel_bins)
        h = self.freq_pool(h)          # (B, C, T, 1)
        h = h.squeeze(-1).transpose(1, 2)  # (B, T, C)
        h = self.proj(h)              # (B, T, out_dim)
        return F.gelu(self.norm(h))


class _Expert(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class _Gate(nn.Module):
    def __init__(self, in_dim, num_experts, hidden_dim=128, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_experts)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = self.dropout(h)
        return self.fc2(h)


# ---------------------------------------------------------------------------
# Main head
# ---------------------------------------------------------------------------
class FrameMultiScaleMoEHead(nn.Module):
    """Frame-level MoE with multi-scale temporal experts + transcription cues.

    Forward returns:
        tonal_logits   (B, T, 4)
        artic_logits   (B, T, 4)
        legato_prob    (B, T, 1)
        gate_tonal     (B, T, 3)
        gate_artic     (B, T, 3)
        gate_legato    (B, T, 3)
    """

    def __init__(self, cfg: FrameMultiScaleMoEConfig = None):
        super().__init__()
        cfg = cfg or FrameMultiScaleMoEConfig()
        self.cfg = cfg
        D = cfg.feat_dim
        P = cfg.proj_dim
        C = cfg.cue_dim
        SP = cfg.spectral_proj_dim

        # Transcription cue projections (scalar → small embedding)
        self.onset_proj = nn.Linear(1, C)
        self.offset_proj = nn.Linear(1, C)
        self.frame_proj = nn.Linear(1, C)

        # Multi-scale temporal context extractors
        self.short_ctx = _TemporalCtx(D, P, cfg.short_kernel)
        self.note_ctx = _TemporalCtx(D, P, cfg.note_kernel)
        self.phrase_ctx = _TemporalCtx(D, P, cfg.phrase_kernel,
                                       dilation=cfg.phrase_dilation)

        # Spectral context extractor (2D CNN on logmel)
        self.spectral_ctx = _SpectralCtx(
            cfg.mel_bins, cfg.spectral_channels, SP)

        # Expert input dims
        exp0_in = P + C           # short_ctx + onset_cue
        exp1_in = P + C           # note_ctx  + frame_cue
        exp2_in = P + 3 * C       # phrase_ctx + all three cues
        exp3_in = SP + C          # spectral_ctx + frame_cue

        self.experts = nn.ModuleList([
            _Expert(exp0_in, cfg.expert_hidden, cfg.shared_dim, cfg.expert_dropout),
            _Expert(exp1_in, cfg.expert_hidden, cfg.shared_dim, cfg.expert_dropout),
            _Expert(exp2_in, cfg.expert_hidden, cfg.shared_dim, cfg.expert_dropout),
            _Expert(exp3_in, cfg.expert_hidden, cfg.shared_dim, cfg.expert_dropout),
        ])

        # Gate input: all 3 temporal contexts + spectral ctx + all 3 cues
        gate_in = 3 * P + SP + 3 * C
        gate_hidden = cfg.expert_hidden // 2

        self.gate_tonal = _Gate(gate_in, cfg.num_experts, gate_hidden, cfg.gate_dropout)
        self.gate_artic = _Gate(gate_in, cfg.num_experts, gate_hidden, cfg.gate_dropout)
        self.gate_legato = _Gate(gate_in, cfg.num_experts, gate_hidden, cfg.gate_dropout)

        self.ln_tonal = nn.LayerNorm(cfg.shared_dim)
        self.ln_artic = nn.LayerNorm(cfg.shared_dim)
        self.ln_legato = nn.LayerNorm(cfg.shared_dim)

        self.tonal_head = nn.Linear(cfg.shared_dim, cfg.num_tonal_classes)
        self.artic_head = nn.Linear(cfg.shared_dim, cfg.num_artic_classes)
        self.legato_head = nn.Linear(cfg.shared_dim, 1)

    def _get_gate_probs(self, gate_module, x):
        logits = gate_module(x)
        if self.cfg.top_k > 0 and self.cfg.top_k < self.cfg.num_experts:
            return self._top_k_gating(logits, self.cfg.top_k)
        return F.softmax(logits, dim=-1)

    def forward(self, features, onset_prob, offset_prob, frame_prob,
                logmel=None):
        """
        Args:
            features:    (B, T, D) acoustic features from dedicated CRNN
            onset_prob:  (B, T, 1) max-pooled onset probability
            offset_prob: (B, T, 1) max-pooled offset probability
            frame_prob:  (B, T, 1) max-pooled frame activity probability
            logmel:      (B, 1, T, mel_bins) raw log-mel spectrogram for spectral expert
        """
        # Project transcription cues
        c_onset = F.gelu(self.onset_proj(onset_prob))    # (B, T, C)
        c_offset = F.gelu(self.offset_proj(offset_prob))
        c_frame = F.gelu(self.frame_proj(frame_prob))

        # Multi-scale temporal contexts
        s_short = self.short_ctx(features)   # (B, T, P)
        s_note = self.note_ctx(features)     # (B, T, P)
        s_phrase = self.phrase_ctx(features)  # (B, T, P)

        # Spectral context from raw logmel
        if logmel is not None:
            s_spec = self.spectral_ctx(logmel)  # (B, T', SP)
            T_feat = s_short.shape[1]
            T_spec = s_spec.shape[1]
            if T_spec != T_feat:
                s_spec = s_spec[:, :T_feat, :] if T_spec > T_feat else \
                    F.pad(s_spec, (0, 0, 0, T_feat - T_spec))
        else:
            B, T_feat, _ = s_short.shape
            s_spec = torch.zeros(B, T_feat, self.cfg.spectral_proj_dim,
                                 device=features.device)

        # Expert-specific inputs
        inp_0 = torch.cat([s_short, c_onset], dim=-1)
        inp_1 = torch.cat([s_note, c_frame], dim=-1)
        inp_2 = torch.cat([s_phrase, c_onset, c_offset, c_frame], dim=-1)
        inp_3 = torch.cat([s_spec, c_frame], dim=-1)

        e0 = self.experts[0](inp_0)
        e1 = self.experts[1](inp_1)
        e2 = self.experts[2](inp_2)
        e3 = self.experts[3](inp_3)
        expert_outs = torch.stack([e0, e1, e2, e3], dim=-2)  # (B, T, 4, shared_dim)

        # Gate input: all contexts + all cues
        gate_in = torch.cat([s_short, s_note, s_phrase, s_spec,
                             c_onset, c_offset, c_frame], dim=-1)

        gp_tonal = self._get_gate_probs(self.gate_tonal, gate_in)
        gp_artic = self._get_gate_probs(self.gate_artic, gate_in)
        gp_legato = self._get_gate_probs(self.gate_legato, gate_in)

        def _mix(gp, ln):
            h = (gp.unsqueeze(-1) * expert_outs).sum(dim=-2)
            return F.relu(ln(h))

        tonal_logits = self.tonal_head(_mix(gp_tonal, self.ln_tonal))
        artic_logits = self.artic_head(_mix(gp_artic, self.ln_artic))
        legato_prob = torch.sigmoid(self.legato_head(_mix(gp_legato, self.ln_legato)))

        return tonal_logits, artic_logits, legato_prob, gp_tonal, gp_artic, gp_legato

    @staticmethod
    def _top_k_gating(logits, k):
        top_vals, top_idx = logits.topk(k, dim=-1)
        mask = torch.zeros_like(logits).scatter(-1, top_idx, 1.0)
        masked_logits = logits.masked_fill(mask == 0, float('-inf'))
        return F.softmax(masked_logits, dim=-1)


# ---------------------------------------------------------------------------
# Losses — frame-level with active-mask
# ---------------------------------------------------------------------------
EXPERT_NAMES = {0: 'Onset', 1: 'Note', 2: 'Phrase', 3: 'Spectral'}


def frame_moe_technique_losses(output_dict, target_dict, device=None):
    """Frame-level technique losses for the multi-scale MoE.

    Output keys: fmoe_tonal_logits, fmoe_artic_logits, fmoe_legato_prob,
                 fmoe_gate_tonal, fmoe_gate_artic, fmoe_gate_legato
    Target keys: tonal_technique (B, T), articulation (B, T), legato (B, T)
    """
    dev = device

    # Active-frame mask: only compute loss on frames where a note is active
    active_mask = None
    if 'frame_roll' in target_dict:
        active_mask = (target_dict['frame_roll'].sum(dim=-1) > 0)  # (B, T)

    def _frame_ce(logits_key, target_key):
        if logits_key not in output_dict or target_key not in target_dict:
            return torch.tensor(0.0, device=dev)
        logits = output_dict[logits_key]
        targets = target_dict[target_key].long()
        T_pred, T_tgt = logits.shape[1], targets.shape[1]
        T = min(T_pred, T_tgt)
        logits = logits[:, :T, :]
        targets = targets[:, :T]
        logits_2d = logits.reshape(-1, logits.shape[-1])
        targets_1d = targets.reshape(-1)
        if active_mask is not None:
            m = active_mask[:, :T].reshape(-1)
            if m.any():
                logits_2d = logits_2d[m]
                targets_1d = targets_1d[m]
            else:
                return torch.tensor(0.0, device=dev)
        return F.cross_entropy(logits_2d, targets_1d)

    loss_tonal = _frame_ce('fmoe_tonal_logits', 'tonal_technique')
    loss_artic = _frame_ce('fmoe_artic_logits', 'articulation')

    # Legato — BCE
    loss_legato = torch.tensor(0.0, device=dev)
    if 'fmoe_legato_prob' in output_dict and 'legato' in target_dict:
        pred = output_dict['fmoe_legato_prob'].squeeze(-1)
        tgt = target_dict['legato'].float()
        T = min(pred.shape[1], tgt.shape[1])
        pred = pred[:, :T]
        tgt = tgt[:, :T]
        if active_mask is not None:
            m = active_mask[:, :T]
            if m.any():
                loss_legato = F.binary_cross_entropy(pred[m], tgt[m])
        else:
            loss_legato = F.binary_cross_entropy(pred.reshape(-1), tgt.reshape(-1))

    # Per-task balance losses
    bal_tonal = _frame_balance_loss(output_dict.get('fmoe_gate_tonal'), active_mask, dev)
    bal_artic = _frame_balance_loss(output_dict.get('fmoe_gate_artic'), active_mask, dev)
    bal_legato = _frame_balance_loss(output_dict.get('fmoe_gate_legato'), active_mask, dev)
    loss_balance = (bal_tonal + bal_artic + bal_legato) / 3.0

    return loss_tonal, loss_artic, loss_legato, loss_balance


def _frame_balance_loss(gate_probs, active_mask, device):
    """Load-balance loss over active frames."""
    if gate_probs is None:
        return torch.tensor(0.0, device=device)
    if active_mask is not None:
        T = min(gate_probs.shape[1], active_mask.shape[1])
        masked = gate_probs[:, :T][active_mask[:, :T]]
    else:
        masked = gate_probs.reshape(-1, gate_probs.shape[-1])
    if masked.shape[0] == 0:
        return torch.tensor(0.0, device=device)
    E = masked.shape[1]
    P = masked.mean(dim=0)
    return E * (P * P).sum()
