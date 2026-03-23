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
    spectral_proj_dim: int = 128     # spectral expert projected dim (→ SP)
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


class _DedicatedSpectralCNN(nn.Module):
    """Dedicated 4-layer ConvBlock stack for spectral technique features.

    Same proven architecture as NoteLevelTechniqueModel's conv stack, but
    outputs **per-frame** features instead of a global-pooled summary.

    All parameters are trained exclusively by the technique loss — no
    transcription gradient contaminates these weights.  This gives the
    spectral expert filters that are specifically tuned for technique-
    discriminative spectral patterns (overtone distribution, attack shape,
    spectral envelope).

    Input : logmel (B, 1, T, 229)  — BN-normalised log-mel spectrogram
    Output: (B, T, out_dim)        — per-frame spectral descriptor
    """

    def __init__(self, out_dim: int = 128, momentum: float = 0.01):
        super().__init__()
        from models_contrast import ConvBlock
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=48, momentum=momentum)
        self.conv_block2 = ConvBlock(in_channels=48, out_channels=64, momentum=momentum)
        self.conv_block3 = ConvBlock(in_channels=64, out_channels=96, momentum=momentum)
        self.conv_block4 = ConvBlock(in_channels=96, out_channels=128, momentum=momentum)
        # 128 channels × 14 freq bins (229 / 2^4) = 1792
        self.proj = nn.Linear(1792, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, logmel):
        # logmel: (B, 1, T, 229)
        x = self.conv_block1(logmel, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        # (B, 128, T, 14) → per-frame flatten + project
        x = x.transpose(1, 2).flatten(2)       # (B, T, 1792)
        return F.gelu(self.norm(self.proj(x)))  # (B, T, out_dim)


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

        # Dedicated spectral CNN — own ConvBlock stack, technique-only gradients
        self.spectral_cnn = _DedicatedSpectralCNN(out_dim=SP)

        # Projection for GRU features fed into spectral expert
        self.feat_proj = nn.Sequential(
            nn.Linear(D, P, bias=False),
            nn.LayerNorm(P),
            nn.GELU(),
        )

        # Expert input dims
        exp0_in = P + C           # short_ctx + onset_cue
        exp1_in = P + C           # note_ctx  + frame_cue
        exp2_in = P + 3 * C       # phrase_ctx + all three cues
        exp3_in = SP + P + C      # spectral_cnn + gru_feat_proj + frame_cue

        self.experts = nn.ModuleList([
            _Expert(exp0_in, cfg.expert_hidden, cfg.shared_dim, cfg.expert_dropout),
            _Expert(exp1_in, cfg.expert_hidden, cfg.shared_dim, cfg.expert_dropout),
            _Expert(exp2_in, cfg.expert_hidden, cfg.shared_dim, cfg.expert_dropout),
            _Expert(exp3_in, cfg.expert_hidden, cfg.shared_dim, cfg.expert_dropout),
        ])

        # Gate input: all 3 temporal contexts + spectral ctx + gru proj + all 3 cues
        gate_in = 3 * P + SP + P + 3 * C
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
                logmel=None, conv2d_map=None):
        """
        Args:
            features:    (B, T, D) acoustic features (GRU output) from dedicated CRNN
            onset_prob:  (B, T, 1) max-pooled onset probability
            offset_prob: (B, T, 1) max-pooled offset probability
            frame_prob:  (B, T, 1) max-pooled frame activity probability
            logmel:      (B, 1, T, 229) BN-normalised log-mel spectrogram
            conv2d_map:  (B, C_in, T, F) (unused, kept for backward compat)
        """
        # Project transcription cues
        c_onset = F.gelu(self.onset_proj(onset_prob))    # (B, T, C)
        c_offset = F.gelu(self.offset_proj(offset_prob))
        c_frame = F.gelu(self.frame_proj(frame_prob))

        # Multi-scale temporal contexts
        s_short = self.short_ctx(features)   # (B, T, P)
        s_note = self.note_ctx(features)     # (B, T, P)
        s_phrase = self.phrase_ctx(features)  # (B, T, P)

        # Dedicated spectral CNN — directly from logmel
        T_feat = s_short.shape[1]
        if logmel is not None:
            s_spec = self.spectral_cnn(logmel)  # (B, T', SP)
            T_spec = s_spec.shape[1]
            if T_spec != T_feat:
                s_spec = s_spec[:, :T_feat, :] if T_spec > T_feat else \
                    F.pad(s_spec, (0, 0, 0, T_feat - T_spec))
        else:
            s_spec = torch.zeros(features.shape[0], T_feat,
                                 self.cfg.spectral_proj_dim,
                                 device=features.device)

        # Project GRU features for spectral expert
        f_proj = self.feat_proj(features)     # (B, T, P)

        # Expert-specific inputs
        inp_0 = torch.cat([s_short, c_onset], dim=-1)
        inp_1 = torch.cat([s_note, c_frame], dim=-1)
        inp_2 = torch.cat([s_phrase, c_onset, c_offset, c_frame], dim=-1)
        inp_3 = torch.cat([s_spec, f_proj, c_frame], dim=-1)

        e0 = self.experts[0](inp_0)
        e1 = self.experts[1](inp_1)
        e2 = self.experts[2](inp_2)
        e3 = self.experts[3](inp_3)
        expert_outs = torch.stack([e0, e1, e2, e3], dim=-2)  # (B, T, 4, shared_dim)

        # Gate input: all contexts + gru proj + all cues
        gate_in = torch.cat([s_short, s_note, s_phrase, s_spec, f_proj,
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


def frame_moe_technique_losses(output_dict, target_dict, device=None, focal_gamma=0.0):
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
        if focal_gamma > 0:
            from losses import focal_cross_entropy
            return focal_cross_entropy(logits_2d, targets_1d, gamma=focal_gamma)
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
