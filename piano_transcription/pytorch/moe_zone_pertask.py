"""Per-Task Gate Zone-Specialized MoE for note-level technique classification.

Same zone-specialized expert structure as moe_zone_specialist.py:
  Expert 0 — Onset specialist:   z_onset ⊕ z_ctx_prev
  Expert 1 — Body specialist:    z_body
  Expert 2 — Offset specialist:  z_offset ⊕ z_ctx_next
  Expert 3 — Holistic expert:    all 5 zones

Key difference: **three independent gating networks** (one per task), so that
  - Articulation gate can freely route spiccato → Onset expert
  - Tonal gate can route harmonics → Body expert
  - Legato gate can route sustained → Offset/Holistic experts
without compromising each other.

Additional features: pitch embedding + log-duration (same as zone specialist).
Default: top_k=2 sparse gating, balance_coeff=0.001.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class PerTaskZoneMoEConfig:
    feat_dim: int = 512
    num_experts: int = 4
    expert_hidden: int = 256
    shared_dim: int = 128
    num_tonal_classes: int = 4
    num_artic_classes: int = 4
    top_k: int = 2
    gate_dropout: float = 0.0
    expert_dropout: float = 0.2
    balance_loss_coeff: float = 0.001
    pitch_vocab: int = 128
    pitch_dim: int = 32
    dur_dim: int = 16


class _Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class _Gate(nn.Module):
    def __init__(self, input_dim, num_experts, hidden_dim=128, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_experts)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = self.dropout(h)
        return self.fc2(h)


class PerTaskZoneMoEHead(nn.Module):
    """Zone-Specialized MoE with **per-task gating**.

    Forward returns:
        tonal_logits, artic_logits, legato_prob,
        gate_probs_tonal, gate_probs_artic, gate_probs_legato
    """

    def __init__(self, cfg: PerTaskZoneMoEConfig = None):
        super().__init__()
        cfg = cfg or PerTaskZoneMoEConfig()
        self.cfg = cfg
        D = cfg.feat_dim
        aux_dim = cfg.pitch_dim + cfg.dur_dim

        self.pitch_emb = nn.Embedding(cfg.pitch_vocab, cfg.pitch_dim)
        self.dur_proj = nn.Linear(1, cfg.dur_dim)

        proj_dim = D // 2
        self.proj_onset    = nn.Linear(D, proj_dim)
        self.proj_body     = nn.Linear(D, proj_dim)
        self.proj_offset   = nn.Linear(D, proj_dim)
        self.proj_ctx_prev = nn.Linear(D, proj_dim)
        self.proj_ctx_next = nn.Linear(D, proj_dim)

        exp0_in = 2 * proj_dim + aux_dim
        exp1_in = 1 * proj_dim + aux_dim
        exp2_in = 2 * proj_dim + aux_dim
        exp3_in = 5 * proj_dim + aux_dim

        self.experts = nn.ModuleList([
            _Expert(exp0_in, cfg.expert_hidden, cfg.shared_dim, cfg.expert_dropout),
            _Expert(exp1_in, cfg.expert_hidden, cfg.shared_dim, cfg.expert_dropout),
            _Expert(exp2_in, cfg.expert_hidden, cfg.shared_dim, cfg.expert_dropout),
            _Expert(exp3_in, cfg.expert_hidden, cfg.shared_dim, cfg.expert_dropout),
        ])

        gate_in = 5 * proj_dim + aux_dim
        gate_hidden = cfg.expert_hidden // 2

        self.gate_tonal  = _Gate(gate_in, cfg.num_experts, gate_hidden, cfg.gate_dropout)
        self.gate_artic  = _Gate(gate_in, cfg.num_experts, gate_hidden, cfg.gate_dropout)
        self.gate_legato = _Gate(gate_in, cfg.num_experts, gate_hidden, cfg.gate_dropout)

        self.ln_tonal  = nn.LayerNorm(cfg.shared_dim)
        self.ln_artic  = nn.LayerNorm(cfg.shared_dim)
        self.ln_legato = nn.LayerNorm(cfg.shared_dim)

        self.tonal_head  = nn.Linear(cfg.shared_dim, cfg.num_tonal_classes)
        self.artic_head  = nn.Linear(cfg.shared_dim, cfg.num_artic_classes)
        self.legato_head = nn.Linear(cfg.shared_dim, 1)

    def _get_gate_probs(self, gate_module, x):
        logits = gate_module(x)
        if self.cfg.top_k > 0 and self.cfg.top_k < self.cfg.num_experts:
            return self._top_k_gating(logits, self.cfg.top_k)
        return F.softmax(logits, dim=-1)

    def forward(self, zone_features: dict, pitches=None, durations=None):
        p_onset    = F.relu(self.proj_onset(zone_features['z_onset']))
        p_body     = F.relu(self.proj_body(zone_features['z_body']))
        p_offset   = F.relu(self.proj_offset(zone_features['z_offset']))
        p_ctx_prev = F.relu(self.proj_ctx_prev(zone_features['z_ctx_prev']))
        p_ctx_next = F.relu(self.proj_ctx_next(zone_features['z_ctx_next']))

        B, N, _ = p_onset.shape

        if pitches is not None:
            pitch_feat = self.pitch_emb(pitches.clamp(0, self.cfg.pitch_vocab - 1))
        else:
            pitch_feat = torch.zeros(B, N, self.cfg.pitch_dim, device=p_onset.device)
        if durations is not None:
            dur_feat = self.dur_proj(durations.unsqueeze(-1))
        else:
            dur_feat = torch.zeros(B, N, self.cfg.dur_dim, device=p_onset.device)
        aux = torch.cat([pitch_feat, dur_feat], dim=-1)

        inp_0 = torch.cat([p_onset, p_ctx_prev, aux], dim=-1)
        inp_1 = torch.cat([p_body, aux], dim=-1)
        inp_2 = torch.cat([p_offset, p_ctx_next, aux], dim=-1)
        inp_3 = torch.cat([p_onset, p_body, p_offset, p_ctx_prev, p_ctx_next, aux], dim=-1)

        e0 = self.experts[0](inp_0)
        e1 = self.experts[1](inp_1)
        e2 = self.experts[2](inp_2)
        e3 = self.experts[3](inp_3)
        expert_outs = torch.stack([e0, e1, e2, e3], dim=-2)  # (B, N, 4, shared_dim)

        gate_in = inp_3

        gp_tonal  = self._get_gate_probs(self.gate_tonal, gate_in)
        gp_artic  = self._get_gate_probs(self.gate_artic, gate_in)
        gp_legato = self._get_gate_probs(self.gate_legato, gate_in)

        def _mix(gp, ln):
            h = (gp.unsqueeze(-1) * expert_outs).sum(dim=-2)
            return F.relu(ln(h))

        tonal_logits = self.tonal_head(_mix(gp_tonal, self.ln_tonal))
        artic_logits = self.artic_head(_mix(gp_artic, self.ln_artic))
        legato_prob  = torch.sigmoid(self.legato_head(_mix(gp_legato, self.ln_legato)))

        return tonal_logits, artic_logits, legato_prob, gp_tonal, gp_artic, gp_legato

    @staticmethod
    def _top_k_gating(logits, k):
        top_vals, top_idx = logits.topk(k, dim=-1)
        mask = torch.zeros_like(logits).scatter(-1, top_idx, 1.0)
        masked_logits = logits.masked_fill(mask == 0, float('-inf'))
        return F.softmax(masked_logits, dim=-1)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def pertask_zone_moe_losses(output_dict, target_dict, device=None, focal_gamma=0.0):
    """Note-level losses for Per-Task Gate Zone MoE.

    Output keys: pt_tonal_logits, pt_artic_logits, pt_legato_prob,
                 pt_gate_tonal, pt_gate_artic, pt_gate_legato
    Target keys: note_tonal_technique, note_articulation, note_legato, num_notes
    """
    num_notes = target_dict['num_notes']
    B = num_notes.shape[0]
    N = target_dict['note_tonal_technique'].shape[1]
    dev = device or num_notes.device

    note_mask = (
        torch.arange(N, device=dev).unsqueeze(0) < num_notes.unsqueeze(1)
    )

    def _masked_ce(logits_key, target_key):
        logits = output_dict[logits_key]
        targets = target_dict[target_key].long()
        flat_logits = logits[note_mask]
        flat_targets = targets[note_mask]
        if flat_logits.shape[0] == 0:
            return torch.tensor(0.0, device=dev)
        if focal_gamma > 0:
            from losses import focal_cross_entropy
            return focal_cross_entropy(flat_logits, flat_targets, gamma=focal_gamma)
        return F.cross_entropy(flat_logits, flat_targets)

    loss_tonal = _masked_ce('pt_tonal_logits', 'note_tonal_technique')
    loss_artic = _masked_ce('pt_artic_logits', 'note_articulation')

    legato_pred = output_dict['pt_legato_prob'].squeeze(-1)
    legato_tgt = target_dict['note_legato'].float()
    pred_flat = legato_pred[note_mask]
    tgt_flat = legato_tgt[note_mask]
    if pred_flat.shape[0] > 0:
        loss_legato = F.binary_cross_entropy(pred_flat, tgt_flat)
    else:
        loss_legato = torch.tensor(0.0, device=dev)

    # Per-task balance losses (averaged)
    bal_tonal  = _load_balance_loss(output_dict.get('pt_gate_tonal'), note_mask, dev)
    bal_artic  = _load_balance_loss(output_dict.get('pt_gate_artic'), note_mask, dev)
    bal_legato = _load_balance_loss(output_dict.get('pt_gate_legato'), note_mask, dev)
    loss_balance = (bal_tonal + bal_artic + bal_legato) / 3.0

    return loss_tonal, loss_artic, loss_legato, loss_balance


def _load_balance_loss(gate_probs, note_mask, device):
    if gate_probs is None:
        return torch.tensor(0.0, device=device)
    masked = gate_probs[note_mask]
    if masked.shape[0] == 0:
        return torch.tensor(0.0, device=device)
    E = masked.shape[1]
    P = masked.mean(dim=0)
    return E * (P * P).sum()
