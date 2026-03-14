"""Zone-Specialized Mixture-of-Experts for note-level technique classification.

Unlike the vanilla MoE where every expert sees the same concatenated 5-zone
input, here each expert is **structurally constrained** to see only a subset
of the note zones, forcing functional specialization:

  Expert 0 — Onset specialist:   z_onset ⊕ z_ctx_prev      (attack + left ctx)
  Expert 1 — Body specialist:    z_body                      (sustain / vibrato)
  Expert 2 — Offset specialist:  z_offset ⊕ z_ctx_next      (release + right ctx)
  Expert 3 — Holistic expert:    all 5 zones concatenated    (global view)

Additional features:
  - Pitch embedding  (nn.Embedding, 128 MIDI pitches → pitch_dim)
  - Log-duration scalar  (nn.Linear 1 → dur_dim)
  These are appended to each expert's input, giving every expert access to
  note identity and length information.

The gating network always sees the full concatenated input (5*D + pitch + dur)
so it can make an informed routing decision.

Load-balance auxiliary loss is identical to the vanilla version.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class ZoneMoEConfig:
    feat_dim: int = 512
    num_experts: int = 4
    expert_hidden: int = 256
    shared_dim: int = 128
    num_tonal_classes: int = 4
    num_artic_classes: int = 4
    top_k: int = 0
    gate_dropout: float = 0.0
    expert_dropout: float = 0.2
    balance_loss_coeff: float = 0.01
    pitch_vocab: int = 128
    pitch_dim: int = 32
    dur_dim: int = 16


class ZoneExpert(nn.Module):
    """Feed-forward expert that receives a zone-specific input slice."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class ZoneGatingNetwork(nn.Module):
    """Gating network operating on the full note representation."""

    def __init__(self, input_dim: int, num_experts: int,
                 hidden_dim: int = 128, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_experts)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = self.dropout(h)
        return self.fc2(h)


class ZoneMoETechniqueHead(nn.Module):
    """Zone-Specialized MoE with pitch/duration features.

    Forward returns (tonal_logits, artic_logits, legato_prob, gate_probs).
    """

    def __init__(self, cfg: ZoneMoEConfig = None):
        super().__init__()
        cfg = cfg or ZoneMoEConfig()
        self.cfg = cfg
        D = cfg.feat_dim
        aux_dim = cfg.pitch_dim + cfg.dur_dim

        self.pitch_emb = nn.Embedding(cfg.pitch_vocab, cfg.pitch_dim)
        self.dur_proj = nn.Linear(1, cfg.dur_dim)

        # Zone-specific projection layers (reduce each zone from D to a smaller dim)
        proj_dim = D // 2
        self.proj_onset    = nn.Linear(D, proj_dim)
        self.proj_body     = nn.Linear(D, proj_dim)
        self.proj_offset   = nn.Linear(D, proj_dim)
        self.proj_ctx_prev = nn.Linear(D, proj_dim)
        self.proj_ctx_next = nn.Linear(D, proj_dim)

        # Expert input dimensions (after projection + aux features)
        exp0_in = 2 * proj_dim + aux_dim   # onset + ctx_prev
        exp1_in = 1 * proj_dim + aux_dim   # body
        exp2_in = 2 * proj_dim + aux_dim   # offset + ctx_next
        exp3_in = 5 * proj_dim + aux_dim   # all zones

        self.experts = nn.ModuleList([
            ZoneExpert(exp0_in, cfg.expert_hidden, cfg.shared_dim, cfg.expert_dropout),
            ZoneExpert(exp1_in, cfg.expert_hidden, cfg.shared_dim, cfg.expert_dropout),
            ZoneExpert(exp2_in, cfg.expert_hidden, cfg.shared_dim, cfg.expert_dropout),
            ZoneExpert(exp3_in, cfg.expert_hidden, cfg.shared_dim, cfg.expert_dropout),
        ])

        gate_in = 5 * proj_dim + aux_dim
        self.gate = ZoneGatingNetwork(
            gate_in, cfg.num_experts,
            hidden_dim=cfg.expert_hidden // 2,
            dropout=cfg.gate_dropout,
        )

        self.layer_norm = nn.LayerNorm(cfg.shared_dim)

        self.tonal_head = nn.Linear(cfg.shared_dim, cfg.num_tonal_classes)
        self.artic_head = nn.Linear(cfg.shared_dim, cfg.num_artic_classes)
        self.legato_head = nn.Linear(cfg.shared_dim, 1)

    def forward(self, zone_features: dict, pitches: torch.Tensor = None,
                durations: torch.Tensor = None):
        """
        Args:
            zone_features: dict with z_onset … z_ctx_next, each (B, N, D)
            pitches:       (B, N) int MIDI pitch per note (0-127)
            durations:     (B, N) float log-duration per note

        Returns:
            tonal_logits, artic_logits, legato_prob, gate_probs
        """
        # Project each zone to lower dimension
        p_onset    = F.relu(self.proj_onset(zone_features['z_onset']))
        p_body     = F.relu(self.proj_body(zone_features['z_body']))
        p_offset   = F.relu(self.proj_offset(zone_features['z_offset']))
        p_ctx_prev = F.relu(self.proj_ctx_prev(zone_features['z_ctx_prev']))
        p_ctx_next = F.relu(self.proj_ctx_next(zone_features['z_ctx_next']))

        B, N, _ = p_onset.shape

        # Auxiliary features
        if pitches is not None:
            pitch_feat = self.pitch_emb(pitches.clamp(0, self.cfg.pitch_vocab - 1))
        else:
            pitch_feat = torch.zeros(B, N, self.cfg.pitch_dim,
                                     device=p_onset.device)
        if durations is not None:
            dur_feat = self.dur_proj(durations.unsqueeze(-1))
        else:
            dur_feat = torch.zeros(B, N, self.cfg.dur_dim,
                                   device=p_onset.device)
        aux = torch.cat([pitch_feat, dur_feat], dim=-1)

        # Build expert-specific inputs
        inp_0 = torch.cat([p_onset, p_ctx_prev, aux], dim=-1)
        inp_1 = torch.cat([p_body, aux], dim=-1)
        inp_2 = torch.cat([p_offset, p_ctx_next, aux], dim=-1)
        inp_3 = torch.cat([p_onset, p_body, p_offset, p_ctx_prev, p_ctx_next, aux], dim=-1)

        # Gate input = full view
        gate_in = inp_3
        gate_logits = self.gate(gate_in)

        if self.cfg.top_k > 0 and self.cfg.top_k < self.cfg.num_experts:
            gate_probs = self._top_k_gating(gate_logits, self.cfg.top_k)
        else:
            gate_probs = F.softmax(gate_logits, dim=-1)

        # Expert outputs — each expert gets its own input slice
        e0 = self.experts[0](inp_0)
        e1 = self.experts[1](inp_1)
        e2 = self.experts[2](inp_2)
        e3 = self.experts[3](inp_3)

        expert_outs = torch.stack([e0, e1, e2, e3], dim=-2)  # (B, N, 4, shared_dim)

        # Weighted combination
        shared = (gate_probs.unsqueeze(-1) * expert_outs).sum(dim=-2)
        shared = self.layer_norm(shared)
        shared = F.relu(shared)

        tonal_logits = self.tonal_head(shared)
        artic_logits = self.artic_head(shared)
        legato_prob = torch.sigmoid(self.legato_head(shared))

        return tonal_logits, artic_logits, legato_prob, gate_probs

    @staticmethod
    def _top_k_gating(logits, k):
        top_vals, top_idx = logits.topk(k, dim=-1)
        mask = torch.zeros_like(logits).scatter(-1, top_idx, 1.0)
        masked_logits = logits.masked_fill(mask == 0, float('-inf'))
        return F.softmax(masked_logits, dim=-1)


# ---------------------------------------------------------------------------
# Losses  (same structure as vanilla MoE, reading from zone_ prefixed keys)
# ---------------------------------------------------------------------------
def zone_moe_note_technique_losses(output_dict, target_dict, device=None):
    """Compute note-level technique losses for the Zone-Specialized MoE head.

    Keys expected:
        output_dict: zone_tonal_logits, zone_artic_logits,
                     zone_legato_prob, zone_gate_probs
        target_dict: note_tonal_technique, note_articulation,
                     note_legato, num_notes
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
        return F.cross_entropy(flat_logits, flat_targets)

    loss_tonal = _masked_ce('zone_tonal_logits', 'note_tonal_technique')
    loss_artic = _masked_ce('zone_artic_logits', 'note_articulation')

    legato_pred = output_dict['zone_legato_prob'].squeeze(-1)
    legato_tgt = target_dict['note_legato'].float()
    pred_flat = legato_pred[note_mask]
    tgt_flat = legato_tgt[note_mask]
    if pred_flat.shape[0] > 0:
        loss_legato = F.binary_cross_entropy(pred_flat, tgt_flat)
    else:
        loss_legato = torch.tensor(0.0, device=dev)

    loss_balance = _load_balance_loss(
        output_dict.get('zone_gate_probs'), note_mask, dev
    )

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
