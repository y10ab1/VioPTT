"""Mixture-of-Experts head for note-level violin technique classification.

Architecture
------------
1.  Five zone embeddings (onset / body / offset / ctx_prev / ctx_next) are
    concatenated into a single note representation  x ∈ R^{5D}.
2.  A shared gating network produces soft expert weights  g ∈ R^E  (dense
    routing; optionally top-k sparse routing).
3.  E experts independently map  x → R^{shared_dim}.
4.  The gated mixture  h = Σ_i g_i · expert_i(x)  is fed to three thin
    task heads:
      • tonal_technique  (4-class CE)
      • articulation      (4-class CE)
      • legato            (binary BCE)

Load-balance auxiliary loss encourages uniform expert utilisation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class MoEConfig:
    feat_dim: int = 512
    num_experts: int = 4
    expert_hidden: int = 256
    shared_dim: int = 128
    num_tonal_classes: int = 4
    num_artic_classes: int = 4
    top_k: int = 0          # 0 = dense (soft) gating; >0 = sparse top-k
    gate_dropout: float = 0.0
    expert_dropout: float = 0.2
    balance_loss_coeff: float = 0.01


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class Expert(nn.Module):
    """Single feed-forward expert:  input_dim -> hidden -> output_dim."""

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


class GatingNetwork(nn.Module):
    """Produces per-token expert weights from the concatenated zone vector."""

    def __init__(self, input_dim: int, num_experts: int,
                 hidden_dim: int = 128, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_experts)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = self.dropout(h)
        return self.fc2(h)  # raw logits (B, N, E)


# ---------------------------------------------------------------------------
# MoE Technique Head
# ---------------------------------------------------------------------------
class MoETechniqueHead(nn.Module):
    """Shared-backbone MoE with three task-specific heads.

    Forward returns (tonal_logits, artic_logits, legato_prob, gate_probs).
    """

    def __init__(self, cfg: MoEConfig = None):
        super().__init__()
        cfg = cfg or MoEConfig()
        self.cfg = cfg

        input_dim = 5 * cfg.feat_dim  # z_onset + z_body + z_offset + z_ctx_prev + z_ctx_next

        self.experts = nn.ModuleList([
            Expert(input_dim, cfg.expert_hidden, cfg.shared_dim,
                   dropout=cfg.expert_dropout)
            for _ in range(cfg.num_experts)
        ])
        self.gate = GatingNetwork(
            input_dim, cfg.num_experts,
            hidden_dim=cfg.expert_hidden // 2,
            dropout=cfg.gate_dropout,
        )

        self.layer_norm = nn.LayerNorm(cfg.shared_dim)

        self.tonal_head = nn.Linear(cfg.shared_dim, cfg.num_tonal_classes)
        self.artic_head = nn.Linear(cfg.shared_dim, cfg.num_artic_classes)
        self.legato_head = nn.Linear(cfg.shared_dim, 1)

    def forward(self, zone_features: dict):
        """
        Args:
            zone_features: dict with z_onset … z_ctx_next, each (B, N, D)

        Returns:
            tonal_logits:  (B, N, num_tonal)
            artic_logits:  (B, N, num_artic)
            legato_prob:   (B, N, 1)
            gate_probs:    (B, N, E)   — for load-balance loss
        """
        x = torch.cat([
            zone_features['z_onset'],
            zone_features['z_body'],
            zone_features['z_offset'],
            zone_features['z_ctx_prev'],
            zone_features['z_ctx_next'],
        ], dim=-1)  # (B, N, 5*D)

        # Gating
        gate_logits = self.gate(x)  # (B, N, E)
        if self.cfg.top_k > 0 and self.cfg.top_k < self.cfg.num_experts:
            gate_probs = self._top_k_gating(gate_logits, self.cfg.top_k)
        else:
            gate_probs = F.softmax(gate_logits, dim=-1)  # (B, N, E)

        # Expert outputs
        expert_outs = torch.stack(
            [expert(x) for expert in self.experts], dim=-2
        )  # (B, N, E, shared_dim)

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
        """Sparse top-k gating: zero out all but top-k experts per token."""
        top_vals, top_idx = logits.topk(k, dim=-1)
        mask = torch.zeros_like(logits).scatter(-1, top_idx, 1.0)
        masked_logits = logits.masked_fill(mask == 0, float('-inf'))
        return F.softmax(masked_logits, dim=-1)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def moe_note_technique_losses(output_dict, target_dict, device=None):
    """Compute note-level technique losses for the MoE head.

    Args:
        output_dict: contains
            note_tonal_logits  (B, N, C_tonal)
            note_artic_logits  (B, N, C_artic)
            note_legato_prob   (B, N, 1)
            moe_gate_probs     (B, N, E)
        target_dict: contains
            note_tonal_technique  (B, N)  int
            note_articulation     (B, N)  int
            note_legato           (B, N)  int {0,1}
            num_notes             (B,)    int

    Returns:
        (loss_tonal, loss_artic, loss_legato, loss_balance)  — each scalar
    """
    num_notes = target_dict['num_notes']  # (B,)
    B = num_notes.shape[0]
    N = target_dict['note_tonal_technique'].shape[1]
    dev = device or num_notes.device

    note_mask = (
        torch.arange(N, device=dev).unsqueeze(0) < num_notes.unsqueeze(1)
    )  # (B, N)

    def _masked_ce(logits_key, target_key):
        logits = output_dict[logits_key]        # (B, N, C)
        targets = target_dict[target_key].long()  # (B, N)
        flat_logits = logits[note_mask]            # (K, C)
        flat_targets = targets[note_mask]          # (K,)
        if flat_logits.shape[0] == 0:
            return torch.tensor(0.0, device=dev)
        return F.cross_entropy(flat_logits, flat_targets)

    loss_tonal = _masked_ce('note_tonal_logits', 'note_tonal_technique')
    loss_artic = _masked_ce('note_artic_logits', 'note_articulation')

    # Legato: binary cross-entropy
    legato_pred = output_dict['note_legato_prob'].squeeze(-1)  # (B, N)
    legato_tgt = target_dict['note_legato'].float()             # (B, N)
    pred_flat = legato_pred[note_mask]
    tgt_flat = legato_tgt[note_mask]
    if pred_flat.shape[0] > 0:
        loss_legato = F.binary_cross_entropy(pred_flat, tgt_flat)
    else:
        loss_legato = torch.tensor(0.0, device=dev)

    # Load-balance auxiliary loss
    loss_balance = _load_balance_loss(
        output_dict.get('moe_gate_probs'), note_mask, dev
    )

    return loss_tonal, loss_artic, loss_legato, loss_balance


def _load_balance_loss(gate_probs, note_mask, device):
    """Encourage uniform expert utilisation.

    balance_loss = E * Σ_i P_i²   where P_i = mean gate prob for expert i
    Minimised when P_i = 1/E  ∀ i.
    """
    if gate_probs is None:
        return torch.tensor(0.0, device=device)
    masked = gate_probs[note_mask]  # (K, E)
    if masked.shape[0] == 0:
        return torch.tensor(0.0, device=device)
    E = masked.shape[1]
    P = masked.mean(dim=0)          # (E,)
    return E * (P * P).sum()
