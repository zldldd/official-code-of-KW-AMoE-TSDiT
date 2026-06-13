"""Sparse Mixture-of-Experts feed-forward network with keyframe-aware gating."""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertFFN(nn.Module):
    """Single expert: two-layer MLP with GELU activation."""

    def __init__(self, hidden_dim: int, hidden_ratio: int):
        super().__init__()
        expand_dim = hidden_dim * hidden_ratio
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, expand_dim),
            nn.GELU(),
            nn.Linear(expand_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SparseMoE(nn.Module):
    """Sparsely-gated Mixture-of-Experts feed-forward network.

    Only Top-K experts are activated per token, keeping inference FLOPs
    comparable to a dense FFN while scaling model capacity.
    """

    def __init__(self, hidden_dim: int, num_experts: int, top_k: int,
                 hidden_ratio: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([
            ExpertFFN(hidden_dim, hidden_ratio) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, D = h.shape
        gate_logits = self.gate(h)
        gate_weights = F.softmax(gate_logits, dim=-1)

        # Top-K selection
        top_k_weights, top_k_indices = torch.topk(gate_weights, self.top_k, dim=-1)
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-8)

        h_out = torch.zeros_like(h)
        for e in range(self.num_experts):
            mask = (top_k_indices == e).any(dim=-1)
            if mask.any():
                # Gather weights for this expert
                e_mask = (top_k_indices == e)
                weights = top_k_weights[e_mask].unsqueeze(-1)
                expert_out = self.experts[e](h[mask])
                h_out[mask] = h_out[mask] + weights * expert_out

        expert_usage = gate_weights.mean(dim=(0, 1))
        return h_out, expert_usage


class KeyframeGate(nn.Module):
    """Keyframe-aware temporal gating module.

    Estimates per-frame importance via temporal self-attention and content
    saliency scoring, then selects Top-K keyframes with straight-through
    gradient estimation for end-to-end training.
    """

    def __init__(self, hidden_dim: int, max_frames: int, keyframe_ratio: float):
        super().__init__()
        self.max_frames = max_frames
        self.keyframe_ratio = keyframe_ratio
        self.top_k = max(1, int(max_frames * keyframe_ratio))

        self.temporal_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=4, batch_first=True
        )
        self.saliency_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.weight_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z_mid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z_mid: intermediate features (B, V, T, D)
        Returns:
            frame_weights: keyframe weights (B, T)
            gate_values: continuous gate values (B, T)
            hard_mask: binary keyframe mask (B, T)
        """
        B, V, T, D = z_mid.shape

        # Cross-view pooling -> frame-level features
        frame_feats = z_mid.mean(dim=1)  # (B, T, D)

        # Temporal self-attention
        attn_out, _ = self.temporal_attn(frame_feats, frame_feats, frame_feats)
        frame_feats = frame_feats + attn_out

        # Content saliency score
        saliency = self.saliency_mlp(frame_feats).squeeze(-1)  # (B, T)

        # Keyframe weights
        frame_weights = self.weight_net(frame_feats).squeeze(-1)
        frame_weights = F.softmax(frame_weights, dim=-1)

        # Top-K selection
        k = min(self.top_k, T)
        _, topk_indices = torch.topk(saliency, k, dim=-1)
        hard_mask = torch.zeros_like(saliency)
        hard_mask.scatter_(1, topk_indices, 1.0)

        # Straight-through gating
        gate_probs = torch.sigmoid(saliency)
        gate_values = hard_mask + gate_probs - gate_probs.detach()

        return frame_weights, gate_values, hard_mask
