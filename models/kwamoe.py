"""Keyframe-consistent sparse MoE with configurable ablation paths."""

from typing import Dict, Tuple

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
    """Route each frame to one expert or to the keyframe expert budget."""

    def __init__(self, hidden_dim: int, num_experts: int, top_k: int,
                 hidden_ratio: int, budget_mode: str = "adaptive"):
        super().__init__()
        if top_k < 1 or top_k > num_experts:
            raise ValueError("top_k must be in [1, num_experts]")
        if budget_mode not in {"adaptive", "uniform", "matched"}:
            raise ValueError(f"Unknown expert budget mode: {budget_mode}")
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.budget_mode = budget_mode
        self.experts = nn.ModuleList([
            ExpertFFN(hidden_dim, hidden_ratio) for _ in range(num_experts)
        ])
        self.router = nn.Linear(hidden_dim * 3, num_experts)

    def forward(
        self,
        h: torch.Tensor,
        frame_features: torch.Tensor,
        global_context: torch.Tensor,
        memory_global: torch.Tensor,
        hard_mask: torch.Tensor,
        num_views: int,
        keyframe_fraction: float = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        B, N, D = h.shape
        T = frame_features.shape[1]
        tokens_per_frame = num_views * (N // (num_views * T))
        if N != T * tokens_per_frame:
            raise ValueError("Token layout is incompatible with frame features")

        router_input = torch.cat([
            global_context.unsqueeze(1).expand(-1, T, -1),
            frame_features,
            memory_global.unsqueeze(1).expand(-1, T, -1),
        ], dim=-1)
        frame_probs = F.softmax(self.router(router_input), dim=-1)
        spatial_tokens = N // (num_views * T)
        routing_probs = (
            frame_probs.unsqueeze(1).unsqueeze(3)
            .expand(-1, num_views, -1, spatial_tokens, -1)
            .reshape(B, N, self.num_experts)
        )
        if self.budget_mode == "uniform":
            budgets = torch.full_like(hard_mask.long(), self.top_k)
        else:
            budgets = 1 + (self.top_k - 1) * hard_mask.long()
        token_budgets = (
            budgets.unsqueeze(1).unsqueeze(3)
            .expand(-1, num_views, -1, spatial_tokens)
            .reshape(B, N)
        )
        if self.budget_mode == "matched":
            if keyframe_fraction is None:
                selected_fraction = hard_mask.mean(dim=1, keepdim=True)
            else:
                selected_fraction = hard_mask.new_full((B, 1), keyframe_fraction)
            extra_expert_tokens = (
                torch.arange(spatial_tokens, device=h.device).float()
                / spatial_tokens < selected_fraction
            )
            token_budgets = (
                1 + (self.top_k - 1) * extra_expert_tokens.long()
            ).unsqueeze(1).unsqueeze(2).expand(-1, num_views, T, -1).reshape(B, N)

        topk_weights, topk_indices = torch.topk(routing_probs, self.top_k, dim=-1)
        ranks = torch.arange(self.top_k, device=h.device).view(1, 1, -1)
        valid = ranks < token_budgets.unsqueeze(-1)
        topk_weights = topk_weights * valid.to(topk_weights.dtype)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-8)

        h_out = torch.zeros_like(h)
        expert_usage = h.new_zeros(self.num_experts)
        for e in range(self.num_experts):
            expert_mask = (topk_indices == e) & valid
            token_mask = expert_mask.any(dim=-1)
            if token_mask.any():
                weights = topk_weights[expert_mask].unsqueeze(-1)
                expert_out = self.experts[e](h[token_mask])
                h_out[token_mask] = h_out[token_mask] + weights * expert_out
                expert_usage[e] = topk_weights[expert_mask].sum()

        expert_calls = valid.sum()
        expert_usage = expert_usage / expert_calls.to(expert_usage.dtype).clamp_min(1)
        frame_token_budgets = token_budgets.reshape(
            B, num_views, T, spatial_tokens
        ).float().mean(dim=(1, 3))
        key_tokens = (
            hard_mask.bool().unsqueeze(1).unsqueeze(3)
            .expand(-1, num_views, -1, spatial_tokens).reshape(B, N)
        )
        key_budgets = frame_token_budgets[hard_mask.bool()].float()
        nonkey_budgets = frame_token_budgets[~hard_mask.bool()].float()
        stats = {
            "expert_calls": expert_calls.to(h.dtype),
            "keyframe_expert_calls": (valid & key_tokens.unsqueeze(-1)).sum().to(h.dtype),
            "nonkeyframe_expert_calls": (valid & ~key_tokens.unsqueeze(-1)).sum().to(h.dtype),
            "mean_keyframe_budget": key_budgets.mean() if key_budgets.numel() else h.new_tensor(0.0),
            "mean_nonkeyframe_budget": nonkey_budgets.mean() if nonkey_budgets.numel() else h.new_tensor(0.0),
            "tokens_per_frame": h.new_tensor(float(tokens_per_frame)),
        }
        return h_out, expert_usage, stats


class KeyframeGate(nn.Module):
    """Derive mask and weights from one score, with explicit ablation modes."""

    def __init__(self, hidden_dim: int, max_frames: int, keyframe_ratio: float,
                 weight_mode: str = "masked"):
        super().__init__()
        if not 0 < keyframe_ratio <= 1:
            raise ValueError("keyframe_ratio must be in (0, 1]")
        if weight_mode not in {"masked", "independent", "unmasked", "shuffled"}:
            raise ValueError(f"Unknown keyframe weight mode: {weight_mode}")
        self.max_frames = max_frames
        self.keyframe_ratio = keyframe_ratio
        self.weight_mode = weight_mode
        self.top_k = max(1, int(torch.ceil(torch.tensor(max_frames * keyframe_ratio)).item()))
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

    def forward(self, z_mid: torch.Tensor):
        B, V, T, D = z_mid.shape
        frame_feats = z_mid.mean(dim=1)
        attn_out, _ = self.temporal_attn(
            frame_feats, frame_feats, frame_feats, need_weights=False
        )
        frame_feats = frame_feats + attn_out
        saliency = self.saliency_mlp(frame_feats).squeeze(-1)
        probabilities = F.softmax(saliency, dim=-1)

        k = min(self.top_k, T)
        _, topk_indices = torch.topk(saliency, k, dim=-1)
        hard_mask = torch.zeros_like(saliency)
        hard_mask.scatter_(1, topk_indices, 1.0)
        gate_probs = torch.sigmoid(saliency)
        gate_values = hard_mask + gate_probs - gate_probs.detach()

        if self.weight_mode == "independent":
            frame_weights = F.softmax(
                self.weight_net(frame_feats).squeeze(-1), dim=-1
            )
        elif self.weight_mode == "unmasked":
            frame_weights = probabilities
        else:
            weighted = gate_values * probabilities
            frame_weights = weighted / (weighted.sum(dim=-1, keepdim=True) + 1e-8)
            if self.weight_mode == "shuffled":
                shuffled = []
                for batch_index in range(B):
                    order = torch.randperm(T, device=z_mid.device)
                    shuffled.append(frame_weights[batch_index, order])
                frame_weights = torch.stack(shuffled)
        return frame_weights, gate_values, hard_mask, probabilities, saliency
