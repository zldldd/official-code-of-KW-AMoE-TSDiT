"""Temporal Stable Memory with configurable keyframe/all/random writes."""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class TemporalStableMemory(nn.Module):
    """Propagate temporal structure across denoising iterations."""

    def __init__(self, hidden_dim: int, alpha: float = 0.8, beta: float = 0.3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.write_net = nn.Linear(hidden_dim, hidden_dim)
        self.read_net = nn.Linear(hidden_dim, hidden_dim)
        self.memory_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=4, batch_first=True
        )
        self.register_buffer("memory_buffer", None, persistent=False)

    def reset_memory(self, init_val: Optional[torch.Tensor] = None):
        self.memory_buffer = init_val

    def forward(
        self,
        z_mid: torch.Tensor,
        frame_weights: torch.Tensor,
        hard_mask: torch.Tensor,
        probabilities: Optional[torch.Tensor] = None,
        write_mode: str = "keyframe",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, V, T, D = z_mid.shape
        frame_feats = z_mid.mean(dim=1)
        if self.memory_buffer is None or self.memory_buffer.shape != (B, T, D):
            self.memory_buffer = torch.zeros_like(frame_feats)

        if write_mode == "keyframe":
            write_weights = frame_weights
        elif write_mode == "all":
            if probabilities is None:
                raise ValueError("All-frame writes require unmasked probabilities")
            write_weights = probabilities
        elif write_mode == "random":
            if probabilities is None:
                raise ValueError("Random writes require unmasked probabilities")
            random_mask = torch.zeros_like(hard_mask)
            key_count = int(hard_mask.sum(dim=-1).max().item())
            for batch_index in range(B):
                order = torch.randperm(T, device=z_mid.device)[:key_count]
                random_mask[batch_index, order] = 1.0
            write_weights = random_mask * probabilities
            write_weights = write_weights / write_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        else:
            raise ValueError(f"Unknown memory write mode: {write_mode}")

        write_scale = torch.clamp(T * write_weights, min=0.0, max=1.0)
        write_gate = self.alpha * write_scale.unsqueeze(-1)
        write_candidate = self.write_net(frame_feats)
        self.memory_buffer = (
            (1 - write_gate) * self.memory_buffer
            + write_gate * write_candidate
        )

        attn_out, _ = self.memory_attn(
            frame_feats, self.memory_buffer, self.memory_buffer, need_weights=False
        )
        read_context = self.read_net(attn_out + self.memory_buffer)
        nonkey_mask = (1 - hard_mask).unsqueeze(-1)
        mem_context = self.beta * nonkey_mask * read_context
        memory_global = (frame_weights.unsqueeze(-1) * read_context).sum(dim=1)
        mem_context = mem_context.unsqueeze(1).expand(-1, V, -1, -1)
        return mem_context, memory_global
