"""Temporal Stable Memory (TS-Mem) module.

Maintains an exponential moving average memory buffer across diffusion
iterations, driven by keyframe-gated writes and residual injection for
non-keyframe timesteps to suppress temporal drift.
"""

from typing import Optional
import torch
import torch.nn as nn


class TemporalStableMemory(nn.Module):
    """Temporal Stable Memory with EMA update, keyframe-driven write,
    and non-keyframe residual injection.

    Args:
        hidden_dim: feature dimension
        alpha: EMA decay rate (default 0.8)
        beta: residual injection strength (default 0.3)
    """

    def __init__(self, hidden_dim: int, alpha: float = 0.8, beta: float = 0.3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.write_net = nn.Linear(hidden_dim, hidden_dim)
        self.read_net = nn.Linear(hidden_dim, hidden_dim)
        self.register_buffer("memory_buffer", None, persistent=False)

    def reset_memory(self, init_val: Optional[torch.Tensor] = None):
        """Reset the memory buffer state."""
        self.memory_buffer = init_val

    def forward(self, z_mid: torch.Tensor, hard_mask: torch.Tensor,
                gate_probs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_mid: intermediate features (B, V, T, D)
            hard_mask: binary keyframe mask (B, T)
            gate_probs: continuous gate probabilities (B, T)
        Returns:
            mem_context: memory context for residual injection (B, V, T, D)
        """
        B, V, T, D = z_mid.shape

        # Frame-level aggregation
        frame_feats = z_mid.mean(dim=1)  # (B, T, D)

        # Initialize memory buffer if needed
        if self.memory_buffer is None or self.memory_buffer.shape != (B, T, D):
            self.memory_buffer = torch.zeros_like(frame_feats)

        # Keyframe-driven write via EMA
        write_candidate = self.write_net(frame_feats)
        gate = hard_mask.unsqueeze(-1) * self.alpha  # (B, T, 1)
        self.memory_buffer = (
            (1 - gate) * self.memory_buffer + gate * write_candidate
        )

        # Non-keyframe residual read
        read_context = self.read_net(self.memory_buffer)  # (B, T, D)
        nonkey_mask = (1 - hard_mask).unsqueeze(-1)  # (B, T, 1)
        mem_context = self.beta * nonkey_mask * read_context

        # Expand to view dimension
        mem_context = mem_context.unsqueeze(1).expand(-1, V, -1, -1)

        return mem_context
