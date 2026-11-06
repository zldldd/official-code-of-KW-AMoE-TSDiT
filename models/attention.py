"""Multi-head self-attention with spatiotemporal relative position bias."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal position embedding for diffusion timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class AdaLN(nn.Module):
    """Adaptive Layer Normalization with zero-initialized modulation (adaLN-Zero)."""

    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, hidden_dim * 2),
        )
        # Zero-initialize for stable training
        nn.init.zeros_(self.mlp[1].weight)
        nn.init.zeros_(self.mlp[1].bias)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h_norm = self.norm(h)
        gamma, beta = self.mlp(cond).chunk(2, dim=-1)
        for _ in range(h_norm.dim() - gamma.dim()):
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        return gamma * h_norm + beta


class MultiHeadSelfAttention(nn.Module):
    """Global multi-head self-attention with memory-efficient bucketed
    spatiotemporal relative position bias."""

    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=False)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

        # Memory-friendly bucketed relative position bias: (heads, 2*buckets+1)
        self.rel_pos_buckets = 64
        self.rel_pos_table = nn.Parameter(
            torch.zeros(num_heads, 2 * self.rel_pos_buckets + 1)
        )
        nn.init.trunc_normal_(self.rel_pos_table, std=0.02)

    def _get_rel_pos_bias(self, N: int, device: torch.device) -> torch.Tensor:
        """Compute bucketed relative position bias on-the-fly: (heads, N, N)."""
        idx = torch.arange(N, device=device)
        rel_idx = idx.unsqueeze(1) - idx.unsqueeze(0)
        num_buckets = self.rel_pos_buckets

        buckets = torch.zeros_like(rel_idx, dtype=torch.long)
        is_small = torch.abs(rel_idx) < num_buckets // 2
        buckets[is_small] = rel_idx[is_small] + num_buckets

        large_rel = rel_idx[~is_small]
        sign = torch.sign(large_rel)
        half = num_buckets // 2
        buckets[~is_small] = (
            half * torch.log(torch.abs(large_rel).float() / half + 1).long() + half
        ) * sign + num_buckets
        buckets = torch.clamp(buckets, 0, 2 * num_buckets)

        return self.rel_pos_table[:, buckets]

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B, N, D = h.shape
        qkv = self.qkv(h).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        rel_bias = self._get_rel_pos_bias(N, h.device).unsqueeze(0)
        attn = attn + rel_bias
        attn = F.softmax(attn, dim=-1)

        h_out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(h_out)
