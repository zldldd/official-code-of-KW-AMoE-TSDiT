"""DiT (Diffusion Transformer) Backbone with KW-AMoE and TS-Mem integration."""

from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from einops import rearrange

from .attention import SinusoidalPosEmb, AdaLN, MultiHeadSelfAttention
from .kwamoe import SparseMoE, KeyframeGate
from .tsmem import TemporalStableMemory


class DiTBlock(nn.Module):
    """Single DiT Transformer Block: adaLN -> MSA -> adaLN -> MoE-FFN."""

    def __init__(self, hidden_dim: int, num_heads: int, cond_dim: int,
                 num_experts: int, top_k: int, hidden_ratio: int):
        super().__init__()
        self.adaln1 = AdaLN(hidden_dim, cond_dim)
        self.attn = MultiHeadSelfAttention(hidden_dim, num_heads)
        self.adaln2 = AdaLN(hidden_dim, cond_dim)
        self.moe = SparseMoE(hidden_dim, num_experts, top_k, hidden_ratio)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = h + self.attn(self.adaln1(h, cond))
        h_out, expert_usage = self.moe(self.adaln2(h, cond))
        h = h + h_out
        return h, expert_usage


class DiTBackbone(nn.Module):
    """Diffusion Transformer Backbone.

    Tokenizes spatiotemporal latent features, injects timestep and view/frame
    conditioning via adaLN-Zero, and processes through L stacked DiT Blocks.

    The middle-layer features are extracted for downstream KW-AMoE and TS-Mem.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Patch projection: 3D conv (no temporal merge, spatial patchify only)
        self.patch_proj = nn.Conv3d(
            config.latent_dim,
            config.dit_hidden_dim,
            kernel_size=(1, config.patch_size, config.patch_size),
            stride=(1, config.patch_size, config.patch_size),
        )

        # Compute max tokens for position embedding
        Hp = config.image_height // config.patch_size
        Wp = config.image_width // config.patch_size
        max_tokens = config.max_frames * config.num_views * Hp * Wp
        self.pos_embed = nn.Parameter(
            torch.zeros(1, max_tokens, config.dit_hidden_dim)
        )

        # Timestep embedding
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(config.dit_hidden_dim),
            nn.Linear(config.dit_hidden_dim, config.dit_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.dit_hidden_dim, config.dit_hidden_dim),
        )

        # View and frame condition embeddings
        self.view_embed = nn.Embedding(config.num_views, config.dit_hidden_dim // 4)
        self.frame_embed = nn.Embedding(config.max_frames, config.dit_hidden_dim // 4)
        self.cond_proj = nn.Linear(config.dit_hidden_dim // 2, config.dit_hidden_dim)

        # DiT Blocks
        cond_dim = config.dit_hidden_dim * 2
        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_dim=config.dit_hidden_dim,
                num_heads=config.dit_heads,
                cond_dim=cond_dim,
                num_experts=config.num_experts,
                top_k=config.top_k,
                hidden_ratio=config.expert_hidden_ratio,
            )
            for _ in range(config.dit_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(
            config.dit_hidden_dim,
            config.latent_dim * config.patch_size * config.patch_size,
        )

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor,
                cond: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z_t: noisy latent (B, V, T, C, H, W)
            t: diffusion timestep (B,)
            cond: {'view_ids': (V,), 'frame_ids': (T,)}
        Returns:
            eps_pred: predicted noise (B, V, T, C, H, W)
            z_mid: intermediate features (B, N_total, D)
            expert_usages: mean expert usage across layers (E,)
        """
        B, V, T, C, H, W_val = z_t.shape
        Hp = self.config.image_height // self.config.patch_size
        Wp = self.config.image_width // self.config.patch_size

        # Tokenization: (B,V,T,C,H,W) -> (B,C,V*T,H,W) -> Conv3d -> (B,N,D,Hp,Wp)
        h = rearrange(z_t, "b v t c h w -> b c (v t) h w")
        h = self.patch_proj(h)  # (B, D, V*T, Hp, Wp)
        _, D, _, _, _ = h.shape
        h = rearrange(h, "b d n h w -> b n (h w) d")  # (B, V*T, Hp*Wp, D)
        h = rearrange(h, "b n s d -> b (n s) d")       # (B, N_total, D)

        # Position embedding
        h = h + self.pos_embed[:, :h.size(1), :]

        # Timestep condition
        t_emb = self.time_embed(t)  # (B, D)

        # View and frame conditions
        v_ids = cond.get("view_ids", torch.arange(V, device=z_t.device))
        f_ids = cond.get("frame_ids", torch.arange(T, device=z_t.device))
        v_emb = self.view_embed(v_ids).unsqueeze(0)  # (1, V, D//4)
        f_emb = self.frame_embed(f_ids).unsqueeze(0)  # (1, T, D//4)

        # Fuse and expand to token level
        vf_emb = torch.cat([
            v_emb.unsqueeze(2).expand(-1, -1, T, -1).reshape(1, V * T, -1),
            f_emb.unsqueeze(1).expand(-1, V, -1, -1).reshape(1, V * T, -1),
        ], dim=-1)  # (1, V*T, D//2)
        vf_emb = self.cond_proj(vf_emb)  # (1, V*T, D)
        vf_emb = vf_emb.unsqueeze(2).expand(-1, -1, Hp * Wp, -1)
        vf_emb = rearrange(vf_emb, "b n s d -> b (n s) d")
        vf_emb = vf_emb.expand(B, -1, -1)  # (B, N_total, D)

        cond_emb = torch.cat([
            t_emb.unsqueeze(1).expand(-1, h.size(1), -1), vf_emb
        ], dim=-1)  # (B, N_total, 2*D)

        # DiT Blocks
        expert_usages = []
        z_mid = None
        mid_layer = self.config.dit_layers // 2
        for i, block in enumerate(self.blocks):
            h, expert_usage = block(h, cond_emb)
            expert_usages.append(expert_usage)
            if i == mid_layer:
                z_mid = h.detach()

        # Output projection and reshape
        h = self.output_proj(h)
        h = rearrange(
            h, "b (n s) (c p1 p2) -> b n s c p1 p2",
            n=V * T, s=Hp * Wp, c=C,
            p1=self.config.patch_size, p2=self.config.patch_size,
        )
        h = rearrange(
            h, "b (v t) (hp wp) c p1 p2 -> b v t c (hp p1) (wp p2)",
            v=V, t=T, hp=Hp, wp=Wp,
        )

        expert_usages = torch.stack(expert_usages).mean(dim=0)  # (E,)
        return h, z_mid, expert_usages
