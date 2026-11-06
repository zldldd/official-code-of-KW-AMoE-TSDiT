"""DiT backbone with configurable dense, vanilla-MoE, and route-A paths."""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .attention import SinusoidalPosEmb, AdaLN, MultiHeadSelfAttention
from .kwamoe import ExpertFFN, SparseMoE, KeyframeGate
from .tsmem import TemporalStableMemory


class DiTBlock(nn.Module):
    """Single DiT block with dense or sparse feed-forward computation."""

    def __init__(self, hidden_dim: int, num_heads: int, cond_dim: int,
                 num_experts: int, top_k: int, hidden_ratio: int,
                 moe_type: str, budget_mode: str):
        super().__init__()
        self.moe_type = moe_type
        self.adaln1 = AdaLN(hidden_dim, cond_dim)
        self.attn = MultiHeadSelfAttention(hidden_dim, num_heads)
        self.adaln2 = AdaLN(hidden_dim, cond_dim)
        if moe_type == "dense":
            self.ffn = ExpertFFN(hidden_dim, hidden_ratio)
        else:
            self.moe = SparseMoE(
                hidden_dim, num_experts, top_k, hidden_ratio, budget_mode
            )

    def forward(self, h: torch.Tensor, cond: torch.Tensor, **moe_kwargs):
        num_experts = moe_kwargs.pop("num_experts")
        h = h + self.attn(self.adaln1(h, cond))
        normalized = self.adaln2(h, cond)
        if self.moe_type == "dense":
            h_out = self.ffn(normalized)
            expert_usage = h.new_zeros(num_experts)
            stats = {
                "expert_calls": h.new_tensor(0.0),
                "keyframe_expert_calls": h.new_tensor(0.0),
                "nonkeyframe_expert_calls": h.new_tensor(0.0),
                "mean_keyframe_budget": h.new_tensor(0.0),
                "mean_nonkeyframe_budget": h.new_tensor(0.0),
                "tokens_per_frame": h.new_tensor(float(h.shape[1])),
            }
        else:
            h_out, expert_usage, stats = self.moe(normalized, **moe_kwargs)
        return h + h_out, expert_usage, stats


class DiTBackbone(nn.Module):
    """Joint multi-view denoising with shared keyframe and memory state."""

    def __init__(self, config, keyframe_gate: KeyframeGate,
                 ts_mem: TemporalStableMemory):
        super().__init__()
        self.config = config
        self.keyframe_gate = keyframe_gate
        self.ts_mem = ts_mem
        self.patch_proj = nn.Conv3d(
            config.latent_dim,
            config.dit_hidden_dim,
            kernel_size=(1, config.patch_size, config.patch_size),
            stride=(1, config.patch_size, config.patch_size),
        )
        Hp = config.image_height // config.patch_size
        Wp = config.image_width // config.patch_size
        max_tokens = config.max_frames * config.num_views * Hp * Wp
        self.pos_embed = nn.Parameter(
            torch.zeros(1, max_tokens, config.dit_hidden_dim)
        )
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(config.dit_hidden_dim),
            nn.Linear(config.dit_hidden_dim, config.dit_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.dit_hidden_dim, config.dit_hidden_dim),
        )
        self.view_embed = nn.Embedding(config.num_views, config.dit_hidden_dim // 4)
        self.frame_pos = SinusoidalPosEmb(config.dit_hidden_dim // 4)
        self.cond_proj = nn.Linear(config.dit_hidden_dim // 2, config.dit_hidden_dim)

        cond_dim = config.dit_hidden_dim * 2
        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_dim=config.dit_hidden_dim,
                num_heads=config.dit_heads,
                cond_dim=cond_dim,
                num_experts=config.num_experts,
                top_k=config.top_k,
                hidden_ratio=config.expert_hidden_ratio,
                moe_type=config.moe_type,
                budget_mode=config.expert_budget_mode,
            )
            for _ in range(config.dit_layers)
        ])
        output_dim = config.latent_dim * config.patch_size * config.patch_size
        self.output_proj = nn.Linear(config.dit_hidden_dim, output_dim)
        self.mem_output_proj = nn.Linear(config.dit_hidden_dim, output_dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _position_embedding(self, V: int, T: int, Hp: int, Wp: int,
                            D: int) -> torch.Tensor:
        max_T = self.config.max_frames
        pos = self.pos_embed.reshape(1, V, max_T, Hp, Wp, D)
        if T > max_T:
            pos = rearrange(pos, "b v t hp wp d -> b (v hp wp) d t")
            pos = F.interpolate(pos, size=T, mode="linear", align_corners=False)
            pos = rearrange(pos, "b (v hp wp) d t -> b v t hp wp d",
                            v=V, hp=Hp, wp=Wp)
        else:
            pos = pos[:, :, :T]
        return pos.reshape(1, V * T * Hp * Wp, D)

    def _reshape_prediction(self, h: torch.Tensor, V: int, T: int,
                            C: int, Hp: int, Wp: int) -> torch.Tensor:
        p = self.config.patch_size
        h = rearrange(
            h, "b (n s) (c p1 p2) -> b n s c p1 p2",
            n=V * T, s=Hp * Wp, c=C, p1=p, p2=p,
        )
        return rearrange(
            h, "b (v t) (hp wp) c p1 p2 -> b v t c (hp p1) (wp p2)",
            v=V, t=T, hp=Hp, wp=Wp,
        )

    def forward(self, z_t: torch.Tensor, t: torch.Tensor,
                cond: Dict) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, V, T, C, H, W = z_t.shape
        p = self.config.patch_size
        Hp, Wp = H // p, W // p
        if (H % p, W % p) != (0, 0):
            raise ValueError("Latent spatial dimensions must be divisible by patch_size")

        h = rearrange(z_t, "b v t c h w -> b c (v t) h w")
        h = self.patch_proj(h)
        _, D, _, _, _ = h.shape
        h = rearrange(h, "b d n hp wp -> b (n hp wp) d")
        h = h + self._position_embedding(V, T, Hp, Wp, D)

        frame_tokens = h.reshape(B, V, T, Hp * Wp, D)
        frame_features = frame_tokens.mean(dim=3)
        if self.config.use_keyframe_gate:
            frame_weights, _, hard_mask, probabilities, saliency_scores = self.keyframe_gate(frame_features)
        else:
            probabilities = frame_features.new_full((B, T), 1.0 / T)
            frame_weights = probabilities
            hard_mask = frame_features.new_ones((B, T))
            saliency_scores = torch.zeros_like(probabilities)

        if self.config.use_tsmem:
            mem_context, memory_global = self.ts_mem(
                frame_features, frame_weights, hard_mask,
                probabilities, self.config.memory_write_mode,
            )
        else:
            mem_context = torch.zeros_like(frame_features)
            memory_global = frame_features.new_zeros(B, D)

        router_frame_features = frame_features.mean(dim=1)
        if self.config.use_keyframe_gate:
            global_context = (frame_weights.unsqueeze(-1) * router_frame_features).sum(dim=1)
        else:
            global_context = router_frame_features.mean(dim=1)

        t_emb = self.time_embed(t)
        v_ids = cond.get("view_ids", torch.arange(V, device=z_t.device))
        if v_ids.ndim > 1:
            v_ids = v_ids[0]
        v_emb = self.view_embed(v_ids).unsqueeze(0)
        frame_ids = cond.get("frame_ids", torch.arange(T, device=z_t.device))
        if frame_ids.ndim > 1:
            frame_ids = frame_ids[0]
        if v_ids.numel() != V or frame_ids.numel() != T:
            raise ValueError(
                f"Condition lengths must match view={V} and latent_time={T}; "
                f"got {v_ids.numel()} and {frame_ids.numel()}"
            )
        f_emb = self.frame_pos(frame_ids).unsqueeze(0)
        vf_emb = torch.cat([
            v_emb.unsqueeze(2).expand(-1, -1, T, -1).reshape(1, V * T, -1),
            f_emb.unsqueeze(1).expand(-1, V, -1, -1).reshape(1, V * T, -1),
        ], dim=-1)
        vf_emb = self.cond_proj(vf_emb)
        vf_emb = vf_emb.unsqueeze(2).expand(-1, -1, Hp * Wp, -1)
        vf_emb = vf_emb.reshape(1, V * T * Hp * Wp, D).expand(B, -1, -1)
        mem_tokens = mem_context.unsqueeze(3).expand(
            -1, -1, -1, Hp * Wp, -1
        ).reshape(B, V * T * Hp * Wp, D)
        cond_emb = torch.cat([
            t_emb.unsqueeze(1).expand(-1, h.shape[1], -1),
            vf_emb + mem_tokens,
        ], dim=-1)

        moe_kwargs = {
            "frame_features": router_frame_features,
            "global_context": global_context,
            "memory_global": memory_global,
            "hard_mask": hard_mask,
            "num_views": V,
            "keyframe_fraction": min(self.keyframe_gate.top_k, T) / T,
            "num_experts": self.config.num_experts,
        }
        expert_usages, expert_stats, z_mid = [], [], None
        mid_layer = max(0, self.config.dit_layers // 2)
        for i, block in enumerate(self.blocks):
            h, usage, stats = block(h, cond_emb, **moe_kwargs)
            expert_usages.append(usage)
            expert_stats.append(stats)
            if i == mid_layer:
                z_mid = h

        base_pred = self._reshape_prediction(self.output_proj(h), V, T, C, Hp, Wp)
        if self.config.use_tsmem:
            mem_pred = self._reshape_prediction(
                self.mem_output_proj(mem_tokens), V, T, C, Hp, Wp
            )
            prediction = base_pred + mem_pred
        else:
            prediction = base_pred

        aux = {
            "z_mid": z_mid.reshape(B, V, T, Hp * Wp, D).mean(dim=3),
            "mem_context": mem_context,
            "frame_weights": frame_weights,
            "hard_mask": hard_mask,
            "probabilities": probabilities,
            "saliency_scores": saliency_scores,
            "expert_usages": torch.stack(expert_usages).mean(dim=0),
        }
        for key in ["expert_calls", "keyframe_expert_calls", "nonkeyframe_expert_calls"]:
            aux[key] = torch.stack([s[key] for s in expert_stats]).sum()
        for key in ["mean_keyframe_budget", "mean_nonkeyframe_budget", "tokens_per_frame"]:
            aux[key] = torch.stack([s[key] for s in expert_stats]).mean()
        return prediction, aux
