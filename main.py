#!/usr/bin/env python
"""KW-AMoE-TSDiT: Keyframe-Weighted Adaptive Mixture-of-Experts
Temporal-Stable Diffusion Transformer for Multi-View Video Generation.

Entry point for training, evaluation, and model verification.

Examples:
    # Quick model structure verification (no data required)
    python main.py --test

    # Full training on nuScenes
    python main.py --train --dataroot /path/to/nuscenes

    # Training with custom config
    python main.py --train --dataroot /path/to/nuscenes --config configs/default.yaml
"""

import os
import sys
import argparse

import torch
import numpy as np

from utils.config import ModelConfig, load_config
from utils.helpers import set_seed, count_parameters, format_param_count


def build_model(config: ModelConfig):
    """Build the full KW-AMoE-TSDiT model from config."""
    from models.dit_backbone import DiTBackbone
    from models.kwamoe import KeyframeGate
    from models.tsmem import TemporalStableMemory
    from models.diffusion import DiffusionScheduler

    class KWAMoETSDiT(torch.nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.dit = DiTBackbone(cfg)
            self.keyframe_gate = KeyframeGate(
                cfg.dit_hidden_dim, cfg.max_frames, cfg.keyframe_ratio
            )
            self.ts_mem = TemporalStableMemory(
                cfg.dit_hidden_dim, cfg.mem_alpha, cfg.mem_beta
            )
            self.diffusion = DiffusionScheduler(cfg.timesteps, cfg.beta_start, cfg.beta_end)

    return KWAMoETSDiT(config)


def run_test(config: ModelConfig):
    """Quick verification: instantiate model and run forward/backward pass.

    Uses reduced dimensions suitable for CPU testing.
    """
    print("=" * 60)
    print("  KW-AMoE-TSDiT Model Structure Verification (Test Mode)")
    print("=" * 60)

    # Reduce dimensions for CPU testing
    config.device = "cpu"
    config.batch_size = 1
    config.max_frames = 4
    config.num_views = 3
    config.image_height = 8
    config.image_width = 12
    config.dit_hidden_dim = 256
    config.dit_layers = 4
    config.dit_heads = 4
    config.num_experts = 4
    config.mem_slots = 4
    config.ddim_steps = 5

    model = build_model(config)
    _init_weights(model)
    model = model.to(config.device)

    total = count_parameters(model)
    print(f"\n  Total parameters: {format_param_count(total)} ({total:,})")

    # Per-module breakdown
    dit_p = sum(p.numel() for p in model.dit.parameters())
    gate_p = sum(p.numel() for p in model.keyframe_gate.parameters())
    mem_p = sum(p.numel() for p in model.ts_mem.parameters())
    print(f"  DiT Backbone:  {format_param_count(dit_p)}")
    print(f"  KeyframeGate:  {format_param_count(gate_p)}")
    print(f"  TS-Mem:        {format_param_count(mem_p)}")

    # Build dummy inputs
    B, V, T, C = config.batch_size, config.num_views, config.max_frames, config.latent_dim
    H, W = config.image_height, config.image_width

    z_0 = torch.randn(B, V, T, C, H, W, device=config.device)
    cond = {
        "view_ids": torch.arange(V, device=config.device),
        "frame_ids": torch.arange(T, device=config.device),
    }
    print(f"\n  Input: z_0 = (B={B}, V={V}, T={T}, C={C}, H={H}, W={W})")

    # 1. DiT Backbone forward
    print("\n  [1/4] DiT Backbone forward...")
    t_rand = torch.randint(0, config.timesteps, (B,), device=config.device)
    noise = torch.randn_like(z_0)
    z_t, _ = model.diffusion.q_sample(z_0, t_rand, noise)
    eps_pred, z_mid, expert_usages = model.dit(z_t, t_rand, cond)
    print(f"    eps_pred:      {list(eps_pred.shape)}")
    print(f"    z_mid:         {list(z_mid.shape)}")
    print(f"    expert_usages: {list(expert_usages.shape)}")

    # 2. KW-AMoE gate
    print("\n  [2/4] KW-AMoE gating...")
    Hp, Wp = H // config.patch_size, W // config.patch_size
    Dh = config.dit_hidden_dim
    z_mid_spatial = z_mid.reshape(B, V * T, Hp * Wp, Dh)
    z_mid_pooled = z_mid_spatial.mean(dim=2)
    z_mid_feat = z_mid_pooled.reshape(B, V, T, Dh)
    fw, gv, hm = model.keyframe_gate(z_mid_feat)
    print(f"    frame_weights: {list(fw.shape)}")
    print(f"    keyframes:     {hm.sum(dim=-1).mean().item():.1f}/{T}")

    # 3. TS-Mem
    print("\n  [3/4] TS-Mem...")
    mc = model.ts_mem(z_mid_feat, hm, gv)
    print(f"    mem_context:   {list(mc.shape)}")

    # 4. Gradient flow
    print("\n  [4/4] Gradient flow check...")
    from training.losses import compute_losses
    losses = compute_losses(eps_pred, noise, z_mid_feat, mc, hm, expert_usages, config)
    losses["loss"].backward()
    grad_norm = sum(
        p.grad.norm().item() for p in model.parameters() if p.grad is not None
    )
    print(f"    Total Loss:    {losses['loss'].item():.4f}")
    print(f"    Gradient norm: {grad_norm:.2f}")

    print("\n" + "=" * 60)
    print("  All checks passed!")
    print("=" * 60)


def _init_weights(m):
    """Initialize model weights with truncated normal distribution."""
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, torch.nn.Embedding):
        torch.nn.init.trunc_normal_(m.weight, std=0.02)
    elif isinstance(m, torch.nn.Conv3d):
        torch.nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


def main():
    parser = argparse.ArgumentParser(
        description="KW-AMoE-TSDiT: Multi-View Video Generation"
    )
    parser.add_argument("--test", action="store_true", default=True,
                        help="Run model structure verification (default)")
    parser.add_argument("--train", action="store_true", default=False,
                        help="Run full training on nuScenes")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")
    parser.add_argument("--dataroot", type=str, default=None,
                        help="Path to nuScenes dataset root")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    config = ModelConfig()
    if args.config:
        config = load_config(args.config)
    if args.dataroot:
        config.dataroot = args.dataroot
    config.seed = args.seed

    set_seed(config.seed)

    if args.train:
        if not config.dataroot:
            print("ERROR: Training requires --dataroot <path_to_nuscenes>")
            sys.exit(1)

        from data import create_nuscenes_dataloader
        from training.trainer import Trainer

        train_loader, val_loader = create_nuscenes_dataloader(
            dataroot=config.dataroot, version=config.version,
            T=config.max_frames, batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        for seed in config.num_seeds:
            print(f"\n{'='*60}")
            print(f"  Experiment: seed = {seed}")
            print(f"{'='*60}")
            set_seed(seed)
            model = build_model(config)
            model.apply(_init_weights)
            trainer = Trainer(model, config, train_loader, val_loader)
            trainer.train()
    else:
        run_test(config)


if __name__ == "__main__":
    main()
