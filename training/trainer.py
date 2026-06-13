"""Training loop for KW-AMoE-TSDiT."""

import os
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from training.losses import compute_losses
from utils.helpers import set_seed


class Trainer:
    """Full training pipeline for KW-AMoE-TSDiT.

    Handles forward pass, loss computation, gradient accumulation,
    checkpointing, and periodic evaluation.
    """

    def __init__(self, model: nn.Module, config, train_loader: DataLoader,
                 val_loader: DataLoader):
        self.model = model.to(config.device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scaler = GradScaler(enabled=config.mixed_precision)
        self.diffusion = model.diffusion
        self.step = 0

        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.output_dir, exist_ok=True)

    def train_step(self, batch: Dict) -> Dict:
        """Single training step with mixed precision."""
        video = batch["video"].to(self.config.device)  # (B, V, 3, T, H, W)
        cond = {
            "view_ids": batch["view_ids"].to(self.config.device),
            "frame_ids": batch["frame_ids"].to(self.config.device),
        }

        with autocast(enabled=self.config.mixed_precision):
            # Simulate latent space (in practice, use pre-trained 3D-VAE)
            B, V, C_in, T, H, W = video.shape
            z_0 = F.avg_pool3d(video, kernel_size=(1, 8, 8))
            H_lat = min(z_0.shape[-2], self.config.image_height)
            W_lat = min(z_0.shape[-1], self.config.image_width)
            z_0 = z_0[:, :, :self.config.latent_dim, :, :H_lat, :W_lat]

            # Forward diffusion
            t = torch.randint(0, self.config.timesteps, (B,), device=z_0.device)
            noise = torch.randn_like(z_0)
            z_t, _ = self.diffusion.q_sample(z_0, t, noise)

            # Model forward
            eps_pred, z_mid, expert_usages = self.model.dit(z_t, t, cond)

            # KW-AMoE gating
            Hp = self.config.image_height // self.config.patch_size
            Wp = self.config.image_width // self.config.patch_size
            D_h = self.config.dit_hidden_dim
            V_m = self.config.num_views
            T_m = self.config.max_frames

            z_mid_spatial = z_mid.reshape(B, V_m * T_m, Hp * Wp, D_h)
            z_mid_pooled = z_mid_spatial.mean(dim=2)
            z_mid_feat = z_mid_pooled.reshape(B, V_m, T_m, D_h)

            frame_weights, gate_values, hard_mask = \
                self.model.keyframe_gate(z_mid_feat)
            mem_context = self.model.ts_mem(z_mid_feat, hard_mask, gate_values)

            # Losses
            losses = compute_losses(
                eps_pred, noise, z_mid_feat, mem_context,
                hard_mask, expert_usages, self.config,
            )

        self.optimizer.zero_grad()
        self.scaler.scale(losses["loss"]).backward()
        if self.config.grad_clip > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip
            )
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return losses

    @torch.no_grad()
    def evaluate(self) -> Dict:
        """Validation: generate samples and compute metrics."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in self.val_loader:
            video = batch["video"].to(self.config.device)
            cond = {
                "view_ids": batch["view_ids"].to(self.config.device),
                "frame_ids": batch["frame_ids"].to(self.config.device),
            }

            B, V, C_in, T, H, W = video.shape
            H_lat = min(H // 8, self.config.image_height)
            W_lat = min(W // 8, self.config.image_width)

            z_T = torch.randn(
                B, V, self.config.latent_dim, T, H_lat, W_lat,
                device=self.config.device,
            )

            def model_fn(z, t_t, c):
                eps, _, _ = self.model.dit(z, t_t, c)
                return eps

            z_gen = self.diffusion.ddim_sample(
                z_T, model_fn, cond, self.config.ddim_steps
            )
            total_loss += F.mse_loss(
                z_gen, F.avg_pool3d(video[:, :, :self.config.latent_dim, :, :H_lat, :W_lat],
                kernel_size=(1, 8, 8))
            ).item()
            n_batches += 1

        self.model.train()
        return {"val_mse": total_loss / max(n_batches, 1)}

    def train(self):
        """Full training loop with logging and checkpointing."""
        print(f"Training on {self.config.device} | "
              f"Total steps: {self.config.total_steps}")

        while self.step < self.config.total_steps:
            for batch in self.train_loader:
                if self.step >= self.config.total_steps:
                    break

                losses = self.train_step(batch)

                if self.step % self.config.log_interval == 0:
                    print(
                        f"[Step {self.step}/{self.config.total_steps}] "
                        f"Loss: {losses['loss'].item():.4f} "
                        f"(diff={losses['loss_diff']:.4f}, "
                        f"st={losses['loss_st']:.4f}, "
                        f"moe={losses['loss_moe']:.4f})"
                    )

                if self.step % self.config.eval_interval == 0 and self.step > 0:
                    metrics = self.evaluate()
                    print(f"  |-- Eval | MSE: {metrics['val_mse']:.4f}")

                if self.step % self.config.save_interval == 0 and self.step > 0:
                    path = os.path.join(
                        self.config.checkpoint_dir, f"checkpoint_step_{self.step}.pt"
                    )
                    torch.save({
                        "step": self.step,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                    }, path)
                    print(f"  |-- Saved: {path}")

                self.step += 1

        print("Training complete.")
