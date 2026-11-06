"""Noise schedule, v-prediction target, and deterministic DDIM sampling."""

from typing import Optional

import torch
import torch.nn as nn


class DiffusionScheduler(nn.Module):
    """Linear beta schedule with v-prediction training and DDIM inference."""

    def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4,
                 beta_end: float = 0.02):
        super().__init__()
        self.timesteps = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    @staticmethod
    def _expand_like(values: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        while values.dim() < target.dim():
            values = values.unsqueeze(-1)
        return values

    def q_sample(self, z_0: torch.Tensor, t: torch.Tensor,
                 noise: Optional[torch.Tensor] = None):
        if noise is None:
            noise = torch.randn_like(z_0)
        alpha_bar = self._expand_like(self.alphas_cumprod[t], z_0)
        z_t = alpha_bar.sqrt() * z_0 + (1 - alpha_bar).sqrt() * noise
        return z_t, noise

    def v_target(self, z_0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor) -> torch.Tensor:
        alpha_bar = self._expand_like(self.alphas_cumprod[t], z_0)
        return alpha_bar.sqrt() * noise - (1 - alpha_bar).sqrt() * z_0

    def velocity_to_epsilon(self, velocity: torch.Tensor, z_t: torch.Tensor,
                            t: torch.Tensor) -> torch.Tensor:
        alpha_bar = self._expand_like(self.alphas_cumprod[t], z_t)
        return (1 - alpha_bar).sqrt() * z_t + alpha_bar.sqrt() * velocity

    @torch.no_grad()
    def ddim_sample(self, z_T: torch.Tensor, model_fn, cond: dict,
                    ddim_steps: int = 50) -> torch.Tensor:
        """Sample with a model returning velocity predictions."""
        if ddim_steps < 1 or ddim_steps > self.timesteps:
            raise ValueError("ddim_steps must be in [1, timesteps]")
        B = z_T.shape[0]
        z = z_T
        timesteps = torch.linspace(
            0, self.timesteps - 1, steps=ddim_steps, device=z.device
        ).round().long().tolist()
        timesteps = list(reversed(timesteps))

        for i, t in enumerate(timesteps):
            t_tensor = torch.full((B,), t, device=z.device, dtype=torch.long)
            velocity = model_fn(z, t_tensor, cond)
            eps_pred = self.velocity_to_epsilon(velocity, z, t_tensor)
            alpha = self.alphas_cumprod[t]
            if i + 1 < len(timesteps):
                alpha_prev = self.alphas_cumprod[timesteps[i + 1]]
            else:
                alpha_prev = torch.ones_like(alpha)

            pred_z0 = (z - torch.sqrt(1 - alpha) * eps_pred) / torch.sqrt(alpha)
            z = torch.sqrt(alpha_prev) * pred_z0 + torch.sqrt(1 - alpha_prev) * eps_pred
        return z
