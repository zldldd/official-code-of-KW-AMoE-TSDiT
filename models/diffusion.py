"""Diffusion noise schedule and DDIM sampler."""

import torch
import torch.nn as nn


class DiffusionScheduler(nn.Module):
    """Linear beta schedule with DDIM deterministic sampling.

    Pre-computes alpha cumulative products and provides utilities
    for forward noising and reverse DDIM sampling.
    """

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

    def q_sample(self, z_0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor = None) -> torch.Tensor:
        """Forward diffusion: z_t = sqrt(alpha_cumprod_t) * z_0 +
           sqrt(1 - alpha_cumprod_t) * epsilon."""
        if noise is None:
            noise = torch.randn_like(z_0)

        sqrt_ac = self.alphas_cumprod[t].sqrt()
        sqrt_1m_ac = (1 - self.alphas_cumprod[t]).sqrt()

        # Reshape for broadcasting
        for _ in range(z_0.dim() - sqrt_ac.dim()):
            sqrt_ac = sqrt_ac.unsqueeze(-1)
            sqrt_1m_ac = sqrt_1m_ac.unsqueeze(-1)

        return sqrt_ac * z_0 + sqrt_1m_ac * noise, noise

    @torch.no_grad()
    def ddim_sample(self, z_T: torch.Tensor, model_fn, cond: dict,
                    ddim_steps: int = 50) -> torch.Tensor:
        """DDIM deterministic reverse sampling.

        Args:
            z_T: initial noise (B, V, T, C, H, W)
            model_fn: callable (z, t, cond) -> eps_pred
            cond: conditioning dictionary
            ddim_steps: number of DDIM steps
        Returns:
            z_0: denoised latent
        """
        B = z_T.shape[0]
        z = z_T

        step_ratio = self.timesteps // ddim_steps
        timesteps = list(range(0, self.timesteps, step_ratio))

        for i, t in enumerate(reversed(timesteps)):
            t_tensor = torch.full((B,), t, device=z.device, dtype=torch.long)
            eps_pred = model_fn(z, t_tensor, cond)

            alpha = self.alphas_cumprod[t]
            alpha_prev = (
                self.alphas_cumprod[t - 1] if t > 0
                else torch.tensor(1.0, device=z.device)
            )

            # DDIM update (sigma=0, deterministic)
            pred_z0 = (z - torch.sqrt(1 - alpha) * eps_pred) / torch.sqrt(alpha)
            z = (
                torch.sqrt(alpha_prev) * pred_z0
                + torch.sqrt(1 - alpha_prev) * eps_pred
            )

        return z
