"""Loss functions for KW-AMoE-TSDiT training."""

import random
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertLoadBalanceLoss(nn.Module):
    """Entropy-based expert load balancing loss.

    Encourages uniform expert utilization to prevent load collapse.
    L_moe = E * sum(rho_e * log(rho_e + eps))
    """

    def __init__(self, num_experts: int):
        super().__init__()
        self.E = num_experts

    def forward(self, expert_usages: torch.Tensor) -> torch.Tensor:
        """Args: expert_usages: mean expert usage probabilities (E,)"""
        rho_e = expert_usages
        return self.E * torch.sum(rho_e * torch.log(rho_e + 1e-8))


class STConsistencyLoss(nn.Module):
    """Spatiotemporal consistency loss with cross-view contrastive alignment
    and temporal smoothness regularization."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_mid: torch.Tensor, mem_context: torch.Tensor,
                hard_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_mid: (B, V, T, D) intermediate features
            mem_context: (B, V, T, D) memory context
            hard_mask: (B, T) keyframe mask
        Returns:
            total loss: cross-view alignment + temporal smoothness
        """
        B, V, T, D = z_mid.shape

        # Cross-view geometric alignment (contrastive)
        loss_align = torch.tensor(0.0, device=z_mid.device)
        n_pairs = 0
        for b in range(B):
            for t_idx in range(T):
                anchor = z_mid[b, 0, t_idx]
                for v_idx in range(1, V):
                    positive = z_mid[b, v_idx, t_idx]
                    sim_pos = F.cosine_similarity(
                        anchor.unsqueeze(0), positive.unsqueeze(0)
                    ).squeeze()
                    neg_t = random.randint(0, T - 1)
                    neg_v = random.randint(0, V - 1)
                    negative = z_mid[b, neg_v, neg_t]
                    sim_neg = F.cosine_similarity(
                        anchor.unsqueeze(0), negative.unsqueeze(0)
                    ).squeeze()
                    loss_align += -torch.log(
                        torch.exp(sim_pos / self.temperature)
                        / (torch.exp(sim_pos / self.temperature)
                           + torch.exp(sim_neg / self.temperature) + 1e-8)
                    )
                    n_pairs += 1
        loss_align = loss_align / max(n_pairs, 1)

        # Temporal smoothness
        loss_smooth = torch.tensor(0.0, device=z_mid.device)
        for t_idx in range(T - 1):
            diff = z_mid[:, :, t_idx + 1] - z_mid[:, :, t_idx]
            loss_smooth += diff.pow(2).mean()
        loss_smooth = loss_smooth / (T - 1)

        return loss_align + loss_smooth


def compute_losses(eps_pred: torch.Tensor, noise: torch.Tensor,
                   z_mid: torch.Tensor, mem_context: torch.Tensor,
                   hard_mask: torch.Tensor, expert_usages: torch.Tensor,
                   config) -> Dict[str, torch.Tensor]:
    """Compute all training losses.

    Returns:
        dict with keys: 'loss', 'loss_diff', 'loss_st', 'loss_moe'
    """
    loss_diff = F.mse_loss(eps_pred, noise)

    st_loss = STConsistencyLoss(config.temperature)
    loss_st = st_loss(z_mid, mem_context, hard_mask)

    moe_loss = ExpertLoadBalanceLoss(config.num_experts)
    loss_moe = moe_loss(expert_usages)

    total = loss_diff + config.lambda_st * loss_st + config.lambda_moe * loss_moe

    return {
        "loss": total,
        "loss_diff": loss_diff.item(),
        "loss_st": loss_st.item(),
        "loss_moe": loss_moe.item(),
    }
