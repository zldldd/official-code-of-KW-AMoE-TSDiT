"""Video generation quality metrics: FVD, FID, LPIPS.

Note: Full FVD/FID/LPIPS computation requires pre-trained feature extractors.
This module provides the computation interface; actual metric values depend on
the installed torchmetrics and lpips packages.
"""

import torch
import torch.nn.functional as F
import numpy as np


def compute_fvd(real_videos: torch.Tensor, gen_videos: torch.Tensor,
                feature_extractor=None) -> float:
    """Compute Frechet Video Distance.

    Args:
        real_videos: (N, T, C, H, W) ground-truth videos
        gen_videos: (N, T, C, H, W) generated videos
        feature_extractor: pre-trained I3D or similar. If None, returns NaN.
    Returns:
        FVD score (lower is better)
    """
    if feature_extractor is None:
        return float("nan")

    with torch.no_grad():
        real_feats = feature_extractor(real_videos)
        gen_feats = feature_extractor(gen_videos)

    mu_r, sigma_r = real_feats.mean(0), torch.cov(real_feats.T)
    mu_g, sigma_g = gen_feats.mean(0), torch.cov(gen_feats.T)

    diff = mu_r - mu_g
    # sqrtm approximation via trace
    trace = torch.trace(sigma_r + sigma_g - 2 * torch.sqrt(sigma_r @ sigma_g))
    fvd = (diff @ diff).item() + trace.item()
    return max(fvd, 0.0)


def compute_fid(real_imgs: torch.Tensor, gen_imgs: torch.Tensor,
                feature_dim: int = 2048) -> float:
    """Compute Frechet Inception Distance on frame-level features.

    Args:
        real_imgs: (N, C, H, W) real frames in [0, 255] uint8
        gen_imgs: (N, C, H, W) generated frames in [0, 255] uint8
        feature_dim: inception feature dimension
    Returns:
        FID score (lower is better)
    """
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        fid = FrechetInceptionDistance(feature=feature_dim, normalize=True)
        fid.update(real_imgs, real=True)
        fid.update(gen_imgs, real=False)
        return fid.compute().item()
    except ImportError:
        return float("nan")


def compute_lpips(real_imgs: torch.Tensor, gen_imgs: torch.Tensor,
                  net: str = "vgg") -> float:
    """Compute Learned Perceptual Image Patch Similarity.

    Args:
        real_imgs: (N, C, H, W) in [-1, 1]
        gen_imgs: (N, C, H, W) in [-1, 1]
        net: backbone network ('vgg' or 'alex')
    Returns:
        LPIPS score (lower is better)
    """
    try:
        import lpips
        lpips_fn = lpips.LPIPS(net=net)
        with torch.no_grad():
            return lpips_fn(real_imgs, gen_imgs).mean().item()
    except ImportError:
        return float("nan")
