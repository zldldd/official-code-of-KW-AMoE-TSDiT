"""Reproducible Fréchet and perceptual metrics with explicit view aggregation."""

from typing import Callable, Dict

import numpy as np
import torch


def frechet_distance(real_features: np.ndarray, gen_features: np.ndarray) -> float:
    """Compute Fréchet distance with symmetric eigenvalue decomposition."""
    real_features = np.asarray(real_features, dtype=np.float64)
    gen_features = np.asarray(gen_features, dtype=np.float64)
    if real_features.shape != gen_features.shape:
        raise ValueError("Real and generated feature shapes must match")
    if real_features.ndim != 2 or real_features.shape[0] < 2:
        raise ValueError("Features must have shape (N>=2, D)")

    mu_r, mu_g = real_features.mean(axis=0), gen_features.mean(axis=0)
    sigma_r = torch.from_numpy(np.cov(real_features, rowvar=False)).double()
    sigma_g = torch.from_numpy(np.cov(gen_features, rowvar=False)).double()
    diff = torch.from_numpy(mu_r - mu_g).double()

    eig_r, vec_r = torch.linalg.eigh(sigma_r)
    sqrt_r = (vec_r * eig_r.clamp_min(1e-10).sqrt()) @ vec_r.T
    product = sqrt_r @ sigma_g @ sqrt_r
    eig_product = torch.linalg.eigvalsh(product).clamp_min(0)
    trace = torch.trace(sigma_r + sigma_g) - 2 * eig_product.sqrt().sum()
    value = (diff @ diff + trace).item()
    return float(max(value, 0.0))


def compute_fvd(real_videos: torch.Tensor, gen_videos: torch.Tensor,
                feature_extractor: Callable) -> float:
    """Compute FVD from ``(N,C,T,H,W)`` videos with a supplied I3D extractor."""
    if feature_extractor is None:
        raise ValueError("A pretrained I3D feature extractor is required")
    if real_videos.shape != gen_videos.shape:
        raise ValueError("Real and generated video shapes must match")
    with torch.no_grad():
        real_feats = feature_extractor(real_videos)
        gen_feats = feature_extractor(gen_videos)
    real_feats = real_feats.reshape(real_feats.shape[0], -1).cpu().numpy()
    gen_feats = gen_feats.reshape(gen_feats.shape[0], -1).cpu().numpy()
    return frechet_distance(real_feats, gen_feats)


def compute_multiview_fvd(real_videos: torch.Tensor, gen_videos: torch.Tensor,
                          feature_extractor: Callable) -> Dict[str, object]:
    """Compute FVD per view from ``(N,V,C,T,H,W)`` and macro average."""
    if real_videos.ndim != 6 or gen_videos.shape != real_videos.shape:
        raise ValueError("Expected matching tensors shaped (N,V,C,T,H,W)")
    per_view = []
    for view in range(real_videos.shape[1]):
        per_view.append(compute_fvd(
            real_videos[:, view], gen_videos[:, view], feature_extractor
        ))
    return {"per_view": per_view, "macro": float(np.mean(per_view))}


def compute_fid(real_imgs: torch.Tensor, gen_imgs: torch.Tensor,
                feature_dim: int = 2048, weights_path: str = None) -> float:
    """Compute FID with normalization and checkpoint provenance made explicit."""
    from torchmetrics.image.fid import FrechetInceptionDistance

    if real_imgs.shape != gen_imgs.shape or real_imgs.dtype != gen_imgs.dtype:
        raise ValueError("Real and generated images must have matching shape and dtype")
    if real_imgs.dtype == torch.uint8:
        normalize = False
    elif real_imgs.is_floating_point() and real_imgs.min() >= 0 and real_imgs.max() <= 1:
        normalize = True
    else:
        raise ValueError("FID expects uint8 images or floating-point images in [0,1]")
    fid = FrechetInceptionDistance(
        feature=feature_dim, normalize=normalize,
        feature_extractor_weights_path=weights_path,
    )
    fid.update(real_imgs, real=True)
    fid.update(gen_imgs, real=False)
    return fid.compute().item()


def compute_lpips(real_imgs: torch.Tensor, gen_imgs: torch.Tensor,
                  net: str = "vgg", device: str = "cpu") -> float:
    """Compute paired LPIPS for inputs in [-1,1] on the declared device."""
    import lpips

    if real_imgs.shape != gen_imgs.shape:
        raise ValueError("Real and generated image shapes must match")
    lpips_fn = lpips.LPIPS(net=net).to(device)
    real_imgs = real_imgs.to(device)
    gen_imgs = gen_imgs.to(device)
    with torch.no_grad():
        return lpips_fn(real_imgs, gen_imgs).mean().item()
