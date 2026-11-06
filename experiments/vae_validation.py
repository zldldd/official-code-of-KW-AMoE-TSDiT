"""Verify the frozen CogVideoX 3D-VAE on a real nuScenes-mini clip."""

import argparse
import json
from pathlib import Path

import torch
from torchvision.utils import save_image

from data import create_nuscenes_dataloader
from models.vae_backend import CogVideoX3DVAE
from utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mini.yaml")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output", default="output/mini/vae_reconstruction")
    parser.add_argument("--disable-tiling", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.device:
        config.device = args.device
    _, loader = create_nuscenes_dataloader(
        dataroot=config.dataroot, version=config.version, T=16,
        batch_size=1, num_workers=0, H=config.data_height, W=config.data_width,
        frame_step=config.frame_step, sync_tolerance_ms=config.sync_tolerance_ms,
    )
    item = loader.dataset[args.index]
    video = item["video"].unsqueeze(0)
    vae = CogVideoX3DVAE(config.vae_path, device=config.device)
    if args.disable_tiling:
        vae.vae.disable_tiling()
    latent = vae.encode(video, view_chunk=1)
    reconstruction = vae.decode(
        latent, view_chunk=1, use_tiling=not args.disable_tiling
    ).clamp(-1, 1)
    real_flat = video[0].permute(0, 2, 1, 3, 4).reshape(-1, 3, config.data_height, config.data_width)
    reconstructed_flat = reconstruction[0].permute(0, 2, 1, 3, 4).reshape(-1, 3, config.data_height, config.data_width).cpu()
    import lpips
    metric = lpips.LPIPS(net="vgg").to(config.device).eval()
    distances = []
    with torch.no_grad():
        for start in range(0, len(real_flat), 4):
            distances.extend(metric(
                real_flat[start:start + 4].to(config.device),
                reconstructed_flat[start:start + 4].to(config.device),
            ).flatten().cpu().tolist())
    mse = torch.mean((reconstructed_flat - real_flat).pow(2)).item()
    frame_mse = (reconstructed_flat - real_flat).pow(2).flatten(1).mean(dim=1)
    result = {
        "compute_device": config.device,
        "tiling_enabled": bool(vae.vae.use_tiling),
        "psnr_per_frame_db": (-10 * frame_mse.log10()).tolist(),
        "scene_token": item["scene_token"],
        "source_index": args.index,
        "input_shape": list(video.shape),
        "latent_shape": list(latent.shape),
        "reconstruction_shape": list(reconstruction.shape),
        "latent_channels": vae.latent_channels,
        "temporal_compression": vae.temporal_compression,
        "spatial_compression": 8,
        "scaling_factor": vae.scaling_factor,
        "posterior": "mode (deterministic)",
        "mse_normalized_minus_one_to_one": mse,
        "psnr_db": -10 * torch.log10(torch.tensor(mse)).item(),
        "lpips_vgg": sum(distances) / len(distances),
        "frame_count": len(real_flat),
        "scope": "one real nuScenes-mini clip; not a full VAE benchmark",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    save_image(torch.cat([real_flat, reconstructed_flat]) * 0.5 + 0.5, output.with_suffix(".png"), nrow=6)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
