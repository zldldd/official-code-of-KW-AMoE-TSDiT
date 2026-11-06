"""Frozen CogVideoX 3D-VAE backend used by MagicDrive-V2."""

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn


class CogVideoX3DVAE(nn.Module):
    """Encode multi-view videos independently with a frozen official VAE."""

    def __init__(self, pretrained_path: str, device: str = "cuda", verbose: bool = False):
        super().__init__()
        from diffusers import AutoencoderKLCogVideoX

        path = Path(pretrained_path)
        required = [path / "config.json", path / "diffusion_pytorch_model.safetensors"]
        if not all(item.is_file() for item in required):
            raise FileNotFoundError(f"Incomplete CogVideoX VAE at {path}: {required}")
        import json
        from safetensors.torch import load_file

        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        config.pop("_class_name", None)
        config.pop("_diffusers_version", None)
        with torch.device("meta"):
            self.vae = AutoencoderKLCogVideoX(**config)
        if verbose:
            print("VAE architecture initialized on meta", flush=True)
        state = load_file(
            str(path / "diffusion_pytorch_model.safetensors"), device="cpu"
        )
        if verbose:
            print("VAE safetensors read on cpu", flush=True)
        self.vae.load_state_dict(state, assign=True)
        del state
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.vae.to(device)
        if verbose:
            print(f"VAE weights loaded on {device}", flush=True)
        self.device = device
        self.scaling_factor = float(self.vae.config.scaling_factor)
        self.temporal_compression = int(self.vae.config.temporal_compression_ratio)
        self.latent_channels = int(self.vae.config.latent_channels)

    @torch.no_grad()
    def encode(self, video: torch.Tensor, view_chunk: int = 1) -> torch.Tensor:
        """Encode pixels to DiT-layout latent ``(B,V,T',C',H',W')``."""
        B, V, C, T, H, W = video.shape
        outputs = []
        for start in range(0, V, view_chunk):
            chunk = video[:, start:start + view_chunk].to(self.device)
            flat = chunk.reshape(-1, C, T, H, W).float()
            posterior = self.vae.encode(flat).latent_dist
            latent = posterior.mode() * self.scaling_factor
            outputs.append(latent.to(video.device))
        latent = torch.cat(outputs, dim=0)
        latent = latent.reshape(B, V, self.latent_channels, *latent.shape[-3:])
        return latent.permute(0, 1, 3, 2, 4, 5)

    @torch.no_grad()
    def decode(self, latent: torch.Tensor, view_chunk: int = 1,
               use_tiling: bool = True) -> torch.Tensor:
        """Decode DiT-layout ``(B,V,T,C,H,W)`` to ``(B,V,3,T,H,W)``."""
        if use_tiling and not self.vae.use_tiling:
            self.vae.enable_tiling(
                tile_sample_min_height=128, tile_sample_min_width=256
            )
        elif not use_tiling and self.vae.use_tiling:
            self.vae.disable_tiling()
        latent = latent.permute(0, 1, 3, 2, 4, 5)
        B, V, C, T, H, W = latent.shape
        outputs = []
        for start in range(0, V, view_chunk):
            chunk = latent[:, start:start + view_chunk].to(self.device)
            flat = chunk.reshape(-1, C, T, H, W).float() / self.scaling_factor
            decoded = self.vae.decode(flat).sample
            outputs.append(decoded.to(latent.device))
        video = torch.cat(outputs, dim=0)
        video = video[..., :H * 8, :W * 8]
        return video.reshape(B, V, 3, *video.shape[-3:])

    def latent_shape(self, frames: int, height: int, width: int) -> Tuple[int, int, int]:
        if frames % self.temporal_compression:
            raise ValueError("Frame count must be divisible by temporal compression")
        return self.latent_channels, frames // self.temporal_compression, height // 8, width // 8
