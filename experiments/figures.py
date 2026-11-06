"""Generate figures only from real run checkpoints, outputs, and histories."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import save_image

from data import create_nuscenes_dataloader
from models.vae_backend import CogVideoX3DVAE
from utils.config import load_config


def _to_uint8(video):
    return ((video + 1) * 127.5).clamp(0, 255).to(torch.uint8)


def _grid(real, generated, destination, title):
    fig, axes = plt.subplots(2, 6, figsize=(18, 5))
    for view in range(6):
        for row, (video, label) in enumerate([(real, "Real"), (generated, "Generated")]):
            image = _to_uint8(video[view, :, 0]).permute(1, 2, 0).numpy()
            axes[row, view].imshow(image)
            axes[row, view].axis("off")
            if row == 0:
                axes[row, view].set_title(str(view))
            if view == 0:
                axes[row, view].text(-0.08, 0.5, label, transform=axes[row, view].transAxes,
                                     rotation=90, va="center", ha="right")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(destination, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _temporal(real, generated, destination):
    steps = [0, 3, 7, 11, 15]
    fig, axes = plt.subplots(2, len(steps), figsize=(16, 5))
    for row, (video, label) in enumerate([(real, "Real"), (generated, "Generated")]):
        for col, step in enumerate(steps):
            image = _to_uint8(video[0, :, step]).permute(1, 2, 0).numpy()
            axes[row, col].imshow(image)
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(f"t={step + 1}")
            if col == 0:
                axes[row, col].text(-0.1, 0.5, label, transform=axes[row, col].transAxes,
                                    rotation=90, va="center", ha="right")
    fig.tight_layout()
    fig.savefig(destination, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _keyframe(run_dir: Path, destination):
    path = run_dir / "routing.pt"
    if not path.is_file():
        return False
    history = torch.load(path, map_location="cpu", weights_only=False)
    weights = history[-1]["frame_weights"][0].numpy()
    mask = history[-1]["hard_mask"][0].numpy()
    budget = history[-1]["keyframe_expert_calls"][0].numpy() if "keyframe_expert_calls" in history[-1] else None
    frames = np.arange(1, len(weights) + 1)
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].bar(frames, weights)
    axes[0].axhline(1 / len(weights), color="black", linestyle="--", label="Uniform")
    axes[0].set_ylabel("Normalized weight")
    axes[0].legend()
    axes[1].step(frames, mask, where="mid", label="Hard mask")
    axes[1].set_ylabel("Selected")
    axes[1].set_xlabel("Latent temporal position")
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].legend()
    fig.suptitle(f"Routing state at diffusion step {int(history[-1]['timestep'][0])}")
    fig.tight_layout()
    fig.savefig(destination, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return True


def _losses(run_dir: Path, destination):
    path = run_dir / "training_history.json"
    if not path.is_file():
        return False
    import json
    history = json.loads(path.read_text(encoding="utf-8"))
    if not history:
        return False
    steps = [item["step"] for item in history]
    fig, ax = plt.subplots(figsize=(8, 5))
    for key in ["loss", "loss_diff", "loss_st", "loss_moe"]:
        ax.plot(steps, [item[key] for item in history], label=key)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return True


def generate_run_figures(config, run_dir: str):
    run_dir = Path(run_dir)
    generated = torch.load(run_dir / "generated.pt", map_location="cpu", weights_only=False)
    _, val_loader = create_nuscenes_dataloader(
        dataroot=config.dataroot, version=config.version, T=16,
        batch_size=1, num_workers=0, H=config.data_height, W=config.data_width,
        frame_step=config.frame_step, sync_tolerance_ms=config.sync_tolerance_ms,
    )
    record = generated[0]
    real = val_loader.dataset[int(record["source_index"])]["video"]
    vae = CogVideoX3DVAE(config.vae_path, device=config.device)
    generated_video = vae.decode(record["latent"].float())[0].cpu()
    output = run_dir / "figures"
    output.mkdir(exist_ok=True)
    title = f"{run_dir.parent.name}/seed{record['noise_seed'] - 10000}"
    _grid(real, generated_video, output / "multiview.png", title)
    _temporal(real, generated_video, output / "temporal.png")
    print("keyframe", _keyframe(run_dir, output / "keyframe.png"))
    print("losses", _losses(run_dir, output / "losses.png"))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mini.yaml")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    print(generate_run_figures(load_config(args.config), args.run_dir))


if __name__ == "__main__":
    main()
