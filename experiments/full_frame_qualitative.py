"""Generate full-field qualitative comparisons without cropping source content."""

import argparse
from dataclasses import replace
import gc
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from data import create_nuscenes_dataloader
from evaluation.video_metrics import to_uint8_video
from main import build_model
from models.vae_backend import CogVideoX3DVAE
from utils.config import config_for_run, load_config
from utils.helpers import set_seed


METHODS = {
    "baseline": "Baseline DiT",
    "vanilla_moe": "+Vanilla MoE",
    "kw_moe": "+KW-MoE",
    "ts_mem": "+TS-Mem",
    "full": "KW-AMoE-TSDiT",
}


def adapted_state(state, config):
    parameters = dict(state["model_state_dict"])
    key = "dit.pos_embed"
    embedding = parameters[key]
    views = config.num_views
    frames = config.max_frames
    height = config.image_height // config.patch_size
    width = config.image_width // config.patch_size
    old_width = 8
    old_height = embedding.shape[1] // (views * frames * old_width)
    if old_height == height and old_width == width:
        return parameters
    grid = embedding.reshape(1, views, frames, old_height, old_width, -1)
    grid = grid.permute(0, 1, 2, 5, 3, 4)
    grid = grid.reshape(views * frames, -1, old_height, old_width)
    grid = F.interpolate(grid, size=(height, width), mode="bilinear", align_corners=False)
    grid = grid.reshape(1, views, frames, -1, height, width)
    parameters[key] = grid.reshape(1, views * frames * height * width, -1)
    return parameters


def generate_variant(config, checkpoint, latent, conditions, seed):
    config = replace(config, image_height=40)
    model = build_model(config)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if int(state["seed"]) != seed:
        raise ValueError(f"Unexpected checkpoint seed in {checkpoint}")
    model.load_state_dict(adapted_state(state, config))
    model = model.to(config.device).eval()
    generator = torch.Generator(device=config.device).manual_seed(seed + 10000)
    initial = torch.randn(latent.shape, generator=generator, device=config.device, dtype=latent.dtype)
    model.reset_memory()

    def model_fn(z, timestep, conditions_):
        velocity, _ = model(z, timestep, conditions_, reset_memory=False)
        return velocity

    with torch.no_grad():
        generated = model.diffusion.ddim_sample(initial, model_fn, conditions, config.ddim_steps)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return generated.half().cpu()


def render(rows, frames, titles, destination, timestamps=None):
    count = len(rows)
    figure, axes = plt.subplots(count, len(frames), figsize=(3 * len(frames), 1.8 * count), squeeze=False)
    for row, (label, video) in enumerate(rows):
        for column, frame in enumerate(frames):
            image = video[column if len(frames) == 6 else 0, :, frame if len(frames) == 6 else frame]
            image = to_uint8_video(image).permute(1, 2, 0).numpy()
            axes[row, column].imshow(image)
            axes[row, column].axis("off")
            if row == 0:
                title = titles[column]
                if timestamps is not None:
                    title += f"\n{timestamps[frame]:.3f} s"
                axes[row, column].set_title(title, fontsize=10)
            if column == 0:
                axes[row, column].text(-0.07, 0.5, label, transform=axes[row, column].transAxes,
                                       rotation=90, va="center", ha="right", fontsize=9)
    figure.subplots_adjust(left=0.115, right=0.998, top=0.965, bottom=0.015, wspace=0.025, hspace=0.10)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=220)
    plt.close(figure)


def generate(config, output, seed, variants):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    _, loader = create_nuscenes_dataloader(
        dataroot=config.dataroot, version=config.version, T=16, batch_size=1,
        num_workers=0, H=288, W=512, frame_step=config.frame_step,
        sync_tolerance_ms=config.sync_tolerance_ms, val_scene_names=["scene-0103"],
    )
    real = loader.dataset[0]
    if real["scene_name"] != "scene-0103":
        raise ValueError("Unexpected full-frame source scene")
    real_video = real["video"]
    padded = torch.cat([
        real_video[..., :1, :].repeat(1, 1, 1, 16, 1),
        real_video,
        real_video[..., -1:, :].repeat(1, 1, 1, 16, 1),
    ], dim=-2).unsqueeze(0)
    vae = CogVideoX3DVAE(config.vae_path, device=config.device)
    with torch.no_grad():
        latent = vae.encode(padded)
    if latent.shape != (1, 6, 4, 16, 40, 64):
        raise ValueError(f"Unexpected full-frame latent shape: {latent.shape}")
    conditions = {
        "view_ids": real["view_ids"].unsqueeze(0).to(config.device),
        "frame_ids": torch.arange(latent.shape[2], device=config.device).unsqueeze(0),
        "intrinsics": real["intrinsics"].unsqueeze(0).to(config.device),
        "extrinsics": real["extrinsics"].unsqueeze(0).to(config.device),
    }
    generated = {}
    for variant in variants:
        checkpoint = Path("output/mini/runs/main") / variant / f"seed{seed}" / "final_model.pt"
        run_config = config_for_run(config, str(checkpoint.parent))
        generated[variant] = generate_variant(run_config, checkpoint, latent, conditions, seed)
        print(f"Generated full-field {variant}", flush=True)
    videos = {"real": real_video}
    for variant, value in generated.items():
        decoded = to_uint8_video(vae.decode(value)[0])[..., 16:-16, :]
        videos[variant] = (decoded.float() / 127.5 - 1.0).clamp(-1, 1)
    del vae, latent
    gc.collect()
    torch.cuda.empty_cache()
    metadata = {
        "scope": "full-field qualitative inference only; not a new metric protocol",
        "source": "uncropped 1600x900 camera frame resized to 512x288",
        "inference_canvas": "replicate-padding of 16 rows at top and bottom to 512x320; removed after VAE decoding",
        "position_embedding": "bilinear interpolation from 4x8 to 5x8 latent patches; other checkpoint weights unchanged",
        "latent_shape": [1, 6, 4, 16, 40, 64],
        "scene_token": real["scene_token"],
        "scene_name": real["scene_name"],
        "source_index": 0,
        "source_split": "val",
        "sample_data_tokens": real["sample_data_tokens"],
        "timestamps_us": real["timestamps"].tolist(),
        "seed": seed,
        "noise_seed": seed + 10000,
        "ddim_steps": config.ddim_steps,
        "checkpoints": {variant: f"output/mini/runs/main/{variant}/seed{seed}/final_model.pt" for variant in variants},
        "external_methods": "not generated; MagicDrive/DiVE require separate verified artifacts",
    }
    torch.save({"real": real_video, "generated_latents": generated, "metadata": metadata}, output / "full_frame_data.pt")
    (output / "manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = [("Real (full frame)", real_video)] + [(METHODS[variant], videos[variant]) for variant in variants]
    render(rows, list(range(6)), ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"], output / "fig6_multiview.png")
    selected = [0, 3, 7, 11, 14]
    times = (real["timestamps"][0] - real["timestamps"][0, 0]).tolist()
    render(rows, selected, [f"frame {index + 1}" for index in selected], output / "fig7_temporal_16.png", [value / 1e6 for value in times])
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mini.yaml")
    parser.add_argument("--output", default="output/mini/report/qualitative/full_frame_seed42")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variants", default=",".join(METHODS))
    args = parser.parse_args()
    variants = args.variants.split(",")
    unknown = set(variants) - set(METHODS)
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}")
    print(generate(load_config(args.config), args.output, args.seed, variants))


if __name__ == "__main__":
    main()
