"""Create paired real-inference qualitative figures from saved model outputs."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from data import create_nuscenes_dataloader
from data.nuscenes_dataset import NuScenesMultiViewDataset
from evaluation.video_metrics import to_uint8_video
from experiments.figures import _losses
from models.vae_backend import CogVideoX3DVAE
from utils.config import config_for_run, load_config


def image(video, view, frame):
    return video[view, :, frame].permute(1, 2, 0).numpy()


def grid(videos, labels, frames, titles, path, timestamp_seconds=None):
    fig, axes = plt.subplots(len(videos), len(frames), figsize=(3 * len(frames), 2.6 * len(videos)), squeeze=False)
    for row, (video, label) in enumerate(zip(videos, labels)):
        for col, frame in enumerate(frames):
            axes[row, col].imshow(image(video, 0 if len(frames) != 6 else frame, frame if len(frames) != 6 else 0))
            axes[row, col].axis("off")
            if row == 0:
                title = titles[col]
                if timestamp_seconds is not None:
                    title += f"\n{timestamp_seconds[frame]:.3f} s"
                axes[row, col].set_title(title)
            if col == 0:
                axes[row, col].text(-0.08, 0.5, label, transform=axes[row, col].transAxes, rotation=90, va="center", ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def keyframe_figure(run_dir, destination):
    from matplotlib.patches import Patch

    path = run_dir / "routing_audit.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Run experiments.routing_audit first: {path}")
    records = torch.load(path, map_location="cpu", weights_only=False)
    scene_tokens = list(dict.fromkeys(item["scene_token"] for item in records))
    selected = []
    for scene_token in scene_tokens:
        for timestep in [999, 500, 0]:
            candidates = [item for item in records if item["scene_token"] == scene_token]
            record = min(candidates, key=lambda item: abs(int(item["timestep"][0]) - timestep))
            selected.append(record)
    fig, axes = plt.subplots(len(scene_tokens) * 2, 3, figsize=(13, 4 * len(scene_tokens)), squeeze=False)
    for index, record in enumerate(selected):
        scene_index = index // 3
        row, col = scene_index * 2 + 1, index % 3
        scores = record["saliency_scores"][0].float().numpy()
        weights = record["frame_weights"][0].float().numpy()
        mask = record["hard_mask"][0].float().numpy()
        budgets = record["expert_budgets"][0].float().numpy()
        timestep = int(record["timestep"][0])
        x = list(range(len(weights)))
        axes[row - 1, col].plot(x, scores, marker="o", color="#23405e")
        axes[row - 1, col].set_xticks(x)
        axes[row - 1, col].set_ylabel(f"{record['scene_token'][:8]}\nraw $s_t$")
        axes[row - 1, col].set_title(f"Diffusion step {timestep}")
        axes[row, col].bar(x, weights, color=["#4472c4" if value else "#c9ced6" for value in mask])
        for position, (weight, budget) in enumerate(zip(weights, budgets)):
            axes[row, col].text(position, weight + 0.035, f"K={int(budget)}", ha="center", fontsize=8)
        axes[row, col].set_xticks(x)
        axes[row, col].set_ylim(0, max(1.1, float(weights.max()) + 0.2))
        axes[row, col].set_ylabel("weight $w_t$; label $K_t$")
        axes[row, col].set_xlabel("Latent temporal position")
    fig.suptitle("Actual keyframe states across scenes and diffusion steps; no semantic event labels")
    fig.legend(handles=[Patch(facecolor="#4472c4", label="Keyframe: $K_e$ experts"),
                        Patch(facecolor="#c9ced6", label="Non-keyframe: one expert")],
               loc="upper right", bbox_to_anchor=(1.0, 0.965))
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(destination, dpi=220)
    plt.close(fig)
    return [
        {
            "scene_token": record["scene_token"],
            "sample_index": record["sample_index"],
            "timestep": int(record["timestep"][0]),
            "saliency_scores": record["saliency_scores"][0].float().tolist(),
            "frame_weights": record["frame_weights"][0].float().tolist(),
            "hard_mask": record["hard_mask"][0].float().tolist(),
            "expert_budgets": record["expert_budgets"][0].float().tolist(),
        }
        for record in selected
    ]


def generate(config, baseline_dir, ours_dir, output, frames, manifest=None, val_scene_names=None):
    baseline_dir, ours_dir, output = Path(baseline_dir), Path(ours_dir), Path(output)
    baseline_config = config_for_run(config, baseline_dir)
    ours_config = config_for_run(config, ours_dir)
    if manifest is None:
        manifest = f"output/mini/latents{frames}/manifest.json"
    _, val_loader = create_nuscenes_dataloader(
        dataroot=config.dataroot, version=config.version, T=frames,
        batch_size=1, num_workers=0, H=config.data_height, W=config.data_width,
        frame_step=config.frame_step, sync_tolerance_ms=config.sync_tolerance_ms,
        val_scene_names=val_scene_names,
    )
    filename = "generated.pt" if frames == 16 else f"generated_long{frames}.pt"
    baseline = torch.load(baseline_dir / filename, map_location="cpu", weights_only=False)[0]
    ours = torch.load(ours_dir / filename, map_location="cpu", weights_only=False)[0]
    if baseline["source_index"] != ours["source_index"] or baseline["scene_token"] != ours["scene_token"]:
        raise ValueError("Qualitative methods must use the same real condition")
    if frames == 16 and baseline["noise_seed"] != ours["noise_seed"]:
        raise ValueError("Qualitative methods must use paired initial noise")
    real_item = val_loader.dataset[baseline["source_index"]]
    if real_item["scene_token"] != baseline["scene_token"]:
        raise ValueError("Real scene token does not match the generated condition")
    real = to_uint8_video(real_item["video"])
    vae = CogVideoX3DVAE(config.vae_path, device=config.device)
    baseline_video = to_uint8_video(vae.decode(baseline["latent"].float())[0])
    ours_video = to_uint8_video(vae.decode(ours["latent"].float())[0])
    output.mkdir(parents=True, exist_ok=True)
    view_titles = NuScenesMultiViewDataset.VIEW_NAMES
    if frames == 16:
        grid([real, ours_video], ["Real", "Ours"], list(range(6)), view_titles, output / "fig1_teaser.png")
        grid([baseline_video, ours_video], ["Baseline", "Ours"], list(range(6)), view_titles, output / "fig6_multiview.png")
        selected = [0, 3, 7, 11, 14]
    else:
        selected = [round(index * (frames - 1) / 4) for index in range(5)]
    times = (real_item["timestamps"][0].numpy() - real_item["timestamps"][0, 0].item()) / 1e6
    grid([baseline_video, ours_video], ["Baseline", "Ours"], selected, [f"frame {index + 1}" for index in selected], output / f"fig7_temporal_{frames}.png", times)
    keyframe_states = None
    if frames == 16:
        if (ours_dir / "routing_audit.pt").is_file():
            keyframe_states = keyframe_figure(ours_dir, output / "fig8_keyframe_state.png")
        _losses(baseline_dir, output / "baseline_losses.png")
        _losses(ours_dir, output / "ours_losses.png")
    metadata = {
        "frames": frames,
        "baseline_checkpoint": str(baseline_dir / "final_model.pt"),
        "ours_checkpoint": str(ours_dir / "final_model.pt"),
        "scene_token": baseline["scene_token"],
        "scene_name": baseline["scene_name"],
        "source_index": baseline["source_index"],
        "source_split": baseline["source_split"],
        "source_manifest": manifest,
        "val_scene_names": val_scene_names,
        "noise_seed": baseline.get("noise_seed"),
        "long_rollout_noise_rule": None if frames == 16 else "training_seed + 20000 + 100*sequence_index + chunk_index",
        "training_seed": baseline_config.seed,
        "selected_frames": selected,
        "keyframe_states": keyframe_states,
        "timestamp_seconds": times.tolist(),
        "baseline_config": vars(baseline_config),
        "ours_config": vars(ours_config),
        "image_source": "real inference decoded with frozen CogVideoX VAE; no artificial shift, noise, or compositing",
    }
    (output / "manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote real-inference figures to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mini.yaml")
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--ours-dir", required=True)
    parser.add_argument("--output", default="output/mini/report/qualitative")
    parser.add_argument("--frames", type=int, choices=[16, 64, 128], default=16)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--val-scenes", default=None)
    args = parser.parse_args()
    generate(load_config(args.config), args.baseline_dir, args.ours_dir, args.output, args.frames,
             manifest=args.manifest,
             val_scene_names=args.val_scenes.split(",") if args.val_scenes else None)


if __name__ == "__main__":
    main()
