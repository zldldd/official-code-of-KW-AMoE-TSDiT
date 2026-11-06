"""Closed-loop 2D detection augmentation using only training-scene conditions."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset

from data import create_nuscenes_dataloader
from data.detection_dataset import NuScenesDetectionDataset
from data.video_frame_dataset import VideoFrameDetectionDataset
from evaluation.detection import (
    build_detector, evaluate_ap50, predict_detector, train_detector,
)
from experiments.latent_pipeline import LatentClipDataset
from experiments.suite import generate_run, variant_config
from models.vae_backend import CogVideoX3DVAE
from utils.config import load_config
from utils.helpers import set_seed


def synthetic_dataset(config, checkpoint: Path, manifest: str, run_dir: str,
                      seed: int, count: int):
    """Generate training clips and retain their conditional 2D annotations."""
    run_dir = Path(run_dir)
    generated_path = generate_run(
        config, checkpoint, manifest, run_dir, seed, count,
        split="train", destination_name="generated_train.pt",
    )
    generated = torch.load(generated_path, map_location="cpu", weights_only=False)
    source = LatentClipDataset(manifest, split="train", limit=count, preload=False)
    vae = CogVideoX3DVAE(config.vae_path, device=config.device)
    videos = [vae.decode(record["latent"].float())[0].cpu() for record in generated]
    annotations = [source[index]["annotations"] for index in range(len(source))]
    tokens = [source[index]["sample_data_tokens"] for index in range(len(source))]
    dataset = VideoFrameDetectionDataset(
        videos, annotations, sample_data_tokens=tokens, deduplicate=True
    )
    return dataset, len(videos)


def run_downstream(config, mode: str, checkpoint: Path, manifest: str,
                   seed: int, steps: int, output_root: str = "output/mini/downstream",
                   train_scene_names=None, val_scene_names=None):
    """Train Real-only or Real+Synthetic and evaluate on real validation only."""
    if mode in {"baseline", "full"}:
        config = variant_config(config, mode)
    output_root = Path(output_root)
    output = output_root / mode / f"seed{seed}"
    result_path = output / "downstream_metrics.json"
    expected_train = train_scene_names or "official mini_train"
    expected_val = val_scene_names or "official mini_val"
    if result_path.is_file():
        try:
            saved = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            saved = {}
        if ((saved.get("mode"), saved.get("seed"), saved.get("steps")) == (mode, seed, steps)
                and saved.get("detector_training_scenes") == expected_train
                and saved.get("detector_validation_scenes") == expected_val):
            print(f"SKIP evaluated downstream {output}", flush=True)
            return saved
    train_loader, val_loader = create_nuscenes_dataloader(
        dataroot=config.dataroot, version=config.version, T=16,
        batch_size=1, num_workers=0, H=config.data_height, W=config.data_width,
        frame_step=config.frame_step, sync_tolerance_ms=config.sync_tolerance_ms,
        train_scene_names=train_scene_names, val_scene_names=val_scene_names,
    )
    real_train = NuScenesDetectionDataset(train_loader.dataset, scene_names=train_scene_names)
    real_val = NuScenesDetectionDataset(val_loader.dataset, scene_names=val_scene_names)
    synthetic_count = 0
    synthetic_frames = 0
    if mode == "real":
        train_dataset = real_train
    else:
        synthetic, synthetic_count = synthetic_dataset(
            config, checkpoint, manifest,
            output_root / mode / f"seed{seed}", seed, 32,
        )
        synthetic_frames = len(synthetic)
        train_dataset = ConcatDataset([real_train, synthetic])
    set_seed(seed)
    model = build_detector("pretrained/Detection/resnet50-0676ba61.pth", device=config.device)
    history_path = output / "history.json"
    completed_training = False
    if (output / "detector.pt").is_file() and history_path.is_file():
        try:
            history = json.loads(history_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            history = []
        completed_training = bool(history) and history[-1].get("step") == steps - 1
    if completed_training:
        model.load_state_dict(torch.load(
            output / "detector.pt", map_location=config.device, weights_only=True,
        ))
        print(f"SKIP trained detector {output}", flush=True)
    else:
        train_detector(model, train_dataset, output, config.device,
                       steps=steps, batch_size=2, seed=seed)
    predictions, ground_truth = predict_detector(model, real_val, config.device, batch_size=2)
    metrics = evaluate_ap50(predictions, ground_truth)
    result = {
        "task": "2D camera object detection with projected nuScenes boxes; not 3D mAP or NDS",
        "detector": "Faster R-CNN ResNet-50-FPN, ResNet-50 ImageNet-1K backbone initialization",
        "detector_training_scenes": train_scene_names or "official mini_train",
        "detector_validation_scenes": val_scene_names or "official mini_val",
        "mode": mode,
        "seed": seed,
        "steps": steps,
        "optimizer": "AdamW",
        "learning_rate": 1e-4,
        "batch_size": 2,
        "synthetic_clip_count": synthetic_count,
        "real_training_frames": len(real_train),
        "synthetic_training_frames": synthetic_frames,
        "synthetic_to_real_frame_ratio": synthetic_frames / len(real_train),
        "real_validation_frames": len(real_val),
        "ap_definition": "non-interpolated AP at 2D IoU=0.5; mean over classes with ground truth",
        "validation_sample_data_tokens": [record["token"] for record in real_val.samples],
        **metrics,
    }
    (output / "downstream_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(result, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mini.yaml")
    parser.add_argument("--manifest", default="output/mini/latents16/manifest.json")
    parser.add_argument("--mode", choices=["real", "baseline", "full"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--output-root", default="output/mini/downstream_controlled")
    parser.add_argument("--train-scenes", default=None)
    parser.add_argument("--val-scenes", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    checkpoint = Path("output/mini/runs/main") / args.mode / f"seed{args.seed}" / "final_model.pt"
    run_downstream(
        config, args.mode, checkpoint, args.manifest, args.seed, args.steps, args.output_root,
        train_scene_names=args.train_scenes.split(",") if args.train_scenes else None,
        val_scene_names=args.val_scenes.split(",") if args.val_scenes else None,
    )


if __name__ == "__main__":
    main()
