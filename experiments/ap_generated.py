"""Evaluate generated-frame AP with one detector trained only on real data."""

import argparse
import json
from pathlib import Path

import torch

from data import create_nuscenes_dataloader
from data.video_frame_dataset import VideoFrameDetectionDataset
from evaluation.detection import build_detector, evaluate_ap50, predict_detector
from models.vae_backend import CogVideoX3DVAE
from utils.config import load_config


def evaluate_generated_ap(config, run_dir: str, detector_checkpoint: Path,
                          videos=None, annotations=None, val_scene_names=None):
    """Evaluate generated frames against annotations from their paired real conditions."""
    from utils.config import config_for_run

    run_dir = Path(run_dir)
    config = config_for_run(config, run_dir)
    if videos is None or annotations is None:
        generated = torch.load(run_dir / "generated.pt", map_location="cpu", weights_only=False)
        _, val_loader = create_nuscenes_dataloader(
            dataroot=config.dataroot, version=config.version, T=16,
            batch_size=1, num_workers=0, H=config.data_height, W=config.data_width,
            frame_step=config.frame_step, sync_tolerance_ms=config.sync_tolerance_ms,
            val_scene_names=val_scene_names,
        )
    if videos is None:
        vae = CogVideoX3DVAE(config.vae_path, device=config.device)
        videos = [vae.decode(record["latent"].float())[0].cpu() for record in generated]
        del vae
    if annotations is None:
        annotations = []
        for record in generated:
            item = val_loader.dataset[int(record["source_index"])]
            if item["scene_token"] != record["scene_token"]:
                raise ValueError("Generated condition does not match the real AP scene")
            annotations.append(item["annotations"])
    from evaluation.video_metrics import to_uint8_video
    videos = [to_uint8_video(video) for video in videos]
    dataset = VideoFrameDetectionDataset(videos, annotations)
    model = build_detector("pretrained/Detection/resnet50-0676ba61.pth", device=config.device)
    model.load_state_dict(torch.load(detector_checkpoint, map_location="cpu", weights_only=True))
    predictions, ground_truth = predict_detector(model, dataset, config.device, batch_size=2)
    metrics = evaluate_ap50(predictions, ground_truth)
    metrics.update({
        "detector": str(detector_checkpoint),
        "sample_frames": len(dataset),
        "input_pixel_format": "uint8/255",
        "input_video_dtype": str(videos[0].dtype),
        "run_dir": str(run_dir),
    })
    (run_dir / "ap.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(metrics, flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mini.yaml")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--detector", default="output/mini/downstream/real/seed42/detector.pt")
    args = parser.parse_args()
    evaluate_generated_ap(load_config(args.config), args.run_dir, Path(args.detector))


if __name__ == "__main__":
    main()
