"""Evaluate generated latents against original nuScenes-mini pixels."""

import argparse
import json
from pathlib import Path

import torch

from data import create_nuscenes_dataloader
from evaluation.video_metrics import VideoMetricSuite
from models.vae_backend import CogVideoX3DVAE
from utils.config import load_config


def evaluate_run(config, run_dir: str, val_scene_names=None, detector_checkpoint=None):
    from evaluation.video_metrics import to_uint8_video
    from utils.config import config_for_run

    run_dir = Path(run_dir)
    config = config_for_run(config, run_dir)
    generated_path = run_dir / "generated.pt"
    generated = torch.load(generated_path, map_location="cpu", weights_only=False)
    print(f"Loaded {len(generated)} generated clips", flush=True)
    _, val_loader = create_nuscenes_dataloader(
        dataroot=config.dataroot,
        version=config.version,
        T=16,
        batch_size=1,
        num_workers=0,
        H=config.data_height,
        W=config.data_width,
        frame_step=config.frame_step,
        sync_tolerance_ms=config.sync_tolerance_ms,
        val_scene_names=val_scene_names,
    )
    real_videos, real_annotations = [], []
    for record in generated:
        item = val_loader.dataset[int(record["source_index"])]
        if item["scene_token"] != record["scene_token"]:
            raise ValueError("Generated condition does not match the real evaluation scene")
        real_videos.append(to_uint8_video(item["video"]))
        real_annotations.append(item["annotations"])
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    print("Initializing VAE", flush=True)
    vae = CogVideoX3DVAE(config.vae_path, device=config.device)
    print("VAE initialized", flush=True)
    generated_videos = []
    for index, record in enumerate(generated):
        decoded = vae.decode(record["latent"].float())[0]
        generated_videos.append(to_uint8_video(decoded))
        print(f"Decoded {index + 1}/{len(generated)}", flush=True)
    descriptions = [record["scene_description"] for record in generated]
    evaluated_scenes = sorted({record["scene_name"] for record in generated})
    noise_seeds = [record["noise_seed"] for record in generated]
    del generated, vae
    gc.collect()
    torch.cuda.empty_cache()
    print("Initializing metric suite", flush=True)
    suite = VideoMetricSuite(
        i3d_checkpoint="pretrained/I3D/i3d_pretrained_400.pt",
        clip_checkpoint="pretrained/CLIP-ViT-B-32-laion2B-s34B-b79K/open_clip_model.safetensors",
        fid_checkpoint="pretrained/Inception/inception_v3_google-0cc3c7bd.pth",
        device=config.device,
    )
    metrics = suite.evaluate(real_videos, generated_videos, descriptions)
    del suite
    gc.collect()
    torch.cuda.empty_cache()
    detector_checkpoint = detector_checkpoint or "output/mini/downstream/real/seed42/detector.pt"
    from experiments.ap_generated import evaluate_generated_ap

    metrics.update({
        "evaluated_scenes": evaluated_scenes,
        "noise_seeds": noise_seeds,
        "scope": "nuScenes-mini exploratory validation; not full-dataset benchmark evidence",
        "config": vars(config),
        "checkpoint": str(run_dir / "final_model.pt"),
        "evaluation_status": "quality_complete_ap_pending",
    })
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    ap = evaluate_generated_ap(
        config, run_dir, Path(detector_checkpoint),
        videos=generated_videos, annotations=real_annotations,
        val_scene_names=val_scene_names,
    )
    metrics["ap"] = ap["mAP"]
    metrics["evaluation_status"] = "complete"
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in metrics.items() if k != "config"}, indent=2), flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mini.yaml")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    evaluate_run(load_config(args.config), args.run_dir)


if __name__ == "__main__":
    main()
