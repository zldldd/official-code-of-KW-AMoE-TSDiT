"""Long-horizon chunked generation and temporally resolved evaluation."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import create_nuscenes_dataloader
from evaluation.video_metrics import VideoMetricSuite
from experiments.latent_pipeline import LatentClipDataset
from main import build_model
from models.vae_backend import CogVideoX3DVAE
from utils.config import config_for_run, load_config
from utils.helpers import set_seed


def generate_long(config, checkpoint: Path, manifest: str, run_dir: str,
                  frames: int, seed: int, eval_limit: int = 2):
    """Roll out 16-frame chunks with one memory state per full sequence."""
    run_dir = Path(run_dir)
    destination = run_dir / f"generated_long{frames}.pt"
    dataset = LatentClipDataset(manifest, split="val", limit=eval_limit, preload=False)
    if destination.is_file():
        existing = torch.load(destination, map_location="cpu", weights_only=False)
        expected_shape = (1, *dataset[0]["latent"].shape)
        if (len(existing) == min(eval_limit, len(dataset)) and
                all(tuple(record["latent"].shape) == expected_shape for record in existing)):
            return destination
        raise RuntimeError(f"Invalid saved long-rollout tensor file: {destination}")
    model = build_model(config)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model = model.to(config.device).eval()
    set_seed(seed + 20000)
    outputs = []
    with torch.no_grad():
        for sequence_index, batch in enumerate(DataLoader(dataset, batch_size=1)):
            full_latent = batch["latent"].to(config.device)
            cond = {
                "view_ids": batch["view_ids"].to(config.device),
                "intrinsics": batch["intrinsics"].to(config.device),
                "extrinsics": batch["extrinsics"].to(config.device),
            }
            model.reset_memory()
            chunks = []
            for chunk_index, start in enumerate(range(0, frames, 16)):
                latent_start = start // 4
                chunk_shape = full_latent[:, :, latent_start:latent_start + 4]
                cond["frame_ids"] = torch.arange(
                    latent_start, latent_start + 4, device=config.device,
                )
                generator = torch.Generator(device=config.device).manual_seed(
                    seed + 20000 + sequence_index * 100 + chunk_index
                )
                z_T = torch.randn(chunk_shape.shape, generator=generator,
                                  device=config.device, dtype=chunk_shape.dtype)

                def model_fn(z, timestep, conditions):
                    velocity, _ = model(z, timestep, conditions, reset_memory=False)
                    return velocity

                generated_chunk = model.diffusion.ddim_sample(
                    z_T, model_fn, cond, config.ddim_steps
                )
                chunks.append(generated_chunk)
                print(f"long {frames}: seq{sequence_index} chunk{chunk_index}", flush=True)
            outputs.append({
                "latent": torch.cat(chunks, dim=2).half().cpu(),
                "real_latent": full_latent.half().cpu(),
                "scene_token": batch["scene_token"][0],
                "scene_name": batch["scene_name"][0],
                "scene_description": batch["scene_description"][0],
                "source_index": int(batch["source_index"][0]),
                "source_split": batch["source_split"][0],
            })
    torch.save(outputs, destination)
    return destination


def evaluate_long(config, run_dir: str, frames: int):
    """Evaluate non-overlapping 16-frame subsegments and the final segment."""
    from evaluation.video_metrics import to_uint8_video
    from utils.config import config_for_run

    run_dir = Path(run_dir)
    config = config_for_run(config, run_dir)
    print(f"Evaluating long rollout: {run_dir}, frames={frames}", flush=True)
    generated = torch.load(run_dir / f"generated_long{frames}.pt",
                           map_location="cpu", weights_only=False)
    summary_path = run_dir / f"metrics_long{frames}.json"
    if summary_path.is_file():
        try:
            saved = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            saved = {}
        if (saved.get("sequence_count") == len(generated) and
                saved.get("training_seed") == config.seed and
                len(saved.get("segments", [])) == frames // 16 and
                all(item.get("sample_count") == len(generated)
                    for item in saved["segments"])):
            print(f"SKIP evaluated long rollout {run_dir} frames={frames}", flush=True)
            return saved
    _, val_loader = create_nuscenes_dataloader(
        dataroot=config.dataroot, version=config.version, T=frames,
        batch_size=1, num_workers=0, H=config.data_height, W=config.data_width,
        frame_step=config.frame_step, sync_tolerance_ms=config.sync_tolerance_ms,
    )
    vae = CogVideoX3DVAE(config.vae_path, device=config.device)
    suite = VideoMetricSuite(
        "pretrained/I3D/i3d_pretrained_400.pt",
        "pretrained/CLIP-ViT-B-32-laion2B-s34B-b79K/open_clip_model.safetensors",
        "pretrained/Inception/inception_v3_google-0cc3c7bd.pth",
        config.device,
    )
    decoded = []
    for record in generated:
        real_item = val_loader.dataset[int(record["source_index"])]
        if real_item["scene_token"] != record["scene_token"]:
            raise ValueError("Long-rollout condition does not match the real evaluation scene")
        timestamps = real_item["timestamps"][0]
        decoded.append({
            "real": to_uint8_video(real_item["video"]),
            "generated": to_uint8_video(vae.decode(record["latent"].float())[0]),
            "description": record["scene_description"],
            "observed_span_seconds": (timestamps[-1] - timestamps[0]).item() / 1e6,
        })
    import gc

    segment_results = []
    for segment_index, start in enumerate(range(0, frames, 16)):
        segment_path = run_dir / f"metrics_long{frames}_segment{segment_index}.json"
        metrics = {}
        if segment_path.is_file():
            try:
                metrics = json.loads(segment_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                metrics = {}
            if metrics and (metrics.get("sample_count") != len(generated) or metrics.get("fid_device") != "cuda"):
                raise RuntimeError(f"Invalid saved long-rollout segment: {segment_path}")
        if metrics:
            print(f"SKIP evaluated long segment {segment_index}", flush=True)
        else:
            real_clips = [item["real"][:, :, start:start + 16] for item in decoded]
            generated_clips = [item["generated"][:, :, start:start + 16] for item in decoded]
            descriptions = [item["description"] for item in decoded]
            metrics = suite.evaluate(real_clips, generated_clips, descriptions)
            metrics.update({
                "segment_index": segment_index,
                "start_frame": start,
                "end_frame": start + 16,
            })
            segment_path.write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            del real_clips, generated_clips, descriptions
            gc.collect()
            torch.cuda.empty_cache()
        segment_results.append(metrics)
    summary = {
        "frames": frames,
        "sequence_count": len(generated),
        "observed_span_seconds": [item["observed_span_seconds"] for item in decoded],
        "initial_noise_seed_rule": "training_seed + 20000 + 100*sequence_index + chunk_index",
        "training_seed": config.seed,
        "rollout_protocol": "non-overlapping 16-frame chunks; TS-Mem reset once per full sequence; no real future latent feedback; absolute four-position latent time encoding within each chunk",
        "segments": segment_results,
        "aggregate": {
            key: float(np.mean([segment[key] for segment in segment_results]))
            for key in ["fvd", "fid", "lpips", "clip_score"]
        },
        "late": {
            key: segment_results[-1][key]
            for key in ["fvd", "fid", "lpips", "clip_score"]
        },
        "checkpoint": str(run_dir / "final_model.pt"),
    }
    (run_dir / f"metrics_long{frames}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mini.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--frames", type=int, choices=[64, 128], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-limit", type=int, default=6)
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    config = config_for_run(load_config(args.config), run_dir)
    if not args.evaluate_only:
        generate_long(config, run_dir / "final_model.pt", args.manifest,
                      run_dir, args.frames, args.seed, args.eval_limit)
    print(evaluate_long(config, run_dir, args.frames)["aggregate"])


if __name__ == "__main__":
    main()
