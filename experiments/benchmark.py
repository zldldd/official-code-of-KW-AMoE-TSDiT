"""Measure DiT FLOPs, denoising throughput, and end-to-end clip latency."""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from experiments.latent_pipeline import LatentClipDataset
from main import build_model
from models.vae_backend import CogVideoX3DVAE
from utils.config import config_for_run, load_config
from utils.helpers import count_parameters


def _latency_stats(times):
    return {
        "mean": float(np.mean(times)),
        "median": float(np.median(times)),
        "p95": float(np.percentile(times, 95, method="nearest")),
        "p95_method": "nearest order statistic",
        "values": times,
    }


def benchmark(config, checkpoint: Path, manifest: str, run_dir: str, profile_only: bool = False):
    """Benchmark batch-one inference after loading models and preparing conditions."""
    config = config_for_run(config, run_dir)
    dataset = LatentClipDataset(manifest, split="val", limit=1, preload=False)
    batch = dataset[0]
    latent = batch["latent"].unsqueeze(0).to(config.device)
    cond = {
        key: batch[key].unsqueeze(0).to(config.device)
        for key in ["view_ids", "frame_ids", "intrinsics", "extrinsics"]
    }
    model = build_model(config)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model = model.to(config.device).eval()
    timestep = torch.full((1,), 500, device=config.device, dtype=torch.long)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],
        with_flops=True,
    ) as profiler:
        with torch.no_grad():
            _, aux = model(latent, timestep, cond, reset_memory=True)
    flops = sum(
        event.flops for event in profiler.key_averages()
        if event.flops > 0 and event.device_type == torch.autograd.DeviceType.CPU
    )

    if profile_only:
        result = {
            "params": count_parameters(model),
            "flops_per_step": float(flops),
            "flops_scope": "one DiT forward; CPU operator records only; excludes VAE",
            "expert_calls": aux["expert_calls"].item(),
            "keyframe_expert_calls": aux["keyframe_expert_calls"].item(),
            "nonkeyframe_expert_calls": aux["nonkeyframe_expert_calls"].item(),
            "profile_status": "operator_profile_only; no latency measurement",
            "config": vars(config),
            "checkpoint": str(checkpoint),
        }
        destination = Path(run_dir) / "profile.json"
        destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({key: value for key, value in result.items() if key != "config"}, indent=2), flush=True)
        return result

    vae = CogVideoX3DVAE(config.vae_path, device=config.device)

    def model_fn(z, t_value, conditions):
        velocity, _ = model(z, t_value, conditions, reset_memory=False)
        return velocity

    with torch.no_grad():
        model.reset_memory()
        for _ in range(3):
            model(latent, timestep, cond, reset_memory=False)
        torch.cuda.synchronize()
        step_times = []
        for _ in range(20):
            start = time.perf_counter()
            model(latent, timestep, cond, reset_memory=False)
            torch.cuda.synchronize()
            step_times.append(time.perf_counter() - start)

        denoising_50_times = []
        for _ in range(3):
            z_T = torch.randn_like(latent)
            model.reset_memory()
            torch.cuda.synchronize()
            start = time.perf_counter()
            model.diffusion.ddim_sample(z_T, model_fn, cond, 50)
            torch.cuda.synchronize()
            denoising_50_times.append(time.perf_counter() - start)

        end_to_end_times, decode_times = [], []
        for index in range(3):
            generator = torch.Generator(device=config.device).manual_seed(12345 + index)
            z_T = torch.randn(
                latent.shape, generator=generator,
                device=config.device, dtype=latent.dtype,
            )
            model.reset_memory()
            torch.cuda.synchronize()
            start = time.perf_counter()
            generated = model.diffusion.ddim_sample(
                z_T, model_fn, cond, config.ddim_steps
            )
            decode_start = time.perf_counter()
            decoded = vae.decode(generated)
            torch.cuda.synchronize()
            decode_end = time.perf_counter()
            decode_times.append(decode_end - decode_start)
            end_to_end_times.append(decode_end - start)

    step = _latency_stats(step_times)
    denoising_50 = _latency_stats(denoising_50_times)
    end_to_end = _latency_stats(end_to_end_times)
    result = {
        "params": count_parameters(model),
        "flops_per_step": float(flops),
        "flops_scope": "one DiT forward; CPU operator records only; excludes VAE",
        "denoising_steps_per_second": 1.0 / step["mean"],
        "step_seconds": step["mean"],
        "step_latency_statistics": step,
        "latency_50_step_denoising": denoising_50["mean"],
        "latency_50_step_denoising_statistics": denoising_50,
        "latency_50_step_measurement": "measured full 50-step sampler, not extrapolated",
        "end_to_end_latency": end_to_end["mean"],
        "end_to_end_latency_statistics": end_to_end,
        "vae_decode_latency_statistics": _latency_stats(decode_times),
        "end_to_end_clips_per_second": 1.0 / end_to_end["mean"],
        "aggregate_output_view_frames_per_second": decoded.shape[1] * decoded.shape[3] / end_to_end["mean"],
        "ddim_steps_measured": config.ddim_steps,
        "timing_status": ("measured_without_concurrent_study_jobs"
                          if os.environ.get("KW_STUDY_EXCLUSIVE") == "1"
                          else "development_or_contended"),
        "timing_scope": "one six-view 16-frame clip; models loaded and conditions prepared; denoising, TS-Mem, VAE decode and CUDA synchronization included; no concurrent job from this study; other system tenants are not claimed to be absent",
        "precision": str(next(model.parameters()).dtype),
        "warmup_iterations": 3,
        "step_repetitions": len(step_times),
        "end_to_end_repetitions": len(end_to_end_times),
        "expert_calls": aux["expert_calls"].item(),
        "keyframe_expert_calls": aux["keyframe_expert_calls"].item(),
        "nonkeyframe_expert_calls": aux["nonkeyframe_expert_calls"].item(),
        "gpu": torch.cuda.get_device_name(0),
        "config": vars(config),
        "checkpoint": str(checkpoint),
    }
    destination = Path(run_dir) / "benchmark.json"
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "config"}, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mini.yaml")
    parser.add_argument("--manifest", default="output/mini/latents16/manifest.json")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    benchmark(load_config(args.config), run_dir / "final_model.pt",
              args.manifest, run_dir, profile_only=args.profile_only)


if __name__ == "__main__":
    main()
