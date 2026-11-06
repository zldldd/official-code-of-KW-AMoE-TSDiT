"""Resumable training and generation for all mini model variants."""

import argparse
import copy
import json
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from experiments.latent_pipeline import LatentClipDataset
from main import _init_weights, build_model
from training.trainer import Trainer
from utils.config import load_config
from utils.helpers import set_seed


VARIANTS = {
    "baseline": dict(moe_type="dense", use_keyframe_gate=False, use_tsmem=False),
    "vanilla_moe": dict(moe_type="adaptive", use_keyframe_gate=False, use_tsmem=False, expert_budget_mode="matched"),
    "matched_budget": dict(expert_budget_mode="matched"),
    "kw_moe": dict(moe_type="adaptive", use_keyframe_gate=True, use_tsmem=False),
    "ts_mem": dict(moe_type="dense", use_keyframe_gate=True, use_tsmem=True),
    "full": dict(moe_type="adaptive", use_keyframe_gate=True, use_tsmem=True),
    "independent": dict(keyframe_weight_mode="independent"),
    "unmasked": dict(keyframe_weight_mode="unmasked"),
    "uniform_budget": dict(expert_budget_mode="uniform"),
    "all_write": dict(memory_write_mode="all"),
    "random_write": dict(memory_write_mode="random"),
    "shuffled": dict(keyframe_weight_mode="shuffled"),
}


def save_json(path: Path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def variant_config(base, variant: str, overrides=None):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    config = replace(base, **VARIANTS[variant])
    if overrides:
        config = replace(config, **overrides)
    return config


def train_run(config, manifest: str, run_dir: str, seed: int, steps: int = None,
              init_checkpoint: Path = None):
    """Train one run or resume from its completed marker."""
    run_dir = Path(run_dir)
    complete = run_dir / "complete.json"
    if complete.is_file():
        print(f"SKIP completed {run_dir}", flush=True)
        return run_dir / "final_model.pt"
    run_dir.mkdir(parents=True, exist_ok=True)
    if steps is not None:
        config.total_steps = steps
    config.save_interval = config.total_steps
    config.eval_interval = config.total_steps + 1
    config.output_dir = str(run_dir)
    config.checkpoint_dir = str(run_dir)
    config.seed = seed
    set_seed(seed)
    train_dataset = LatentClipDataset(manifest, split="train")
    val_dataset = LatentClipDataset(manifest, split="val", limit=2)
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=0, generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    model = build_model(config)
    model.apply(_init_weights)
    if init_checkpoint is not None:
        state = torch.load(init_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        print(f"Initialized from {init_checkpoint}", flush=True)
    trainer = Trainer(model, config, train_loader, val_loader)
    trainer.train()
    checkpoint = run_dir / "final_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": vars(config),
        "seed": seed,
        "initialized_from": str(init_checkpoint) if init_checkpoint else None,
    }, checkpoint)
    save_json(complete, {
        "seed": seed,
        "steps": config.total_steps,
        "config": vars(config),
        "checkpoint": checkpoint.name,
        "training_history": "training_history.json",
        "initialized_from": str(init_checkpoint) if init_checkpoint else None,
    })
    return checkpoint


def generate_run(config, checkpoint: Path, manifest: str, run_dir: str,
                 seed: int, eval_limit: int = None, split: str = "val",
                 destination_name: str = None):
    """Generate latents from the declared split with paired noise seeds."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    destination = run_dir / (destination_name or "generated.pt")
    if destination.is_file():
        print(f"SKIP generated {run_dir}", flush=True)
        return destination
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    limit = eval_limit or config.eval_limit
    dataset = LatentClipDataset(manifest, split=split, limit=limit, preload=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    model = build_model(config)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model = model.to(config.device).eval()
    set_seed(seed + 10000)
    generated = []
    routing_history = []
    with torch.no_grad():
        for index, batch in enumerate(loader):
            z_0 = batch["latent"].to(config.device)
            cond = {
                "view_ids": batch["view_ids"].to(config.device),
                "frame_ids": batch["frame_ids"].to(config.device),
                "intrinsics": batch["intrinsics"].to(config.device),
                "extrinsics": batch["extrinsics"].to(config.device),
            }
            generator = torch.Generator(device=z_0.device).manual_seed(seed + 10000 + index)
            z_T = torch.randn(z_0.shape, generator=generator, device=z_0.device, dtype=z_0.dtype)
            model.reset_memory()

            def model_fn(z, timestep, conditions):
                velocity, aux = model(z, timestep, conditions, reset_memory=False)
                routing_history.append({
                    "sample_index": index,
                    "scene_token": batch["scene_token"][0],
                    "saliency_scores": aux["saliency_scores"].cpu(),
                    "frame_weights": aux["frame_weights"].cpu(),
                    "hard_mask": aux["hard_mask"].cpu(),
                    "probabilities": aux["probabilities"].cpu(),
                    "expert_usages": aux["expert_usages"].cpu(),
                    "expert_budgets": 1 + (config.top_k - 1) * aux["hard_mask"].cpu(),
                    "expert_calls": aux["expert_calls"].cpu(),
                    "keyframe_expert_calls": aux["keyframe_expert_calls"].cpu(),
                    "nonkeyframe_expert_calls": aux["nonkeyframe_expert_calls"].cpu(),
                    "timestep": timestep.cpu(),
                })
                return velocity

            z_gen = model.diffusion.ddim_sample(
                z_T, model_fn, cond, config.ddim_steps
            )
            generated.append({
                "latent": z_gen.half().cpu(),
                "real_latent": z_0.half().cpu(),
                "scene_token": batch["scene_token"][0],
                "scene_name": batch["scene_name"][0],
                "scene_description": batch["scene_description"][0],
                "source_index": int(batch["source_index"][0]),
                "source_split": batch["source_split"][0],
                "noise_seed": seed + 10000 + index,
            })
            print(f"generated {run_dir.name}: {index + 1}/{len(loader)}", flush=True)
    torch.save(generated, destination)
    torch.save(routing_history, run_dir / "routing.pt")
    return destination


def build_jobs(group: str):
    jobs = []
    if group in {"main", "all"}:
        for variant in ["baseline", "vanilla_moe", "kw_moe", "ts_mem", "full"]:
            jobs.append({"variant": variant, "steps": 1000, "eval_limit": 16, "group": "main"})
    if group in {"mechanism", "all"}:
        for variant in ["independent", "unmasked", "matched_budget", "all_write", "random_write", "shuffled", "kw_moe", "full"]:
            jobs.append({"variant": variant, "steps": 500, "eval_limit": 4, "group": "mechanism"})
    if group in {"rho", "all"}:
        for rho in [0.1, 0.2, 0.3, 0.4, 0.5]:
            jobs.append({"variant": "full", "steps": 500, "eval_limit": 4,
                         "group": "rho", "overrides": {"keyframe_ratio": rho},
                         "name": f"rho_{rho}"})
    if group in {"expert", "all"}:
        for experts in [4, 8, 16]:
            for top_k in [1, 2, 4]:
                jobs.append({"variant": "full", "steps": 500, "eval_limit": 4,
                             "group": "expert", "overrides": {"num_experts": experts, "top_k": top_k},
                             "name": f"E{experts}_K{top_k}"})
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mini.yaml")
    parser.add_argument("--manifest", default="output/mini/latents16/manifest.json")
    parser.add_argument("--group", choices=["main", "mechanism", "rho", "expert", "all"], default="all")
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--parts", type=int, default=1)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--output", default="output/mini/runs")
    args = parser.parse_args()

    base = load_config(args.config)
    jobs = build_jobs(args.group)
    if args.variant:
        jobs = [job for job in jobs if (job.get("name", job["variant"]) == args.variant)]
    if not jobs:
        raise ValueError("No jobs selected")
    jobs = [job for i, job in enumerate(jobs) if i % args.parts == args.part]
    for job in jobs:
        name = job.get("name", job["variant"])
        seeds = [args.seed] if args.seed is not None else base.num_seeds
        for seed in seeds:
            config = variant_config(base, job["variant"], job.get("overrides"))
            if args.eval_limit is not None:
                config.eval_limit = args.eval_limit
            else:
                config.eval_limit = job["eval_limit"]
            run_dir = Path(args.output) / job["group"] / name / f"seed{seed}"
            checkpoint = train_run(config, args.manifest, run_dir, seed, args.steps or job["steps"])
            generate_run(config, checkpoint, args.manifest, run_dir, seed, config.eval_limit)
            print(f"COMPLETE {job['group']} {name} seed{seed}", flush=True)


if __name__ == "__main__":
    main()
