"""Audit actual keyframe states and expert use throughout saved DDIM runs."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments.latent_pipeline import LatentClipDataset
from main import build_model
from utils.config import config_for_run, load_config
from utils.helpers import set_seed


def audit_run(config, run_dir, manifest):
    run_dir = Path(run_dir)
    config = config_for_run(config, run_dir)
    generated = torch.load(run_dir / "generated.pt", map_location="cpu", weights_only=False)
    checkpoint = torch.load(run_dir / "final_model.pt", map_location="cpu", weights_only=False)
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(config.device).eval()
    set_seed(checkpoint["seed"] + 10000)
    datasets = {}
    records = []
    entropy = []
    differences = []
    with torch.no_grad():
        for index, record in enumerate(generated):
            split = record["source_split"]
            if split not in datasets:
                datasets[split] = LatentClipDataset(manifest, split=split, preload=False)
            batch = next(iter(DataLoader(
                torch.utils.data.Subset(datasets[split], [record["source_index"]]),
                batch_size=1, shuffle=False,
            )))
            z_0 = batch["latent"].to(config.device)
            cond = {
                key: batch[key].to(config.device)
                for key in ["view_ids", "frame_ids", "intrinsics", "extrinsics"]
            }
            generator = torch.Generator(device=config.device).manual_seed(record["noise_seed"])
            z_T = torch.randn(z_0.shape, generator=generator, device=config.device, dtype=z_0.dtype)
            model.reset_memory()

            def model_fn(z, timestep, conditions):
                velocity, aux = model(z, timestep, conditions, reset_memory=False)
                if config.moe_type == "dense":
                    budgets = torch.zeros_like(aux["hard_mask"])
                else:
                    budgets = 1 + (config.top_k - 1) * aux["hard_mask"]
                usage = aux["expert_usages"]
                if usage.sum() > 0 and config.num_experts > 1:
                    usage = usage / usage.sum()
                    entropy.append((-(usage * usage.clamp_min(1e-12).log()).sum() / np.log(config.num_experts)).item())
                records.append({
                    "sample_index": index,
                    "scene_token": record["scene_token"],
                    "saliency_scores": aux["saliency_scores"].half().cpu(),
                    "frame_weights": aux["frame_weights"].half().cpu(),
                    "hard_mask": aux["hard_mask"].half().cpu(),
                    "probabilities": aux["probabilities"].half().cpu(),
                    "expert_budgets": budgets.half().cpu(),
                    "expert_usages": usage.half().cpu(),
                    "expert_calls": aux["expert_calls"].cpu(),
                    "keyframe_expert_calls": aux["keyframe_expert_calls"].cpu(),
                    "nonkeyframe_expert_calls": aux["nonkeyframe_expert_calls"].cpu(),
                    "timestep": timestep.cpu(),
                })
                return velocity

            reproduced = model.diffusion.ddim_sample(z_T, model_fn, cond, config.ddim_steps)
            differences.append((reproduced.half().cpu() - record["latent"]).abs().max().item())
            print(f"routing audit {index + 1}/{len(generated)}", flush=True)
    torch.save(records, run_dir / "routing_audit.pt")
    weights = torch.stack([item["frame_weights"].float() for item in records])
    masks = torch.stack([item["hard_mask"].float() for item in records])
    summary = {
        "samples": len(generated),
        "states": len(records),
        "weight_sum_max_abs_error": (weights.sum(dim=-1) - 1).abs().max().item(),
        "nonkeyframe_max_abs_weight": weights[masks == 0].abs().max().item() if (masks == 0).any() else None,
        "keyframe_min_weight": weights[masks == 1].min().item() if (masks == 1).any() else None,
        "normalized_usage_entropy": float(np.mean(entropy)) if entropy else None,
        "mean_expert_calls": float(np.mean([item["expert_calls"].item() for item in records])),
        "mean_keyframe_expert_calls": float(np.mean([item["keyframe_expert_calls"].item() for item in records])),
        "mean_nonkeyframe_expert_calls": float(np.mean([item["nonkeyframe_expert_calls"].item() for item in records])),
        "max_abs_reproduction_difference_from_saved_half": max(differences),
        "scope": "all saved validation clips and all DDIM steps; not a semantic event annotation",
        "config": vars(config),
    }
    (run_dir / "routing_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "config"}, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mini.yaml")
    parser.add_argument("--manifest", default="output/mini/latents16/manifest.json")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    audit_run(load_config(args.config), args.run_dir, args.manifest)


if __name__ == "__main__":
    main()
