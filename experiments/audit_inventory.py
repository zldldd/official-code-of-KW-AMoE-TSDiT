"""Inventory run artifacts and optionally correct seed metadata without touching weights."""

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from experiments.provenance import sha256


def inventory(root, repair_seed_metadata=False):
    root = Path(root)
    rows = []
    for checkpoint_path in sorted(root.glob("*/*/seed*/final_model.pt")):
        run_dir = checkpoint_path.parent
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        seed = int(state["seed"])
        complete_path = run_dir / "complete.json"
        if complete_path.is_file():
            complete = json.loads(complete_path.read_text(encoding="utf-8"))
            if repair_seed_metadata and complete.get("config", {}).get("seed") != seed:
                complete["config"]["seed"] = seed
                complete_path.write_text(json.dumps(complete, ensure_ascii=False, indent=2), encoding="utf-8")
        metrics = {}
        metrics_path = run_dir / "metrics.json"
        if metrics_path.is_file():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        ap_path = run_dir / "ap.json"
        ap = json.loads(ap_path.read_text(encoding="utf-8")) if ap_path.is_file() else {}
        benchmark_path = run_dir / "benchmark.json"
        benchmark = json.loads(benchmark_path.read_text(encoding="utf-8")) if benchmark_path.is_file() else {}
        relative = run_dir.relative_to(root)
        rows.append({
            "run_dir": str(run_dir),
            "group": relative.parts[0],
            "variant": relative.parts[1],
            "training_seed": seed,
            "training_steps": state["config"]["total_steps"],
            "checkpoint_sha256": sha256(checkpoint_path),
            "generated": (run_dir / "generated.pt").is_file(),
            "metrics": bool(metrics),
            "fid_device": metrics.get("fid_device"),
            "ap_protocol_valid": "input_pixel_format" in ap and ap.get("input_video_dtype") == "torch.uint8",
            "benchmark": benchmark_path.is_file(),
            "benchmark_timing_status": benchmark.get("timing_status"),
            "routing_audit": (run_dir / "routing_summary.json").is_file(),
        })
    counts = Counter((row["group"], row["variant"]) for row in rows)
    payload = {
        "root": str(root),
        "checkpoint_count": len(rows),
        "counts_by_variant": {f"{group}/{variant}": count for (group, variant), count in sorted(counts.items())},
        "runs": rows,
        "metadata_note": "training_seed comes from checkpoint['seed']; no model weights or metric values are modified",
    }
    destination = root / "run_inventory.json"
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {destination}: {len(rows)} checkpoints")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="output/mini/runs")
    parser.add_argument("--repair-seed-metadata", action="store_true")
    args = parser.parse_args()
    inventory(args.root, args.repair_seed_metadata)


if __name__ == "__main__":
    main()
