"""Run a matched second-stage convergence probe from completed mini checkpoints."""

import argparse
import json
from pathlib import Path

from experiments.evaluate import evaluate_run
from experiments.suite import generate_run, train_run, variant_config
from utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mini.yaml")
    parser.add_argument("--manifest", default="output/mini/latents16/manifest.json")
    parser.add_argument("--variant", choices=["baseline", "full"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    config = variant_config(load_config(args.config), args.variant)
    config.eval_limit = 16
    source = Path("output/mini/runs/main") / args.variant / f"seed{args.seed}" / "final_model.pt"
    if not source.is_file():
        raise FileNotFoundError(source)
    run_dir = Path("output/mini/convergence") / args.variant / f"seed{args.seed}"
    checkpoint = train_run(
        config, args.manifest, str(run_dir), args.seed,
        steps=args.steps, init_checkpoint=source,
    )
    generate_run(config, checkpoint, args.manifest, str(run_dir), args.seed, 16)
    protocol = {
        "source_checkpoint": str(source),
        "source_steps": 1000,
        "additional_steps": args.steps,
        "optimizer_state_restored": False,
        "schedule": "new 100-step warmup followed by cosine decay over the additional stage",
        "seed": args.seed,
        "comparison": "matched additional-stage budget for Baseline and Ours",
    }
    (run_dir / "convergence_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    if args.evaluate:
        evaluate_run(config, str(run_dir))


if __name__ == "__main__":
    main()
