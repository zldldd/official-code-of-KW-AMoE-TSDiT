"""Daytime-to-nighttime zero-shot and target-domain adaptation on mini."""

import argparse
from pathlib import Path

from experiments.evaluate import evaluate_run
from experiments.latent_pipeline import prepare_latent_cache
from experiments.suite import generate_run, train_run, variant_config
from utils.config import load_config


SOURCE_DAY = ["scene-0061", "scene-0553", "scene-0655", "scene-0757", "scene-0796"]
TARGET_NIGHT = ["scene-1094"]
ADAPT_NIGHT = ["scene-1077", "scene-1100"]


def save_scene_split(config):
    import json
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.splits import mini_train, mini_val

    destination = Path("output/mini/domain/scene_split.json")
    if destination.is_file():
        return
    nusc = NuScenes(version=config.version, dataroot=config.dataroot, verbose=False)
    lookup = {scene["name"]: scene for scene in nusc.scene}
    assert not (set(SOURCE_DAY) & set(ADAPT_NIGHT + TARGET_NIGHT))
    assert not (set(ADAPT_NIGHT) & set(TARGET_NIGHT))
    assert set(SOURCE_DAY + ADAPT_NIGHT) <= set(mini_train)
    assert set(TARGET_NIGHT) <= set(mini_train)
    assert not set(TARGET_NIGHT) & set(mini_val)
    assert all("night" not in lookup[name]["description"].lower() for name in SOURCE_DAY)
    assert all("night" in lookup[name]["description"].lower() for name in ADAPT_NIGHT + TARGET_NIGHT)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({
        "source_day": [lookup[name] for name in SOURCE_DAY],
        "adaptation_night": [lookup[name] for name in ADAPT_NIGHT],
        "target_validation_night": [lookup[name] for name in TARGET_NIGHT],
        "target_protocol": "explicitly held out from all domain training; not an official mini_val scene",
        "domain_detector_training_scenes": SOURCE_DAY,
        "scope": "within-nuScenes illumination transfer, not cross-dataset generalization",
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def run_domain(config, protocol: str, variant: str, seed: int, manifest: str,
               evaluate: bool = True):
    """Train, generate, and optionally evaluate one domain-transfer run."""
    import json

    if protocol == "zero-shot":
        train_scenes = SOURCE_DAY
    elif protocol == "adapt":
        train_scenes = ADAPT_NIGHT
    else:
        raise ValueError(protocol)
    save_scene_split(config)
    manifest_path = Path(manifest)
    if not manifest_path.is_file():
        prepare_latent_cache(
            config, str(manifest_path.parent), frames=16,
            train_scene_names=train_scenes,
            val_scene_names=TARGET_NIGHT,
            resume=True,
        )
    config = variant_config(config, variant)
    config.eval_limit = 16
    run_dir = Path("output/mini/domain") / protocol / variant / f"seed{seed}"
    initialized_from = None
    if protocol == "adapt":
        initialized_from = Path("output/mini/domain/zero-shot") / variant / f"seed{seed}" / "final_model.pt"
        if not initialized_from.is_file() and not (run_dir / "complete.json").is_file():
            raise FileNotFoundError(
                f"Run zero-shot {variant} seed {seed} before target adaptation: {initialized_from}"
            )
    checkpoint = train_run(
        config, manifest, run_dir, seed,
        steps=200 if protocol == "adapt" else 1000,
        init_checkpoint=initialized_from,
    )
    generate_run(config, checkpoint, manifest, run_dir, seed, config.eval_limit)
    protocol_record = {
        "protocol": protocol,
        "seed": seed,
        "source_day_scenes": SOURCE_DAY,
        "training_scenes": train_scenes,
        "target_validation_scenes": TARGET_NIGHT,
        "adaptation_scenes": ADAPT_NIGHT if protocol == "adapt" else [],
        "training_steps": 200 if protocol == "adapt" else 1000,
        "initialized_from": str(initialized_from) if initialized_from else None,
    }
    (run_dir / "domain_protocol.json").write_text(
        json.dumps(protocol_record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not evaluate:
        return protocol_record
    metrics = evaluate_run(
        config, run_dir, TARGET_NIGHT,
        detector_checkpoint="output/mini/domain_detector/real/seed42/detector.pt",
    )
    metrics.update(protocol_record)
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mini.yaml")
    parser.add_argument("--protocol", choices=["zero-shot", "adapt"], required=True)
    parser.add_argument("--variant", choices=["baseline", "full"], required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--skip-evaluation", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    manifest = args.manifest or f"output/mini/domain_{args.protocol}_latents/manifest.json"
    seeds = [args.seed] if args.seed is not None else config.num_seeds
    for seed in seeds:
        print(run_domain(config, args.protocol, args.variant, seed, manifest, not args.skip_evaluation), flush=True)


if __name__ == "__main__":
    main()
