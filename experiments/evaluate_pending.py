"""Evaluate all completed mini runs without repeating validated metrics."""

import argparse
import json
import traceback
from pathlib import Path

from experiments.ap_generated import evaluate_generated_ap
from experiments.evaluate import evaluate_run
from utils.config import load_config


def _read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def evaluate_isolated(config_path, run_dir, scenes, detector, ap_only):
    import subprocess
    import sys

    if ap_only:
        script = (
            "import json,sys; from utils.config import load_config; "
            "from experiments.ap_generated import evaluate_generated_ap as fn; "
            "fn(load_config(sys.argv[1]),sys.argv[2],sys.argv[3],"
            "val_scene_names=json.loads(sys.argv[4]))"
        )
    else:
        script = (
            "import json,sys; from utils.config import load_config; "
            "from experiments.evaluate import evaluate_run as fn; "
            "fn(load_config(sys.argv[1]),sys.argv[2],json.loads(sys.argv[4]),sys.argv[3])"
        )
    subprocess.run(
        [sys.executable, "-c", script, config_path, str(run_dir), detector, json.dumps(scenes)],
        check=True,
    )


def pending(root, config, detector, group=None, isolated=False, config_path="configs/mini.yaml"):
    if detector is None:
        detector = (
            "output/mini/domain_detector/real/seed42/detector.pt"
            if root.name == "domain"
            else "output/mini/downstream/real/seed42/detector.pt"
        )
    failures = []
    completed = []
    pattern = "*/seed*/generated.pt" if root.name == "convergence" else "*/*/seed*/generated.pt"
    generated_paths = sorted(root.glob(pattern))
    for generated_path in generated_paths:
        if group is not None and generated_path.relative_to(root).parts[0] != group:
            continue
        run_dir = generated_path.parent
        if not (run_dir / "complete.json").is_file():
            print(f"SKIP unfinished {run_dir}", flush=True)
            continue
        metrics_path = run_dir / "metrics.json"
        ap_path = run_dir / "ap.json"
        metrics = _read_json(metrics_path)
        ap = _read_json(ap_path)
        scenes = ["scene-1094"] if root.name == "domain" else None
        try:
            expected_clips = 13 if root.name == "domain" else 16 if root.name == "main" or group == "main" else 4
            quality_missing = (
                any(key not in metrics for key in ["fvd", "fvd_feature", "fid", "lpips", "clip_score"])
                or metrics.get("fid_device") != "cuda"
                or metrics.get("sample_count") != expected_clips
                or metrics.get("frames_per_view") != 16
                or metrics.get("image_count_per_distribution") != expected_clips * 96
                or len(metrics.get("fvd_per_view", [])) != 6
                or len(metrics.get("clip_score_per_clip", [])) != expected_clips
                or (scenes is not None and metrics.get("evaluated_scenes") != scenes)
            )
            ap_missing = (
                "ap" not in metrics
                or "input_pixel_format" not in ap
                or ap.get("input_video_dtype") != "torch.uint8"
                or ap.get("mAP") != metrics.get("ap")
                or Path(ap.get("detector", "")).resolve() != Path(detector).resolve()
            )
            if isolated and (quality_missing or ap_missing):
                evaluate_isolated(config_path, run_dir, scenes, detector, not quality_missing)
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                ap = _read_json(ap_path)
                if ("mAP" not in ap or "input_pixel_format" not in ap
                        or ap.get("input_video_dtype") != "torch.uint8"):
                    raise RuntimeError("Evaluation did not produce protocol-valid AP")
                metrics["ap"] = ap["mAP"]
                metrics["evaluation_status"] = "complete"
                metrics_path.write_text(
                    json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                print(f"Verified AP from {ap_path}", flush=True)
            elif quality_missing:
                metrics = evaluate_run(config, str(run_dir), scenes, detector)
            elif ap_missing:
                ap = evaluate_generated_ap(config, str(run_dir), Path(detector), val_scene_names=scenes)
                metrics["ap"] = ap["mAP"]
                metrics["evaluation_status"] = "complete"
                metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
            else:
                if metrics.get("evaluation_status") != "complete":
                    metrics["evaluation_status"] = "complete"
                    metrics_path.write_text(
                        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                print(f"SKIP evaluated {run_dir}", flush=True)
            completed.append(str(run_dir))
        except Exception as error:
            failures.append({"run_dir": str(run_dir), "error": str(error)})
            traceback.print_exc()
    status = {"root": str(root), "completed": completed, "failures": failures}
    (root / "evaluation_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(f"Evaluated/verified {len(completed)} runs; failures={len(failures)}", flush=True)
    if failures:
        raise RuntimeError(f"Evaluation failures recorded in {root / 'evaluation_status.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mini.yaml")
    parser.add_argument("--root", default="output/mini/runs")
    parser.add_argument("--detector", default=None)
    parser.add_argument("--group", choices=["main", "mechanism", "rho", "expert"], default=None)
    parser.add_argument("--isolated", action="store_true")
    args = parser.parse_args()
    pending(Path(args.root), load_config(args.config), args.detector, args.group, args.isolated, args.config)


if __name__ == "__main__":
    main()
