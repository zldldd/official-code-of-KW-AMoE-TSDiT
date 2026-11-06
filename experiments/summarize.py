"""Aggregate raw mini metrics and recompute paired statistics from seed records."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats


METRICS = ["fvd", "fid", "lpips", "ap", "clip_score"]
LOWER_BETTER = {"fvd", "fid", "lpips"}


def collect_runs(root: Path):
    runs = []
    for path in root.glob("*/*/seed*/metrics.json"):
        relative = path.relative_to(root)
        group, variant = relative.parts[0], relative.parts[1]
        seed = int(relative.parts[2].replace("seed", ""))
        metrics = json.loads(path.read_text(encoding="utf-8"))
        ap_path = path.with_name("ap.json")
        if not ap_path.is_file():
            continue
        ap = json.loads(ap_path.read_text(encoding="utf-8"))
        complete = metrics.get("evaluation_status") == "complete"
        valid_protocol = (metrics.get("fid_device") == "cuda" and "input_pixel_format" in ap
                          and ap.get("input_video_dtype") == "torch.uint8")
        if not complete or not valid_protocol or not all(metric in metrics for metric in METRICS):
            continue
        if not np.isclose(metrics["ap"], ap["mAP"], rtol=0, atol=1e-12):
            continue
        runs.append({"group": group, "variant": variant, "seed": seed, "metrics": metrics})
    return runs


def aggregate_runs(runs):
    groups = {}
    for run in runs:
        groups.setdefault((run["group"], run["variant"]), []).append(run)
    output = {}
    for (group, variant), items in sorted(groups.items()):
        values = {}
        for metric in METRICS:
            series = [item["metrics"][metric] for item in items if metric in item["metrics"]]
            if not series:
                continue
            values[metric] = {
                "mean": float(np.mean(series)),
                "std": float(np.std(series, ddof=1)) if len(series) > 1 else 0.0,
                "values": series,
            }
        output[f"{group}/{variant}"] = {"seeds": [item["seed"] for item in items], **values}
    return output


def paired_statistics(ours, baseline, metrics=None):
    seeds = sorted(set(ours["seeds"]) & set(baseline["seeds"]))
    result = {}
    for metric in metrics or METRICS:
        ours_map = dict(zip(ours["seeds"], ours[metric]["values"]))
        baseline_map = dict(zip(baseline["seeds"], baseline[metric]["values"]))
        differences = np.array([ours_map[seed] - baseline_map[seed] for seed in seeds])
        n = len(differences)
        mean = float(differences.mean())
        sd = float(differences.std(ddof=1)) if n > 1 else 0.0
        if n > 1 and sd > 0:
            t_value = mean / (sd / np.sqrt(n))
            p_value = float(2 * stats.t.sf(abs(t_value), n - 1))
            ci = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
            dz = mean / sd
        else:
            t_value, p_value, ci, dz = 0.0, 1.0, 0.0, 0.0
        result[metric] = {
            "n": n,
            "mean_difference": mean,
            "std_difference": sd,
            "t": float(t_value),
            "df": n - 1,
            "p": p_value,
            "cohen_dz": float(dz),
            "ci95": [float(mean - ci), float(mean + ci)],
        }
    ordered = sorted(result, key=lambda key: result[key]["p"])
    running_max = 0.0
    for rank, metric in enumerate(ordered, start=1):
        running_max = max(
            running_max, result[metric]["p"] * (len(ordered) - rank + 1)
        )
        result[metric]["p_holm"] = min(1.0, running_max)
    return result


def write_summary(aggregate, statistics, output: Path):
    lines = ["# nuScenes-mini experiment summary", "",
             "All values are computed from raw per-seed metrics; no values are copied from the full-dataset manuscript.", ""]
    for variant, values in aggregate.items():
        seeds = values["seeds"]
        lines.append(f"## {variant} (seeds={seeds})")
        lines.append("| Metric | Mean | SD | Raw values |")
        lines.append("|---|---:|---:|---|")
        for metric in METRICS:
            item = values[metric]
            raw = ", ".join(f"{value:.6g}" for value in item["values"])
            lines.append(f"| {metric} | {item['mean']:.6g} | {item['std']:.6g} | {raw} |")
        lines.append("")
    if statistics:
        lines.append("## Full vs. Baseline: paired statistics")
        lines.append("| Metric | Mean Δ | t(df) | Exact p | Holm p | Cohen dz | 95% CI |")
        lines.append("|---|---:|---:|---:|---:|---:|---|")
        for metric, item in statistics.items():
            lines.append(
                f"| {metric} | {item['mean_difference']:.6g} | {item['t']:.4g}({item['df']}) | "
                f"{item['p']:.4g} | {item['p_holm']:.4g} | {item['cohen_dz']:.4g} | "
                f"[{item['ci95'][0]:.6g}, {item['ci95'][1]:.6g}] |"
            )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="output/mini/runs")
    parser.add_argument("--output", default="output/mini/summary")
    args = parser.parse_args()
    root, output = Path(args.root), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    runs = collect_runs(root)
    aggregate = aggregate_runs(runs)
    statistics = None
    paired_groups = [aggregate.get("main/full"), aggregate.get("main/baseline")]
    expected_seeds = {42, 123, 456, 789, 1024}
    if all(item is not None and set(item["seeds"]) == expected_seeds for item in paired_groups):
        statistics = paired_statistics(aggregate["main/full"], aggregate["main/baseline"])
    (output / "metrics_by_seed.json").write_text(
        json.dumps(runs, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "aggregate.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    if statistics:
        (output / "statistics.json").write_text(
            json.dumps(statistics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary(aggregate, statistics, output / "summary.md")
    print(f"Collected {len(runs)} runs into {output}")


if __name__ == "__main__":
    main()
