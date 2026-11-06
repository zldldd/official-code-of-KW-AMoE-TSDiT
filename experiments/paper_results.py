"""Build mini-only tables and figures from raw run artifacts, never manuscript values."""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from experiments.summarize import METRICS, aggregate_runs, collect_runs, paired_statistics


EXPECTED = {
    "main": ["baseline", "vanilla_moe", "kw_moe", "ts_mem", "full"],
    "mechanism": ["independent", "unmasked", "matched_budget", "all_write", "random_write", "shuffled", "kw_moe", "full"],
    "rho": ["rho_0.1", "rho_0.2", "rho_0.3", "rho_0.4", "rho_0.5"],
    "expert": [f"E{e}_K{k}" for e in [4, 8, 16] for k in [1, 2, 4]],
}
SEEDS = {42, 123, 456, 789, 1024}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def mean_sd(values):
    values = list(values)
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def cell(item, metric, digits=4):
    if metric not in item:
        return "N/A"
    value = item[metric]
    if len(item.get("seeds", [])) < 2:
        return f"{value['mean']:.{digits}g} (single seed)"
    return f"{value['mean']:.{digits}g} ± {value['std']:.{digits}g}"


def quality_table(aggregate, group, names):
    lines = ["| Variant | FVD ↓ | FID ↓ | LPIPS ↓ | 2D AP@0.5 ↑ | CLIP ↑ | Completed seeds |", "|---|---:|---:|---:|---:|---:|---|"]
    for name in names:
        item = aggregate.get(f"{group}/{name}")
        if item is None:
            lines.append(f"| {name} | pending | pending | pending | pending | pending | 0/5 |")
        else:
            lines.append(f"| {name} | " + " | ".join(cell(item, metric) for metric in METRICS) + f" | {len(item['seeds'])}/5 |")
    return lines


def expert_grid_table(root, aggregate):
    lines = ["| Variant | FVD ↓ | FID ↓ | LPIPS ↓ | 2D AP@0.5 ↑ | CLIP ↑ | Rel. FLOPs | Usage entropy | Seeds |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    reference_path = root / "expert" / "E8_K2" / "seed42" / "profile.json"
    reference = read_json(reference_path).get("flops_per_step") if reference_path.is_file() else None
    for name in EXPECTED["expert"]:
        item = aggregate.get(f"expert/{name}")
        if item is None:
            lines.append(f"| {name} | pending | pending | pending | pending | pending | pending | pending | 0/5 |")
            continue
        profile_path = root / "expert" / name / "seed42" / "profile.json"
        profile = read_json(profile_path) if profile_path.is_file() else {}
        ratio = (f"{profile['flops_per_step'] / reference:.4f}×"
                 if reference and profile.get("profile_status") == "operator_profile_only; no latency measurement"
                 else "pending")
        paths = sorted((root / "expert" / name).glob("seed*/routing_summary.json"))
        entropies = [read_json(path)["normalized_usage_entropy"] for path in paths]
        entropy = f"{np.mean(entropies):.4f}" if entropies else "pending"
        lines.append(f"| {name} | " + " | ".join(cell(item, metric) for metric in METRICS)
                     + f" | {ratio} | {entropy} | {len(item['seeds'])}/5 |")
    return lines


def valid_quality_record(path, sample_count, seed):
    if not path.is_file() or not path.with_name("ap.json").is_file():
        return False
    item = read_json(path)
    ap = read_json(path.with_name("ap.json"))
    return (
        item.get("evaluation_status") == "complete"
        and all(metric in item for metric in METRICS)
        and item.get("config", {}).get("seed") == seed
        and item.get("evaluated_scenes") == (["scene-0103", "scene-0916"] if sample_count == 16 else ["scene-0103"])
        and item.get("sample_count") == sample_count
        and item.get("frames_per_view") == 16
        and item.get("image_count_per_distribution") == sample_count * 96
        and item.get("fid_device") == "cuda"
        and len(item.get("fvd_per_view", [])) == 6
        and len(item.get("clip_score_per_clip", [])) == sample_count
        and item.get("noise_seeds") == [seed + 10000 + index for index in range(sample_count)]
        and "input_pixel_format" in ap
        and ap.get("input_video_dtype") == "torch.uint8"
        and Path(ap.get("detector", "")).as_posix().endswith("output/mini/downstream/real/seed42/detector.pt")
        and math.isclose(item["ap"], ap.get("mAP", float("nan")), rel_tol=0, abs_tol=1e-12)
    )


def valid_long_record(path, seed, frames):
    if not path.is_file():
        return False
    item = read_json(path)
    count = 6 if frames == 64 else 2
    if (item.get("frames") != frames or item.get("training_seed") != seed
            or item.get("sequence_count") != count
            or len(item.get("observed_span_seconds", [])) != count
            or len(item.get("segments", [])) != frames // 16):
        return False
    for index, segment in enumerate(item["segments"]):
        if (segment.get("sample_count") != count
                or segment.get("frames_per_view") != 16
                or segment.get("image_count_per_distribution") != count * 96
                or segment.get("fid_device") != "cuda"
                or segment.get("segment_index") != index
                or segment.get("start_frame") != index * 16
                or segment.get("end_frame") != (index + 1) * 16
                or len(segment.get("fvd_per_view", [])) != 6
                or len(segment.get("clip_score_per_clip", [])) != count
                or any(metric not in segment for metric in ["fvd", "fvd_feature", "fid", "lpips", "clip_score"])):
            return False
    return all(key in item.get("aggregate", {}) and key in item.get("late", {})
               for key in ["fvd", "fid", "lpips", "clip_score"])


def check_complete(aggregate, root, downstream_root, domain_root):
    missing = []
    for group, variants in EXPECTED.items():
        for variant in variants:
            key = f"{group}/{variant}"
            if key not in aggregate or set(aggregate[key]["seeds"]) != SEEDS:
                missing.append(key)
            elif any(metric not in aggregate[key] for metric in METRICS):
                missing.append(key + ":metrics")
            for seed in SEEDS:
                run_dir = root / group / variant / f"seed{seed}"
                metrics_path = run_dir / "metrics.json"
                if not valid_quality_record(metrics_path, 16 if group == "main" else 4, seed):
                    missing.append(f"{key}/seed{seed}:protocol")
            benchmark_path = root / group / variant / "seed42" / "benchmark.json"
            if not benchmark_path.is_file() or read_json(benchmark_path).get("timing_status") != "measured_without_concurrent_study_jobs":
                missing.append(key + ":benchmark_protocol")
            if group in {"expert", "rho"}:
                profile_path = root / group / variant / "seed42" / "profile.json"
                if not profile_path.is_file() or read_json(profile_path).get("profile_status") != "operator_profile_only; no latency measurement":
                    missing.append(key + ":profile_protocol")
        for variant in (["baseline", "full"] if group == "main" else variants if group in {"rho", "expert"} else ["matched_budget", "full"]):
            for seed in SEEDS:
                if not (root / group / variant / f"seed{seed}" / "routing_summary.json").is_file():
                    missing.append(f"{group}/{variant}/seed{seed}:routing")
    for mode in ["real", "baseline", "full"]:
        for seed in SEEDS:
            path = downstream_root / mode / f"seed{seed}" / "downstream_metrics.json"
            item = read_json(path) if path.is_file() else {}
            if (item.get("mode") != mode or item.get("seed") != seed or item.get("steps") != 200
                    or item.get("detector_training_scenes") != "official mini_train"
                    or item.get("detector_validation_scenes") != "official mini_val"):
                missing.append(f"downstream/{mode}/seed{seed}")
    for protocol in ["zero-shot", "adapt"]:
        for variant in ["baseline", "full"]:
            for seed in SEEDS:
                path = domain_root / protocol / variant / f"seed{seed}" / "metrics.json"
                if not valid_domain_record(path):
                    missing.append(f"domain/{protocol}/{variant}/seed{seed}")
    for variant in ["baseline", "full"]:
        for seed in SEEDS:
            for frames in [64, 128]:
                path = root / "main" / variant / f"seed{seed}" / f"metrics_long{frames}.json"
                if not valid_long_record(path, seed, frames):
                    missing.append(f"long/{variant}/seed{seed}/{frames}")
    if missing:
        raise RuntimeError("Incomplete experiment evidence: " + ", ".join(missing))


def benchmark_table(root):
    lines = ["| Group/variant | Params (M) | FLOPs/step (G) | Expert calls | step/s | Measured 50-step denoising (s) | 20-step end-to-end (s) | clips/s | Timing status |", "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for group, variants in EXPECTED.items():
        for variant in variants:
            path = root / group / variant / "seed42" / "benchmark.json"
            if not path.is_file():
                continue
            item = read_json(path)
            lines.append(
                f"| {group}/{variant} | {item['params']/1e6:.4f} | {item['flops_per_step']/1e9:.4f} | "
                f"{item['expert_calls']:.0f} | {item['denoising_steps_per_second']:.4f} | "
                f"{item['latency_50_step_denoising']:.4f} | {item['end_to_end_latency']:.4f} | {item['end_to_end_clips_per_second']:.5f} | {item.get('timing_status', 'development_or_contended')} |"
            )
    return lines


def downstream_table(root):
    lines = ["| Training data | 2D mAP ↑ | Car AP ↑ | Pedestrian AP ↑ | Synthetic frames | Seeds |", "|---|---:|---:|---:|---:|---|"]
    values = {}
    for mode in ["real", "baseline", "full"]:
        items = [read_json(path) for path in sorted(root.glob(f"{mode}/seed*/downstream_metrics.json"))]
        if not items:
            continue
        values[mode] = items
        m, sd = mean_sd([item["mAP"] for item in items])
        car, car_sd = mean_sd([item["per_class"].get("vehicle.car", 0) for item in items])
        ped, ped_sd = mean_sd([item["per_class"].get("human.pedestrian.adult", 0) for item in items])
        lines.append(f"| {mode} | {m:.6g} ± {sd:.3g} | {car:.6g} ± {car_sd:.3g} | {ped:.6g} ± {ped_sd:.3g} | {items[0]['synthetic_training_frames']} | {len(items)} |")
    return lines, values


def valid_domain_record(path):
    if not path.is_file() or not path.with_name("ap.json").is_file():
        return False
    item = read_json(path)
    ap = read_json(path.with_name("ap.json"))
    return (
        item.get("evaluation_status") == "complete"
        and all(metric in item for metric in METRICS)
        and item.get("config", {}).get("seed") == int(path.parent.name[4:])
        and item.get("evaluated_scenes") == ["scene-1094"]
        and item.get("noise_seeds") == [int(path.parent.name[4:]) + 10000 + index for index in range(13)]
        and item.get("sample_count") == 13
        and item.get("frames_per_view") == 16
        and item.get("image_count_per_distribution") == 1248
        and item.get("fid_device") == "cuda"
        and item.get("fvd_view_aggregation") == "macro_average_of_six_separate_view_fvds"
        and item.get("fid_view_aggregation") == "pool_all_six_views_and_sixteen_frames"
        and len(item.get("fvd_per_view", [])) == 6
        and len(item.get("clip_score_per_clip", [])) == 13
        and ap.get("sample_frames") == 168
        and "input_pixel_format" in ap
        and ap.get("input_video_dtype") == "torch.uint8"
        and Path(ap.get("detector", "")).as_posix().endswith("output/mini/domain_detector/real/seed42/detector.pt")
        and math.isclose(item["ap"], ap.get("mAP", float("nan")), rel_tol=0, abs_tol=1e-12)
    )


def domain_table(root):
    lines = ["| Protocol | Method | FVD ↓ | FID ↓ | LPIPS ↓ | 2D AP ↑ | Completed seeds |", "|---|---|---:|---:|---:|---:|---:|"]
    for protocol in ["zero-shot", "adapt"]:
        for variant in ["baseline", "full"]:
            paths = [path for path in sorted(root.glob(f"{protocol}/{variant}/seed*/metrics.json")) if valid_domain_record(path)]
            if not paths:
                continue
            items = [read_json(path) for path in paths]
            row = []
            for metric in ["fvd", "fid", "lpips", "ap"]:
                mean, sd = mean_sd([item[metric] for item in items])
                row.append(f"{mean:.5g}" + (f" ± {sd:.3g}" if len(items) > 1 else " (single seed)"))
            lines.append(f"| {protocol} | {variant} | " + " | ".join(row) + f" | {len(items)}/5 |")
    return lines


def mechanism_paired_statistics(aggregate):
    comparisons = {}
    full = aggregate.get("mechanism/full")
    if full is None or set(full["seeds"]) != SEEDS:
        return comparisons
    for variant in EXPECTED["mechanism"]:
        if variant == "full":
            continue
        item = aggregate.get(f"mechanism/{variant}")
        if item is not None and set(item["seeds"]) == SEEDS:
            comparisons[variant] = paired_statistics(item, full)
    return comparisons


def domain_paired_statistics(root, protocol):
    groups = {}
    for variant in ["baseline", "full"]:
        paths = [path for path in sorted(root.glob(f"{protocol}/{variant}/seed*/metrics.json"))
                 if valid_domain_record(path)]
        items = {int(path.parent.name[4:]): read_json(path) for path in paths}
        if set(items) != SEEDS:
            return None
        group = {"seeds": sorted(items)}
        for metric in METRICS:
            group[metric] = {"values": [items[seed][metric] for seed in sorted(items)]}
        groups[variant] = group
    return paired_statistics(groups["full"], groups["baseline"])


def domain_adaptation_statistics(root, variant):
    groups = {}
    for protocol in ["zero-shot", "adapt"]:
        paths = [path for path in sorted(root.glob(f"{protocol}/{variant}/seed*/metrics.json"))
                 if valid_domain_record(path)]
        items = {int(path.parent.name[4:]): read_json(path) for path in paths}
        if set(items) != SEEDS:
            return None
        groups[protocol] = {"seeds": sorted(items)}
        for metric in METRICS:
            groups[protocol][metric] = {"values": [items[seed][metric] for seed in sorted(items)]}
    return paired_statistics(groups["adapt"], groups["zero-shot"])


def long_paired_statistics(root, frames):
    groups = {}
    for variant in ["baseline", "full"]:
        paths = [path for path in sorted(root.glob(f"main/{variant}/seed*/metrics_long{frames}.json"))
                 if valid_long_record(path, int(path.parent.name[4:]), frames)]
        items = {int(path.parent.name[4:]): read_json(path) for path in paths}
        if set(items) != SEEDS:
            return None
        groups[variant] = {"seeds": sorted(items)}
        for metric, scope, key in [
            ("mean_fvd", "aggregate", "fvd"), ("late_fvd", "late", "fvd"),
            ("mean_fid", "aggregate", "fid"), ("mean_lpips", "aggregate", "lpips"),
            ("late_lpips", "late", "lpips"), ("clip_score", "aggregate", "clip_score"),
        ]:
            groups[variant][metric] = {"values": [items[seed][scope][key] for seed in sorted(items)]}
    return paired_statistics(groups["full"], groups["baseline"],
                             ["mean_fvd", "late_fvd", "mean_fid", "mean_lpips", "late_lpips", "clip_score"])


def domain_statistics_table(comparison, protocol, title=None):
    heading = title or f"Full − Baseline: {protocol}"
    lines = [f"### {heading} paired statistics", "",
             "| Metric | Mean Δ | t(df) | Exact p | Holm p | Cohen dz | 95% CI |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for metric, item in comparison.items():
        lines.append(f"| {metric} | {item['mean_difference']:.6g} | {item['t']:.4g}({item['df']}) | {item['p']:.4g} | {item['p_holm']:.4g} | {item['cohen_dz']:.4g} | [{item['ci95'][0]:.5g}, {item['ci95'][1]:.5g}] |")
    return lines


def long_table(root):
    lines = ["| Frames | Method | Sequences/seed | Timestamp span (s) | Mean segment FVD ↓ | Late FVD ↓ | Mean FID ↓ | Mean LPIPS ↓ | Late LPIPS ↓ | Seeds |", "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for frames in [64, 128]:
        for variant in ["baseline", "full"]:
            paths = [path for path in sorted(root.glob(f"main/{variant}/seed*/metrics_long{frames}.json"))
                     if valid_long_record(path, int(path.parent.name[4:]), frames)]
            if not paths:
                continue
            items = [read_json(path) for path in paths]
            values = []
            for key in ["fvd", "fid", "lpips"]:
                mean, sd = mean_sd([item["aggregate"][key] for item in items])
                values.append(f"{mean:.5g}" + (f" ± {sd:.3g}" if len(items) > 1 else " (single seed)"))
            late_fvd, late_fvd_sd = mean_sd([item["late"]["fvd"] for item in items])
            late_lpips, late_lpips_sd = mean_sd([item["late"]["lpips"] for item in items])
            sequence_count = items[0]["sequence_count"]
            spans = sorted({span for item in items for span in item["observed_span_seconds"]})
            span_text = f"{min(spans):.2f}–{max(spans):.2f}" if spans else "N/A"
            suffix = f" ± {late_fvd_sd:.3g}" if len(items) > 1 else " (single seed)"
            lpips_suffix = f" ± {late_lpips_sd:.3g}" if len(items) > 1 else " (single seed)"
            lines.append(f"| {frames} | {variant} | {sequence_count} | {span_text} | {values[0]} | {late_fvd:.5g}{suffix} | {values[1]} | {values[2]} | {late_lpips:.5g}{lpips_suffix} | {len(items)} |")
    return lines


def routing_table(root, group, variants):
    lines = ["| Variant | States | Weight sum max error | Non-key max weight | Max half reproduction diff | Usage entropy | Mean expert calls (key/non-key) |", "|---|---:|---:|---:|---:|---:|---:|"]
    for variant in variants:
        paths = sorted(root.glob(f"{group}/{variant}/seed*/routing_summary.json"))
        if not paths:
            continue
        items = [read_json(path) for path in paths]
        entropies = [item["normalized_usage_entropy"] for item in items if item["normalized_usage_entropy"] is not None]
        entropy = f"{np.mean(entropies):.5g}" if entropies else "N/A"
        lines.append(f"| {variant} | {sum(item['states'] for item in items)} | {max(item['weight_sum_max_abs_error'] for item in items):.3g} | {max(item['nonkeyframe_max_abs_weight'] or 0 for item in items):.3g} | {max(item['max_abs_reproduction_difference_from_saved_half'] for item in items):.3g} | {entropy} | {np.mean([item['mean_keyframe_expert_calls'] for item in items]):.5g}/{np.mean([item['mean_nonkeyframe_expert_calls'] for item in items]):.5g} |")
    return lines


def scalar_comparison(control, treatment):
    control_values = {item["seed"]: item["mAP"] for item in control}
    treatment_values = {item["seed"]: item["mAP"] for item in treatment}
    seeds = sorted(set(control_values) & set(treatment_values))
    delta = np.array([treatment_values[seed] - control_values[seed] for seed in seeds])
    if len(seeds) < 2:
        return {"n": len(seeds), "mean_delta": float(delta.mean()) if seeds else None,
                "p": None, "cohen_dz": None, "ci95": None}
    sd = delta.std(ddof=1)
    test = stats.ttest_rel([treatment_values[seed] for seed in seeds], [control_values[seed] for seed in seeds]) if sd > 0 else None
    ci = stats.t.ppf(0.975, len(seeds) - 1) * sd / np.sqrt(len(seeds)) if sd > 0 else 0.0
    return {
        "n": len(seeds), "mean_delta": float(delta.mean()),
        "p": float(test.pvalue) if test else 1.0,
        "cohen_dz": float(delta.mean() / sd) if sd > 0 else 0.0,
        "ci95": [float(delta.mean() - ci), float(delta.mean() + ci)],
    }


def save_csv(path, header, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def plot_main(aggregate, output):
    names = EXPECTED["main"]
    if not all(f"main/{name}" in aggregate and set(aggregate[f"main/{name}"]["seeds"]) == SEEDS for name in names):
        return
    fig, axes = plt.subplots(1, 5, figsize=(19, 4))
    for axis, metric in zip(axes, METRICS):
        means = [aggregate[f"main/{name}"][metric]["mean"] for name in names]
        errors = [aggregate[f"main/{name}"][metric]["std"] for name in names]
        axis.bar(range(len(names)), means, yerr=errors, capsize=3)
        axis.set_xticks(range(len(names)), names, rotation=25, ha="right")
        axis.set_title(metric)
    fig.suptitle("nuScenes-mini: mean ± SD over five independent training seeds")
    fig.tight_layout()
    fig.savefig(output / "fig9_component_ablation.png", dpi=220)
    plt.close(fig)

    angles = np.linspace(0, 2 * np.pi, len(METRICS), endpoint=False).tolist()
    angles += angles[:1]
    fig, axis = plt.subplots(figsize=(9, 7), subplot_kw={"polar": True})
    for name in names:
        scores = []
        for metric in METRICS:
            values = [aggregate[f"main/{variant}"][metric]["mean"] for variant in names]
            low, high = min(values), max(values)
            value = aggregate[f"main/{name}"][metric]["mean"]
            score = 0.5 if high == low else ((high - value) if metric in {"fvd", "fid", "lpips"} else (value - low)) / (high - low)
            scores.append(score)
        axis.plot(angles, scores + scores[:1], label=name)
    axis.set_xticks(angles[:-1], ["FVD", "FID", "LPIPS", "2D AP", "CLIP"])
    axis.set_title("Within-mini min–max normalization\nNot a cross-paper comparison", fontsize=11)
    axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1.12), fontsize=9)
    fig.subplots_adjust(left=0.06, right=0.78, top=0.84, bottom=0.08)
    fig.savefig(output / "fig5_radar.png", dpi=220)
    plt.close(fig)


def plot_efficiency(root, aggregate, output):
    items = []
    for name in EXPECTED["main"]:
        path = root / "main" / name / "seed42" / "benchmark.json"
        if path.is_file() and f"main/{name}" in aggregate:
            items.append((name, read_json(path), aggregate[f"main/{name}"]))
    if len(items) != len(EXPECTED["main"]) or any(set(item[2]["seeds"]) != SEEDS for item in items):
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    names = [item[0] for item in items]
    axes[0].bar(names, [item[1]["denoising_steps_per_second"] for item in items])
    axes[0].set_ylabel("Denoising steps/s (higher is faster)")
    axes[1].bar(names, [item[1]["end_to_end_latency"] for item in items])
    axes[1].set_ylabel("20-step end-to-end seconds/clip (lower is faster)")
    for axis in axes:
        axis.tick_params(axis="x", rotation=25)
    fig.suptitle("Measured seed42 inference; not an equivalent frame-rate claim")
    fig.tight_layout()
    fig.savefig(output / "fig10_efficiency.png", dpi=220)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(7, 5))
    points = sorted((item[1]["end_to_end_latency"], item[2]["fvd"]["mean"], item[0]) for item in items)
    frontier = []
    best = float("inf")
    for x, y, name in points:
        axis.scatter(x, y)
        axis.annotate(name, (x, y))
        if y < best:
            frontier.append((x, y))
            best = y
    if frontier:
        axis.plot(*zip(*frontier), linestyle="--", label="Observed mini Pareto set")
        axis.legend()
    axis.set_xlabel("End-to-end latency/clip (s)")
    axis.set_ylabel("FVD (five-seed mean)")
    fig.tight_layout()
    fig.savefig(output / "fig11_pareto.png", dpi=220)
    plt.close(fig)


def plot_long(root, output):
    for frames in [64, 128]:
        fig, axis = plt.subplots(figsize=(8, 5))
        found = False
        for variant in ["baseline", "full"]:
            paths = [path for path in sorted(root.glob(f"main/{variant}/seed*/metrics_long{frames}.json"))
                     if valid_long_record(path, int(path.parent.name[4:]), frames)]
            if not paths:
                continue
            items = [read_json(path) for path in paths]
            x = np.arange(len(items[0]["segments"]))
            means = np.mean([[segment["fvd"] for segment in item["segments"]] for item in items], axis=0)
            errors = np.std([[segment["fvd"] for segment in item["segments"]] for item in items], axis=0, ddof=1) if len(items) > 1 else None
            error_options = {"yerr": errors, "capsize": 3} if errors is not None else {}
            axis.errorbar(x, means, marker="o", label=f"{variant} (n={len(items)})", **error_options)
            found = True
        if found:
            axis.set_xlabel("Non-overlapping 16-frame segment index")
            axis.set_ylabel("FVD")
            axis.set_title(f"Exploratory mini {frames}-frame rollout; sample SD shown only for n≥2")
            axis.legend()
            fig.tight_layout()
            fig.savefig(output / f"long_{frames}_fvd.png", dpi=220)
        plt.close(fig)


def plot_hyperparameters(root, aggregate, output):
    if all(f"expert/E{e}_K{k}" in aggregate and set(aggregate[f"expert/E{e}_K{k}"]["seeds"]) == SEEDS for e in [4, 8, 16] for k in [1, 2, 4]):
        values = np.array([[aggregate[f"expert/E{e}_K{k}"]["fvd"]["mean"] for k in [1, 2, 4]] for e in [4, 8, 16]])
        fig, axis = plt.subplots(figsize=(7, 5))
        image = axis.imshow(values, aspect="auto", cmap="viridis_r")
        for row in range(3):
            for col in range(3):
                value = values[row, col]
                normalized = (value - values.min()) / (values.max() - values.min())
                color = "black" if normalized < 0.55 else "white"
                axis.text(col, row, f"{value:.1f}", ha="center", va="center", color=color)
        axis.set_xticks(range(3), [1, 2, 4])
        axis.set_yticks(range(3), [4, 8, 16])
        axis.set_xlabel("Keyframe expert budget $K_e$")
        axis.set_ylabel("Number of experts $E$")
        axis.set_title("500-step mini grid: five-seed mean FVD")
        fig.colorbar(image, ax=axis)
        fig.tight_layout()
        fig.savefig(output / "expert_grid_fvd.png", dpi=220)
        plt.close(fig)
    items = []
    reference = None
    for name in EXPECTED["rho"]:
        path = root / "rho" / name / "seed42" / "profile.json"
        if not path.is_file():
            continue
        item = read_json(path)
        items.append((name, item))
        if name == "rho_0.3":
            reference = item["flops_per_step"]
    if items and reference:
        fig, axis = plt.subplots(figsize=(8, 5))
        labels = [name for name, _ in items]
        ratios = [item["flops_per_step"] / reference for _, item in items]
        axis.bar(labels, ratios)
        axis.axhline(1, color="black", linestyle="--")
        axis.set_ylabel("Measured FLOPs relative to ρ=0.3")
        axis.set_title("Effective keyframe counts are quantized to ceil(4ρ)")
        axis.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        fig.savefig(output / "rho_flops.png", dpi=220)
        plt.close(fig)


def write_report(root, output, downstream_root, domain_root, require_complete=False):
    aggregate = aggregate_runs(collect_runs(root))
    if require_complete:
        check_complete(aggregate, root, downstream_root, domain_root)
    output.mkdir(parents=True, exist_ok=True)
    lines = [
        "# KW-AMoE-TSDiT — nuScenes-mini experimental evidence", "",
        "> 本文件由 `experiments/paper_results.py` 从逐种子原始 JSON 自动生成。所有结果仅是降规模 mini 验证，不是原 700/150 场景基准，也不支持与外部论文不同协议数字直接比较。负面结果和零 AP 均保留。", "",
        "**Protocol:** six views, 16 pixel frames, frozen CogVideoX-2b 3D-VAE, four latent temporal positions. Main models use 1,000 optimization steps and 16 validation clips; grids/mechanism controls use 500 steps and four clips. All use five independent training seeds and 20-step DDIM. FID pools 1,536 images for main models or 384 images for supplemental models per distribution.", "",
        "## Main component comparison", "",
    ]
    lines += quality_table(aggregate, "main", EXPECTED["main"])
    paired_groups = [aggregate.get("main/full"), aggregate.get("main/baseline")]
    if all(item is not None and set(item["seeds"]) == SEEDS for item in paired_groups):
        comparison = paired_statistics(aggregate["main/full"], aggregate["main/baseline"])
        lines += ["", "## Full vs. Baseline paired statistics", "", "| Metric | Mean Δ (Ours−Baseline) | t(df) | Exact p | Holm p | Cohen dz | 95% CI |", "|---|---:|---:|---:|---:|---:|---|"]
        for metric, item in comparison.items():
            lines.append(f"| {metric} | {item['mean_difference']:.6g} | {item['t']:.4g}({item['df']}) | {item['p']:.4g} | {item['p_holm']:.4g} | {item['cohen_dz']:.4g} | [{item['ci95'][0]:.5g}, {item['ci95'][1]:.5g}] |")
    lines += ["", "## Mechanism controls (matched 500-step reference)", ""]
    lines += quality_table(aggregate, "mechanism", EXPECTED["mechanism"])
    for variant, comparison in mechanism_paired_statistics(aggregate).items():
        lines += [""] + domain_statistics_table(comparison, "", title=f"{variant} − Full (500 steps)")
    lines += ["", "## Keyframe ratio", "", "With four latent temporal positions, `ceil(4ρ)` gives only two effective ratios here: 0.25 for ρ=0.1/0.2 and 0.5 for ρ=0.3/0.4/0.5. Duplicate effective configurations must not be interpreted as independent evidence of a ρ trend.", ""]
    lines += quality_table(aggregate, "rho", EXPECTED["rho"])
    lines += ["", "## Complete E × keyframe-budget grid", ""]
    lines += expert_grid_table(root, aggregate)
    lines += ["", "## Measured efficiency (seed42; final timing requires no concurrent jobs from this study; other system tenants are not claimed to be absent)", ""]
    lines += benchmark_table(root)
    lines += ["", "## Long-horizon rollout", "", "Chunks contain 16 frames with zero overlap and absolute latent time indices in their temporal encoding. TS-Mem resets once per full sequence, not per chunk. No real future latent is fed back. The observed mini median camera interval is 100 ms; nominal first-to-last spans are 6.3 s (64 frames) and 12.7 s (128 frames), while actual per-sequence timestamp spans are reported in the table and raw JSON. Six long validation sequences exist at 64 frames (measured spans 5.30–5.75 s) and two at 128 frames (measured spans 10.75 and 11.25 s). These are exploratory temporal diagnostics, not autoregressive pixel-frame generation.", ""]
    lines += long_table(root)
    for frames in [64, 128]:
        comparison = long_paired_statistics(root, frames)
        if comparison is not None:
            lines += [""] + domain_statistics_table(comparison, f"{frames} frames")
    lines += ["", "## Downstream 2D detection augmentation", "", "Faster R-CNN ResNet-50-FPN is trained from the same per-seed initialization with 200 steps. Synthetic clips use only training-scene conditions and are deduplicated by sample_data token. Evaluation is exclusively on real validation keyframes. This is **2D camera AP at IoU=0.5**, not 3D mAP/NDS.", ""]
    downstream_lines, downstream = downstream_table(downstream_root)
    lines += downstream_lines
    if "real" in downstream and "baseline" in downstream and "full" in downstream:
        comparisons = {control: scalar_comparison(downstream[control], downstream["full"])
                       for control in ["real", "baseline"]}
        ordered = sorted((control for control in comparisons if comparisons[control]["p"] is not None),
                         key=lambda control: comparisons[control]["p"])
        running_max = 0.0
        for rank, control in enumerate(ordered, start=1):
            running_max = max(running_max, comparisons[control]["p"] * (len(ordered) - rank + 1))
            comparisons[control]["p_holm"] = min(1.0, running_max)
        for control, result in comparisons.items():
            if result["p"] is None:
                lines.append(f"\nFull − {control}: paired seeds={result['n']}; statistical test withheld until at least two pairs are available.")
            else:
                lines.append(f"\nFull − {control}: ΔmAP={result['mean_delta']:.6g}, exact p={result['p']:.4g}, Holm p={result['p_holm']:.4g}, Cohen dz={result['cohen_dz']:.4g}, 95% CI={result['ci95']}.")
    lines += ["", "## Daytime→nighttime transfer within nuScenes", "", "Source daytime scenes: scene-0061/0553/0655/0757/0796. Target validation: mini_train night scene-1094, explicitly held out from every domain-training stage (13 clips; not an official mini_val scene). Adaptation: disjoint scene-1077/1100, initialized from the corresponding zero-shot checkpoint for 200 steps. Domain AP uses a source-day-only detector (1,080 training frames, 164 unique target real keyframes), not a detector trained on the target scene. AP-on-generated contains 168 synchronized annotation selections representing those 164 unique keyframes. This is not cross-dataset generalization.", ""]
    lines += domain_table(domain_root)
    for protocol in ["zero-shot", "adapt"]:
        comparison = domain_paired_statistics(domain_root, protocol)
        if comparison is not None:
            lines += [""] + domain_statistics_table(comparison, protocol)
    for variant in ["baseline", "full"]:
        comparison = domain_adaptation_statistics(domain_root, variant)
        if comparison is not None:
            lines += [""] + domain_statistics_table(comparison, "", title=f"{variant}: adaptation − zero-shot")
    convergence = aggregate_runs([
        run for run in collect_runs(root.parent)
        if run["group"] == "convergence"
    ])
    if convergence:
        lines += ["", "## Second-stage convergence probes (seed42)", "", "Each probe starts from its 1000-step checkpoint and adds 4000 steps with a reinitialized optimizer. It is not a continuous 5000-step run and is not mixed with the main table.", ""]
        lines += quality_table(convergence, "convergence", ["baseline", "full"])
    lines += ["", "## Routing consistency and usage entropy", "", "### Main models", ""]
    lines += routing_table(root, "main", ["baseline", "full"])
    lines += ["", "### Matched-budget mechanism controls", ""]
    lines += routing_table(root, "mechanism", ["matched_budget", "full"])
    lines += ["", "### Keyframe ratios", ""]
    lines += routing_table(root, "rho", EXPECTED["rho"])
    lines += ["", "### Expert grid", ""]
    lines += routing_table(root, "expert", EXPECTED["expert"])
    lines += ["", "## Evidence boundaries", "", "- No MagicDrive-V2 or DiVE numbers are copied into these mini tables because no same-protocol mini checkpoints were run.", "- FVD/FID on this small mini subset have unstable high-dimensional covariance estimates; treat them as exploratory, not full-benchmark replacements.", "- Keyframe visualizations show learned states, not human-verified turn/lane-change event labels.", "- Full-dataset reruns, external-method same-protocol comparisons, and author-confirmed metadata remain submission tasks.", ""]
    (output / "mini_results.md").write_text("\n".join(lines), encoding="utf-8")
    plot_main(aggregate, output)
    plot_efficiency(root, aggregate, output)
    plot_hyperparameters(root, aggregate, output)
    plot_long(root, output)
    print(f"Wrote {output / 'mini_results.md'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="output/mini/runs")
    parser.add_argument("--downstream-root", default="output/mini/downstream_controlled")
    parser.add_argument("--domain-root", default="output/mini/domain")
    parser.add_argument("--output", default="output/mini/report")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    write_report(Path(args.root), Path(args.output), Path(args.downstream_root), Path(args.domain_root), args.require_complete)


if __name__ == "__main__":
    main()
