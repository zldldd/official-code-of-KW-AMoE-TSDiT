"""Resume local mini experiments without downloads or duplicate stage runners."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


SEEDS = [42, 123, 456, 789, 1024]
VARIANTS = ["baseline", "full"]
GROUPS = {
    "main": ["baseline", "full"],
    "mechanism": ["matched_budget", "full"],
    "rho": [f"rho_{rho}" for rho in [0.1, 0.2, 0.3, 0.4, 0.5]],
    "expert": [f"E{e}_K{k}" for e in [4, 8, 16] for k in [1, 2, 4]],
}


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def routing_complete(path):
    if not path.is_file():
        return False
    item = read_json(path)
    return (item.get("states", 0) > 0 and
            abs(item.get("max_abs_reproduction_difference_from_saved_half", 1)) < 1e-3)


def downstream_complete(path, mode, seed):
    if not path.is_file():
        return False
    item = read_json(path)
    return (item.get("mode") == mode and item.get("seed") == seed and
            item.get("steps") == 200 and
            item.get("detector_training_scenes") == "official mini_train" and
            item.get("detector_validation_scenes") == "official mini_val")


def timing_complete(path):
    return path.is_file() and read_json(path).get("timing_status") == "measured_without_concurrent_study_jobs"


def jobs(stage):
    commands = []
    if stage in {"main-metrics", "all"}:
        commands.append(("main_metrics", ["experiments.evaluate_pending", "--group", "main", "--isolated"], None))
    if stage in {"supplement-metrics", "all"}:
        for group in ["expert", "rho", "mechanism"]:
            commands.append((f"{group}_metrics", ["experiments.evaluate_pending", "--group", group, "--isolated"], None))
    if stage in {"domain-training", "all"}:
        for protocol in ["zero-shot", "adapt"]:
            for variant in VARIANTS:
                commands.append((f"domain_{protocol}_{variant}", ["experiments.domain", "--protocol", protocol, "--variant", variant, "--skip-evaluation"], None))
        commands.append(("domain_metrics", ["experiments.evaluate_pending", "--root", "output/mini/domain", "--isolated"], None))
    if stage in {"downstream", "all"}:
        for mode in ["real", "baseline", "full"]:
            for seed in SEEDS:
                output = Path("output/mini/downstream_controlled") / mode / f"seed{seed}" / "downstream_metrics.json"
                argv = ["experiments.downstream", "--mode", mode, "--seed", str(seed), "--steps", "200", "--output-root", "output/mini/downstream_controlled"]
                commands.append((f"downstream_{mode}_{seed}", argv, (downstream_complete, output, mode, seed)))
    if stage in {"long", "all"}:
        for seed in SEEDS:
            for variant in VARIANTS:
                for frames in [64, 128]:
                    run_dir = Path("output/mini/runs/main") / variant / f"seed{seed}"
                    argv = ["experiments.long_rollout", "--manifest", f"output/mini/latents{frames}/manifest.json", "--run-dir", str(run_dir), "--frames", str(frames), "--seed", str(seed)]
                    commands.append((f"long_{variant}_{seed}_{frames}", argv, None))
    if stage in {"routing", "all"}:
        for group, variants in GROUPS.items():
            for variant in variants:
                for seed in SEEDS:
                    run_dir = Path("output/mini/runs") / group / variant / f"seed{seed}"
                    path = run_dir / "routing_summary.json"
                    argv = ["experiments.routing_audit", "--run-dir", str(run_dir)]
                    commands.append((f"routing_{group}_{variant}_{seed}", argv, (routing_complete, path)))
    if stage in {"efficiency", "all"}:
        for group, variants in {
            "main": ["baseline", "vanilla_moe", "kw_moe", "ts_mem", "full"],
            "mechanism": ["independent", "unmasked", "matched_budget", "all_write", "random_write", "shuffled", "kw_moe", "full"],
            "rho": GROUPS["rho"],
            "expert": GROUPS["expert"],
        }.items():
            for variant in variants:
                run_dir = Path("output/mini/runs") / group / variant / "seed42"
                path = run_dir / "benchmark.json"
                argv = ["experiments.benchmark", "--run-dir", str(run_dir)]
                commands.append((f"benchmark_{group}_{variant}", argv, (timing_complete, path)))
    if stage in {"report", "all"}:
        commands.append(("provenance", ["experiments.provenance"], None))
        commands.append(("paper_results", ["experiments.paper_results", "--require-complete"], None))
    return commands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["main-metrics", "supplement-metrics", "domain-training", "downstream", "long", "routing", "efficiency", "report", "all"], default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", default="output/mini/orchestrator")
    args = parser.parse_args()
    output = Path(args.output) / args.stage
    output.mkdir(parents=True, exist_ok=True)
    planned = jobs(args.stage)
    pending = []
    for name, argv, check in planned:
        if check is not None:
            function, *parameters = check
            if function(*parameters):
                continue
        pending.append((name, argv))
    print(f"STAGE {args.stage}: {len(pending)} pending commands; {len(planned) - len(pending)} validated skips", flush=True)
    if args.dry_run:
        for name, _ in pending:
            print(name, flush=True)
        return
    environment = os.environ.copy()
    environment.update({key: "1" for key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]})
    environment["KW_STUDY_EXCLUSIVE"] = "1"
    results = []
    for index, (name, argv) in enumerate(pending, start=1):
        command = [sys.executable, "-m", *argv]
        log_path = output / f"{index:03d}_{name}.log"
        print(f"START {index}/{len(pending)} {name}; log={log_path}", flush=True)
        with log_path.open("w", encoding="utf-8") as handle:
            handle.write("COMMAND " + " ".join(command) + "\n")
            handle.flush()
            completed = subprocess.run(command, cwd=Path(__file__).resolve().parent.parent,
                                       env=environment, stdout=handle, stderr=subprocess.STDOUT)
        result = {"name": name, "returncode": completed.returncode, "log": str(log_path)}
        results.append(result)
        if completed.returncode:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-3000:]
            print(f"FAILED {name}: {completed.returncode}\n{tail}", flush=True)
        else:
            print(f"DONE {index}/{len(pending)} {name}", flush=True)
    status = {
        "stage": args.stage,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "results": results,
        "failures": [item["name"] for item in results if item["returncode"]],
    }
    (output / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    if status["failures"]:
        raise RuntimeError(f"Failed independent jobs: {status['failures']}; see {output / 'status.json'}")
    print(f"STAGE COMPLETE {args.stage}", flush=True)


if __name__ == "__main__":
    main()
