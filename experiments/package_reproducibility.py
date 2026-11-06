"""Package mini evidence without datasets, large weights, secrets, or absolute private paths."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile


PROJECT = Path(__file__).resolve().parent.parent
TEXT_SUFFIXES = {".py", ".json", ".yaml", ".yml", ".md", ".txt", ".tex", ".bib", ".cfg"}
RESULT_NAMES = {"metrics.json", "ap.json", "benchmark.json", "profile.json", "routing_summary.json", "complete.json", "run_inventory.json"}
DRIVE_PATH = re.compile(r"\b[A-Za-z]:[\\/]")


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sanitized(path):
    data = path.read_bytes()
    if path.suffix not in TEXT_SUFFIXES:
        return data
    text = data.decode("utf-8")
    replacements = [
        ("H:/数据集/NuScenes mini", "${NUSCENES_MINI_ROOT}"),
        ("H:\\数据集\\NuScenes mini", "${NUSCENES_MINI_ROOT}"),
        ("H:/数据集", "${DATASET_ROOT}"),
        ("H:\\数据集", "${DATASET_ROOT}"),
        (PROJECT.as_posix(), "<PROJECT>"),
        (str(PROJECT), "<PROJECT>"),
        (str(PROJECT).replace("\\", "\\\\"), "<PROJECT>"),
        (PROJECT.parent.as_posix(), "<PROJECT_PARENT>"),
        (str(PROJECT.parent), "<PROJECT_PARENT>"),
        (str(PROJECT.parent).replace("\\", "\\\\"), "<PROJECT_PARENT>"),
        ("C:/Users/Administrator", "<HOME>"),
        ("C:\\Users\\Administrator", "<HOME>"),
        ("C:\\\\Users\\\\Administrator", "<HOME>"),
    ]
    for source, target in replacements:
        variants = {
            source,
            source.replace("\\", "\\\\"),
            json.dumps(source, ensure_ascii=True)[1:-1],
            json.dumps(source.replace("\\", "\\\\"), ensure_ascii=True)[1:-1],
        }
        for variant in variants:
            text = text.replace(variant, target)
    if path.name != "package_reproducibility.py" and (DRIVE_PATH.search(text) or "Users/Administrator" in text or "论文/2026视频专家生成" in text):
        raise ValueError(f"Unsanitized private path in {path}")
    return text.encode("utf-8")


def source_files():
    files = []
    for name in ["main.py", "README.md", "requirements.txt", "sitecustomize.py"]:
        files.append(PROJECT / name)
    files.extend((PROJECT / "configs").glob("*.yaml"))
    for directory in ["data", "evaluation", "experiments", "models", "training", "utils"]:
        files.extend((PROJECT / directory).rglob("*.py"))
    root = PROJECT / "output" / "mini"
    files.append(root / "provenance.json")
    files.extend(root.glob("evaluation_manifest*.json"))
    files.extend(root.glob("latents*/manifest.json"))
    files.extend(root.glob("domain*_latents/manifest.json"))
    files.extend((root / "summary").glob("*.json"))
    for name in ["mini_results.md", "routing_audit_summary.json", "fig5_radar.png", "fig9_component_ablation.png", "fig10_efficiency.png", "fig11_pareto.png", "expert_grid_fvd.png", "rho_flops.png", "long_64_fvd.png", "long_128_fvd.png"]:
        files.append(root / "report" / name)
    files.extend((root / "report" / "qualitative").rglob("*manifest.json"))
    files.extend((root / "report" / "qualitative").rglob("fig*.png"))
    files.extend([
        PROJECT / "external" / "dive" / "vis_cmp.png",
        PROJECT / "external" / "dive" / "sunny_604d12eb.mp4",
    ])
    files.append(root / "reproducibility_package" / "README.md")
    for directory in ["runs", "domain", "convergence"]:
        for path in (root / directory).rglob("*.json"):
            if path.name in RESULT_NAMES or path.name.startswith("metrics_long"):
                files.append(path)
    for directory in ["downstream_controlled", "downstream", "domain_detector"]:
        files.extend((root / directory).rglob("*metrics.json"))
    files.append(root / "domain" / "scene_split.json")
    return sorted(set(files))


def write_detector_inventory():
    root = PROJECT / "output" / "mini"
    records = []
    for directory in ["downstream_controlled", "downstream", "domain_detector"]:
        for path in sorted((root / directory).rglob("detector.pt")):
            data = path.read_bytes()
            records.append({"checkpoint": path.relative_to(PROJECT).as_posix(), "sha256": sha256_bytes(data), "bytes": len(data)})
    destination = root / "detector_inventory.json"
    destination.write_text(json.dumps({"detector_count": len(records), "weights_not_packaged": True, "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination


def build_package(destination):
    inventory = write_detector_inventory()
    files = sorted(set(source_files() + [inventory]))
    root = PROJECT / "output" / "mini"
    if (root / "runs" / "run_inventory.json") not in files:
        raise FileNotFoundError(root / "runs" / "run_inventory.json")
    if (root / "domain" / "run_inventory.json") not in files:
        raise FileNotFoundError(root / "domain" / "run_inventory.json")
    manifest = []
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            if not path.is_file():
                raise FileNotFoundError(path)
            relative = path.relative_to(PROJECT)
            if any(part in {".git", ".venv", ".venv39", "__pycache__"} for part in relative.parts):
                raise ValueError(f"Excluded path encountered: {relative}")
            data = sanitized(path)
            name = relative.as_posix()
            archive.writestr(name, data)
            manifest.append({"file": name, "sha256": sha256_bytes(data), "bytes": len(data)})
        archive.writestr("package_manifest.json", json.dumps({"file_count": len(manifest), "scope": "nuScenes-mini evidence; no large data or weights; no public DOI asserted", "files": manifest}, ensure_ascii=False, indent=2))
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("ZIP integrity check failed")
        text_names = [name for name in archive.namelist() if Path(name).suffix in TEXT_SUFFIXES]
        for name in text_names:
            text = archive.read(name).decode("utf-8")
            if name != "experiments/package_reproducibility.py" and (DRIVE_PATH.search(text) or "Users/Administrator" in text or "论文/2026视频专家生成" in text):
                raise ValueError(f"Private path remained in archive: {name}")
    print(json.dumps({"archive": str(destination), "source_file_count": len(files), "bytes": destination.stat().st_size}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(PROJECT.parent / "CAAI Transactions on Intelligence Technology" / "KW-AMoE-TSDiT_mini_reproducibility.zip"))
    args = parser.parse_args()
    build_package(Path(args.output))


if __name__ == "__main__":
    main()
