"""Export immutable sample-level evaluation manifests from cached latent records."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from data.nuscenes_dataset import NuScenesMultiViewDataset
from experiments.latent_pipeline import LatentClipDataset
from experiments.provenance import sha256


def make_manifest(cache_path, output, count, frames, seeds, role):
    dataset = LatentClipDataset(cache_path, split="val", limit=count, preload=False)
    records = []
    for index in range(len(dataset)):
        item = dataset[index]
        record = dataset.records[index]
        cache_file = dataset.manifest_path.parent / record["file"]
        records.append({
            "evaluation_index": index,
            "role": role,
            "scene_token": item["scene_token"],
            "scene_name": item["scene_name"],
            "scene_description": item["scene_description"],
            "source_split": item["source_split"],
            "source_index": item["source_index"],
            "cache_file": str(cache_file),
            "cache_sha256": sha256(cache_file),
            "sample_data_tokens": item["sample_data_tokens"],
            "timestamps_us": item["timestamps"].tolist(),
            "latent_frame_ids": item["frame_ids"].tolist(),
            "noise_seeds": {str(seed): seed + (20000 if frames > 16 else 10000) + index * (100 if frames > 16 else 1) for seed in seeds},
            "latent_shape": list(item["latent"].shape),
        })
    source = json.loads(Path(cache_path).read_text(encoding="utf-8"))
    payload = {
        "protocol": role,
        "dataset_version": source["dataset_version"],
        "pixel_frames": frames,
        "latent_frames": frames // 4,
        "view_names": NuScenesMultiViewDataset.VIEW_NAMES,
        "num_clips": len(records),
        "vae": source["vae"],
        "vae_scaling_factor": source["scaling_factor"],
        "source_manifest": str(cache_path),
        "source_manifest_declared_sha256": source.get("manifest_sha256"),
        "records": records,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    payload["evaluation_manifest_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {output}: {len(records)} clips")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default="output/mini/latents16/manifest.json")
    parser.add_argument("--output", default="output/mini/evaluation_manifest.json")
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--role", default="main_16clip")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1024])
    args = parser.parse_args()
    make_manifest(args.cache, args.output, args.count, args.frames, args.seeds, args.role)


if __name__ == "__main__":
    main()
