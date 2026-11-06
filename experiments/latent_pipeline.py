"""Shared CogVideoX latent caching for all mini experiments."""

import gc
import hashlib
import json
from pathlib import Path
from typing import Optional, Sequence

import torch
from torch.utils.data import Dataset

from data import create_nuscenes_dataloader
from models.vae_backend import CogVideoX3DVAE


class LatentClipDataset(Dataset):
    """Load precomputed latent clips with conditions and source metadata."""

    def __init__(self, manifest_path: str, limit: Optional[int] = None,
                 preload: bool = False, split: Optional[str] = None):
        self.manifest_path = Path(manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.records = self.manifest["records"]
        if split is not None:
            self.records = [r for r in self.records if r["source_split"] == split]
        if limit is not None:
            self.records = self.records[:limit]
        self._cache = {}
        if preload:
            for record in self.records:
                self._cache[record["index"]] = self._load_record(record)

    def _load_record(self, record):
        path = self.manifest_path.parent / record["file"]
        return torch.load(path, map_location="cpu", weights_only=True)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        item = self._cache.get(record["index"])
        if item is None:
            item = self._load_record(record)
            if len(self._cache) < 16:
                self._cache[record["index"]] = item
        latent = item["latent"].float()
        return {
            "latent": latent,
            "view_ids": item["view_ids"],
            "frame_ids": torch.arange(latent.shape[1]),
            "intrinsics": item["intrinsics"],
            "extrinsics": item["extrinsics"],
            "timestamps": item["timestamps"],
            "annotations": item["annotations"],
            "sample_data_tokens": item["sample_data_tokens"],
            "scene_token": item["scene_token"],
            "scene_name": item["scene_name"],
            "scene_description": item["scene_description"],
            "source_index": item["source_index"],
            "source_split": item["source_split"],
        }


def _metadata(item, latent, source_index, source_split):
    return {
        "latent": latent.half().cpu(),
        "view_ids": item["view_ids"].cpu(),
        "frame_ids": torch.arange(latent.shape[1]),
        "intrinsics": item["intrinsics"].cpu(),
        "extrinsics": item["extrinsics"].cpu(),
        "timestamps": item["timestamps"].cpu(),
        "annotations": item["annotations"],
        "sample_data_tokens": item["sample_data_tokens"],
        "scene_token": item["scene_token"],
        "scene_name": item["scene_name"],
        "scene_description": item["scene_description"],
        "source_index": source_index,
        "source_split": source_split,
    }


def prepare_latent_cache(
    config,
    output_dir: str,
    frames: int = 16,
    train_scene_names: Optional[Sequence[str]] = None,
    val_scene_names: Optional[Sequence[str]] = None,
    limit_per_split: Optional[int] = None,
    view_chunk: int = 2,
    resume: bool = False,
) -> dict:
    """Encode train/validation clips in 16-frame VAE chunks and save a manifest."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader = create_nuscenes_dataloader(
        dataroot=config.dataroot,
        version=config.version,
        T=frames,
        batch_size=1,
        num_workers=config.num_workers,
        H=config.data_height,
        W=config.data_width,
        frame_step=config.frame_step,
        sync_tolerance_ms=config.sync_tolerance_ms,
        train_scene_names=train_scene_names,
        val_scene_names=val_scene_names,
    )
    vae = CogVideoX3DVAE(config.vae_path, device=config.device)
    expected_shape = (
        config.num_views, frames // 4, config.latent_dim,
        config.image_height, config.image_width,
    )
    records = []
    for split, dataset in [("train", train_loader.dataset), ("val", val_loader.dataset)]:
        split_dir = output / split
        split_dir.mkdir(exist_ok=True)
        count = len(dataset) if limit_per_split is None else min(len(dataset), limit_per_split)
        for index in range(count):
            filename = f"{split}_{index:05d}.pt"
            destination = split_dir / filename
            if resume and destination.is_file():
                try:
                    cached = torch.load(destination, map_location="cpu", weights_only=True)
                except (RuntimeError, EOFError):
                    cached = None
                if cached is not None and tuple(cached["latent"].shape) == expected_shape:
                    records.append({
                        "index": len(records), "file": f"{split}/{filename}",
                        "scene_token": dataset.sample_scenes[index]["token"],
                        "scene_name": dataset.sample_scenes[index]["name"],
                        "source_index": index, "source_split": split,
                    })
                    continue
                destination.rename(destination.with_suffix(".invalid.pt"))
            item = dataset[index]
            video = item["video"].unsqueeze(0)
            latent_chunks = []
            for start in range(0, frames, 16):
                chunk = video[:, :, :, start:start + 16]
                latent_chunks.append(vae.encode(chunk, view_chunk=view_chunk).cpu())
            latent = torch.cat(latent_chunks, dim=2)[0]
            if tuple(latent.shape) != expected_shape:
                raise RuntimeError(
                    f"Latent shape {tuple(latent.shape)} does not match {expected_shape}"
                )
            metadata = _metadata(item, latent, index, split)
            torch.save(metadata, destination)
            records.append({
                "index": len(records),
                "file": f"{split}/{filename}",
                "scene_token": item["scene_token"],
                "scene_name": item["scene_name"],
                "source_index": index,
                "source_split": split,
            })
            if (index + 1) % 10 == 0 or index + 1 == count:
                gc.collect()
                torch.cuda.empty_cache()
                print(f"latent cache {split}: {index + 1}/{count}", flush=True)
        del dataset
    manifest = {
        "dataset_version": config.version,
        "pixel_frames": frames,
        "latent_frames": frames // 4,
        "vae": "THUDM/CogVideoX-2b AutoencoderKLCogVideoX",
        "scaling_factor": vae.scaling_factor,
        "records": records,
    }
    text = json.dumps(manifest, ensure_ascii=False, indent=2)
    manifest["manifest_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest
