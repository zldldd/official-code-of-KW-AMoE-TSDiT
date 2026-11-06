"""Record the exact mini experiment environment, weights, and evaluation protocol."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from collections import Counter
from pathlib import Path

import torch

from utils.config import load_config


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(output):
    config = load_config("configs/mini.yaml")
    package_names = [
        "torch", "torchvision", "numpy", "scipy", "diffusers", "transformers",
        "safetensors", "open_clip_torch", "lpips", "nuscenes-devkit",
        "Pillow", "PyYAML", "einops", "matplotlib", "opencv-python",
    ]
    packages = {}
    for name in package_names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    weights = [
        "pretrained/CogVideoX-2b/vae/config.json",
        "pretrained/CogVideoX-2b/vae/diffusion_pytorch_model.safetensors",
        "pretrained/I3D/i3d_pretrained_400.pt",
        "pretrained/Inception/inception_v3_google-0cc3c7bd.pth",
        "pretrained/CLIP-ViT-B-32-laion2B-s34B-b79K/open_clip_model.safetensors",
        "pretrained/Detection/resnet50-0676ba61.pth",
        "output/mini/downstream/real/seed42/detector.pt",
        "output/mini/downstream_controlled/real/seed42/detector.pt",
        "output/mini/domain_detector/real/seed42/detector.pt",
    ]
    import lpips
    weights.append(str(Path(lpips.__file__).parent / "weights/v0.1/vgg.pth"))
    vgg = Path(torch.hub.get_dir()) / "checkpoints/vgg16-397923af.pth"
    if vgg.is_file():
        weights.append(str(vgg))
    vae_config = json.loads(Path(config.vae_path, "config.json").read_text(encoding="utf-8"))
    manifest = json.loads(Path("output/mini/latents16/manifest.json").read_text(encoding="utf-8"))
    counts = Counter(record["source_split"] for record in manifest["records"])
    evaluation_manifests = {}
    for path in [Path("output/mini/evaluation_manifest.json"), Path("output/mini/evaluation_manifest_4clip.json"),
                 Path("output/mini/evaluation_manifest_long64.json"), Path("output/mini/evaluation_manifest_long128.json")]:
        if path.is_file():
            evaluation_manifests[str(path)] = json.loads(path.read_text(encoding="utf-8"))["evaluation_manifest_sha256"]
    domain_split = Path("output/mini/domain/scene_split.json")
    domain_split_sha256 = sha256(domain_split) if domain_split.is_file() else None
    code = {}
    for directory in ["models", "experiments", "evaluation", "data", "training", "utils"]:
        for path in Path(directory).glob("*.py"):
            code[str(path)] = sha256(path)
    for path in [Path("main.py"), Path("configs/mini.yaml")]:
        code[str(path)] = sha256(path)
    payload = {
        "scope": "nuScenes v1.0-mini validation only; not comparable with the original full-dataset manuscript tables",
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpus": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        "model_config": vars(config),
        "evaluation_manifests": evaluation_manifests,
        "domain_split_sha256": domain_split_sha256,
        "dataset": {
            "version": config.version,
            "train_clips": counts["train"],
            "val_clips": counts["val"],
            "camera_views": 6,
            "pixel_frames_per_clip": 16,
            "latent_temporal_positions": 4,
            "pixel_resolution": [config.data_height, config.data_width],
            "frame_step": config.frame_step,
            "declared_nuscenes_nominal_camera_rate_hz": 12,
            "observed_mini_median_camera_interval_ms": 100.0,
            "observed_mini_median_camera_rate_hz": 10.0,
            "nominal_mini_frame_span_seconds_at_median_interval": {"16": 1.5, "64": 6.3, "128": 12.7},
            "measured_long_rollout_span_seconds": {
                str(frames): sorted({span for path in Path("output/mini/runs/main").glob(f"main/*/seed*/metrics_long{frames}.json")
                                     for span in json.loads(path.read_text(encoding="utf-8"))["observed_span_seconds"]})
                for frames in [64, 128]
            },
            "train_clip_stride": 4,
            "val_clip_stride": 16,
            "view_synchronization_tolerance_ms": config.sync_tolerance_ms,
            "image_preprocessing": "fixed center crop to 2:1 followed by isotropic bilinear resize; K and projected boxes use the same crop and scale",
        },
        "vae": {
            "source": "THUDM/CogVideoX-2b AutoencoderKLCogVideoX, also used by MagicDrive-V2",
            "repository": "https://github.com/THUDM/CogVideo",
            "config": vae_config,
            "frozen": True,
            "loading": "meta architecture, one CPU safetensors read, assign=True, then device transfer",
            "temporal_chunk_concatenation_dim": 2,
            "initialization_check": json.loads(Path("output/mini/vae_initialization_check.json").read_text(encoding="utf-8")),
            "tiled_reconstruction": json.loads(Path("output/mini/vae_reconstruction_full.json").read_text(encoding="utf-8")),
        },
        "training": {
            "second_stage_convergence": {
                "source_steps": 1000,
                "additional_steps": 4000,
                "optimizer_state_restored": False,
                "scope": "matched Baseline/Ours seed42 probes, not a continuous 5000-step run",
            },
            "main_steps": 1000,
            "supplemental_grid_and_mechanism_steps": 500,
            "domain_adaptation_steps": 200,
            "independent_training_seeds": config.num_seeds,
            "optimizer": "AdamW",
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "mixed_precision_training": config.mixed_precision,
        },
        "generation": {
            "sampler": "deterministic DDIM",
            "ddim_steps": config.ddim_steps,
            "initial_noise_seed": "training_seed + 10000 + validation_clip_index",
            "generation_dtype": "float32",
            "condition_frame_ids": "contiguous indices derived from cached (V,T,C,H,W) latent.shape[1]; long rollout uses absolute latent chunk offsets",
            "memory_reset": "once per clip or once per full long-horizon sequence",
        },
        "evaluation": {
            "fvd": "I3D Kinetics-400 400-dimensional temporally averaged logits; separate FVD per camera view followed by macro average",
            "i3d_preprocessing": "isotropic resize of short side to 224, center crop 224x224, scale to [-1,1]",
            "fid": "torchvision Inception-v3 pool3 2048-dimensional features; GPU extraction with batch size 8; pool six views and all 16 frames",
            "lpips": "lpips v0.1 VGG; paired generated and real frames",
            "clip": "OpenCLIP ViT-B/32 laion2B-s34B-b79K; raw cosine between scene description and normalized mean image feature",
            "ap": "fixed Real-only Faster R-CNN ResNet-50-FPN seed42; non-interpolated 2D AP at IoU=0.5 on annotated projected keyframes",
            "domain_ap": "separate source-day-only detector seed42; scene-1094 is excluded from both generative and detector training; 1,080 source frames and 164 real target validation frames",
            "real_pair_validation": "scene_token equality is required before quality metrics, AP, long-rollout evaluation and qualitative comparison",
            "domain_protocol": "within-nuScenes daytime-to-nighttime zero-shot and disjoint-scene adaptation; 13 target clips; not cross-dataset generalization",
            "main_generated_and_real_clips": 16,
            "supplemental_generated_and_real_clips": 4,
            "generated_pixel_quantization": "round(clamp(x,-1,1)*127.5+127.5), uint8",
            "small_sample_warning": "mini FVD/FID estimates have few clips and high-dimensional feature covariance; they are exploratory and not full-benchmark replacements",
        },
        "weights": {path: {"sha256": sha256(path), "bytes": Path(path).stat().st_size} for path in weights},
        "code_sha256": code,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_worktree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Recorded environment and weight provenance in {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="output/mini/provenance.json")
    arguments = parser.parse_args()
    record(arguments.output)
