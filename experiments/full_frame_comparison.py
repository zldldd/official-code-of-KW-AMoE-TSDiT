"""Render full-frame internal inference and explicitly sourced external examples."""

import argparse
import gc
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from PIL import Image

from evaluation.video_metrics import to_uint8_video
from experiments.full_frame_qualitative import METHODS
from experiments.provenance import sha256
from models.vae_backend import CogVideoX3DVAE
from utils.config import load_config


SELECTED = [0, 3, 7, 11, 14]
VIEW_NAMES = [
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
]
MAGIC_X = [(56, 811), (820, 1575), (1585, 2340)]
MAGIC_Y = [(55, 268), (269, 483)]
DIVE_URL = (
    "https://raw.githubusercontent.com/LiAutoAD/DIVE/master/static/videos/"
    "dive/sunny/604d12ebcf784c3f945189f79262f19c.mp4"
)
COMPARE_URL = "https://raw.githubusercontent.com/LiAutoAD/DIVE/master/static/images/vis_cmp.png"


def array(image):
    return image.permute(1, 2, 0).numpy().copy()


def internal_videos(config, source, output):
    source_hash = sha256(source)
    cache = output / "decoded_videos.pt"
    if cache.is_file():
        saved = torch.load(cache, map_location="cpu", weights_only=False)
        if saved["source_sha256"] == source_hash:
            return saved["videos"], source_hash
    saved = torch.load(source, map_location="cpu", weights_only=False)
    if saved["metadata"]["seed"] != 42 or saved["metadata"]["noise_seed"] != 10042:
        raise ValueError("The paired seed42 full-field inference record is required")
    videos = {"real": to_uint8_video(saved["real"])}
    vae = CogVideoX3DVAE(config.vae_path, device=config.device)
    with torch.no_grad():
        for name in METHODS:
            latent = saved["generated_latents"][name].float()
            if latent.shape != (1, 6, 4, 16, 40, 64):
                raise ValueError(f"Unexpected cached latent shape for {name}: {latent.shape}")
            video = to_uint8_video(vae.decode(latent)[0])[..., 16:-16, :]
            if video.shape != (6, 3, 16, 288, 512):
                raise ValueError(f"Unexpected decoded video shape for {name}: {video.shape}")
            videos[name] = video.cpu()
            print(f"Decoded saved full-field latent: {name}", flush=True)
    torch.save({"source_sha256": source_hash, "videos": videos}, cache)
    del vae, saved
    gc.collect()
    torch.cuda.empty_cache()
    return videos, source_hash


def magicdrive_examples(path):
    image = Image.open(path).convert("RGB")
    if image.size != (2347, 1005):
        raise ValueError(f"Unexpected DiVE comparison image size: {image.size}")
    multiview, temporal, rectangles = [], [], []
    for example, (x0, x1) in enumerate(MAGIC_X, start=1):
        split = (x0 + x1) // 2
        for moment, (y0, y1) in enumerate(MAGIC_Y, start=1):
            panels = []
            for half, (left, right) in enumerate([(x0, split), (split, x1)]):
                rectangle = [left, y0, right, y1]
                panel = np.asarray(image.crop(rectangle)).copy()
                panels.append(panel)
                rectangles.append({"example": example, "moment": moment, "half": half, "box": rectangle})
            temporal.append((f"E{example}-A / T{moment}", panels[0]))
            temporal.append((f"E{example}-B / T{moment}", panels[1]))
            if moment == 1:
                multiview.extend([(f"E{example}-A / T1", panels[0]), (f"E{example}-B / T1", panels[1])])
    return multiview, temporal, rectangles


def dive_examples(path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Cannot read official DiVE video: {path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width % 6 or count <= max(SELECTED) or fps <= 0:
        raise ValueError("The DiVE video does not provide the required six views and frame indices")
    frames = {}
    index = 0
    while index <= max(SELECTED):
        valid, frame = capture.read()
        if not valid:
            raise ValueError(f"Cannot decode DiVE video frame {index}")
        if index in SELECTED:
            frames[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        index += 1
    capture.release()
    view_width = width // 6
    multiview = []
    for view in range(6):
        panel = frames[0][:, view * view_width:(view + 1) * view_width].copy()
        multiview.append((f"V{view + 1} / public 0.000 s", panel))
    temporal = []
    for frame in SELECTED:
        panel = frames[frame][:, :view_width].copy()
        temporal.append((f"public {frame / fps:.3f} s", panel))
    metadata = {
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": count,
        "views_per_video_frame": 6,
        "full_view_width": view_width,
        "full_view_height": height,
        "selected_frame_indices": SELECTED,
        "selected_seconds": [index / fps for index in SELECTED],
        "first_view_used_for_temporal_row": True,
        "camera_names_not_claimed": True,
    }
    return multiview, temporal, metadata


def render(rows, base_columns, titles, destination, footnote):
    height_ratios = [1.0 if len(row["panels"]) <= base_columns else 0.55 for row in rows]
    figure = plt.figure(figsize=(2.85 * base_columns, 1.70 * sum(height_ratios)))
    grid = GridSpec(len(rows), 60, figure=figure, height_ratios=height_ratios)
    for row_index, row in enumerate(rows):
        panels = row["panels"]
        if 60 % len(panels):
            raise ValueError(f"Unsupported panel count: {len(panels)}")
        span = 60 // len(panels)
        for column, panel in enumerate(panels):
            if panel.dtype != np.uint8 or panel.ndim != 3 or panel.shape[2] != 3:
                raise ValueError("All panels must retain their complete uint8 RGB frames")
            axes = figure.add_subplot(grid[row_index, column * span:(column + 1) * span])
            axes.imshow(panel, aspect="equal", interpolation="bilinear")
            axes.axis("off")
            if row_index == 0:
                axes.set_title(titles[column], fontsize=12, pad=6)
            elif row.get("notes"):
                axes.text(0.5, 1.025, row["notes"][column], transform=axes.transAxes,
                          ha="center", va="bottom", fontsize=8.5)
            if column == 0:
                axes.text(-0.095, 0.5, row["label"], transform=axes.transAxes,
                          rotation=90, ha="right", va="center", fontsize=11)
    figure.subplots_adjust(left=0.105, right=0.998, top=0.952, bottom=0.070,
                           wspace=0.025, hspace=0.17)
    figure.text(0.105, 0.014, footnote, ha="left", va="bottom", fontsize=9.5)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=220)
    plt.close(figure)


def generate(config, source, output, magicdrive_path, dive_path):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data = torch.load(source, map_location="cpu", weights_only=False)
    videos, source_hash = internal_videos(config, source, output)
    magic_multi, magic_temporal, rectangles = magicdrive_examples(magicdrive_path)
    dive_multi, dive_temporal, dive_metadata = dive_examples(dive_path)

    real_multi = [array(videos["real"][view, :, 0]) for view in range(6)]
    rows = [{"label": "Real reference\n(mini)", "panels": real_multi}]
    for name, label in METHODS.items():
        rows.append({
            "label": label,
            "panels": [array(videos[name][view, :, 0]) for view in range(6)],
        })
    rows.append({"label": "MagicDrive\n(public stills)", "panels": [p for _, p in magic_multi],
                 "notes": [n for n, _ in magic_multi]})
    rows.append({"label": "DiVE\n(public video)", "panels": [p for _, p in dive_multi],
                 "notes": [n for n, _ in dive_multi]})
    if len(rows) != 8:
        raise ValueError("Seven method rows plus one separate real-reference row are required")
    render(rows, 6, VIEW_NAMES, output / "fig6_multiview.png",
           "Mini: same validation condition/noise for the five internal models; 1000-step checkpoints, display-only spatial adaptation.\n"
           "External rows: different official public scenes/conditions. MagicDrive E1--E3 are DiVE comparison stills (A/B halves, T1); DiVE V1--V6 are full video panels. Not a matched ranking.")

    times = (data["metadata"]["timestamps_us"][0])
    times = [(value - times[0]) / 1e6 for value in times]
    rows = [{"label": "Real reference\n(mini)", "panels": [array(videos["real"][0, :, i]) for i in SELECTED]}]
    for name, label in METHODS.items():
        rows.append({"label": label, "panels": [array(videos[name][0, :, i]) for i in SELECTED]})
    rows.append({"label": "MagicDrive\n(public stills)", "panels": [p for _, p in magic_temporal],
                 "notes": [n for n, _ in magic_temporal]})
    rows.append({"label": "DiVE\n(public video)", "panels": [p for _, p in dive_temporal],
                 "notes": [n for n, _ in dive_temporal]})
    if len(magic_temporal) != 12:
        raise ValueError("All three public examples, two source halves and two moments must be shown")
    titles = [f"Frame {i + 1}\nmini {times[i]:.3f} s" for i in SELECTED]
    render(rows, 5, titles, output / "fig7_temporal_16.png",
           "Mini: jointly denoised CAM_FRONT; frame indices 1/4/8/12/15, with mini timestamps in the titles. No autoregressive-error claim.\n"
           "MagicDrive: three examples x two source halves x two moments (12 full frames, not a five-time trajectory). DiVE: public 12 Hz frames 1/4/8/12/15. External conditions differ.")

    manifest = {
        "scope": "Full-field qualitative panel; not a matched external benchmark or a new quantitative protocol",
        "row_order": ["Real reference (additional row)"] + list(METHODS.values()) + ["MagicDrive", "DiVE"],
        "method_row_count": 7,
        "real_reference_is_additional": True,
        "internal": data["metadata"],
        "source_data_sha256": source_hash,
        "decoded_videos_sha256": sha256(output / "decoded_videos.pt"),
        "full_frame_policy": "Every displayed array contains a complete camera frame. Only source-panel boundary extraction and display resizing are used; no local content crop, added noise, artificial shift, or duplicated time frame.",
        "external_methods": {
            "local_official_reproduction": False,
            "matched_scene_conditions_noise_or_camera_identity": False,
            "permission_for_public_example_display": "Use only as explicitly sourced public qualitative examples; no same-protocol quality claim",
            "magicdrive": {
                "path": str(magicdrive_path),
                "url": COMPARE_URL,
                "project_page": "https://github.com/LiAutoAD/DIVE",
                "source_label": "(a) MagicDrive in the official DiVE comparison figure",
                "source_image_sha256": sha256(magicdrive_path),
                "source_image_size": [2347, 1005],
                "source_rectangles": rectangles,
                "figure6_layout": "E1-A, E1-B, E2-A, E2-B, E3-A, E3-B, each from public T1",
                "figure7_layout": "For each E1-E3: A/T1, B/T1, A/T2, B/T2 (12 complete frames total)",
                "limitation": "Three public examples with two complete source halves and two released moments; twelve full source frames, not a six-camera match to the mini scene or a five-time generated trajectory. Camera identities and exact public time intervals are not claimed.",
            },
            "dive": {
                "path": str(dive_path),
                "url": DIVE_URL,
                "project_page": "https://github.com/LiAutoAD/DIVE",
                "source_label": "Official DiVE sunny-day generated multi-view video on nuScenes",
                "source_video_sha256": sha256(dive_path),
                **dive_metadata,
            },
        },
        "figure7_mini_selected_indices": SELECTED,
        "figure7_mini_front_timestamp_seconds": [times[index] for index in SELECTED],
        "external_source_verified_on": "2026-11-02",
        "external_software": {"opencv": cv2.__version__},
    }
    (output / "comparison_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"figure6": str(output / "fig6_multiview.png"),
                      "figure7": str(output / "fig7_temporal_16.png"),
                      "method_rows": 7, "real_reference_rows": 1}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mini.yaml")
    parser.add_argument("--source", default="output/mini/report/qualitative/full_frame_seed42/full_frame_data.pt")
    parser.add_argument("--output", default="output/mini/report/qualitative/full_frame_seed42")
    parser.add_argument("--magicdrive-image", default="external/dive/vis_cmp.png")
    parser.add_argument("--dive-video", default="external/dive/sunny_604d12eb.mp4")
    args = parser.parse_args()
    generate(load_config(args.config), Path(args.source), Path(args.output),
             Path(args.magicdrive_image), Path(args.dive_video))


if __name__ == "__main__":
    main()
