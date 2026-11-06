"""Synchronized nuScenes camera clips with geometry-consistent preprocessing."""

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from pyquaternion import Quaternion
from torch.utils.data import DataLoader, Dataset

from nuscenes.utils.geometry_utils import view_points


def crop_resize_with_intrinsics(
    image: Image.Image,
    intrinsics: np.ndarray,
    output_h: int,
    output_w: int,
) -> Tuple[Image.Image, np.ndarray, Tuple[int, int, int, int]]:
    """Apply a fixed center crop and isotropic resize, updating ``K`` exactly."""
    width, height = image.size
    target_ratio = output_w / output_h
    source_ratio = width / height
    if source_ratio > target_ratio:
        crop_w = int(round(height * target_ratio))
        crop_h = height
        x0 = (width - crop_w) // 2
        y0 = 0
    else:
        crop_w = width
        crop_h = int(round(width / target_ratio))
        x0 = 0
        y0 = (height - crop_h) // 2
    crop_box = (x0, y0, x0 + crop_w, y0 + crop_h)
    cropped = image.crop(crop_box)
    resized = cropped.resize((output_w, output_h), Image.Resampling.BILINEAR)

    sx, sy = output_w / crop_w, output_h / crop_h
    shift = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], dtype=np.float64)
    scale = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float64)
    updated_k = scale @ shift @ np.asarray(intrinsics, dtype=np.float64)
    return resized, updated_k, crop_box


class NuScenesMultiViewDataset(Dataset):
    """Create timestamp-synchronized six-camera clips from the full camera stream."""

    VIEW_NAMES = [
        "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
        "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
    ]

    def __init__(
        self,
        nusc,
        scene_names: Sequence[str],
        T: int = 16,
        H: int = 256,
        W: int = 512,
        is_train: bool = True,
        frame_step: int = 1,
        clip_stride: Optional[int] = None,
        sync_tolerance_ms: float = 40.0,
    ):
        self.nusc = nusc
        self.T = T
        self.H = H
        self.W = W
        self.frame_step = frame_step
        self.clip_stride = clip_stride or (4 if is_train else T * frame_step)
        self.sync_tolerance_us = int(sync_tolerance_ms * 1000)
        self.samples = []
        self.sample_scenes = []
        scene_lookup = {scene["name"]: scene for scene in nusc.scene}
        for scene_name in scene_names:
            if scene_name not in scene_lookup:
                raise ValueError(f"Unknown nuScenes scene: {scene_name}")
            scene = scene_lookup[scene_name]
            synchronized = self._synchronize_scene(scene)
            last_start = len(synchronized) - (T - 1) * frame_step
            for start in range(0, last_start, self.clip_stride):
                indices = range(start, start + T * frame_step, frame_step)
                self.samples.append([synchronized[i] for i in indices])
                self.sample_scenes.append(scene)

    def _view_track(self, scene: Dict, view_name: str) -> List[Dict]:
        first_sample = self.nusc.get("sample", scene["first_sample_token"])
        record = self.nusc.get("sample_data", first_sample["data"][view_name])
        while record["prev"]:
            record = self.nusc.get("sample_data", record["prev"])
        track = []
        while True:
            track.append(record)
            if not record["next"]:
                break
            record = self.nusc.get("sample_data", record["next"])
        return track

    def _synchronize_scene(self, scene: Dict) -> List[Dict]:
        tracks = {view: self._view_track(scene, view) for view in self.VIEW_NAMES}
        timestamps = {
            view: np.array([item["timestamp"] for item in track], dtype=np.int64)
            for view, track in tracks.items()
        }
        synchronized = []
        reference_view = self.VIEW_NAMES[0]
        for reference in tracks[reference_view]:
            selected = {reference_view: reference}
            max_error = 0
            for view in self.VIEW_NAMES[1:]:
                times = timestamps[view]
                pos = int(np.searchsorted(times, reference["timestamp"]))
                candidates = [p for p in [pos - 1, pos] if 0 <= p < len(times)]
                if not candidates:
                    selected = {}
                    break
                nearest = min(candidates, key=lambda p: abs(
                    int(times[p]) - int(reference["timestamp"])
                ))
                error = abs(int(times[nearest]) - int(reference["timestamp"]))
                if error > self.sync_tolerance_us:
                    selected = {}
                    break
                selected[view] = tracks[view][nearest]
                max_error = max(max_error, error)
            if len(selected) == len(self.VIEW_NAMES):
                synchronized.append({"views": selected, "sync_error_us": max_error})
        return synchronized

    def _visible_boxes(self, cam_data: Dict, intrinsics: np.ndarray) -> Dict:
        if not cam_data["is_key_frame"]:
            return {"boxes": torch.zeros((0, 4)), "labels": [], "has_annotations": False}
        sample = self.nusc.get("sample", cam_data["sample_token"])
        ego = self.nusc.get("ego_pose", cam_data["ego_pose_token"])
        camera = self.nusc.get(
            "calibrated_sensor", cam_data["calibrated_sensor_token"]
        )
        boxes, labels = [], []
        for annotation_token in sample["anns"]:
            box = self.nusc.get_box(annotation_token)
            box.translate(-np.array(ego["translation"]))
            box.rotate(Quaternion(ego["rotation"]).inverse)
            box.translate(-np.array(camera["translation"]))
            box.rotate(Quaternion(camera["rotation"]).inverse)
            corners = box.corners()
            if not np.all(corners[2, :] > 0.1):
                continue
            projected = view_points(corners, intrinsics, normalize=True)[:2, :]
            x0 = max(float(projected[0, :].min()), 0.0)
            y0 = max(float(projected[1, :].min()), 0.0)
            x1 = min(float(projected[0, :].max()), float(self.W))
            y1 = min(float(projected[1, :].max()), float(self.H))
            if x1 - x0 >= 1 and y1 - y0 >= 1:
                boxes.append([x0, y0, x1, y1])
                labels.append(box.name)
        tensor = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4))
        return {"boxes": tensor, "labels": labels, "has_annotations": True}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        clip = self.samples[idx]
        videos, all_intrinsics, all_extrinsics = [], [], []
        timestamps, sync_errors, token_records, annotation_records = [], [], [], []
        for view_name in self.VIEW_NAMES:
            view_frames, view_k, view_t, view_times = [], [], [], []
            view_tokens, view_annotations = [], []
            for record in clip:
                cam_data = record["views"][view_name]
                path = os.path.join(self.nusc.dataroot, cam_data["filename"])
                camera = self.nusc.get(
                    "calibrated_sensor", cam_data["calibrated_sensor_token"]
                )
                with Image.open(path) as source:
                    image, updated_k, _ = crop_resize_with_intrinsics(
                        source.convert("RGB"), camera["camera_intrinsic"], self.H, self.W
                    )
                array = np.asarray(image).copy()
                tensor = torch.from_numpy(array).float() / 127.5 - 1.0
                view_frames.append(tensor.permute(2, 0, 1))
                view_k.append(torch.tensor(updated_k, dtype=torch.float32))
                extrinsic = np.eye(4, dtype=np.float64)
                extrinsic[:3, :3] = Quaternion(camera["rotation"]).rotation_matrix
                extrinsic[:3, 3] = np.asarray(camera["translation"])
                view_t.append(torch.tensor(extrinsic, dtype=torch.float32))
                view_times.append(cam_data["timestamp"])
                view_tokens.append(cam_data["token"])
                view_annotations.append(self._visible_boxes(cam_data, updated_k))

            videos.append(torch.stack(view_frames, dim=1))
            all_intrinsics.append(torch.stack(view_k))
            all_extrinsics.append(torch.stack(view_t))
            timestamps.append(torch.tensor(view_times, dtype=torch.int64))
            sync_errors.append(torch.tensor([
                record["sync_error_us"] for record in clip
            ], dtype=torch.int64))
            token_records.append(view_tokens)
            annotation_records.append(view_annotations)

        return {
            "video": torch.stack(videos, dim=0),
            "intrinsics": torch.stack(all_intrinsics),
            "extrinsics": torch.stack(all_extrinsics),
            "view_ids": torch.arange(len(self.VIEW_NAMES)),
            "frame_ids": torch.arange(self.T),
            "timestamps": torch.stack(timestamps),
            "sync_errors": torch.stack(sync_errors),
            "sample_data_tokens": token_records,
            "annotations": annotation_records,
            "scene_token": self.sample_scenes[idx]["token"],
            "scene_name": self.sample_scenes[idx]["name"],
            "scene_description": self.sample_scenes[idx]["description"],
        }


def lighting_domain_scenes(nusc) -> Tuple[List[str], List[str]]:
    """Split scenes into source daytime and target nighttime domains."""
    daytime, nighttime = [], []
    for scene in nusc.scene:
        target = nighttime if "night" in scene["description"].lower() else daytime
        target.append(scene["name"])
    return daytime, nighttime


def create_nuscenes_dataloader(
    dataroot: str,
    version: str = "v1.0-trainval",
    T: int = 16,
    batch_size: int = 4,
    num_workers: int = 4,
    H: int = 256,
    W: int = 512,
    frame_step: int = 1,
    sync_tolerance_ms: float = 40.0,
    train_scene_names: Optional[Sequence[str]] = None,
    val_scene_names: Optional[Sequence[str]] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Create loaders from official scene splits or explicit domain scene lists."""
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils import splits as official_splits

    nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
    if train_scene_names is None:
        train_scene_names = list(official_splits.mini_train) if version == "v1.0-mini" else list(official_splits.train)
    if val_scene_names is None:
        val_scene_names = list(official_splits.mini_val) if version == "v1.0-mini" else list(official_splits.val)

    train_dataset = NuScenesMultiViewDataset(
        nusc, train_scene_names, T=T, H=H, W=W, is_train=True,
        frame_step=frame_step, sync_tolerance_ms=sync_tolerance_ms,
    )
    val_dataset = NuScenesMultiViewDataset(
        nusc, val_scene_names, T=T, H=H, W=W, is_train=False,
        frame_step=frame_step, sync_tolerance_ms=sync_tolerance_ms,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    print(f"Train scenes: {len(train_scene_names)}, Val scenes: {len(val_scene_names)}")
    print(f"Train clips: {len(train_dataset)}, Val clips: {len(val_dataset)}")
    return train_loader, val_loader
