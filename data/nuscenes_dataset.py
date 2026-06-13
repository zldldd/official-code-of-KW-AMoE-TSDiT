"""nuScenes multi-view video dataset loader.

Loads synchronized 6-camera surround-view video clips of length T frames,
along with camera intrinsics/extrinsics for geometric consistency cues.
"""

import os
from typing import Dict, List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class NuScenesMultiViewDataset(Dataset):
    """nuScenes multi-view video dataset.

    Each sample is a V-view × T-frame clip with associated camera parameters.

    Args:
        nusc: NuScenes instance
        scene_names: list of scene names to include
        T: number of frames per clip
        H, W: target image resolution
        is_train: if True, use stride=4 sampling (more clips); else stride=T
    """

    VIEW_NAMES = [
        "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
        "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
    ]

    def __init__(self, nusc, scene_names: List[str], T: int = 16,
                 H: int = 256, W: int = 384, is_train: bool = True):
        self.nusc = nusc
        self.T = T
        self.H = H
        self.W = W
        self.is_train = is_train

        # Build sample index: sliding windows of T frames per scene
        self.samples = []
        for scene_name in scene_names:
            scene = nusc.scene[nusc.scene_names.index(scene_name)]
            sample = nusc.get("sample", scene["first_sample_token"])

            scene_samples = []
            while sample:
                scene_samples.append(sample)
                if sample["next"] == "":
                    break
                sample = nusc.get("sample", sample["next"])

            stride = 4 if is_train else T
            for i in range(0, len(scene_samples) - T + 1, stride):
                self.samples.append(scene_samples[i:i + T])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        samples = self.samples[idx]

        frames = []
        intrinsics = []
        extrinsics = []

        for view_name in self.VIEW_NAMES:
            view_frames = []
            for sample in samples:
                cam_data = self.nusc.get(
                    "sample_data", sample["data"][view_name]
                )
                img_path = os.path.join(self.nusc.dataroot, cam_data["filename"])

                img = Image.open(img_path).convert("RGB").resize(
                    (self.W, self.H), Image.BILINEAR
                )
                img = torch.from_numpy(np.array(img)).float() / 127.5 - 1.0
                img = img.permute(2, 0, 1)  # (C, H, W)
                view_frames.append(img)

                # Camera parameters
                cam = self.nusc.get(
                    "calibrated_sensor", cam_data["calibrated_sensor_token"]
                )
                intrinsics.append(torch.tensor(cam["camera_intrinsic"], dtype=torch.float32))
                extrinsics.append(
                    torch.tensor(cam["rotation"] + cam["translation"], dtype=torch.float32)
                )

            frames.append(torch.stack(view_frames, dim=1))

        video = torch.stack(frames, dim=0)  # (V, C, T, H, W)

        return {
            "video": video,
            "intrinsics": torch.stack(intrinsics),
            "extrinsics": torch.stack(extrinsics),
            "view_ids": torch.arange(len(self.VIEW_NAMES)),
            "frame_ids": torch.arange(len(samples)),
        }


def create_nuscenes_dataloader(
    dataroot: str, version: str = "v1.0-trainval",
    T: int = 16, batch_size: int = 4, num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """Create train/val DataLoaders for nuScenes.

    Automatically splits scenes: first 700 for training, 700-850 for validation.
    """
    from nuscenes.nuscenes import NuScenes

    nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)

    all_scene_names = [s["name"] for s in nusc.scene]
    train_scenes = all_scene_names[:700]
    val_scenes = all_scene_names[700:850]

    train_dataset = NuScenesMultiViewDataset(
        nusc, train_scenes, T=T, is_train=True
    )
    val_dataset = NuScenesMultiViewDataset(
        nusc, val_scenes, T=T, is_train=False
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    print(f"Train scenes: {len(train_scenes)}, Val scenes: {len(val_scenes)}")
    print(f"Train clips: {len(train_dataset)}, Val clips: {len(val_dataset)}")
    return train_loader, val_loader
