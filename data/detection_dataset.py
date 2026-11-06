"""nuScenes-mini camera 2D detection with shared geometric preprocessing."""

import os

import torch
from PIL import Image
from torch.utils.data import Dataset

from data.nuscenes_dataset import crop_resize_with_intrinsics


NUSCENES_2D_CATEGORIES = {
    "vehicle.car": 1,
    "vehicle.truck": 2,
    "vehicle.bus": 3,
    "vehicle.trailer": 4,
    "vehicle.construction": 5,
    "human.pedestrian.adult": 6,
    "vehicle.motorcycle": 7,
    "vehicle.bicycle": 8,
    "vehicle.traffic_cone": 9,
    "vehicle.barrier": 10,
}


class NuScenesDetectionDataset(Dataset):
    """Deduplicated keyframe images with projected 2D boxes."""

    def __init__(self, base_dataset, scene_names=None):
        self.base = base_dataset
        self.records = {}
        for clip_index, clip in enumerate(base_dataset.samples):
            scene = base_dataset.sample_scenes[clip_index]
            if scene_names is not None and scene["name"] not in scene_names:
                continue
            for record in clip:
                for view_name in base_dataset.VIEW_NAMES:
                    sample_data = record["views"][view_name]
                    if sample_data["is_key_frame"]:
                        self.records[sample_data["token"]] = sample_data
        self.samples = list(self.records.values())
        self.samples.sort(key=lambda item: item["token"])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_data = self.samples[index]
        camera = self.base.nusc.get(
            "calibrated_sensor", sample_data["calibrated_sensor_token"]
        )
        path = os.path.join(self.base.nusc.dataroot, sample_data["filename"])
        with Image.open(path) as source:
            image, updated_k, _ = crop_resize_with_intrinsics(
                source.convert("RGB"), camera["camera_intrinsic"],
                self.base.H, self.base.W,
            )
        annotation = self.base._visible_boxes(sample_data, updated_k)
        boxes, labels = [], []
        for box, name in zip(annotation["boxes"], annotation["labels"]):
            if name in NUSCENES_2D_CATEGORIES:
                boxes.append(box)
                labels.append(NUSCENES_2D_CATEGORIES[name])
        if boxes:
            boxes = torch.stack(boxes)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        image = torch.from_numpy(__import__("numpy").array(image).copy()).float() / 255.0
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor(index),
            "sample_data_token": sample_data["token"],
        }
        return image.permute(2, 0, 1), target


def detection_collate(batch):
    return tuple(zip(*batch))
