"""Camera frames and conditional 2D annotations for downstream augmentation."""

import torch
from torch.utils.data import Dataset

from data.detection_dataset import NUSCENES_2D_CATEGORIES


class VideoFrameDetectionDataset(Dataset):
    """Expose annotated frames from real or generated video tensors."""

    def __init__(self, videos, annotations, sample_data_tokens=None, deduplicate=False):
        if len(videos) != len(annotations):
            raise ValueError("Video and annotation counts differ")
        if sample_data_tokens is not None and len(sample_data_tokens) != len(videos):
            raise ValueError("Token and video counts differ")
        self.samples = []
        seen = set()
        for clip_index, (video, annotation_views) in enumerate(zip(videos, annotations)):
            for view_index, frame_annotations in enumerate(annotation_views):
                for frame_index, annotation in enumerate(frame_annotations):
                    if not annotation["has_annotations"]:
                        continue
                    if deduplicate:
                        if sample_data_tokens is None:
                            raise ValueError("Deduplication requires sample_data_tokens")
                        token = sample_data_tokens[clip_index][view_index][frame_index]
                        if token in seen:
                            continue
                        seen.add(token)
                    self.samples.append((video[view_index, :, frame_index], annotation))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image, annotation = self.samples[index]
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
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor(index),
            "sample_data_token": f"frame_{index}",
        }
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        else:
            image = ((image.float() + 1) / 2).clamp(0, 1)
        return image, target
