"""Faster R-CNN training and AP@0.5 evaluation for mini downstream experiments."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.detection_dataset import NUSCENES_2D_CATEGORIES, detection_collate


def build_detector(checkpoint: str, num_classes: int = 11, device: str = "cuda"):
    """Build Faster R-CNN for bbox-only nuScenes evaluation."""
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if Path(checkpoint).name.startswith("resnet50-"):
        model.backbone.body.load_state_dict(state, strict=False)
    else:
        model.load_state_dict(state)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model.to(device)


def train_detector(model, train_dataset, output_dir: str, device: str,
                   steps: int = 200, batch_size: int = 2, lr: float = 1e-4,
                   seed: int = 42):
    """Fine-tune a detector and save raw loss history and final weights."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, collate_fn=detection_collate, generator=generator,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history = []
    model.train()

    def batches():
        while True:
            for batch in loader:
                yield batch

    for step, (images, targets) in enumerate(batches()):
        if step >= steps:
            break
        images = [image.to(device) for image in images]
        targets = [{key: value.to(device) if torch.is_tensor(value) else value
                    for key, value in target.items()} for target in targets]
        losses = model(images, targets)
        loss = sum(losses.values())
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite detection loss at step {step}: {losses}")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 20 == 0 or step + 1 == steps:
            entry = {"step": step, "loss": loss.item(),
                     **{key: value.item() for key, value in losses.items()}}
            history.append(entry)
            print(f"detector step {step}/{steps}: {loss.item():.6f}", flush=True)
    torch.save(model.state_dict(), output / "detector.pt")
    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return model


def _move_targets(targets, device):
    return [{key: value.to(device) if torch.is_tensor(value) else value
             for key, value in target.items()} for target in targets]


@torch.no_grad()
def predict_detector(model, dataset, device: str, batch_size: int = 2):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0, collate_fn=detection_collate)
    predictions, ground_truth = [], []
    model.eval()
    for images, targets in loader:
        outputs = model([image.to(device) for image in images])
        predictions.extend([{key: value.cpu() for key, value in output.items()}
                            for output in outputs])
        ground_truth.extend(targets)
    return predictions, ground_truth


def _average_precision(records, gt_count):
    if gt_count == 0:
        return None
    if not records:
        return 0.0
    records.sort(key=lambda item: item[0], reverse=True)
    true_positives = np.array([item[1] for item in records], dtype=np.float64)
    false_positives = np.array([item[2] for item in records], dtype=np.float64)
    tp_cumsum = np.cumsum(true_positives)
    fp_cumsum = np.cumsum(false_positives)
    precision = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, 1e-12)
    recall = tp_cumsum / gt_count
    precision = np.concatenate(([1.0], precision))
    recall = np.concatenate(([0.0], recall))
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def evaluate_ap50(predictions, ground_truth):
    """Compute one-to-one IoU=0.5 AP for each declared category."""
    per_class = {}
    for class_name, class_id in NUSCENES_2D_CATEGORIES.items():
        records, gt_count = [], 0
        for prediction, target in zip(predictions, ground_truth):
            pred_mask = prediction["labels"] == class_id
            gt_mask = target["labels"] == class_id
            pred_boxes = prediction["boxes"][pred_mask]
            pred_scores = prediction["scores"][pred_mask]
            gt_boxes = target["boxes"][gt_mask]
            gt_count += len(gt_boxes)
            matched = torch.zeros(len(gt_boxes), dtype=torch.bool)
            if len(pred_boxes) == 0:
                continue
            if len(gt_boxes) == 0:
                records.extend([(float(score), 0, 1) for score in pred_scores])
                continue
            iou = _box_iou(pred_boxes, gt_boxes)
            for pred_index, score in enumerate(pred_scores):
                best_iou, best_gt = iou[pred_index].max(dim=0)
                if best_iou >= 0.5 and not matched[best_gt]:
                    matched[best_gt] = True
                    records.append((float(score), 1, 0))
                else:
                    records.append((float(score), 0, 1))
        if gt_count == 0:
            continue
        records = [(score, tp, fp) for score, tp, fp in records]
        ap = _average_precision(records, gt_count)
        per_class[class_name] = ap
    valid = [value for value in per_class.values() if value is not None]
    return {"per_class": per_class, "mAP": float(np.mean(valid)) if valid else None}


def _box_iou(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    intersection = wh[..., 0] * wh[..., 1]
    return intersection / (area1[:, None] + area2[None, :] - intersection + 1e-8)
