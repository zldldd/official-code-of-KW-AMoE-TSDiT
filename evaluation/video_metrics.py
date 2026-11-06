"""Shared, memory-bounded FVD/FID/LPIPS/CLIP evaluation for mini clips."""

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from evaluation.i3d import InceptionI3d
from evaluation.inception_fid import InceptionV3FID
from evaluation.metrics import compute_multiview_fvd, frechet_distance


def to_uint8_video(video):
    if video.dtype == torch.uint8:
        return video.cpu()
    return ((video.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).cpu()


class I3DFeatureExtractor(torch.nn.Module):
    """I3D Kinetics-400 extractor with the reference VideoGPT preprocessing."""

    def __init__(self, checkpoint: str, device: str):
        super().__init__()
        self.device = device
        model = InceptionI3d(400, in_channels=3)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state)
        self.model = model.to(device).eval()

    @staticmethod
    def _single(video, resolution=224):
        video = video.permute(0, 3, 1, 2).float() / 255.0
        _, _, height, width = video.shape
        scale = resolution / min(height, width)
        if height < width:
            target = (resolution, math.ceil(width * scale))
        else:
            target = (math.ceil(height * scale), resolution)
        video = F.interpolate(video, size=target, mode="bilinear", align_corners=False)
        _, _, height, width = video.shape
        x0, y0 = (width - resolution) // 2, (height - resolution) // 2
        video = video[:, :, y0:y0 + resolution, x0:x0 + resolution]
        return video.permute(1, 0, 2, 3).contiguous() - 0.5

    def preprocess(self, videos):
        if videos.dtype == torch.uint8:
            videos = videos.permute(0, 2, 3, 4, 1)
        else:
            videos = ((videos + 1) / 2).clamp(0, 1)
            videos = (videos.permute(0, 2, 3, 4, 1) * 255).to(torch.uint8)
        processed = [self._single(video) * 2 for video in videos]
        return torch.stack(processed)

    @torch.no_grad()
    def forward(self, videos):
        videos = self.preprocess(videos)
        outputs = []
        for start in range(0, len(videos), 2):
            outputs.append(self.model(videos[start:start + 2].to(self.device)))
        return torch.cat(outputs)


class CLIPScorer:
    """OpenCLIP ViT-B/32 raw-cosine scorer using scene descriptions."""

    def __init__(self, checkpoint: str, device: str):
        import open_clip

        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained=str(Path(checkpoint).resolve())
        )
        self.model = self.model.to(device).eval()
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")

    @torch.no_grad()
    def score(self, videos, descriptions):
        text_features = self.model.encode_text(self.tokenizer(descriptions).to(self.device))
        scores = []
        for index, video in enumerate(videos):
            frames = video.permute(0, 2, 3, 4, 1).reshape(
                -1, *video.shape[1:3], *video.shape[4:]
            )
            images = []
            for frame in frames:
                if frame.dtype == torch.uint8:
                    array = frame.permute(1, 2, 0).numpy()
                else:
                    array = ((frame.permute(1, 2, 0).clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).numpy()
                images.append(self.preprocess(Image.fromarray(array)))
            images = torch.stack(images)
            values = []
            for start in range(0, len(images), 8):
                features = self.model.encode_image(images[start:start + 8].to(self.device))
                features = features / features.norm(dim=-1, keepdim=True)
                values.append(features)
            image_feature = torch.cat(values).mean(dim=0)
            image_feature = image_feature / image_feature.norm()
            text_feature = text_features[index] / text_features[index].norm()
            scores.append((image_feature @ text_feature).item())
        return scores


class VideoMetricSuite:
    """Compute per-view FVD and pooled-frame metrics with shared evaluators."""

    def __init__(self, i3d_checkpoint: str, clip_checkpoint: str,
                 fid_checkpoint: str, device: str, fid_device: str = None):
        import lpips

        self.device = device
        self.i3d = I3DFeatureExtractor(i3d_checkpoint, device)
        self.clip = CLIPScorer(clip_checkpoint, device)
        self.fid = InceptionV3FID(fid_checkpoint, device=fid_device or device)
        self.lpips = lpips.LPIPS(net="vgg").to(device).eval().requires_grad_(False)

    @torch.no_grad()
    def evaluate(self, real_videos, generated_videos, descriptions):
        if len(real_videos) != len(generated_videos):
            raise ValueError("Real and generated clip counts differ")
        if len(real_videos) < 2:
            raise ValueError("FVD and FID require at least two clips")
        real = torch.stack([to_uint8_video(video) for video in real_videos])
        generated = torch.stack([to_uint8_video(video) for video in generated_videos])
        if real.shape != generated.shape or real.ndim != 6:
            raise ValueError("Expected matching (N,V,C,T,H,W) videos")
        fvd = compute_multiview_fvd(real, generated, self.i3d)
        print(f"FVD complete: {fvd['macro']:.6f}", flush=True)

        real_features, generated_features, lpips_values = [], [], []
        for index, (real_clip, generated_clip) in enumerate(zip(real, generated)):
            real_images = real_clip.permute(0, 2, 1, 3, 4).reshape(-1, *real_clip.shape[1:3], *real_clip.shape[4:])
            generated_images = generated_clip.permute(0, 2, 1, 3, 4).reshape(-1, *generated_clip.shape[1:3], *generated_clip.shape[4:])
            real_features.append(self.fid(real_images))
            generated_features.append(self.fid(generated_images))
            for start in range(0, len(real_images), 4):
                real_batch = real_images[start:start + 4].float().to(self.device) / 127.5 - 1
                generated_batch = generated_images[start:start + 4].float().to(self.device) / 127.5 - 1
                lpips_values.extend(self.lpips(real_batch, generated_batch).flatten().cpu().tolist())
            print(f"FID/LPIPS clips: {index + 1}/{len(real)}", flush=True)
        fid = frechet_distance(torch.cat(real_features).numpy(), torch.cat(generated_features).numpy())
        del real_features, generated_features, real_images, generated_images, real_batch, generated_batch
        torch.cuda.empty_cache()
        clip_scores = self.clip.score(generated, descriptions)
        print("CLIP complete (image batches of 8)", flush=True)
        return {
            "fvd": fvd["macro"],
            "fvd_per_view": fvd["per_view"],
            "fvd_feature": "I3D Kinetics-400 temporally averaged 400-dimensional logits",
            "fvd_view_aggregation": "macro_average_of_six_separate_view_fvds",
            "fid": fid,
            "fid_view_aggregation": "pool_all_six_views_and_sixteen_frames",
            "fid_device": self.fid.device,
            "fid_batch_size": 8,
            "lpips": float(np.mean(lpips_values)),
            "lpips_batch_size": 4,
            "clip_score": float(np.mean(clip_scores)),
            "clip_score_per_clip": clip_scores,
            "clip_image_batch_size": 8,
            "sample_count": len(real),
            "frames_per_view": real.shape[3],
            "image_count_per_distribution": real.shape[0] * real.shape[1] * real.shape[3],
            "pixel_dtype": "uint8",
            "pixel_quantization": "round(clamp(x,-1,1)*127.5+127.5)",
        }
