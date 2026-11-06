"""Direct torchvision Inception-v3 FID without torch-fidelity native hooks."""

import numpy as np
import torch

from evaluation.metrics import frechet_distance


class InceptionV3FID(torch.nn.Module):
    """Extract 2048-dimensional pool features using official torchvision weights."""

    def __init__(self, checkpoint: str, device: str = "cpu"):
        super().__init__()
        from torchvision.models import inception_v3

        self.device = device
        self.model = inception_v3(weights=None, transform_input=True)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.fc = torch.nn.Identity()
        self.model = self.model.to(device).eval()

    @torch.no_grad()
    def forward(self, images):
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        elif images.max() > 1:
            images = images / 255.0
        images = torch.nn.functional.interpolate(
            images, size=(299, 299), mode="bilinear", align_corners=False
        )
        outputs = []
        for start in range(0, len(images), 8):
            outputs.append(self.model(images[start:start + 8].to(self.device)))
        return torch.cat(outputs).cpu()


def compute_fid_torchvision(real_imgs: torch.Tensor, gen_imgs: torch.Tensor,
                            extractor: InceptionV3FID) -> float:
    """Compute FID from real/generated uint8 or [0,1] floating-point images."""
    if real_imgs.shape != gen_imgs.shape:
        raise ValueError("Real and generated image shapes must match")
    if real_imgs.dtype != gen_imgs.dtype:
        raise ValueError("Real and generated image dtypes must match")
    real_features = extractor(real_imgs).numpy()
    gen_features = extractor(gen_imgs).numpy()
    return frechet_distance(real_features, gen_features)
