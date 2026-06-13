"""Model configuration for KW-AMoE-TSDiT."""

from dataclasses import dataclass, field
from typing import List
import yaml


@dataclass
class ModelConfig:
    """Configuration for KW-AMoE-TSDiT model and training."""

    # ── Device ──
    device: str = "cuda"

    # ── DiT Backbone ──
    dit_layers: int = 12
    dit_hidden_dim: int = 768
    dit_heads: int = 12
    patch_size: int = 2
    max_frames: int = 16
    num_views: int = 6
    image_height: int = 32   # latent spatial H (after 8x downsampling)
    image_width: int = 48    # latent spatial W
    latent_dim: int = 4

    # ── KW-AMoE ──
    num_experts: int = 8
    top_k: int = 2
    keyframe_ratio: float = 0.3
    expert_hidden_ratio: int = 4

    # ── TS-Mem ──
    mem_alpha: float = 0.8
    mem_beta: float = 0.3
    mem_slots: int = 16

    # ── Diffusion ──
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    ddim_steps: int = 50

    # ── Loss weights ──
    lambda_st: float = 0.1
    lambda_moe: float = 0.01
    temperature: float = 0.07

    # ── Training ──
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    total_steps: int = 100000
    warmup_steps: int = 5000
    log_interval: int = 100
    eval_interval: int = 5000
    save_interval: int = 10000
    grad_clip: float = 1.0
    mixed_precision: bool = True

    # ── Data ──
    dataroot: str = ""
    version: str = "v1.0-trainval"
    num_workers: int = 4

    # ── Reproducibility ──
    seed: int = 42
    num_seeds: List[int] = field(default_factory=lambda: [40, 41, 42, 43, 44])

    # ── Logging ──
    output_dir: str = "./output"
    checkpoint_dir: str = "./checkpoints"


def load_config(path: str) -> ModelConfig:
    """Load configuration from YAML file, overriding defaults."""
    with open(path, "r") as f:
        yaml_config = yaml.safe_load(f)
    base = ModelConfig()
    for k, v in yaml_config.items():
        if hasattr(base, k):
            setattr(base, k, v)
    return base
