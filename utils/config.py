"""Model, data, and reproducibility configuration."""

from dataclasses import dataclass, field, replace
from typing import List

import yaml


@dataclass
class ModelConfig:
    device: str = "cuda"

    dit_layers: int = 16
    dit_hidden_dim: int = 768
    dit_heads: int = 12
    patch_size: int = 2
    max_frames: int = 16
    num_views: int = 6
    image_height: int = 32
    image_width: int = 64
    latent_dim: int = 4

    num_experts: int = 8
    top_k: int = 2
    keyframe_ratio: float = 0.3
    expert_hidden_ratio: int = 4

    mem_alpha: float = 0.8
    mem_beta: float = 0.3

    timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    ddim_steps: int = 50

    lambda_st: float = 0.1
    lambda_moe: float = 0.01

    batch_size: int = 4
    learning_rate: float = 0.0001
    weight_decay: float = 0.0
    total_steps: int = 100000
    warmup_steps: int = 5000
    log_interval: int = 100
    eval_interval: int = 5000
    save_interval: int = 10000
    grad_clip: float = 1.0
    mixed_precision: bool = True

    dataroot: str = ""
    version: str = "v1.0-trainval"
    num_workers: int = 4
    data_height: int = 256
    data_width: int = 512
    frame_step: int = 1
    sync_tolerance_ms: float = 40.0
    allow_proxy_latent: bool = False
    vae_path: str = ""
    moe_type: str = "adaptive"
    expert_budget_mode: str = "adaptive"
    keyframe_weight_mode: str = "masked"
    memory_write_mode: str = "keyframe"
    use_keyframe_gate: bool = True
    use_tsmem: bool = True
    eval_limit: int = 16

    seed: int = 42
    num_seeds: List[int] = field(
        default_factory=lambda: [42, 123, 456, 789, 1024]
    )

    output_dir: str = "./output"
    checkpoint_dir: str = "./checkpoints"


def load_config(path: str) -> ModelConfig:
    """Load only declared fields from a YAML configuration."""
    with open(path, "r", encoding="utf-8") as handle:
        yaml_config = yaml.safe_load(handle) or {}
    base = ModelConfig()
    unknown = set(yaml_config) - set(vars(base))
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    for key, value in yaml_config.items():
        setattr(base, key, value)
    return base


def config_for_run(config: ModelConfig, run_dir: str) -> ModelConfig:
    """Restore the exact variant configuration stored with a trained checkpoint."""
    from pathlib import Path
    import torch

    checkpoint = Path(run_dir) / "final_model.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved = state.get("config", {})
    unknown = set(saved) - set(vars(config))
    if unknown:
        raise ValueError(f"Unknown checkpoint configuration fields: {sorted(unknown)}")
    saved["seed"] = int(state["seed"])
    return replace(config, **saved)
