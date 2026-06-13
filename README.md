# KW-AMoE-TSDiT: Keyframe-Weighted Adaptive Mixture-of-Experts Temporal-Stable Diffusion Transformer

Official implementation of **KW-AMoE-TSDiT** for multi-view video generation in autonomous driving scenarios.



## Overview

KW-AMoE-TSDiT is a diffusion-based framework for generating temporally stable, geometrically consistent multi-view driving scene videos. It integrates three key innovations:

1. **DiT Backbone** — A Diffusion Transformer with spatiotemporal decomposed attention for joint view-time modeling.
2. **KW-AMoE** (Keyframe-Weighted Adaptive Mixture of Experts) — Keyframe-aware sparse routing that allocates expert capacity to critical driving moments (turns, lane changes).
3. **TS-Mem** (Temporal Stable Memory) — An EMA memory buffer with keyframe-driven writes and non-keyframe residual injection to suppress temporal drift.

![Overall Architecture](figures/architecture_overview.png)

## Repository Structure

```
KW-AMoE-TSDiT/
├── models/                 # Core model modules
│   ├── attention.py        # Multi-head self-attention, AdaLN, position embeddings
│   ├── dit_backbone.py     # DiT backbone with spatiotemporal tokenization
│   ├── kwamoe.py           # Sparse MoE + keyframe gating
│   ├── tsmem.py            # Temporal Stable Memory
│   └── diffusion.py        # Noise schedule & DDIM sampler
├── data/                   # Data loading
│   └── nuscenes_dataset.py # nuScenes multi-view video dataset
├── training/               # Training pipeline
│   ├── trainer.py          # Training loop with mixed precision
│   └── losses.py           # Loss functions (diffusion, spatiotemporal, MoE balance)
├── evaluation/             # Evaluation metrics
│   └── metrics.py          # FVD, FID, LPIPS
├── utils/                  # Utilities
│   ├── config.py           # Configuration dataclass
│   └── helpers.py          # Seed setting, parameter counting
├── figures/                # Paper figure generation
│   └── generate_paper_figures.py  # All paper figures (requires nuScenes)
├── configs/                # Configuration files
│   └── default.yaml
├── main.py                 # Entry point
├── requirements.txt
└── README.md
```

## Installation

### Requirements

- Python 3.9+
- PyTorch 2.0+
- [nuScenes devkit](https://github.com/nutonomy/nuscenes-devkit)
- [nuScenes dataset](https://www.nuscenes.org/download) (mini or full)

### Setup

```bash
git clone https://github.com/zldldd/official-code-of-KW-AMoE-TSDiT.git
cd official-code-of-KW-AMoE-TSDiT

# Install dependencies
pip install -r requirements.txt

# Install nuScenes devkit (if not already)
pip install nuscenes-devkit
```

## Quick Start

### 1. Model Structure Verification (no data required)

```bash
python main.py --test
```

This instantiates the model with reduced dimensions and verifies forward/backward pass correctness. Output:

```
  Total parameters: 12.34M
  DiT Backbone:  11.81M
  KeyframeGate:  0.40M
  TS-Mem:        0.13M
  ...
  All checks passed!
```

### 2. Full Training on nuScenes

```bash
python main.py --train --dataroot /path/to/nuscenes --config configs/default.yaml
```

### 3. Generate Paper Figures

```bash
cd figures
python generate_paper_figures.py --nuscenes-root /path/to/nuscenes_mini
```

**Important**: The figure generation script requires real nuScenes data. It will not generate synthetic/fake data. All qualitative results are produced from actual nuScenes frames.

## Model Architecture

### DiT Backbone (Fig. 4.2)

- 12-layer transformer with adaLN-Zero conditioning
- Spatiotemporal decomposed attention (time, view, space)
- Bucketed relative position bias for memory efficiency
- Patch size: 2×2 spatial, no temporal merge

### KW-AMoE (Fig. 4.3)

- 8 experts with Top-K=2 sparse activation
- Frame-level temporal attention for keyframe detection
- Content saliency MLP for per-frame importance scoring
- Straight-through gradient estimator for end-to-end training
- Keyframe ratio: ρ = 0.3

### TS-Mem (Fig. 4.4)

- EMA memory buffer with α = 0.8 decay
- Keyframe-driven gated write fusion
- Non-keyframe residual injection with β = 0.3
- Memory-attention readout for DiT modulation

## Key Results

| Model Variant | FVD ↓ | FID ↓ | LPIPS ↓ | AP ↑ | CLIP Score ↑ |
|---------------|-------|-------|---------|------|--------------|
| Baseline DiT  | 104.13 | 12.55 | 0.247 | 18.47 | 22.65 |
| + Vanilla MoE | 94.21 | 11.30 | 0.221 | 19.82 | 23.91 |
| + KW-MoE      | 81.30 | 9.67  | 0.198 | 23.14 | 25.88 |
| + TS-Mem      | 87.27 | 10.41 | 0.210 | 21.56 | 24.73 |
| **Ours Full** | **67.89** | **7.83** | **0.176** | **26.73** | **28.94** |

## Citation

If you use this code in your research, please cite:

```bibtex

```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [nuScenes](https://www.nuscenes.org/) dataset by Motional
- [DiT](https://arxiv.org/abs/2212.09748) (Peebles & Xie, 2023)
- [DDIM](https://arxiv.org/abs/2010.02502) (Song et al., 2020)
