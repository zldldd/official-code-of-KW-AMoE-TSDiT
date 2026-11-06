# KW-AMoE-TSDiT

Route-A implementation of keyframe-weighted adaptive mixture-of-experts and temporal-stable memory for multi-view video diffusion.

## Computation path

- **DiT backbone:** joint view-time-space tokens, adaLN conditioning, and v-prediction diffusion.
- **Unified keyframe state:** one temporal saliency score produces the hard Top-K frame mask and mask-normalized weights. Non-keyframes have zero forward routing/write weight.
- **Actual adaptive computation:** keyframe tokens activate `K_e` experts and non-keyframe tokens activate one expert. Measured keyframe and non-keyframe expert calls are returned by the model.
- **Temporal Stable Memory:** mask-weighted EMA writes, cross-attention reads, memory-conditioned adaLN/router inputs, and non-keyframe residual prediction. Memory is reset for every training sample and generated sequence.

## Frozen 3D-VAE

`models/vae_backend.py` integrates **AutoencoderKLCogVideoX**, the same 3D-VAE used by [MagicDrive-V2](https://github.com/flymin/MagicDrive-V2), with local weights from `THUDM/CogVideoX-2b`.

- Latent channels: 16
- Temporal compression: 4×
- Spatial compression: 8×
- Scaling factor: 1.15258426
- Layout: DiT uses `(B,V,T,C,H,W)`; pixel videos use `(B,V,3,T,H,W)`

The backend loads local files only and is frozen during diffusion training. VAE reconstruction is verified on real nuScenes-mini clips before latent-cache generation. Long clips are split into 16-frame VAE chunks and concatenated along **time dimension 2**; the latent shape is `(V,T_pixel/4,16,H/8,W/8)` and is checked before every cache write. `LatentClipDataset` derives `frame_ids` from the latent time axis `(V,T,C,H,W)` at index 1, not the channel axis; long rollouts supply absolute latent-time offsets explicitly.

## Data and preprocessing

The nuScenes loader synchronizes six camera streams by timestamp, uses complete camera frames rather than only 2 Hz key samples, and performs center-crop plus isotropic resize. Intrinsics are updated with crop offsets and scale factors; extrinsics are stored as 4×4 matrices. Visible 3D boxes are reprojected with the updated intrinsics. Custom validation scenes do not reset the training scene list. Generated and real pairs are checked by `scene_token` before quality metrics, AP and qualitative comparison.

## Experiment configuration

`configs/mini.yaml` defines a reduced nuScenes-mini protocol for engineering validation. It is not a replacement for the full 700/150-scene benchmark.

- 16 pixel frames → 4 latent temporal positions
- 6 views, 512×256 input, 32×64 latent spatial shape
- Small DiT for local validation
- 20-step DDIM for mini experiments
- Five independent training seeds

All experiments use shared latent caches and resumable run directories containing configuration, final model, training history, generated latents, routing history, and raw metrics.

## Main commands

Run route-A invariants:

```bash
python main.py --test
```

Generate a shared latent cache:

```bash
python - <<'PY'
from utils.config import load_config
from experiments.latent_pipeline import prepare_latent_cache
config = load_config('configs/mini.yaml')
prepare_latent_cache(config, 'output/mini/latents16', frames=16, resume=True)
PY
```

The cache used by the current mini pipeline is created with `prepare_latent_cache` in `experiments/latent_pipeline.py`.

Run main and ablation groups:

```bash
python -m experiments.suite --group main
python -m experiments.suite --group mechanism
python -m experiments.suite --group rho
python -m experiments.suite --group expert
```

Evaluate one run:

```bash
python -m experiments.evaluate --run-dir output/mini/runs/main/full/seed42
```

Evaluate completed runs in isolated per-seed processes and audit actual routing. The evaluator skips validated outputs, records failures without discarding successful runs, and supports `--group main|expert|mechanism|rho`:

```bash
python -m experiments.evaluate_pending --isolated
python -m experiments.routing_audit --run-dir output/mini/runs/main/full/seed42
```

Run matched second-stage convergence probes. Each probe starts from the corresponding 1000-step checkpoint, adds 4000 steps, and reinitializes the optimizer; it is not a single continuous 5000-step run:

```bash
python -m experiments.convergence --variant baseline --steps 4000
python -m experiments.convergence --variant full --steps 4000
```

Run all long-horizon variants and seeds after building the corresponding shared caches. There are six non-overlapping validation sequences at 64 frames and two at 128 frames; the latter must not be described as six-sequence evidence:

```bash
for frames in 64 128; do
  for variant in baseline full; do
    for seed in 42 123 456 789 1024; do
      python -m experiments.long_rollout --manifest output/mini/latents${frames}/manifest.json --run-dir output/mini/runs/main/${variant}/seed${seed} --frames ${frames} --seed ${seed}
    done
  done
done
```

The domain split explicitly holds out mini_train night scene `scene-1094`; it is not an official mini_val night split. Train the source-day-only detector before domain evaluation:

```bash
python -m experiments.downstream --mode real --seed 42 --steps 200 --output-root output/mini/domain_detector --train-scenes scene-0061,scene-0553,scene-0655,scene-0757,scene-0796 --val-scenes scene-1094
```

Run daytime-to-nighttime zero-shot and target adaptation in this order (all five seeds by default):

```bash
python -m experiments.domain --protocol zero-shot --variant baseline --skip-evaluation
python -m experiments.domain --protocol zero-shot --variant full --skip-evaluation
python -m experiments.domain --protocol adapt --variant baseline --skip-evaluation
python -m experiments.domain --protocol adapt --variant full --skip-evaluation
python -m experiments.evaluate_pending --root output/mini/domain --isolated
```

Run controlled downstream 2D detection with the same detector initialization and 200-step budget:

```bash
for mode in real baseline full; do
  for seed in 42 123 456 789 1024; do
    python -m experiments.downstream --mode ${mode} --seed ${seed} --steps 200
  done
done
```

Generate manifests, provenance, summaries and real-run figures. Final timing should be performed without competing jobs from this study; other system GPU tenants are not claimed to be absent:

```bash
python -m experiments.manifest_audit --count 16
python -m experiments.manifest_audit --count 4 --role supplemental_4clip --output output/mini/evaluation_manifest_4clip.json
python -m experiments.provenance
python -m experiments.summarize
python -m experiments.benchmark --run-dir output/mini/runs/main/full/seed42
python -m experiments.benchmark --run-dir output/mini/runs/expert/E4_K4/seed42 --profile-only
python -m experiments.qualitative --baseline-dir output/mini/runs/main/baseline/seed42 --ours-dir output/mini/runs/main/full/seed42
python -m experiments.paper_results --require-complete
```

`output/mini/summary` contains per-seed raw metrics, aggregate values and paired statistics. `output/mini/report/mini_results.md` and the figures under `output/mini/report` are generated from raw result files only. `--profile-only` records operator FLOPs and expert calls in `profile.json` without measuring latency or loading the VAE; it does not replace the final exclusive-timing `benchmark.json`.

## Resumable execution

Inspect or resume the complete local experiment sequence with no dataset downloads:

```bash
python -m experiments.resume --stage all --dry-run
python -m experiments.resume --stage all
```

Use `--stage main-metrics`, `supplement-metrics`, `domain-training`, `downstream`, `long`, `routing`, `efficiency`, or `report` for an individual phase. Do not launch two runners for the same stage concurrently; wait for active study jobs to finish before running `--stage all`. Each command has a separate log under `output/mini/orchestrator/`; independent failures are recorded without discarding successful artifacts. Validated metrics, generated latents and completed detector training are reused. Run the efficiency phase only after other jobs from this study have finished; its output records `timing_status=measured_without_concurrent_study_jobs`. Other system GPU tenants are not claimed to be absent.

## Evaluation

- FVD: I3D Kinetics-400 temporally averaged 400-dimensional logits, computed per camera view and macro-averaged.
- FID: direct torchvision Inception-v3, 2048-dimensional features, GPU extraction with batches of eight.
- LPIPS: VGG, paired frames in `[-1,1]`, batches of four.
- CLIP Score: OpenCLIP ViT-B/32 raw cosine between scene descriptions and **generated** clip features.
- AP: bbox AP@0.5 using projected nuScenes annotations and a shared Real-only detector. Both full and standalone AP evaluations quantize decoded frames to uint8 with the same rule as the quality metrics; detector paths are normalized before accepting cached results.

The downstream closed loop is explicitly a **2D detection augmentation** experiment; it is not reported as 3D mAP/NDS.

## Result status

Only results from completed, provenance-verified runs are reported. Short smoke runs are engineering checks and are not manuscript evidence. No full-dataset claim is made from nuScenes results.
