# Phase 1 — Perception (README)

> Phase-1-scoped docs, kept **separate from the root [`README.md`](../../README.md) /
> [`RUNBOOK.md`](../../RUNBOOK.md)** so this work doesn't clash with upstream PRs.
> Run all commands from the **project root**. This file supersedes the root docs
> for Phase-1 details (cross-platform setup + the training features below).

Occlusion-robust **road extraction** from LISS-IV (5.8 m, G/R/NIR): ingest → tiles
→ segmentation (`[G,R,NIR,NDVI]`) → `pred_mask.tif`. Loss = BCE + Dice + clDice;
headline metric = **Occlusion-Recall**.

## Setup (cross-platform)
Works on **macOS (MPS)**, **Linux/NVIDIA (CUDA)**, and **Windows (CPU, or CUDA)** —
the geo stack and code are identical across all three (`micromamba`, `mamba`, or
`conda` are interchangeable).

**macOS / Linux (bash/zsh):**
```bash
micromamba create -f environment.yml -y
micromamba activate rr
export PYTORCH_ENABLE_MPS_FALLBACK=1     # macOS only (no-op elsewhere)
```

**Windows (PowerShell):**
```powershell
micromamba create -f environment.yml -y   # or: conda env create -f environment.yml
micromamba activate rr
# No MPS on Windows. For NVIDIA, swap in a CUDA torch build (match your CUDA;
# cu128 for CUDA 12.8 / Blackwell sm_120 RTX 50-series, older cuXXX otherwise):
#   pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
```
VS Code (any OS): **Python: Select Interpreter → rr**.

## Quick start
Commands are identical on every OS (only env-var syntax differs — see Setup).
```bash
# smoke test (synthetic, no data)
python -m src.phase1.train --config config/phase1/smoke.yaml
# Step 1 — ingest: OSM labels + LISS-IV tiles (paths in config/phase1/config.yaml -> data.liss4)
python -m src.phase1.preprocess.ingest_liss4 --config config/phase1/config.yaml
# train the baseline (set data.source: tiles + paste data.norm first)
python -m src.phase1.train --config config/phase1/config.yaml
# GPU: python -m src.phase1.train --config config/phase1/config_gpu.yaml
```

## Models (`cfg.model.arch` / `cfg.model.encoder`)
| setting | what | role |
|---|---|---|
| `arch: miniunet` | dep-free U-Net (+ optional Dblock center) | smoke/CI |
| `arch: smp` + `encoder: resnet34` | UNet++ / ResNet (stem-inflated 4-ch) | **baseline** |
| `arch: smp` + `encoder: mit_b2` | SegFormer / Transformer (attention) | **advanced** (toggle in `config_gpu.yaml`) |
| `arch: dinov3` | DINOv3 SAT-493M (timm) | optional |

## Phase-1 training features (added)
All config-driven; see `config/phase1/config.yaml` for the knobs.

| feature | where | config |
|---|---|---|
| **LR schedule** cosine(+warmup)/plateau | `train.py:_build_scheduler` | `train.scheduler.{name,warmup_epochs,min_lr,...}` |
| **Early stopping** (monitors `eval.monitor`) | `train.py` loop | `train.early_stop.{enabled,patience,min_delta}` |
| **Longer runs** | — | `config.yaml` epochs 50, `config_gpu.yaml` 60 |
| **Augmentation on** + upgrades | `data/augment.py` | `augment.enabled: true` |
| · road-targeted CoarseDropout | `RoadCoarseDropout` | `augment.coarse_dropout` |
| · per-band radiometric jitter | `RadiometricJitter` | `augment.radiometric` |
| · intra-image road copy-paste | `CopyPasteRoads` | `augment.copy_paste` (off by default) |
| · D4-flip **TTA** at eval | `train.py:_tta_logits` | `eval.tta: true` |
| **Spatial-block CV** (no leak) | `train.py:_spatial_block_split` | `data.cv.{scheme: spatial_block, block_size_m}` |
| **Warm-start** from any checkpoint | `train.py:load_pretrained` | `train.init_from`, `train.init_inflate_stem` |
| **DeepGlobe pretrain** (0.5→5.8 m) | `data/deepglobe.py`, `pretrain.py` | `config/phase1/pretrain.yaml` |

Recursive `extends:` is supported in configs (e.g. `config_gpu.yaml` →
`config.yaml`; deeper chains resolve too).

## Status (Phase 1)
- ✅ Step 1 ingest (OSM labels) + segmentation baseline on real data.
- ✅ Training stability: cosine/plateau LR schedule + warmup + early-stop; longer runs.
- ✅ Spatial-block CV wired into training (whole blocks held out; no leak).
- ✅ Augmentation on by default + upgrades (CoarseDropout, radiometric jitter, copy-paste, TTA).
- ✅ SegFormer advanced path (toggle in `config_gpu.yaml`); warm-start hook (`train.init_from`).
- ✅ DeepGlobe pretraining (`pretrain.yaml`; 0.5 m→5.8 m degrade → warm-start).
- ⬜ Next: `pred_mask.tif` export (Phase 1→2 contract), Phase 2 `src/phase2/graph/`.

Run steps in [`RUNBOOK.md`](RUNBOOK.md) (this folder). Methodology in
[`METHODOLOGY.md`](../../METHODOLOGY.md).
