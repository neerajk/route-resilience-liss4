# rr — Route Resilience

Occlusion-robust **road extraction** from medium-resolution **LISS-IV** satellite
imagery (5.8 m, Green/Red/NIR), turned into a routable **road graph** and analysed
for **criticality / resilience**.

Config-driven, device-dynamic (MPS / CUDA / CPU). Two decoupled phases:
**Phase 1** = perception (segmentation), **Phase 2** = graph (skeleton → heal →
network analysis), connected by a single artifact (`pred_mask.tif`).

## Methodology (overview)
```
  INPUTS:  LISS-IV G/R/NIR  ·  OSM roads  ·  AOI

  ┌──────────────── PHASE 1 — perception ─────────────────┐
  │ ingest → tiles → train → best.pt → predict            │
  │ [G/R/NIR→NDVI, OSM→mask]   [smp UNet++ / SegFormer]    │
  │ loss BCE+Dice+clDice  →  Occlusion-Recall             │
  └───────────────────────┬───────────────────────────────┘
                          ▼  pred_mask.tif (georeferenced)  ◄ the contract
  ┌──────────────── PHASE 2 — graph ──────────────────────┐
  │ read → clean → skeletonize → build graph (sknw)       │
  │ [ tiled over the whole city ] → georeference          │
  │ → HEAL (Union-Find + MST) → weight → graph.graphml    │
  └───────────────────────┬───────────────────────────────┘
                          ▼
  ┌──────────────── PHASE 3 / 4 — resilience (next) ──────┐
  │ betweenness → Resilience Index → dashboard            │
  └────────────────────────────────────────────────────────┘
```
- **Occlusion** is handled by **context-aware deep learning** (Transformer attention
  infers road continuity across gaps) + occlusion augmentation + a connectivity
  (clDice) loss — not an explicit height prior.
- **Labels** = OpenStreetMap roads, auto-rasterised onto the LISS-IV grid (no manual
  labelling). **Input stack** = `[G, R, NIR, NDVI]`.
- Full detail in [`METHODOLOGY.md`](METHODOLOGY.md); run steps in [`RUNBOOK.md`](RUNBOOK.md).

## Setup
```bash
micromamba create -f environment.yml -y
micromamba activate rr
export PYTORCH_ENABLE_MPS_FALLBACK=1     # macOS
```
VS Code: **Python: Select Interpreter → rr**.

## Quick start
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
| `arch: smp` + `encoder: mit_b2` | SegFormer / Transformer (attention) | **advanced** |
| `arch: dinov3` | DINOv3 SAT-493M (timm) | optional |

## Layout
```
config/
  phase1/  config.yaml · config_gpu.yaml · smoke.yaml
  phase2/  config_phase2.yaml
src/
  common/   runtime (device/seed/amp) · config (extends loader) · viz (figures)
  phase1/   train.py · data/ · preprocess/ (ingest_liss4) · models/ · losses/ · metrics/ · eda/
  phase2/   graph/  (tiled mask → skeleton → graph → heal → export)
data/raw/liss4/  data/raw/aoi/  data/tiles/   (gitignored)
runs/   (gitignored)
METHODOLOGY.md · RUNBOOK.md · REFERENCES.md · CONTRIBUTING.md
```

## Status
- ✅ **Phase 1** — ingest (OSM labels) + baseline trained (Occlusion-Recall ≈ 0.39).
- ✅ **Export** `pred_mask.tif` (`src/phase1/predict.py`) — the Phase 1→2 contract.
- ✅ **Phase 2** — tiled mask → graph → heal → export (`src/phase2/graph/`).
- ⬜ **Next** — improve the model (more epochs, SegFormer, DeepGlobe pretrain) so its
  mask is vectorizable; then **Phase 3** (betweenness → Resilience Index) + **Phase 4** dashboard.

See [`METHODOLOGY.md`](METHODOLOGY.md) and [`RUNBOOK.md`](RUNBOOK.md).
