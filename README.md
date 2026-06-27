# rr — Route Resilience

Occlusion-robust **road extraction** from medium-resolution **LISS-IV** satellite
imagery (5.8 m, Green/Red/NIR), turned into a routable **road graph** and analysed
for **criticality / resilience**.

Config-driven, device-dynamic (MPS / CUDA / CPU). Two decoupled phases:
**Phase 1** = perception (segmentation), **Phase 2** = graph (skeleton → heal →
network analysis), connected by a single artifact (`pred_mask.tif`).

## Methodology (overview)
```
LISS-IV G/R/NIR + AOI
   └─ ingest: NDVI · canopy=NDVI>thr · OSM roads → rasterised mask · tile
        └─ segmentation model (smp ResNet baseline → SegFormer/Transformer advanced)
             loss = BCE + Dice + clDice (+ optional canopy-weighting)
             metric = Occlusion-Recall (recall on canopy-occluded roads)
                └─ export → pred_mask.tif (georeferenced)
                     └─ PHASE 2: skeletonize → graph (sknw/NetworkX) → heal
                          (Union-Find + MST, distance×angle) → weighted graph
                               └─ criticality (betweenness) → resilience index
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
  phase2/   graph/  (mask → skeleton → graph → heal)   ← in progress
data/raw/liss4/  data/raw/aoi/  data/tiles/   (gitignored)
runs/   (gitignored)
METHODOLOGY.md · RUNBOOK.md · REFERENCES.md · CONTRIBUTING.md
```

## Status
- ✅ Step 1 ingest (OSM labels) + segmentation baseline trained on real data.
- ⬜ Next: spatial-block CV, LR schedule + more epochs, DeepGlobe pretrain, SegFormer,
  `pred_mask.tif` export (Phase 1→2 contract), Phase 2 `src/phase2/graph/`.

See [`METHODOLOGY.md`](METHODOLOGY.md) and [`RUNBOOK.md`](RUNBOOK.md).
