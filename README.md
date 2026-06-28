# rr — Route Resilience

Occlusion-robust **road extraction** from medium-resolution **LISS-IV** satellite
imagery (5.8 m, Green/Red/NIR), turned into a routable **road graph** and analysed
for **criticality / resilience**.

Config-driven, device-dynamic (macOS MPS · NVIDIA CUDA · Windows/CPU). Four
decoupled phases — **Phase 1** perception (segmentation) → **Phase 2** graph
(skeleton → heal) → **Phase 3** resilience (criticality + stress test) →
**Phase 4** dashboard — connected by single hand-off artifacts (`pred_mask.tif`,
`graph.graphml`).

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
  ┌──────────────── PHASE 3 — resilience ─────────────────┐
  │ betweenness → ablation → Resilience Index             │
  └───────────────────────┬───────────────────────────────┘
                          ▼
  ┌──────────────── PHASE 4 — dashboard ──────────────────┐
  │ Streamlit / Leaflet (criticality + flood simulator)   │
  └────────────────────────────────────────────────────────┘
```
- **Occlusion** is handled by **context-aware deep learning** (Transformer attention
  infers road continuity across gaps) + an occlusion-augmentation suite + a
  connectivity (clDice) loss — not an explicit height prior.
- **Labels** = OpenStreetMap roads, auto-rasterised onto the LISS-IV grid (no manual
  labelling). **Input stack** = `[G, R, NIR, NDVI]`.
- **Generalisation** is hardened by **DeepGlobe pretraining** (0.5 m → 5.8 m degrade,
  RGB warm-start via stem inflation), **spatial-block CV**, a **cosine LR schedule +
  early-stop**, and optional **D4 test-time augmentation**.
- Full detail in [`METHODOLOGY.md`](METHODOLOGY.md); run steps in [`RUNBOOK.md`](RUNBOOK.md).

## Setup
```bash
micromamba create -f environment.yml -y
micromamba activate rr
export PYTORCH_ENABLE_MPS_FALLBACK=1     # macOS only
```
VS Code: **Python: Select Interpreter → rr**. The same `environment.yml` works on
macOS (MPS), Windows (CPU), and Linux; for an NVIDIA GPU swap in a CUDA PyTorch
build (`cu128` for RTX 50-series / Blackwell — see the header of `environment.yml`).

## Quick start
```bash
# smoke test (synthetic, no data)
python -m src.phase1.train --config config/phase1/smoke.yaml
# Step 1 — ingest: OSM labels + LISS-IV tiles (paths in config/phase1/config.yaml -> data.liss4)
python -m src.phase1.preprocess.ingest_liss4 --config config/phase1/config.yaml
# (optional) Stage A — DeepGlobe pretrain (0.5 m -> 5.8 m), then set train.init_from
python -m src.phase1.pretrain --config config/phase1/pretrain.yaml
# train the baseline (set data.source: tiles + paste data.norm first)
python -m src.phase1.train --config config/phase1/config.yaml
# GPU: python -m src.phase1.train --config config/phase1/config_gpu.yaml
```
Full pipeline end-to-end (predict → graph → resilience → dashboard) is in
[`RUNBOOK.md`](RUNBOOK.md); per-phase detail in `src/phase1/README.md` and `docs/`.

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
  phase1/  config.yaml · config_gpu.yaml · pretrain.yaml · smoke.yaml
  phase2/  config_phase2.yaml
  phase3/  config_phase3.yaml
  phase4/  config_phase4.yaml
src/
  common/   runtime (device/seed/amp) · config (extends loader) · viz (figures)
  phase1/   train.py · pretrain.py · predict.py · data/ (datasets · augment · deepglobe)
            · preprocess/ (ingest_liss4) · models/ · losses/ · metrics/ · eda/
  phase2/   graph/  (tiled mask → skeleton → graph → heal → export)
  phase3/   resilience/  (betweenness → ablation → Resilience Index)
  phase4/   dashboard.py  (Streamlit: criticality map · curves · flood simulator)
data/raw/liss4/  data/raw/aoi/  data/raw/deepglobe/  data/tiles/   (gitignored)
runs/   (gitignored)
METHODOLOGY.md · RUNBOOK.md · REFERENCES.md · CONTRIBUTING.md · docs/ · src/phase1/README.md
```

## Status (all four phases merged to `main`)
- ✅ **Phase 1** — ingest (OSM labels) + training stack: **augmentation suite**,
  **spatial-block CV**, **cosine LR + warmup + early-stop**, **TTA**, **DeepGlobe
  pretrain** with warm-start stem inflation; baseline trained (Occlusion-Recall ≈ 0.39,
  pre-upgrade). Windows/CPU + NVIDIA ready.
- ✅ **Export** `pred_mask.tif` (`src/phase1/predict.py`) — the Phase 1→2 contract.
- ✅ **Phase 2** — tiled mask → graph → heal → export (`src/phase2/graph/`).
- ✅ **Phase 3** — criticality (betweenness) + resilience stress-test (`src/phase3/resilience/`).
- ✅ **Phase 4** — Streamlit dashboard: criticality map, resilience curves, flood simulator (`src/phase4/`).
- ⬜ **Next** — full training run with the upgraded stack (pretrain → fine-tune,
  SegFormer) for a vectorizable mask; then run Phases 2→3→4 end-to-end on the real graph.

See [`METHODOLOGY.md`](METHODOLOGY.md) and [`RUNBOOK.md`](RUNBOOK.md).
