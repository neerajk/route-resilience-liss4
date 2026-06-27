# rr — Methodology

**Occlusion-robust road extraction from LISS-IV → routable graph → resilience.**
Step-by-step methodology. Citations in [`REFERENCES.md`](REFERENCES.md); run steps
in [`RUNBOOK.md`](RUNBOOK.md).

---

## 0. Framing & key decisions

Per-pixel **binary road segmentation** from Resourcesat-2/2A **LISS-IV** (5.8 m;
3 bands B2/B3/B4 = G/R/NIR; no Blue/SWIR) where roads are hidden under canopy,
shadow, clutter — then graph healing + criticality (Phases 2–4).

**Occlusion is handled by context-aware deep learning** (Transformer attention
"sees through" gaps) — not an explicit height prior. NDVI is the only derived
occlusion cue.

| Tier | Model | Role | Status |
|---|---|---|---|
| **Baseline** | smp UNet++ / **ResNet34** (stem-inflated) | guaranteed result | **trained** (OccRec ≈ 0.39, 3-epoch) |
| **Advanced** | smp **SegFormer / MiT** (`mit_b2`) — Transformer/attention | context-aware | wired, not yet run |
| dev/CI | MiniUNet (dep-free, optional D-LinkNet center) | smoke tests | runs |
| optional | DINOv3-SAT-493M (timm) | stretch arm | optional |

**Input stack = 4-channel `[G, R, NIR, NDVI]`** (CHM dropped). Labels = **OSM**
(auto-rasterized — zero manual labelling). Pretraining = **DeepGlobe** (downsampled).

Code layout: shared helpers in `src/common/`, perception in `src/phase1/`, graph in
`src/phase2/`.

---

## 1. Step-by-step pipeline (input → operation → output)

### Step 1 — Ingest → OSM-labelled tiles  (`src/phase1/preprocess/ingest_liss4.py`) ✅ built+run
- *In:* LISS-IV B2/B3/B4 GeoTIFFs + AOI shapefile.
- *Op:* reference grid = Green band (CRS/transform); Red/NIR aligned via WarpedVRT;
  **NDVI** = (NIR−Red)/(NIR+Red) per tile; **canopy = NDVI > thr** (occlusion proxy);
  **OSM roads auto-pulled (osmnx) for the AOI → buffered → rasterised** onto the grid
  → per-tile road `mask`; tile to 256² `.npz`; band-statistics written.
- *Out:* `data/tiles/*.npz` {bands[3], ndvi, canopy, mask, bounds} + `data.norm` stats.

### Step 2 — Normalize  (`src/phase1/data/dataset.py`)
- *Op:* per-channel standardise raw DN via `cfg.data.norm.{mean,std}`. → standardised input.

### Step 3 — Augment (train only)  (`src/phase1/data/augment.py`)
- *Op:* canopy-driven **OcclusionAugment** (hide roads under canopy → teaches gap
  inference), **ScaleAugment** (MTF blur-downsample), albumentations. → harder tiles.

### Step 4 — Model forward  (`src/phase1/models/factory.py`)
- *Baseline:* smp encoder (ImageNet, **stem inflated** to 4-ch — RGB conv1 copied to
  G/R/NIR, mean-init NDVI; Carreira & Zisserman 2017) → decoder → logits `[1,H,W]`.
- *Advanced:* swap encoder to **`mit_b2`** (SegFormer) — long-range attention is the
  "see through occlusion" mechanism. Optional D-LinkNet **Dblock** center.

### Step 5 — Loss  (`src/phase1/losses/losses.py`)
- *Op:* `L = 0.3·BCE + 0.4·Dice + 0.3·clDice`. clDice = topology/connectivity
  (Shit 2021). Optional **canopy-weighted BCE** (`loss.canopy_weight`) penalises
  missed *occluded* roads → pushes Occlusion-Recall.

### Step 6 — Train + validate  (`src/phase1/train.py`)
- *Op:* AdamW; CUDA AMP+GradScaler (no-op on MPS). Validation metrics pooled over
  **global pixel counts** (unbiased): IoU, Dice, **Occlusion-Recall** (headline),
  relaxed IoU/F1 at 3–5 px. Checkpoint on best Occlusion-Recall.
- *Out:* `runs/train/<ts>/` {best.pt, metrics.csv, loss_curve, prediction panel}.

### Step 7 — Export (the Phase 1→2 contract)  ⬜ TODO `src/phase1/predict.py`
- *Op:* load `best.pt` → run inference over the whole scene (windowed) → **stitch →
  georeferenced `pred_mask.tif`** (CRS + transform).
- *Out:* `pred_mask.tif` — the single artifact Phase 2 consumes.

### Data-flow
```
LISS-IV G/R/NIR + AOI ─► [1] ingest: NDVI · canopy=NDVI>thr · OSM→mask · tile
                                   │
              [2] normalize ─► [3] augment (occlusion+scale, train only)
                                   ▼
   [4] model  smp ResNet (baseline) | smp SegFormer/MiT (advanced)  ─► logits
                                   │
   [5] LOSS BCE+Dice+clDice (+canopy-weight)   [6] sigmoid→thr→mask
                                   │             METRICS (global): IoU, Dice,
                                   ▼             Occlusion-Recall + relaxed
   runs/train/<ts>/  ──► [7] export pred_mask.tif (georeferenced) ──► Phase 2
```

---

## 2. Phases 2–4 (graph + resilience) — extensible by contract

**Contract:** Phase 1 emits **`pred_mask.tif`** (or use the OSM mask for dev). Each
phase consumes only the previous phase's artifact, so work parallelises.

- **Phase 2 — graph** (`src/phase2/graph/`): binarize+clean → `skeletonize` → `sknw`
  → NetworkX → georeference → **heal** (Union-Find + MST, distance×angle) → weight →
  GeoJSON + graph + Connectivity Ratio. Ref: CRESI.
- **Phase 3 — resilience:** NetworkX **betweenness** → Gatekeeper nodes; node ablation
  (targeted vs random) → **Resilience Index** `R = L_base / L_perturbed`.
- **Phase 4 — dashboard:** Streamlit + Leaflet (criticality heatmap, click-to-flood).

---

## 3. Experiments / ablations
- **Occlusion ablation (headline):** baseline → +occlusion-aug → +clDice →
  +canopy-weight → SegFormer, reported with **Occlusion-Recall** + relaxed IoU.
- **Backbone:** smp ResNet vs smp SegFormer (vs optional DINOv3).
- **Pretraining:** scratch vs DeepGlobe-pretrained.
- **Generalisation:** leave-one-terrain-out (needs ≥2 terrains).
- Report **mean ± std over spatial-block folds** (Roberts 2017).

## 4. Status & outstanding
- ✅ Step 1 ingest (OSM labels) + baseline trained on real data.
- ⬜ **Export `pred_mask.tif`** (Phase 1→2 contract).
- ⬜ Spatial-block CV wired into `train.py` (currently contiguous split → leaky metric).
- ⬜ LR scheduler + early-stop + more epochs (metrics bounce at 3 epochs).
- ⬜ DeepGlobe pretraining; SegFormer run; augmentation upgrades; multi-terrain.
- ⬜ Phase 2 `src/phase2/graph/` (start on OSM masks).
- ⏸ Parked: CHM/DINOv3/Clay/distillation, OCOI, Sentinel-2.

> Note: where OSM already covers the area, the model's value is **generalisation**
> (areas without OSM) + **occlusion recovery**. Phase 2/3 may run on the OSM graph
> directly; the model graph is the automated / "no-OSM" demonstration.
