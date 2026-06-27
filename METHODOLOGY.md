# RR Pipeline — Phase I Methodology

**Occlusion-robust road extraction from medium-resolution LISS-IV imagery.**
Target: InGARSS 2026 + ISRO hackathon. This document is the step-by-step
methodology for the `rr` Phase-I pipeline (the `src/` code). Citations in
[`REFERENCES.md`](REFERENCES.md).

---

## 0. Framing

Per-pixel **binary road segmentation** from Resourcesat-2/2A **LISS-IV** (5.8 m
GSD; 3 bands B2/B3/B4 = Green/Red/NIR; **no Blue, no SWIR**) where roads are often
hidden under **tree canopy**. We hand the network physical priors (NDVI, CHM) and
evaluate headline on **Occlusion-Recall** — recall on canopy-occluded roads.

**Architecture decision (deadline-driven):**

| Tier | Model | Role | Status |
|---|---|---|---|
| **B0 baseline** | smp UNet++/Linknet, ResNet34/EffNet-b0, **stem inflated** | the GUARANTEED paper number | runnable when env installed |
| **B1 hero (stretch)** | DINOv3 ViT-L **SAT-493M** (timm, non-gated) + aux + decoder | foundation-model arm | wired; needs `timm` + weights download |
| **B2 stretch** | Clay v1.5 (GSD/wavelength-aware, ingests G/R/NIR natively) | non-RGB foundation arm | guarded stub |
| dev/CI | MiniUNet (dep-free, optional D-LinkNet center) | smoke tests only | runs today |

The paper's headline (the **input-stack ablation** `RGB → +NDVI → +CHM`) and the
**Occlusion-Recall** metric ride on B0 — they do **not** require DINOv3 or Clay.
LUPI/Clay distillation (the `liss4_dinov3_explorer` teacher–student idea) is
explicitly **out of scope** for Phase I; it would attach as a `λ·L_distill` term.

---

## 1. Step-by-step pipeline (inputs → operation → outputs)

### Step 1 — Raw inputs
- **LISS-IV** G/R/NIR (Bhoonidhi STAC + JWT) — primary imagery, 5.8 m.
- **CHM** (canopy height) — co-registered openCHm/CHMv2 GeoTIFF (occlusion prior).
- **OSM roads** (osmnx) — the labels (noisy).
- *(optional)* **Sentinel-2** (Planetary Computer) — spectral/temporal context.

### Step 2 — Put everything on one grid
*Operation:* reproject all layers to **EPSG:32643** (UTM 43N); resample CHM/S2 to
the LISS-IV 5.8 m grid (`bilinear` for continuous, `nearest` for labels).
`src/preprocess/coregister.py`. → *aligned layers.*

### Step 3 — Derive NDVI
*Operation:* `NDVI = (NIR − Red) / (NIR + Red)` on **reflectance** (not raw DN).
`src/data/indices.py` (Rouse et al. 1974). → *vegetation/occlusion channel.*

### Step 4 — Build the 5-channel stack
*Operation:* concatenate `[Green, Red, NIR, NDVI, CHM]` → `[5, H, W]`.
`src/data/dataset.py::_stack_channels`. → *model input.*

### Step 5 — Make the label
*Operation:* rasterise OSM road lines onto the LISS-IV grid; **buffer** to a
config width and burn with `rasterio.features.rasterize(all_touched=True)` so thin
5.8 m roads survive. `src/data/sources/osm.py` (osmnx; Boeing 2017). → *binary mask.*

### Step 6 — Tile + spatial-block split
*Operation:* cut into 256×256 tiles (`.npz`); split **by spatial blocks** (whole
~1.5 km blocks to one fold), NOT random pixels, to prevent autocorrelation leakage
inflating metrics (Roberts et al. 2017). `cfg.data.cv`. → *train / val / test tiles.*

### Step 7 — Normalize (real data)
*Operation:* per-channel standardise raw 10-bit DN via `cfg.data.norm.{mean,std}`
(synthetic is already [0,1] → no-op). `src/data/dataset.py::_to_tensors`. → *standardised input.*

### Step 8 — Augment (train only)
*Operation:* canopy-driven **OcclusionAugment** (hide road pixels under canopy →
teaches inpainting), **ScaleAugment** (MTF blur-downsample), optional
albumentations. `src/data/augment.py`. → *harder training tiles.*

### Step 9 — Model forward
*Operation (B0):* smp encoder (pretrained, **stem inflated** to 5 ch — copies RGB
conv1 onto G/R/NIR, mean-inits NDVI/CHM, rescaled 3/5; Carreira & Zisserman 2017)
→ decoder → **logits [1, H, W]**. Optional D-LinkNet **Dblock** center (dilation
1/2/4/8) bridges canopy gaps (Zhou et al. 2018). `src/models/factory.py`.
*Operation (B1):* DINOv3 SAT-493M (frozen, SAT normalization) on G/R/NIR → patch
tokens → `[B,1024,h,w]`; NDVI/CHM via a parallel aux CNN; concat → light decoder.

### Step 10 — Loss
*Operation:* `L = 0.3·BCE + 0.4·Dice + 0.3·clDice`.
- BCE = per-pixel (Milletari 2016 for Dice; class imbalance).
- **clDice** = topology/connectivity (Shit et al. 2021).
- **Canopy-weighted BCE** (optional, `loss.canopy_weight`): pixel weight
  `1 + w·canopy` so missing an *occluded* road costs more — pushes Occlusion-Recall.
`src/losses/losses.py`. → *scalar loss.*

### Step 11 — Optimise
*Operation:* AdamW; CUDA AMP + GradScaler (no-op on MPS/CPU). `src/train.py`.

### Step 12 — Validate
*Operation:* `logits → sigmoid → threshold → mask`; metrics **pooled over global
pixel counts** (unbiased) — IoU, Dice, **Occlusion-Recall** (headline) — plus
buffered/relaxed IoU & P/R/F1 at 3–5 px (Wiedemann 1998; Demir 2018). Checkpoint
on best Occlusion-Recall. → *metrics row.*

### Step 13 — Artifacts
`runs/train/<timestamp>/`: `best.pt`, `metrics.csv`, `figures/loss_curve.*`,
`figures/prediction.*` (FCC | GT | pred | occlusion overlay). *These are paper figures.*

### Inference
LISS-IV tile → Steps 2–4, 7, 9 → probability → threshold → **road mask** → Phase II
(skeletonise → graph → healing) → Phase III (betweenness criticality, resilience).

---

## 2. Data-flow graph

```
 LISS-IV G/R/NIR    CHM      OSM roads        Sentinel-2 (opt)
      │              │           │                  │
      ▼              ▼           ▼                  ▼
   [2] REPROJECT EPSG:32643 + RESAMPLE to 5.8 m grid
      │              │                          context
      ▼              │
   [3] NDVI          │
      └────┬─────────┘
           ▼
   [4] STACK [G,R,NIR,NDVI,CHM] [5,H,W]      [5] RASTERIZE OSM -> mask [1,H,W]
           │                                         │
           └──────────────┬──────────────────────────┘
                          ▼
   [6] TILE 256 + SPATIAL-BLOCK SPLIT  ──►  train / val / test
                          │
              [7] NORMALIZE (real DN)   [8] AUGMENT (occlusion+scale, train only)
                          ▼
   [9] MODEL  smp(stem-inflated)+[Dblock]  |  DINOv3 SAT-493M + aux  ─► logits
                          │
   [10] LOSS BCE+Dice+clDice (+canopy-weight)   [12] sigmoid->thr->mask
                          │                       METRICS (global): IoU, Dice,
   [11] AdamW (+GradScaler on CUDA)               Occlusion-Recall + relaxed
                          ▼
   [13] runs/train/<ts>/ : best.pt, metrics.csv, loss_curve, prediction panel
                          ▼
            INFERENCE -> road mask -> Phase II graph -> Phase III resilience
```

---

## 3. Experiments (the paper)
- **Input-stack ablation (headline):** `RGB → +NDVI → +CHM (→ +S2)`, reported with
  Occlusion-Recall + IoU.
- **Loss ablation:** BCE → +Dice → +clDice; canopy-weight on/off.
- **Backbone arms:** B0 smp baseline vs B1 DINOv3-SAT vs (B2 Clay).
- **Occlusion-Recall vs OCOI:** stratify test segments by the per-segment OCOI
  (`src/canopy/`) — the figure that proves the under-canopy claim.
- Report **mean ± std over spatial-block folds** + paired significance.

## 4. Known gaps / next steps (honest)
- Real-data hooks (`preprocess/pipeline.py` Stage-1 LISS-IV unpack, Bhoonidhi
  endpoints) are skeletons → activate when a product is in hand.
- `cfg.data.norm` must be filled from EDA band statistics before any real run.
- DINOv3 SAT features looked **noisy** in the explorer — gate B1 on confirming
  clean features before relying on it; B0 is the safe paper.
- D-LinkNet Dblock is wired into MiniUNet; for smp insert it between encoder and
  decoder (a small wrapper) — near-term task.
