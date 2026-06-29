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
| **Advanced** | smp **SegFormer / MiT** (`mit_b2`) — Transformer/attention | context-aware | wired (pretrain config ships with `mit_b2`) |
| dev/CI | MiniUNet (dep-free, optional D-LinkNet center) | smoke tests | runs |
| optional | DINOv3-SAT-493M (timm) | stretch arm | optional |

**Input stack = 4-channel `[G, R, NIR, NDVI]`** (CHM dropped). Labels = **OSM**
(auto-rasterized — zero manual labelling). Pretraining = **DeepGlobe** (0.5 m → 5.8 m
blur-downsample; RGB → 4-ch warm-start via stem inflation — `src/phase1/vista/pretrain.py`).

Code layout: shared helpers in `src/common/`, perception in `src/phase1/`, graph in
`src/phase2/`, resilience in `src/phase3/`, dashboard in `src/phase4/`. Runs cross-platform
(macOS MPS · Windows CPU · NVIDIA CUDA, `cu128` for Blackwell/RTX-50).

---

## 0b. Two arms: VISTA & GROVE

Phase 1 perception is organised as **two road-extraction arms over one shared core**
(`src/phase1/shared/` — data, models, losses, metrics, preprocess). The active arm is
set by `cfg.arm.name`; **all** artifact names are derived from it by `src/common/naming.py`
(run dirs `runs/train/<arm>__<model>__<stage>__<stamp>/`, masks `data/<arm>__pred_mask.tif`),
so the two arms never overwrite each other and Phase 2 can glob `*__pred_mask.tif`.

| Arm | Package / config | What it does | Research gap it closes |
|---|---|---|---|
| **VISTA** — VIsible-Surface road segmenTAtion (Arm A) | `src/phase1/vista/` · `config/phase1/config.yaml` | smp UNet++/SegFormer over `[G,R,NIR,NDVI]`; segments the road surface the sensor **sees** | the working baseline; strong where the road is visible |
| **GROVE** — occlusion completion (Arm B) | `src/phase1/grove/` · `grove.yaml` (`extends:` VISTA) | topology/continuity-aware recovery of roads **under canopy** | a per-pixel segmenter can't tell an occluded pixel "continues the line"; GROVE fuses the road-continuity prior HA-RoadFormer (Zhang et al. 2022) names as unsolved |

**GROVE design** (papers in [`REFERENCES.md`](REFERENCES.md)): **HA-RoadFormer** backbone
(overlapping multi-scale patch embedding + linear-complexity hybrid attention; Zhang et al.
2022) + **sinusoidal positional encoding** on tokens + a **sin(2θ)/cos(2θ) orientation head**
(axial road direction, mod-180°; Batra et al. 2019) + **clDice** topology loss (Shit et al.
2021), supervised on **under-canopy road pixels** (OSM road ∩ NDVI canopy). **CoANet**
(Mei et al. 2021 — the tree-occlusion prior art) contributes strip-conv + connectivity
attention; **CSWin** (Dong et al. 2022) is the optional stripe-attention backbone. Long,
fully-canopied gaps are bridged at the **graph layer** (reuse the Phase-2 angle-gate heal,
or Sat2Graph/RNGDet) — honest limit: optical-only inference recovers anchored gaps, not
arbitrarily long fully-hidden stretches.

**Graph heal — one step, two modes** (`config/phase2/config_phase2.yaml → graph.heal.mode`):
both arms pass through the *same* Phase-2 heal (Union-Find + MST + angle-gate); the mode
just changes what drives the angle-gate. **VISTA → `geometric`** (endpoint geometry only).
**GROVE → `orientation`** (the predicted `grove__orientation.tif` field). It is **not** two
stacked passes — GROVE swaps the angle signal, it doesn't add a second heal. `geometric` is
the only implemented mode today; `orientation` activates in GROVE Stage 6.

**Pretraining.** GROVE's backbone reuses VISTA's **DeepGlobe** pretrain pipeline (0.5 m→5.8 m
degrade, warm-start) for a fair ablation; its arm-specific heads (orientation, under-canopy
focal) are **LISS-IV-only** (DeepGlobe is RGB — no NIR/NDVI canopy proxy). A standard
Swin/CSWin/MiT encoder additionally gets ImageNet init; pure HA-RoadFormer has no public
weights (DeepGlobe-from-scratch, or pretrain on Massachusetts 1.2 m).

**Build stages.** **0–5 built**: naming + Phase-2 mask contract (0); supervision
`under_canopy_road` + `orient` (1, `grove/build_supervision.py`); **pluggable backbone**
`vista_mit` | `vista_resnet` | `cswin` | `haroadformer` behind one feature-pyramid
interface + sinusoidal PE (2, `grove/backbones/`); **orientation head** (3, `grove/heads.py`);
**GROVE loss** = clDice + under-canopy focal + orientation (4, `grove/losses.py`); optional
**CoANet** strip-conv + connectivity (5, `grove/modules/coanet.py`). The three backbones are
benchmarked under **identical heads/loss** (only the backbone varies). **Remaining**: the
graph bridge `heal.mode: orientation` (6) and the ablation run (7).

---

## Pipeline at a glance (boxes & arrows)

```
                ┌─────────────── INPUTS ────────────────┐
                │ LISS-IV G/R/NIR · OSM roads · AOI .shp  │
                └────────────────────┬───────────────────┘
                                     ▼
  ╔══════════════════════ PHASE 1 — perception ═══════════════════════╗
  ║  ingest ──► tiles(.npz) ──► train ──► best.pt ──► predict           ║
  ║  G/R/NIR→NDVI→canopy;        smp UNet++ / SegFormer                 ║
  ║  OSM→mask; tile             loss BCE+Dice+clDice → Occlusion-Recall ║
  ╚════════════════════════════════╤═══════════════════════════════════╝
                                     ▼
                  ┌──────── pred_mask.tif  (georeferenced) ─────────┐
                  │   the PHASE 1→2 CONTRACT  (or OSM mask for dev)  │
                  └────────────────────┬────────────────────────────┘
                                     ▼
  ╔══════════════════════ PHASE 2 — graph ═════════════════════════════╗
  ║  read ─► binarize+clean ─► skeletonize ─► build graph (sknw)        ║
  ║                  [ TILED over blocks — whole city ]                 ║
  ║  ─► georeference (pixel→world) ─► HEAL (Union-Find + MST, dist×ang) ║
  ║  ─► weight (length→time) ─► graph.graphml + roads.geojson           ║
  ╚════════════════════════════════╤═══════════════════════════════════╝
                                     ▼
  ╔══════════════════════ PHASE 3 — resilience ═══════════════════════╗
  ║  betweenness → Gatekeeper nodes → ablation (targeted/degree/random) ║
  ║  → global efficiency → Resilience Index + decay curves             ║
  ╚════════════════════════════════╤═══════════════════════════════════╝
                                     ▼
  ┌──────────── PHASE 4 — dashboard ─────────────────────────┐
  │ Streamlit + Folium/Leaflet + Plotly                       │
  │ criticality map · resilience curves · flood simulator     │
  └───────────────────────────────────────────────────────────┘
```

---

## 1. Step-by-step pipeline (input → operation → output)

### Step 0 — (optional) DeepGlobe pretraining — Stage A  (`src/phase1/vista/pretrain.py`) ✅ built
- *In:* DeepGlobe Road Extraction set (0.5 m RGB + road masks) at `data/raw/deepglobe/`.
- *Op:* **degrade 0.5 m → 5.8 m** with the sensor-realistic blur-downsample, then run the
  full `train.run` machinery in 3-ch RGB (`config/phase1/pretrain.yaml`, monitored on
  relaxed-F1 since DeepGlobe has no canopy). Closes the *resolution* gap; the *spectral*
  gap (RGB→G/R/NIR) is closed at fine-tune time by warm-start stem inflation.
- *Out:* `runs/train/<ts>/best.pt` → set `train.init_from` in `config/phase1/config.yaml`.

### Step 1 — Ingest → OSM-labelled tiles  (`src/phase1/shared/preprocess/ingest_liss4.py`) ✅ built+run
- *In:* LISS-IV B2/B3/B4 GeoTIFFs + AOI shapefile.
- *Op:* reference grid = Green band (CRS/transform); Red/NIR aligned via WarpedVRT;
  **NDVI** = (NIR−Red)/(NIR+Red) per tile; **canopy = NDVI > thr** (occlusion proxy);
  **OSM roads auto-pulled (osmnx) for the AOI → buffered → rasterised** onto the grid
  → per-tile road `mask`; tile to 256² `.npz`; band-statistics written.
- *Out:* `data/tiles/*.npz` {bands[3], ndvi, canopy, mask, bounds} + `data.norm` stats.

### Step 2 — Normalize  (`src/phase1/shared/data/dataset.py`)
- *Op:* per-channel standardise raw DN via `cfg.data.norm.{mean,std}`. → standardised input.

### Step 3 — Augment (train only)  (`src/phase1/shared/data/augment.py`)
- *Op:* a composable suite (config `augment.*`): **OcclusionAugment** (canopy-driven —
  hide roads under canopy → teaches gap inference), **RoadCoarseDropout** (mask road
  patches → occlusion recovery), **ScaleAugment** (MTF blur-downsample, GSD jitter),
  **RadiometricJitter** (per-band gain/bias), **CopyPasteRoads** (graft road patches),
  **PhotometricGeometric** (albumentations flips/rotations). → harder, more varied tiles.

### Step 4 — Model forward  (`src/phase1/shared/models/factory.py`)
- *Baseline:* smp encoder (ImageNet, **stem inflated** to 4-ch — RGB conv1 copied to
  G/R/NIR, mean-init NDVI; Carreira & Zisserman 2017) → decoder → logits `[1,H,W]`.
  When `train.init_from` is set, the DeepGlobe-pretrained 3-ch stem is **inflated**
  to the 4-ch `[G,R,NIR,NDVI]` input (warm-start).
- *Advanced:* swap encoder to **`mit_b2`** (SegFormer) — long-range attention is the
  "see through occlusion" mechanism. Optional D-LinkNet **Dblock** center.

### Step 5 — Loss  (`src/phase1/shared/losses/losses.py`)
- *Op:* `L = 0.3·BCE + 0.4·Dice + 0.3·clDice`. clDice = topology/connectivity
  (Shit 2021). Optional **canopy-weighted BCE** (`loss.canopy_weight`) penalises
  missed *occluded* roads → pushes Occlusion-Recall.

### Step 6 — Train + validate  (`src/phase1/vista/train.py`)
- *Op:* AdamW; CUDA AMP+GradScaler (no-op on MPS); **cosine LR + linear warmup**
  (or plateau) and **early-stop** on the monitored metric (`train.scheduler` /
  `train.early_stop`). **Spatial-block CV** (`cv.scheme: spatial_block`, Roberts 2017)
  holds out whole spatial blocks to stop the autocorrelation leak a random/contiguous
  split causes. Validation metrics pooled over **global pixel counts** (unbiased):
  IoU, Dice, **Occlusion-Recall** (headline), relaxed IoU/F1 at 3–5 px; optional **D4
  test-time augmentation** (`eval.tta`). Checkpoint on the best monitored metric.
- *Out:* `runs/train/<ts>/` {best.pt, metrics.csv, loss_curve, prediction panel}.

### Step 7 — Export (the Phase 1→2 contract)  ✅ `src/phase1/vista/predict.py`
- *Op:* load `best.pt` → windowed inference over the whole scene → **georeferenced
  `<arm>__pred_mask.tif`** (CRS + transform; probability or `--binary`; name from
  `src/common/naming.py`).
- *Out:* `data/<arm>__pred_mask.tif` (e.g. `vista__pred_mask.tif`) — the single artifact
  Phase 2 consumes (`config/phase2/config_phase2.yaml → graph.mask`).

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

- **Phase 2 — graph** (`src/phase2/graph/`) ✅: binarize+clean → `skeletonize` → `sknw`
  → NetworkX → georeference → **heal** (Union-Find + MST, distance×angle) → weight →
  GeoJSON + graph + Connectivity Ratio. **Tiled** over blocks for the whole city
  (`tile.py`; the global heal stitches the seams). Config-driven (`config/phase2/`).
  Sources: sknw (MIT) · CRESI/APLS (refs).
- **Phase 3 — resilience** (`src/phase3/resilience/`) ✅: **betweenness** → Gatekeeper
  nodes (Freeman 1977); **global efficiency** (Latora–Marchiori 2001); node ablation
  (targeted vs degree vs random) → **Resilience Index** (efficiency retained) + decay
  curves (Albert–Barabási 2000). Config-driven (`config/phase3/`).
- **Phase 4 — dashboard** (`src/phase4/`) ✅: Streamlit + Folium/Leaflet + Plotly.
  Four tabs: **Criticality Map** (betweenness heatmap on Leaflet), **Resilience Curves**
  (interactive Plotly ablation chart), **Gatekeepers** (sortable table),
  **Flood Simulator** (select top-N junctions → remove them → report fragmentation
  in real time from the Phase 2 graph). Config-driven (`config/phase4/`).

---

## 3. Experiments / ablations
- **Occlusion ablation (headline):** baseline → +occlusion-aug → +clDice →
  +canopy-weight → SegFormer, reported with **Occlusion-Recall** + relaxed IoU.
- **Backbone:** smp ResNet vs smp SegFormer (vs optional DINOv3).
- **Pretraining:** scratch vs DeepGlobe-pretrained.
- **Generalisation:** leave-one-terrain-out (needs ≥2 terrains).
- Report **mean ± std over spatial-block folds** (Roberts 2017).

### 3a. VISTA-v2 — positional-encoding study  (see [`docs/vista_v2.md`](docs/vista_v2.md))
**Arch:** ResNet-101 + UNet++ (smp) on NIR-free `[G, R, NGRDI]` (`(G−R)/(G+R)`, a
domain-invariant DeepGlobe↔LISS-IV input; weaker canopy cue than NDVI — accepted trade-off).
A PE-pluggable model (`arch: vista_v2`, `model.pe.type`) compares **4 variants under
identical heads/loss/data**: **botnet** (relative PE, attention bottleneck — *default*),
**rope** (2-D RoPE, bottleneck), **sincos** (sinusoidal at the **input**; encoder stem
patched for the +8 channels — a transfer caveat), **nope** (control). PE placement:
botnet/rope are attention-internal at the **stride-32 bottleneck**; sincos is input-level —
so the study isolates **PE type *and* location** (relative PEs are translation-robust under
spatial-block CV; absolute input PE risks tile-overfit — that's the hypothesis).
**Protocol** (`grove`-style, `vista_v2/bench.py`+`plots.py`): per-fold OccRec over **7
spatial-block folds** → mean ± 95% CI, **paired Wilcoxon vs botnet, Holm-corrected,
Cohen's d**. *Caveat:* 7 folds = low power → lead with effect sizes + CIs, p-values
indicative. Pretrain once on DeepGlobe (`vista_v2_pretrain.yaml`) → warm-start all 4.

### 3b. VISTA vs GROVE — two-level comparison (`src/phase1/grove/bench.py`)
Compare **at two levels**, never conflated:
- **Level 1 — mask (clean, no graph):** Occlusion-Recall **stratified by canopy/OCOI**,
  IoU, Dice, clDice/connectivity on the raw masks. The perceptual claim.
- **Level 2 — graph (system):** mask → Phase 2 → Connectivity Ratio / APLS / Resilience
  Index, each arm in its own `heal.mode` (VISTA `geometric`, GROVE `orientation`); report
  also (2a) both with `geometric` to isolate mask-effect from heal-effect.

**Backbone benchmark** (identical heads/loss, only the backbone varies — `grove_<bb>.yaml`):
`vista` baseline → `grove:vista_mit` → `grove:cswin` → `grove:haroadformer`, plus params &
latency. **Ablation ladder:** VISTA → GROVE-B1 → +orientation head → +clDice/under-canopy
focal → +CoANet → +orientation-guided heal — each row one marginal contribution, same
spatial-block split. Controls held constant: CV split · pretrain policy · `[G,R,NIR,NDVI]` ·
GT (OSM ∪ Microsoft) · metric code.

## 4. Status & outstanding  (all four phases merged to `main`)
- ✅ Phase 1 ingest (OSM labels) + baseline trained on real data (OccRec ≈ 0.39, pre-upgrade).
- ✅ Phase 1 training stack: **augmentation suite, spatial-block CV, cosine LR + warmup +
  early-stop, TTA, DeepGlobe pretrain (warm-start stem inflation)** — all built; Windows/CPU + NVIDIA ready.
- ✅ Export `pred_mask.tif` (`src/phase1/predict.py`) — the Phase 1→2 contract.
- ✅ Phase 2 graph (`src/phase2/graph/`) — mask → skeleton → **tiled** graph → heal → export.
- ✅ Phase 3 — resilience (`src/phase3/resilience/`): betweenness → ablation → Resilience Index.
- ✅ Phase 4 — Streamlit/Leaflet/Plotly dashboard (`src/phase4/dashboard.py`).
- 🟧 GROVE (Arm B) — Stages **0–5 built** (`src/phase1/grove/`): naming/contract, supervision,
  pluggable backbone (vista_mit·vista_resnet·cswin·haroadformer), orientation head, GROVE loss,
  optional CoANet. **Needs smoke-test + benchmark**; Stage 6 (`heal.mode: orientation`) and
  Stage 7 (ablation) remain.
- ⬜ **Run** the upgraded training stack end-to-end (pretrain → fine-tune, SegFormer) to
  produce a vectorizable `pred_mask.tif`, then Phases 2→3→4 on the real graph.
- ⏸ Parked: CHM/DINOv3/Clay/distillation, OCOI, Sentinel-2.

> Note: where OSM already covers the area, the model's value is **generalisation**
> (areas without OSM) + **occlusion recovery**. Phase 2/3 may run on the OSM graph
> directly; the model graph is the automated / "no-OSM" demonstration.
