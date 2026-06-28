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
python -m src.phase1.vista.train --config config/phase1/smoke.yaml
# Step 1 — ingest: OSM labels + LISS-IV tiles (paths in config/phase1/config.yaml -> data.liss4)
python -m src.phase1.shared.preprocess.ingest_liss4 --config config/phase1/config.yaml
# (optional) Stage A — DeepGlobe pretrain (0.5 m -> 5.8 m), then set train.init_from
python -m src.phase1.vista.pretrain --config config/phase1/pretrain.yaml
# train VISTA (Arm A) baseline (set data.source: tiles + paste data.norm first)
python -m src.phase1.vista.train --config config/phase1/config.yaml
# GPU: python -m src.phase1.vista.train --config config/phase1/config_gpu.yaml
# GROVE (Arm B) Stage 1 — supervision targets onto existing tiles:
python -m src.phase1.grove.build_supervision --config config/phase1/grove.yaml
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

## Two arms: VISTA & GROVE
Phase 1 perception has **two road-extraction arms** that share one infrastructure
and both export the Phase-2 `pred_mask.tif` contract. The active arm is set by
`cfg.arm.name`, which drives every artifact name via `src/common/naming.py`
(so the arms never overwrite each other).

| arm | config | what it does | research gap it closes |
|---|---|---|---|
| **VISTA** — VIsible-Surface road segmenTAtion | `config/phase1/config.yaml` | smp UNet++/SegFormer segments the road surface the sensor **can see** | the working baseline; strong where the road is visible |
| **GROVE** — occlusion-completion (Arm B) | `config/phase1/grove.yaml` (`extends:` VISTA) | topology/continuity-aware recovery of roads **hidden under canopy** | a per-pixel segmenter can't tell that an occluded pixel "continues the line"; GROVE fuses the road-continuity prior that HA-RoadFormer (Zhang et al. 2022) names as unsolved |

**Mental model — what each arm *reasons about*.** Both arms output a pixel mask, but they
solve different problems:

| | unit of reasoning | problem type | question |
|---|---|---|---|
| **VISTA** | the **pixel** | image segmentation (dense classification) | "does *this pixel* look like road?" |
| **GROVE** | the **structure** (curve → network) | structure / topology-aware prediction | "where does the *road* go — even where it's hidden?" |

Think of it as three levels, with GROVE the bridge:
```
1. IMAGE PROCESSING      pixels, appearance         → VISTA      "is this pixel road?"
2. STRUCTURE-AWARE       continuity, direction      → GROVE      "where does the road go (even hidden)?"
3. SPATIAL NETWORK       graph, routing, resilience → Phase 2-3  "how do roads connect; what fails?"
```
VISTA classifies pixels by appearance (so canopy → "leaves" → no road). GROVE adds the priors
that make a road a *network object* — **continuity** (clDice), **direction** (sin/cos
orientation), a **graph bridge** for long gaps — so it can infer road under canopy from
context. The actual spatial-network analysis (intersections, routing, what breaks if a node
floods) is **Phase 2–3**; GROVE is structure-aware *perception* that feeds it, not the network
analysis itself.

**How GROVE works** (see [`METHODOLOGY.md`](METHODOLOGY.md) "Two arms"): a **backbone-pluggable
dual-head network** — a swappable backbone (`vista_mit` | `vista_resnet` | `cswin` |
`haroadformer`) + a shared FPN + **seg** and **sin(2θ)/cos(2θ) orientation** heads, trained
with **clDice** topology loss + **under-canopy focal** supervision (OSM road ∩ NDVI canopy) +
**sinusoidal positional encoding** on tokens. Long fully-canopied gaps are bridged at the
graph layer (Phase-2 `heal.mode: orientation`, or Sat2Graph/RNGDet). The three backbones are
**benchmarked under identical heads/loss** so the comparison isolates the backbone. References:
HA-RoadFormer (Zhang et al. 2022), CSWin (Dong et al. 2022), CoANet (Mei et al. 2021,
tree-occlusion prior art), Batra et al. (2019) orientation — see [`REFERENCES.md`](REFERENCES.md).

**Artifact naming** (`src/common/naming.py`): `runs/train/<arm>__<backbone>__<stage>__<stamp>/`,
masks `data/<arm>__pred_mask.tif` and `data/<arm>__orientation.tif`.

```bash
# Stage 1 — supervision targets (under-canopy + orientation) onto existing tiles:
python -m src.phase1.grove.build_supervision --config config/phase1/grove.yaml
# Stages 2-5 — backbone benchmark (seg-only, full VISTA pipeline):
python -m src.phase1.vista.train --config config/phase1/grove_haroadformer.yaml   # | grove_cswin | grove_vista_mit
# Full GROVE (seg + orientation + focal):
python -m src.phase1.grove.train --config config/phase1/grove.yaml
# Collate the comparison table:
python -m src.phase1.grove.bench --runs "runs/train/*liss4*" --out runs/bench
```
> **Built: Stages 0–5** (scaffold/naming · supervision · pluggable backbone · orientation
> head · GROVE loss · optional CoANet). **Remaining: Stage 6** (`heal.mode: orientation`
> in Phase 2) and **Stage 7** (run the benchmark + ablation). Smoke-test first:
> `python -m src.phase1.grove.train --config config/phase1/grove_smoke.yaml`.

## Layout
```
config/
  phase1/  config.yaml · config_gpu.yaml · pretrain.yaml · smoke.yaml
  phase2/  config_phase2.yaml
  phase3/  config_phase3.yaml
  phase4/  config_phase4.yaml
src/
  common/   runtime (device/seed/amp) · config (extends loader) · naming (arm-aware) · viz
  phase1/   ── two arms over one shared core ──
    shared/   data/ (datasets · augment · deepglobe) · preprocess/ (ingest_liss4)
              · models/ · losses/ · metrics/ · eda/ · canopy/
    vista/    train.py · pretrain.py · predict.py        (Arm A — visible-surface seg)
    grove/    backbones/ (vista·cswin·haroadformer) · decoder · heads · model · losses
              · modules/coanet · data · train · predict · bench · supervision   (Arm B)
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
- ✅ **Export** `<arm>__pred_mask.tif` (`src/phase1/vista/predict.py`) — the Phase 1→2 contract.
- ✅ **Phase 2** — tiled mask → graph → heal → export (`src/phase2/graph/`).
- ✅ **Phase 3** — criticality (betweenness) + resilience stress-test (`src/phase3/resilience/`).
- ✅ **Phase 4** — Streamlit dashboard: criticality map, resilience curves, flood simulator (`src/phase4/`).
- 🟧 **GROVE (Arm B)** — Stages **0–5 built** (naming/contract · supervision · pluggable
  backbone `vista_mit`/`vista_resnet`/`cswin`/`haroadformer` · orientation head · GROVE loss ·
  optional CoANet); **needs smoke-test + benchmark run**. Stage 6 (`heal.mode: orientation`)
  and Stage 7 (ablation) remain.
- ⬜ **Next** — full VISTA training run (pretrain → fine-tune, SegFormer); smoke + benchmark
  GROVE backbones; then Phases 2→3→4 end-to-end.

See [`METHODOLOGY.md`](METHODOLOGY.md) and [`RUNBOOK.md`](RUNBOOK.md).
