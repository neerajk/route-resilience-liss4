# VISTA-v2 — explained from scratch

A self-contained guide. If you've never seen this project, start here.

## 1. The problem (one paragraph)
We extract **roads** from **LISS-IV** satellite images (5.8 m/pixel, only Green/Red/NIR
bands). A model labels each pixel road / not-road. **VISTA** is the baseline that does
this. **VISTA-v2** is an upgraded version that (a) changes the input channels, (b) uses a
deeper backbone, and (c) adds **positional encoding** — and lets us **compare three kinds
of positional encoding** head-to-head.

## 2. What changes vs VISTA
| | VISTA (baseline) | VISTA-v2 |
|---|---|---|
| input channels | `[Green, Red, NIR, NDVI]` (4) | **`[Green, Red, NGRDI]`** (3, **no NIR**) |
| backbone | ResNet-34 / SegFormer | **ResNet-101** |
| decoder | UNet++ | UNet++ (same) |
| positional encoding | none | **3 options** (the experiment) |

### 2a. Why `NGRDI` and no NIR
`NGRDI = (Green − Red) / (Green + Red)` is an **RGB-only greenness index**. Why use it:
- The pretraining dataset **DeepGlobe is RGB** (no NIR). LISS-IV has Green and Red. So
  `[Green, Red, NGRDI]` is computable on **both** → the model pretrained on DeepGlobe
  transfers to LISS-IV with **identical input channels** (a "domain-invariant" input).
- **Trade-off (be honest):** NGRDI is weaker than NDVI at spotting canopy (NDVI uses NIR,
  which vegetation reflects strongly). We accept a weaker canopy cue to get clean transfer.
- The "is this pixel under canopy?" mask used for the **Occlusion-Recall** metric still
  comes from the pre-computed canopy layer in the tiles (or NGRDI for a fresh ingest).

### 2b. Why ResNet-101
A deeper CNN (101 layers) → larger **receptive field** (it "sees" a wider area per pixel),
which helps infer a road continuing under a tree gap. Cost: more compute, more overfit
risk on small data — handled by DeepGlobe pretraining + spatial-block cross-validation.

## 3. What is positional encoding (PE), and why
A piece of the network called **attention** can relate far-apart pixels — useful for
"this road segment continues over there." But attention is **position-blind** by itself;
it needs to be *told* where each pixel is. **Positional encoding is that location signal.**
PE only matters where attention is used.

We compare three PEs:
| PE | where it's added | idea | strength |
|---|---|---|---|
| **BoTNet** (default) | attention **bottleneck** | learned **relative** position bias | best for continuity, position-robust |
| **2-D RoPE** | attention **bottleneck** | rotate features by an angle ∝ position | relative, no extra params, size-robust |
| **sinusoidal** | **input layer** | fixed sin/cos coordinate channels added to the input | simple; absolute (can overfit tile layout) |
| *nope* (control) | — | no PE | tells us if PE helps at all |

"**Bottleneck**" = the deepest, smallest, most compressed feature map in the U-shaped
network (here 8×8 for a 256 tile) — the cheap, global place to run attention.
"**Input layer**" = before the encoder, alongside the 3 image channels.

> Note: BoTNet and RoPE live *inside* attention (the bottleneck). Sinusoidal is the only
> one that can sit at the input — that's why placement differs, and the benchmark also
> answers "does PE *location* matter?"

## 4. The architecture
```
[G, R, NGRDI]  (+ sinusoidal PE channels, only for the 'sincos' variant)
   │
   ▼  ResNet-101 encoder (downsamples; ImageNet-pretrained)
   │      deepest feature = 8×8 × 2048
   │      └─ if PE = botnet/rope: self-attention here, with that PE   ← attention bottleneck
   ▼  UNet++ decoder (upsamples back to full resolution)
   ▼
 road mask (1 channel)
```
One model class picks the PE from config — so all four variants share code and run the
**same way**.

## 5. How to run (all variants identical)
```bash
# pretrain once on DeepGlobe (shared encoder), then warm-start each variant:
python -m src.phase1.vista.pretrain --config config/phase1/vista_v2_pretrain.yaml

# train the four variants — same command, different config:
python -m src.phase1.vista.train --config config/phase1/vista_v2_botnet.yaml
python -m src.phase1.vista.train --config config/phase1/vista_v2_rope.yaml
python -m src.phase1.vista.train --config config/phase1/vista_v2_sincos.yaml
python -m src.phase1.vista.train --config config/phase1/vista_v2_nope.yaml
```
Each writes `runs/train/vista__vista_v2-<pe>__liss4__<ts>/` with `best.pt` + `metrics.csv`.

## 6. Comparing them (benchmark + statistics + plots)
```bash
python -m src.phase1.vista_v2.bench  --runs "runs/train/*vista_v2-*" --out runs/vista_v2_bench
python -m src.phase1.vista_v2.plots  --runs "runs/train/*vista_v2-*" --out runs/vista_v2_bench/figures
```
- **bench** → a table (mean ± 95% CI per PE) + **paired Wilcoxon signed-rank** tests
  across the 7 spatial-block folds (BoTNet vs each other PE), **Holm-corrected**, with
  **Cohen's d** effect sizes. *Caveat:* 7 folds = low statistical power, so we lead with
  effect sizes and confidence intervals, not p-values.
- **plots** → publication figures (PNG+PDF): grouped bars with CIs, per-fold paired lines,
  training-curve overlays.

## 7. Is VISTA-v2 still "VISTA"?
Yes — same job (per-pixel road segmentation), same arm. The attention bottleneck adds a
little global context but no road-graph reasoning, so it stays this side of **GROVE** (the
separate structure/topology-aware arm). Think: **VISTA-v2 = VISTA + depth + a PE study.**

## 8. Files
```
src/phase1/vista_v2/  pe.py · attention.py · model.py · bench.py · plots.py
config/phase1/        vista_v2.yaml (base) · vista_v2_{botnet,rope,sincos,nope}.yaml · vista_v2_pretrain.yaml
src/phase1/data/indices.py  → ngrdi()      src/phase1/models/factory.py → arch: vista_v2
```
