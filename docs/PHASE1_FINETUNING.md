# Phase 1 — Fine-Tuning & Occlusion-Recall Improvement Guide

How to push **Occlusion-Recall (OccRec)** — the model's ability to recover road
pixels hidden under tree canopy — and how to **benchmark changes rigorously** so
each improvement is attributable. Targets the LISS-IV SegFormer fine-tune
(`config/phase1/config_gpu.yaml`), warm-started from the DeepGlobe pretrain.

> OccRec = recall on the *small, hard* subset of road pixels that fall under
> canopy. It is the noisiest metric by design (small denominator, pure recall at a
> fixed 0.5 threshold). Always read it alongside precision / F1 / IoU and the
> qualitative panels — never a single epoch.

---

## 1. Improvement roadmap (prioritized by impact ÷ effort)

Status: `[ ]` todo · `[~]` in progress · `[x]` done. Map each to the file you'd touch.

### Tier 1 — loss & eval (few-line changes; do first)
- `[~]` **`loss.canopy_weight` 0 → 1.5–3** — up-weights occluded-road pixels in BCE.
  *(file: config; loss already supports it.)* **You are running 1.5.**
- `[ ]` **Add a Tversky / Focal-Tversky term** (β>α, e.g. β=0.7) — penalizes false
  negatives more than false positives → directly raises recall on thin structures.
  *(file: `src/phase1/losses/losses.py` → add to `CombinedRoadLoss`.)*
- `[~]` **`eval.tta: true`** — D4-flip test-time augmentation; steadier, usually
  higher recall. Eval-only (no retrain). **You enabled this.**
- `[ ]` **Stronger road-targeted masking** — raise `augment.coarse_dropout.{p,max_frac}`;
  this is the cheap version of Masked Image Modeling (below). *(file: config.)*

### Tier 2 — architecture / auxiliary tasks (moderate; highest research support)
- `[ ]` **Masked Image Modeling (MIM) head** — add an auxiliary branch that
  reconstructs masked *road* patches from surrounding context (RemainNet /
  RoadFocusNet). Explicitly trains "infer occluded road from visible road."
- `[ ]` **Joint orientation learning** (Batra et al. CVPR'19) — auxiliary head
  predicting per-pixel road **direction** (binned angles, labels from the mask
  skeleton). Shared features encode connectivity → bridges canopy gaps.
- `[ ]` **Dilated / strip-conv bottleneck** — D-LinkNet `Dblock` or strip kernels
  (OARENet) between encoder and head → larger receptive field to "see" both sides
  of a covered road. *(file: `src/phase1/models/factory.py`.)*

### Tier 3 — bigger bets
- `[ ]` Topology-aware adversarial loss (TopoAL) / multi-receptive-field (TopoRF-Net).
- `[ ]` Stronger RS-pretrained backbone (SatlasPretrain Swin-v2 / DINOv3-SAT stub).
- `[ ]` **Occlusion-conditioned attention** using NIR/NDVI — tell the network *where*
  it's occluded (NDVI-derived) so it knows when to trust context over direct evidence.
  (You already carry NIR/NDVI — most RGB-only methods cannot do this.)

> Ceiling note: where canopy *fully* occludes a road with no visible cue on either
> side, segmentation cannot recover it — that gap is **Phase 2's** job (graph
> skeleton + MST healing). Pixel-model + graph-healing together is the real win.

---

## 2. Benchmarking methodology

### 2.1 Golden rule — change ONE variable at a time
Keep these **identical** across compared runs so the delta is attributable:
`runtime.seed`, `data.cv` (same spatial-block split → same val tiles), `epochs` +
`early_stop`, `train.init_from` (same pretrain ckpt), `eval.threshold`, augmentation,
LR schedule. Vary only the knob under test.

### 2.2 TTA is separable — don't conflate it with training changes
TTA changes **inference only**, not weights. So:
- To measure **canopy_weight**, hold TTA constant (ideally **off**) in both runs.
- To measure **TTA**, evaluate the **same** `best.pt` with `tta:false` vs `tta:true` —
  **no retrain needed**.

⚠️ **Your current run has BOTH canopy_weight=1.5 and TTA=on.** If your baseline run
had TTA off, the metrics.csv are not apples-to-apples. Fix it one of two cheap ways:
1. (preferred) Use a small eval harness to re-score each `best.pt` with TTA off
   **and** on, on the same val split → isolates both effects (see 2.6).
2. Or re-run the cw=1.5 model with `tta:false` for the training comparison and treat
   TTA as a separate final measurement.

### 2.3 Recommended experiment matrix
Minimal clean set (TTA measured at eval, not as separate trainings):

| Run | `canopy_weight` | train TTA | Purpose |
|-----|-----------------|-----------|---------|
| A — baseline | 0.0 | off | reference (DeepGlobe warm-start only) |
| B — recall-weighted | 1.5 | off | does occluded-pixel weighting help? |
| (eval pass) | — | off **and** on per ckpt | isolates TTA's contribution |

Optionally add C: `canopy_weight: 3.0` to see if more weight helps or over-predicts.

### 2.4 Metrics to report (NOT just OccRec)
Recall-weighting trades precision for recall — you must see the whole picture:

| Metric | Why |
|--------|-----|
| **occlusion_recall** | primary objective |
| relaxed_precision / recall / **F1** | F1 balances the trade-off (buffered) |
| iou / dice | overall pixel agreement |
| best **epoch** | report the early-stop checkpoint, not the last/noisy epoch |

**Genuine occlusion recovery** = OccRec ↑ **while** precision/F1/IoU hold. If OccRec
↑ but F1/IoU drop sharply, the model just predicts *more road everywhere* (cheap
recall), not real under-canopy recovery.

### 2.5 Control for the operating point (threshold)
OccRec at a fixed 0.5 mixes "better model" with "different calibration." For a fair
read, sweep the threshold on the val set and compare **OccRec at matched precision**
(or report PR-AUC / the PR curve). A model that gives higher OccRec at the *same*
precision is genuinely better.

### 2.6 Where the numbers live + a quick compare
Each run writes `runs/train/<ts>/`:
- `metrics.csv` — per-epoch (incl. `lr`, all metrics).
- `best.pt` — has `["val"]` (best-epoch metrics) + `["epoch"]`.
- `figures/prediction*.png` — 3 random val patches (now source-labeled).

Pull the best-epoch metrics of any run:
```bash
python -c "import torch; d=torch.load('runs/train/<ts>/best.pt', map_location='cpu', weights_only=False); print('ep', d['epoch'], d['val'])"
```
> A small `benchmark.py` that loads N checkpoints, re-evaluates the **same** val
> split with `tta` on/off and a threshold sweep, and prints one comparison table is
> the rigorous way to do 2.2–2.5. (Ask to have it scaffolded.)

### 2.7 Handle the noise
OccRec bounces epoch-to-epoch. To compare fairly:
- Compare **best-checkpoint** metrics (early-stop already selects these), not single epochs.
- Or report **mean ± std over the last 5 epochs**.
- Metrics are globally **pooled** over the val set already (unbiased) — good.
- For a publication claim, repeat the key runs over **2–3 seeds** and report mean ± std.

### 2.8 Qualitative check (don't skip)
Open `figures/prediction*.png`. The 4th panel is the **occlusion overlay**:
GREEN = occluded road **recovered**, RED = occluded road **missed**. More green / less
red across the 3 random patches is the direct visual read of OccRec improvement —
and catches "cheap recall" (roads predicted everywhere) that a number alone hides.

---

## 3. Results table (fill as you go)

| Run id (ts) | canopy_w | TTA | thr | OccRec | relF1 | relPrec | IoU | best ep | notes |
|-------------|----------|-----|-----|--------|-------|---------|-----|---------|-------|
| `<A ts>` | 0.0 | off | 0.5 |  |  |  |  |  | baseline |
| `<B ts>` | 1.5 | off | 0.5 |  |  |  |  |  | + canopy weight |
| `<B ts>` (eval) | 1.5 | **on** | 0.5 |  |  |  |  |  | + TTA on same ckpt |

Decision rule: keep the change if **OccRec ↑ and relF1/IoU not materially worse**.

---

## 4. Reproducibility / logging
For each experiment record: run timestamp, **git commit**, `runtime.seed`, the exact
config diff (canopy_weight, tta, …), and the `init_from` checkpoint. Configs +
`best.pt["cfg"]` already capture most of this — note the git commit alongside.

---

## 5. References
- DL road extraction review (2025): https://www.sciencedirect.com/science/article/pii/S0924271625002758
- RemainNet — masked image modeling for occlusion: https://www.mdpi.com/2072-4292/15/17/4215
- RoadFocusNet — focused MIM + transformer: https://www.tandfonline.com/doi/full/10.1080/17538947.2025.2549435
- Occlusion-Aware Road Extraction Net (strip conv): https://www.researchgate.net/publication/379845557
- Batra et al. — orientation + segmentation (connectivity): https://www.researchgate.net/publication/338095973
- clDice — topology-preserving loss: https://ar5iv.labs.arxiv.org/html/2003.07311
- TopoAL — adversarial topology: https://arxiv.org/pdf/2007.09084
- SatlasPretrain backbones: https://huggingface.co/allenai/satlas-pretrain
