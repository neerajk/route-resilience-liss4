# Phase 1 — Runbook

> Phase-1-scoped runbook, kept **separate from the root [`RUNBOOK.md`](../../RUNBOOK.md)**
> so this work doesn't clash with upstream PRs. Run all commands from the
> **project root** (the repo folder, not this directory).

Layout: `src/common` (shared), `src/phase1` (perception), `src/phase2` (graph);
configs in `config/phase1/`. Stack: smp **ResNet baseline → SegFormer advanced**,
**OSM labels**.

Runs on macOS (MPS), Linux/NVIDIA (CUDA), and Windows (CPU or CUDA). The Python
commands are identical on every OS; only the env-var prefix differs:
- **bash/zsh (macOS/Linux):** `PYTORCH_ENABLE_MPS_FALLBACK=1 python -m ...`
- **PowerShell (Windows):** `$env:PYTORCH_ENABLE_MPS_FALLBACK = "1"` (once per session, macOS-only var — harmless on Windows), then `python -m ...`

## 0. Create the env (once)
```bash
# macOS / Linux (bash/zsh) and Windows (PowerShell) — same commands:
micromamba create -f environment.yml -y     # or: conda env create -f environment.yml
micromamba activate rr
```
VS Code: **Python: Select Interpreter → rr** (`Cmd+Shift+P` on macOS, `Ctrl+Shift+P` on Windows/Linux).

## 1. Smoke test (no data)
```bash
# macOS/Linux:
PYTORCH_ENABLE_MPS_FALLBACK=1 python -m src.phase1.train --config config/phase1/smoke.yaml
# Windows (PowerShell):
python -m src.phase1.train --config config/phase1/smoke.yaml
```

## 2. Step 1 — ingest (OSM labels → tiles)
Inputs wired in `config/phase1/config.yaml → data.liss4`:
`data/raw/liss4/B2,B3,B4.tif` + `data/raw/aoi/blore_urban.shp`.
```bash
# quick test: set  data.liss4.max_tiles: 6  in config/phase1/config.yaml, then:
python -m src.phase1.preprocess.ingest_liss4 --config config/phase1/config.yaml
# full run: set max_tiles: 0
```
→ `data/tiles/*.npz` (bands, ndvi, canopy, **mask**) + `data/band_statistics.csv` + prints `data.norm`.

## 3. Wire tiles for training — edit `config/phase1/config.yaml`
```yaml
data:
  source: tiles
  root: data/tiles
  norm: { mean: [g,r,nir,ndvi], std: [g,r,nir,ndvi] }   # from Step 2 printout
  cv: { scheme: spatial_block, block_size_m: 1500 }      # whole blocks held out (no leak)
```

## 4. EDA (optional)
```bash
python -m src.phase1.eda.run_eda --config config/phase1/config.yaml
```

## 5. Train — baseline (dev: Mac/Windows CPU)
`config/phase1/config.yaml`: `model.arch: smp`, `decoder: unetplusplus`, `encoder: resnet34`.
Now ships with cosine LR + warmup, early-stop, augmentation on, and spatial-block CV.
```bash
# macOS/Linux:
PYTORCH_ENABLE_MPS_FALLBACK=1 python -m src.phase1.train --config config/phase1/config.yaml
# Windows (PowerShell):
python -m src.phase1.train --config config/phase1/config.yaml
```
Headline metric = **Occlusion-Recall**; artifacts in `runs/train/<ts>/`
(`best.pt`, `metrics.csv` with per-epoch `lr`, `figures/loss_curve.*`, prediction panel).

## 6. Advanced (Transformer) — SegFormer / MiT-B2
The SegFormer path lives as a documented toggle inside `config_gpu.yaml`: uncomment
the `model:` block at the bottom and set `train.batch_size: 12`, then run the GPU
config as usual:
```bash
python -m src.phase1.train --config config/phase1/config_gpu.yaml
```
(Attention infers road continuity across canopy gaps. Keep the encoder == your
DeepGlobe-pretrain encoder so warm-start weights transfer.)

### Warm-start from a pretrained checkpoint (optional)
Set `train.init_from` to ANY checkpoint to load compatible weights before training
(non-strict, shape-checked: matches copied, mismatches reported & skipped):
```yaml
train:
  init_from: runs/train/<ts>/best.pt      # null => from initialisation
  init_inflate_stem: true                 # inflate a 3-ch RGB stem onto [G,R,NIR,NDVI] (I3D)
```
Accepts our `{"model": ...}` checkpoints, Lightning `{"state_dict": ...}`, or a raw
state_dict. This is the hook the DeepGlobe pretrain (below) plugs into.

## 6b. DeepGlobe pretraining → warm-start → train (full recipe)

**What this does, in plain terms:** road shapes look the same everywhere, so we first
teach the model "what a road looks like" on the big DeepGlobe dataset, then move that
knowledge to our LISS-IV model and fine-tune it on our actual data. Result: a better
starting point than ImageNet alone.

**The chain (there are TWO warm-starts — this trips people up):**
```
ImageNet  --(encoder_weights: imagenet)-->  DeepGlobe pretrain (RGB, 0.5->5.8 m)
          --(train.init_from = that best.pt)-->  LISS-IV fine-tune (4-ch G/R/NIR/NDVI)
```
- Stage 1 is warm-started from **ImageNet** (automatic).
- Stage 2 is warm-started from **your Stage-1 checkpoint** (you set `init_from`).

> ⚠️ **THE ONE RULE: the pretrain architecture must MATCH the fine-tune architecture**
> (same `encoder` AND `decoder`). Weights transfer by name+shape — mismatch ⇒ nothing
> transfers. This recipe uses the **advanced** model (`mit_b2` / `segformer`) on both
> sides. (For the ResNet baseline, use `resnet34` / `unetplusplus` on both instead.)

### Step 0 — data in place
- DeepGlobe at `data/raw/deepglobe/` (standard layout `<id>_sat.jpg` + `<id>_mask.png`;
  only the `train/` split has masks). Road-Extraction track, ~6 GB — e.g. Kaggle
  `balraj98/deepglobe-road-extraction-dataset`. The loader globs `data.deepglobe.root`
  **recursively** and auto-skips images without a mask (so `valid/`+`test/` are ignored).
- LISS-IV tiles already at `data/tiles/` (from Step 2) and `data.norm` set (Step 3).

**Wiring a downloaded Kaggle archive into place.** Say the archive extracted to
`D:\some\path\archive\` with `train/ valid/ test/`. You only need `train/` under
`data/raw/deepglobe/`. Pick one (data/ is gitignored, so none of this is committed):
```powershell
# Windows — directory junction (instant, no copy, keeps the archive intact):
New-Item -ItemType Junction -Path data\raw\deepglobe\train -Target D:\some\path\archive\train
```
```bash
# macOS/Linux — symlink (instant) OR move:
ln -s /some/path/archive/train data/raw/deepglobe/train      # symlink
# mv /some/path/archive/train data/raw/deepglobe/train       # or move
```
Verify the loader sees the pairs (expect ~6226):
```bash
python -c "from src.phase1.data.deepglobe import DeepGlobeDataset as D; print(len(D(root='data/raw/deepglobe')), 'pairs')"
```

### Step 1 — pretrain on DeepGlobe (mit_b2 / SegFormer)
Edit `config/phase1/pretrain.yaml` so the model **matches your target**:
```yaml
model:
  decoder: segformer       # match the LISS-IV model
  encoder: mit_b2          # match the LISS-IV model
  encoder_weights: imagenet
  in_channels: 3           # DeepGlobe is RGB
  stem_init: smp_default
train:
  batch_size: 12           # mit_b2 ~2x memory; drop to 8/4 if CUDA OOM
data:
  deepglobe:
    root: data/raw/deepglobe
    limit: 0               # 0 = all. TIP: use 200 for a fast first dry-run.
```
Run it:
```bash
python -m src.phase1.pretrain --config config/phase1/pretrain.yaml
```
→ writes `runs/train/<timestamp>/best.pt`. **Copy that path** — Step 2 needs it.
Sanity at start: log shows `encoder=mit_b2 decoder=segformer in_ch=3`,
`monitor=relaxed_f1`, and `normalization: OFF` (correct — RGB is already 0–1).

### Step 2 — fine-tune on LISS-IV, warm-started from Step 1
In `config/phase1/config_gpu.yaml`: uncomment the SegFormer `model:` block, set
`train.batch_size: 12`, and add the warm-start (paste your Step-1 path):
```yaml
train:
  batch_size: 12
  init_from: runs/train/<timestamp>/best.pt   # <- from Step 1
  init_inflate_stem: true                      # inflate 3-ch RGB stem -> 4-ch [G,R,NIR,NDVI]
model:
  encoder_weights: null    # optional: init_from supplies the encoder, skip the ImageNet redownload
```
Run training:
```bash
python -m src.phase1.train --config config/phase1/config_gpu.yaml
```

### Step 3 — confirm it actually warm-started (read the first ~10 log lines)
- ✅ `normalization: ON` and `in_ch=4`.
- ✅ a line like `[warm-start] ... loaded N tensors verbatim + inflated 1 stem conv(s)`
  and `inflate encoder...conv1: RGB(3) -> 4ch`. That's proof the DeepGlobe weights moved over.
- ❌ `[warm-start] ... loaded 0 tensors` ⇒ **architecture mismatch** — your `pretrain.yaml`
  wasn't `mit_b2/segformer`. Fix Step 1 and re-run.

### Notes
- **First-timer tip:** do a tiny dry run first — `data.deepglobe.limit: 200` + a few
  epochs in Step 1 — to confirm the whole chain end-to-end before the multi-hour run.
- **Does it help?** Also train Step 2 once with `init_from: null` (ImageNet-only) and
  compare Occlusion-Recall. That ablation is what justifies the pretrain.
- Windows: the DeepGlobe image-IO path sets `KMP_DUPLICATE_LIB_OK=TRUE` for you
  (torch/skimage OpenMP clash) — no action needed.

## 7. Train on GPU
```bash
# on the GPU box, after creating the env, swap in a CUDA torch build.
# Match the cuXXX to the box's CUDA: cu128 for CUDA 12.8 (and required for
# Blackwell GPUs / sm_120, e.g. RTX 50-series); older cu124/cu121 for older CUDA.
micromamba run -n rr pip install --force-reinstall torch torchvision \
  --index-url https://download.pytorch.org/whl/cu128
# then run with the GPU config (extends config.yaml):
python -m src.phase1.train --config config/phase1/config_gpu.yaml
```

---

## Dev ↔ GPU at a glance
| knob | Dev: Mac (MPS) / Windows·Linux (CPU) | GPU (CUDA, train) |
|---|---|---|
| config | `config/phase1/config.yaml` | `config/phase1/config_gpu.yaml` |
| `train.batch_size` | 2–4 | 16–32 |
| `train.num_workers` | 0 (also safest on Windows) | 4–8 |
| `train.amp` | no-op | true |
| env var | `PYTORCH_ENABLE_MPS_FALLBACK=1` (macOS only) | — |

## Notes
- Step 1 (ingest) is CPU/geo — runs on any OS, no GPU needed.
- Windows: keep `train.num_workers: 0` (reliable default); entrypoints are guarded
  with `if __name__ == "__main__":`.
- `data/` + `runs/` live at the repo root (shared) — relative paths in configs resolve when run from root.
- Phase II graph: `python -m src.phase2.graph.run_graph --config config/phase2/config_phase2.yaml` (once `src/phase2/graph` exists).
