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

## 6b. DeepGlobe pretraining (0.5 m → 5.8 m → warm-start)
Pretrain a 3-ch RGB road model on DeepGlobe, degraded to the LISS-IV GSD, then
warm-start the LISS-IV model from it. DeepGlobe is just another `data.source`, so
this reuses the whole training pipeline.
```bash
# 0) put DeepGlobe at data/raw/deepglobe (standard layout <id>_sat.jpg + <id>_mask.png)
#    Road-Extraction track, ~6 GB; e.g. Kaggle balraj98/deepglobe-road-extraction-dataset.
#    Set the path in config/phase1/pretrain.yaml -> data.deepglobe.root.
# 1) pretrain (degrades 0.5 m -> 5.8 m via the MTF blur-downsample, monitors relaxed-F1):
python -m src.phase1.pretrain --config config/phase1/pretrain.yaml
#    -> runs/train/<ts>/best.pt   (RGB, in_channels=3)
# 2) warm-start LISS-IV training: set train.init_from to that best.pt in config.yaml, then:
python -m src.phase1.train --config config/phase1/config.yaml
```
The 3-ch→4-ch stem inflation is automatic (`init_inflate_stem: true`). Tune the
encoder in `pretrain.yaml` to match your target (`resnet34` baseline / `mit_b2`
advanced) so weights transfer cleanly. Windows: the DeepGlobe image-IO path sets
`KMP_DUPLICATE_LIB_OK=TRUE` for you (torch/skimage OpenMP clash).

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
