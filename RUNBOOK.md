# rr — Runbook (Phase I)

Run from the project root `~/Desktop/Projects/rr` (so `python -m src.…` resolves).
Stack: PS-minimal — smp **ResNet baseline → SegFormer advanced**, **OSM labels**,
DeepGlobe pretrain. Occlusion solved by context-aware DL (CHM optional, NDVI proxy).

## 0. Create the env (once)
```bash
cd ~/Desktop/Projects/rr
micromamba create -f environment.yml -y      # geo + torch + smp (few min)
micromamba activate rr
```
VS Code: `Cmd+Shift+P → Python: Select Interpreter → rr`.

## 1. Smoke test (no data)
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python -m src.train --config config/smoke.yaml
```
Writes `runs/train/<ts>/` — confirms torch + smp + the loop work.

## 2. Step 1 — ingest (OSM labels → tiles)
Inputs already wired in `config.yaml → data.liss4`:
`data/raw/liss4/B2,B3,B4.tif` + `data/raw/aoi/bangalore_urban.shp`.
```bash
# quick test: set  data.liss4.max_tiles: 6  in config/config.yaml, then:
python -m src.preprocess.ingest_liss4 --config config/config.yaml
# full run: set max_tiles: 0
```
→ `data/tiles/*.npz` (bands, ndvi, canopy, **mask**) + `data/band_statistics.csv`,
and prints the `data.norm` to paste in.

## 3. Wire tiles for training — edit `config/config.yaml`
```yaml
data:
  source: tiles
  root: data/tiles
  norm: { mean: [g,r,nir,ndvi], std: [g,r,nir,ndvi] }   # from Step 2 printout
```

## 4. EDA (optional sanity)
```bash
python -m src.eda.run_eda --config config/config.yaml
```
→ `runs/eda/<ts>/` (histograms, NDVI separability, FCC previews, stats).

## 5. Train — baseline (Mac, small)
`config.yaml`: `model.arch: smp`, `decoder: unetplusplus`, `encoder: resnet34`.
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python -m src.train --config config/config.yaml
```
Headline metric = **Occlusion-Recall**; artifacts in `runs/train/<ts>/`.

## 6. Advanced (Transformer) — swap the encoder
`config.yaml`: `decoder: segformer` (or unetplusplus), `encoder: mit_b2`. Same command.

## 7. Train on GPU
1. Create the env on the GPU box, then swap in a CUDA torch build:
   ```bash
   micromamba run -n rr pip install --force-reinstall torch torchvision \
     --index-url https://download.pytorch.org/whl/cu124   # match the box CUDA
   ```
2. Run with the GPU config (inherits config.yaml, bigger batch/workers/amp):
   ```bash
   python -m src.train --config config/config_gpu.yaml
   ```
`runtime.device: auto` picks CUDA automatically — no code change.

---

## Mac ↔ GPU at a glance
| knob | Mac (MPS, dev) | GPU (CUDA, train) |
|---|---|---|
| config file | `config.yaml` | `config_gpu.yaml` (extends it) |
| `train.batch_size` | 2–4 | 16–32 |
| `train.num_workers` | 0 | 4–8 |
| `train.amp` | no-op | true |
| env var | `PYTORCH_ENABLE_MPS_FALLBACK=1` | — |

## Notes
- Step 1 (ingest) is CPU/geo — runs on Mac, no GPU needed.
- `route-resilience` env has the geo deps + osmnx (ingest works there) but lacks
  torch — use the `rr` env for training.
- Occlusion-Recall needs a `canopy` mask; with no CHM it's the **NDVI proxy**.
