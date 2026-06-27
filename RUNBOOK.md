# rr — Runbook

Run from the project root `~/Desktop/Projects/rr`. Layout: `src/common` (shared),
`src/phase1` (perception), `src/phase2` (graph); configs in `config/phase1/` and
`config/phase2/`. Stack: smp **ResNet baseline → SegFormer advanced**, **OSM labels**.

## 0. Create the env (once)
```bash
cd ~/Desktop/Projects/rr
micromamba create -f environment.yml -y
micromamba activate rr
```
VS Code: `Cmd+Shift+P → Python: Select Interpreter → rr`.

## 1. Smoke test (no data)
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python -m src.phase1.train --config config/phase1/smoke.yaml
```

## 2. Step 1 — ingest (OSM labels → tiles)
Inputs wired in `config/phase1/config.yaml → data.liss4`:
`data/raw/liss4/B2,B3,B4.tif` + `data/raw/aoi/bangalore_urban.shp`.
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
```

## 4. EDA (optional)
```bash
python -m src.phase1.eda.run_eda --config config/phase1/config.yaml
```

## 5. Train — baseline (Mac)
`config/phase1/config.yaml`: `model.arch: smp`, `decoder: unetplusplus`, `encoder: resnet34`.
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python -m src.phase1.train --config config/phase1/config.yaml
```
Headline metric = **Occlusion-Recall**; artifacts in `runs/train/<ts>/`.

## 6. Advanced (Transformer) — swap encoder
`config`: `decoder: segformer` (or unetplusplus), `encoder: mit_b2`. Same command.

## 7. Train on GPU
```bash
# on the GPU box, after creating the env, swap in a CUDA torch build:
micromamba run -n rr pip install --force-reinstall torch torchvision \
  --index-url https://download.pytorch.org/whl/cu124
# then run with the GPU config (extends config.yaml):
python -m src.phase1.train --config config/phase1/config_gpu.yaml
```

## 8. Export pred_mask.tif (Phase 1 → 2 contract)
```bash
python -m src.phase1.predict --ckpt runs/train/<ts>/best.pt --out data/pred_mask.tif
# --binary for 0/1 instead of probability
```
Writes a georeferenced full-scene road-probability GeoTIFF (model + norm read from the checkpoint).

## 9. Phase 2 — graph (config-driven; clean CLI)
```bash
python -m src.phase2.graph.run_graph --config config/phase2/config_phase2.yaml
```
All options live in `config/phase2/config_phase2.yaml → graph`:
- **input** — `mask:` a GeoTIFF (e.g. `data/pred_mask.tif`, or a saved `osm_mask.tif`),
  or `null` to auto-build an OSM mask.
- **mode** — `tiling.enabled: true` (default) processes the **whole scene in blocks**
  (the global heal stitches the seams); set `false` + `window: [row,col,h,w]` for one region.
- **de-noise a model mask** — raise `min_object_size` / `threshold`.

→ `runs/graph/<ts>/`: `graph.graphml` (Phase 3 input), `roads.geojson` (QGIS),
`metrics.csv` (Connectivity Ratio), and a healing overlay (window/single mode).

---

## Mac ↔ GPU at a glance
| knob | Mac (MPS, dev) | GPU (CUDA, train) |
|---|---|---|
| config | `config/phase1/config.yaml` | `config/phase1/config_gpu.yaml` |
| `train.batch_size` | 2–4 | 16–32 |
| `train.num_workers` | 0 | 4–8 |
| `train.amp` | no-op | true |
| env var | `PYTORCH_ENABLE_MPS_FALLBACK=1` | — |

## Notes
- Step 1 (ingest) is CPU/geo — runs on Mac, no GPU needed.
- `data/` + `runs/` live at the repo root (shared) — relative paths in configs resolve when run from root.
