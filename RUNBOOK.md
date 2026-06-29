# rr — Runbook

Run everything from the project root `~/Desktop/Projects/rr`. Layout: `src/common`
(shared) · `src/phase1` (perception) · `src/phase2` (graph) · `src/phase3`
(resilience) · `src/phase4` (dashboard); configs in `config/phase{1,2,3,4}/`.
Stack: smp **ResNet baseline → SegFormer advanced**, **OSM labels**, **DeepGlobe
pretrain**. Cross-platform: macOS (MPS) · Windows (CPU) · NVIDIA (CUDA).

> **Note:** the user runs all jobs themselves. Every block below is copy-paste; nothing
> here is auto-executed.

---

## 0. Create the env (once)
```bash
cd ~/Desktop/Projects/rr
micromamba create -f environment.yml -y
micromamba activate rr
```
VS Code: `Cmd/Ctrl+Shift+P → Python: Select Interpreter → rr`.

- **macOS (Apple Silicon):** ships a PyTorch MPS build. Prefix runs with
  `PYTORCH_ENABLE_MPS_FALLBACK=1`.
- **Windows (CPU dev):** the same file works (`conda`/`mamba` are equivalent to
  `micromamba`); PyTorch resolves to a CPU build.
- **NVIDIA GPU (Linux/Windows):** create the env, then swap in a CUDA PyTorch build:
  ```bash
  micromamba run -n rr pip install --force-reinstall torch torchvision \
    --index-url https://download.pytorch.org/whl/cu128   # cu128 for RTX-50/Blackwell; cu124/cu121 otherwise
  ```

---

## 1. Smoke test (no data)
Confirms the install + training loop on a synthetic fixture.
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python -m src.phase1.vista.train --config config/phase1/smoke.yaml
```

---

## 2. Step 1 — ingest (OSM labels → tiles)
Inputs wired in `config/phase1/config.yaml → data.liss4`:
`data/raw/liss4/B2,B3,B4.tif` + `data/raw/aoi/blore_urban.shp`.
```bash
# quick test: set  data.liss4.max_tiles: 6  (or a small stride) first, then:
python -m src.phase1.shared.preprocess.ingest_liss4 --config config/phase1/config.yaml
# full run: remove the cap
```
→ `data/tiles/*.npz` (bands, ndvi, canopy, **mask**) + `data/band_statistics.csv`
(prints the `data.norm` mean/std).

> **GROVE (Arm B) Stage 1** — after ingest, add the occlusion-completion targets
> (`under_canopy_road`, `orient`) onto the same tiles (idempotent, no re-ingest):
> ```bash
> python -m src.phase1.grove.build_supervision --config config/phase1/grove.yaml
> ```

## 3. Wire tiles for training — edit `config/phase1/config.yaml`
```yaml
data:
  source: tiles                 # switch off synthetic
  root: data/tiles
  norm:                         # paste the per-channel [G,R,NIR,NDVI] numbers from Step 2
    mean: [106.059, 95.983, 189.757, 0.322]
    std:  [21.483, 27.295, 40.894, 0.143]
```
> **Why norm matters:** raw 10-bit DN fed into an ImageNet encoder collapses training.
> Always paste real `mean`/`std` before a real run.

## 4. EDA (optional)
```bash
python -m src.phase1.shared.eda.run_eda --config config/phase1/config.yaml
```

---

## 5. Choosing the model — config changes per architecture

**All model selection happens in `config/phase1/config.yaml → model`.** Pick a row,
set those keys, then run the Step 7 train command. `in_channels: 4` and
`stem_init: inflate` apply to every smp model (4-ch `[G,R,NIR,NDVI]` input).

| Model | `model.arch` | `model.decoder` | `model.encoder` | extra | batch (MPS / CUDA) |
|---|---|---|---|---|---|
| **MiniUNet** (dev/CI, dep-free) | `miniunet` | — | — | `center: none` or `dblock` | 4 / 32 |
| **Baseline** (smp UNet++ / ResNet34) | `smp` | `unetplusplus` | `resnet34` | `encoder_weights: imagenet` | 4 / 24 |
| **Advanced** (SegFormer / MiT-B2) | `smp` | `segformer` | `mit_b2` | attention bridges gaps | 2 / 12 |
| **DINOv3** (optional stretch) | `dinov3` | — | — | see `model.dinov3` block | 2 / 8 |

Baseline (default in the repo):
```yaml
model:
  arch: smp
  in_channels: 4
  decoder: unetplusplus
  encoder: resnet34
  encoder_weights: imagenet
  stem_init: inflate
```
Advanced — switch to the transformer (SegFormer's long-range attention is the
"see-through-occlusion" mechanism):
```yaml
model:
  arch: smp
  in_channels: 4
  decoder: segformer        # all-MLP SegFormer head (needs smp>=0.4)
  encoder: mit_b2           # MiT-B2 transformer backbone
  encoder_weights: imagenet
  stem_init: inflate
# also lower train.batch_size (mit_b2 uses ~2x memory): 2 on MPS, ~12 on CUDA.
```

### Other Phase-1 knobs (same `config.yaml`)
| Goal | Key | Note |
|---|---|---|
| LR schedule | `train.scheduler.name: cosine` | `warmup_epochs`, `min_lr`; or `plateau` / `none` |
| Early stop | `train.early_stop.enabled: true` | `patience`, `min_delta` (monitors `eval.monitor`) |
| Unbiased CV | `data.cv.scheme: spatial_block` | `block_size_m` (~1500 m); `random` leaks in one AOI |
| Push Occlusion-Recall | `loss.canopy_weight: 1.0–3.0` | up-weights BCE on occluded road pixels |
| Steadier val metric | `eval.tta: true` | D4-flip test-time aug (~4× val cost) |
| Augmentation | `augment.enabled: true` | occlusion / coarse-dropout / scale / radiometric / copy-paste |
| Checkpoint metric | `eval.monitor: occlusion_recall` | the headline number; `relaxed_f1` for no-canopy data |

---

## 6. Stage A — DeepGlobe pretraining (optional, recommended)
Closes the resolution gap: trains a 3-ch RGB road model on DeepGlobe **degraded
0.5 m → 5.8 m**, then warm-starts the LISS-IV model (stem inflation maps RGB →
`[G,R,NIR,NDVI]`).

1. Put the DeepGlobe set at `data/raw/deepglobe/` (layout `<id>_sat.jpg` +
   `<id>_mask.png`; ~6 GB, not shipped).
2. Confirm `config/phase1/pretrain.yaml → data.deepglobe.root` and the encoder —
   it ships as **SegFormer / mit_b2**. **Keep the pretrain encoder == your fine-tune
   encoder** so the warm-start weights transfer cleanly.
3. Run:
   ```bash
   python -m src.phase1.vista.pretrain --config config/phase1/pretrain.yaml
   ```
   → `runs/train/<ts>/best.pt`.
4. Warm-start the LISS-IV model — edit `config/phase1/config.yaml`:
   ```yaml
   train:
     init_from: runs/train/<ts>/best.pt   # the pretrain checkpoint
     init_inflate_stem: true              # 3-ch RGB stem -> 4-ch [G,R,NIR,NDVI]
   ```

---

## 7. Train the LISS-IV model
After picking the model (Step 5) and (optionally) warm-start (Step 6):

**macOS (MPS, dev):**
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python -m src.phase1.vista.train --config config/phase1/config.yaml
```
**NVIDIA GPU (CUDA, real training)** — `config_gpu.yaml` extends `config.yaml` and
overrides only the machine knobs (bigger batch/workers/epochs + AMP):
```bash
python -m src.phase1.vista.train --config config/phase1/config_gpu.yaml
```
> To train the **advanced SegFormer on GPU**: uncomment the `model:` block at the
> bottom of `config_gpu.yaml` AND set `train.batch_size: 12` (OOM-safe). Otherwise it
> trains the ResNet34 baseline inherited from `config.yaml`.

Headline metric = **Occlusion-Recall**; artifacts in `runs/train/<ts>/`
{`best.pt`, `metrics.csv`, `loss_curve`, prediction panel}.

---

## 8. Export pred_mask.tif (Phase 1 → 2 contract)
```bash
python -m src.phase1.vista.predict --ckpt runs/train/<ts>/best.pt
# default out = data/<arm>__pred_mask.tif (e.g. data/vista__pred_mask.tif); --out to override
# --binary for 0/1 instead of probability
```
Writes a georeferenced full-scene road GeoTIFF (model + norm read from the checkpoint).
Point `config/phase2/config_phase2.yaml → graph.mask` at this file (e.g. `data/vista__pred_mask.tif`).

---

## 8b. GROVE (Arm B) — occlusion-completion arm  [Stages 0–5 built]
Same tiles as VISTA; adds occlusion-completion. **Smoke-test first** (CPU, synthetic):
```bash
pip install einops      # cswin/haroadformer need it
python -m src.phase1.grove.train --config config/phase1/grove_smoke.yaml
# swap grove.backbone: haroadformer -> cswin -> vista_mit to smoke each
```
**Stage 1 — supervision** (under-canopy + sin/cos orientation onto existing tiles):
```bash
python -m src.phase1.grove.build_supervision --config config/phase1/grove.yaml
```
**Backbone benchmark** (seg-only → full VISTA pipeline w/ CV/TTA):
```bash
python -m src.phase1.vista.train --config config/phase1/grove_vista_mit.yaml
python -m src.phase1.vista.train --config config/phase1/grove_cswin.yaml
python -m src.phase1.vista.train --config config/phase1/grove_haroadformer.yaml
```
**Full GROVE** (seg + orientation + under-canopy focal):
```bash
python -m src.phase1.grove.train --config config/phase1/grove.yaml
```
**Export + compare:**
```bash
python -m src.phase1.grove.predict --ckpt runs/train/grove__<bb>__<ts>/best.pt
#   -> data/grove__pred_mask.tif (+ data/grove__orientation.tif)
python -m src.phase1.grove.bench --runs "runs/train/*liss4*" --out runs/bench
```
> Caveats: CSWin/HA-RoadFormer have no public weights (train from scratch); if `vista_mit`
> errors on smp's MiT feature shape, use `backbone: vista_resnet`; run GROVE multi-task with
> geometric augmentation OFF (orientation-aware aug not yet wired). Stage 6 = Phase 2
> `heal.mode: orientation` consuming `data/grove__orientation.tif`.

---

## 8b. VISTA-v2 — ResNet-101 + UNet++ + pluggable PE  (see `docs/vista_v2.md`)
NIR-free `[G,R,NGRDI]`; benchmarks 3 PEs + control. Needs `einops` (+ `scipy`/`matplotlib`
for bench/plots, already in env). Same command for every variant — only the config changes.
```bash
pip install einops
# (optional) shared DeepGlobe pretrain -> warm-start; then set train.init_from in each config:
python -m src.phase1.vista.pretrain --config config/phase1/vista_v2_pretrain.yaml
# train the 4 variants (run each across your 7 spatial-block folds):
python -m src.phase1.vista.train --config config/phase1/vista_v2_botnet.yaml   # relative PE (default)
python -m src.phase1.vista.train --config config/phase1/vista_v2_rope.yaml     # 2-D RoPE
python -m src.phase1.vista.train --config config/phase1/vista_v2_sincos.yaml   # sinusoidal at input
python -m src.phase1.vista.train --config config/phase1/vista_v2_nope.yaml     # control
# benchmark + publication plots:
python -m src.phase1.vista_v2.bench --runs "runs/train/*vista_v2-*" --out runs/vista_v2_bench
python -m src.phase1.vista_v2.plots --runs "runs/train/*vista_v2-*" --out runs/vista_v2_bench/figures
```
Run dirs: `runs/train/vista__vista_v2-<pe>__liss4__<ts>/`. `bench` → mean±95%CI + paired Wilcoxon
(Holm) + Cohen's d; `plots` → bars/CI, per-fold lines, training curves (PNG+PDF).
> Caveats: NIR dropped (canopy mask reuses the precomputed tile layer); `sincos` patches the
> encoder stem for +8 channels; verify DeepGlobe routes `ngrdi` (see pretrain config note).

---

## 9. Phase 2 — graph (config-driven; clean CLI)
```bash
python -m src.phase2.graph.run_graph --config config/phase2/config_phase2.yaml
```
All options live in `config/phase2/config_phase2.yaml → graph`:
- **input** — `mask:` a GeoTIFF (e.g. `data/pred_mask.tif`, or a saved `osm_mask.tif`),
  or `null` to auto-build an OSM mask.
- **mode** — `tiling.enabled: true` (default) processes the **whole scene in blocks**
  (the global heal stitches the seams); set `false` + `window: [row,col,h,w]` for one region.
- **de-noise a model mask** — raise `min_object_size` / `threshold` (an under-trained
  over-predicting mask needs higher values, or use the OSM mask for a clean demo).

→ `runs/graph/<ts>/`: `graph.graphml` (Phase 3 input), `roads.geojson` (QGIS),
`metrics.csv` (Connectivity Ratio), and a healing overlay (window/single mode).

---

## 10. Phase 3 — resilience (config-driven; clean CLI)
Point the config at a Phase 2 graph, then run:
```bash
# config/phase3/config_phase3.yaml -> resilience.graph: runs/graph/<ts>/graph.graphml
python -m src.phase3.resilience.run_resilience --config config/phase3/config_phase3.yaml
```
Options in `config/phase3/config_phase3.yaml → resilience`: `weight` (travel_time_s),
`betweenness_k`, `efficiency_samples`, `ablation.{strategies,max_fraction,steps}`.
Big/slow graph → lower `efficiency_samples` / `betweenness_k`. CPU only.

→ `runs/resilience/<ts>/`: `criticality.geojson` (betweenness heatmap), `gatekeepers.csv`,
`resilience_curves.csv` + `figures/resilience_curves`, `resilience_summary.csv` (Resilience Index).

---

## 11. Phase 4 — dashboard (Streamlit + Leaflet)

**One-time: install Phase 4 deps** (if the env predates Phase 4):
```bash
micromamba activate rr
pip install "streamlit>=1.32" "folium>=0.17" "streamlit-folium>=0.22" "plotly>=5.20"
```

**Run** (always from project root):
```bash
streamlit run src/phase4/dashboard.py
```
Opens at `http://localhost:8501`. The sidebar auto-discovers every Phase 3 run in
`runs/resilience/`.

**Enable the Flood Simulator** — edit `config/phase4/config_phase4.yaml`:
```yaml
dashboard:
  graph_path: runs/graph/<ts>/graph.graphml   # the Phase 2 graph to stress-test
  max_map_nodes: 5000                          # cap on Leaflet nodes (top-N by betweenness)
```

→ Four tabs: **Criticality Map** (Leaflet, nodes coloured by betweenness),
**Resilience Curves** (interactive Plotly ablation chart + Resilience Index table),
**Gatekeepers** (sortable table of top junctions),
**Flood Simulator** (pick nodes → remove → see fragmentation live).

---

## Platform & batch-size at a glance
| knob | macOS (MPS, dev) | Windows (CPU, dev) | NVIDIA (CUDA, train) |
|---|---|---|---|
| Phase-1 config | `config/phase1/config.yaml` | `config/phase1/config.yaml` | `config/phase1/config_gpu.yaml` |
| `train.batch_size` (ResNet34) | 2–4 | 1–2 | 16–32 |
| `train.batch_size` (mit_b2) | 2 | 1 | ~12 |
| `train.num_workers` | 0 | 0 | 4–8 |
| `train.amp` | no-op | no-op | true |
| env var | `PYTORCH_ENABLE_MPS_FALLBACK=1` | — | — |

## Notes
- Step 1 (ingest), Phase 2, Phase 3, Phase 4 are CPU/geo — no GPU needed; only Phase 1
  training benefits from CUDA.
- `data/` + `runs/` live at the repo root (shared) — relative paths in configs resolve
  when you run from the root.
- Keep the **pretrain encoder == fine-tune encoder** so warm-start weights transfer.
