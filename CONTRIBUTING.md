# Contributing — rr (two-person workflow)

Goal: **Phase I (model) and Phase II (graph) progress in parallel**, connected by
ONE interface. Friend improves the model; you build the graph; neither blocks the other.

## 1. Branches & PRs
- `main` is always runnable. **No direct pushes** — change via Pull Request.
- One branch per person:
  - Phase I (model): `phase1/baseline` (or `feature/<thing>`)
  - Phase II (graph): `phase2/graph`
- Small, frequent PRs → quick review → merge. Each finished item = one PR.

## 2. THE CONTRACT — `pred_mask.tif` (do not break without agreement)
Phase II consumes exactly one artifact; Phase I produces it:

| Field | Spec |
|---|---|
| File | `pred_mask.tif` (GeoTIFF) |
| Shape | single band `[H, W]` |
| Values | road **probability** in `[0,1]` (or binary 0/1) |
| Georef | valid **CRS + affine transform** (e.g. EPSG:32643) |
| Producer | Phase 1 `src/phase1/predict.py` (stitched full-scene inference) |
| Consumer | Phase 2 `src/phase2/graph/io.read_mask()` |

For development, Phase II may use the **OSM-rasterized mask** (same format) until a
real `pred_mask.tif` exists. As long as the format above holds, the model can change
freely (ResNet → SegFormer → DINOv3, pretraining, losses) with **zero Phase-II rework**.

## 3. Code ownership (keep files disjoint → no merge conflicts)
- **Phase 1 (friend):** `src/phase1/` (models, losses, metrics, train.py, preprocess,
  data, eda) + `config/phase1/`.
- **Phase 2 (you):** `src/phase2/graph/` + `config/phase2/`.
- **Shared (coordinate before editing):** `src/common/`, `environment.yml`,
  `README.md`, `METHODOLOGY.md`.

## 4. Config — keep per-phase configs separate
- Friend: `config/phase1/config.yaml` (+ `config_gpu.yaml`, which `extends:` it).
- You: `config/phase2/config_phase2.yaml` (standalone — Phase 2 only needs the mask).

## 5. Data is OUT-OF-BAND (not in git)
`.gitignore` excludes `data/`, `runs/`, `*.npz`, `*.tif`. So:
- Each person keeps imagery/tiles locally.
- You develop Phase II on your **OSM masks**.
- For integration, friend sends you one `pred_mask.tif` (drive/scp); you point
  `--mask` at it. No data ever goes through git.

## 6. Definition of done
A PR is mergeable when: it runs from a clean `rr` env, doesn't touch the other
person's files, and (for Phase I) still emits a valid `pred_mask.tif`.
