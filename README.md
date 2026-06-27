# rr — Route Resilience (Phase I)

Occlusion-robust **road extraction** from Resourcesat-2/2A **LISS-IV** (5.8 m,
G/R/NIR) under tree canopy. Phase I = perception (segmentation). Phases II–III
(graph healing + criticality/resilience) follow.

Built for **InGARSS 2026 + ISRO hackathon**. Engineering rules: config-driven,
device-dynamic (MPS/CUDA/CPU), graceful degradation.

## Setup (micromamba, from VS Code CLI)
```bash
micromamba create -f environment.yml -y
micromamba activate rr
# macOS: allow CPU fallback for ops not yet on Metal
export PYTORCH_ENABLE_MPS_FALLBACK=1
```
In VS Code: **Python: Select Interpreter** → the `rr` env (`.vscode/settings.json`
points at `~/micromamba/envs/rr/bin/python` — adjust if your micromamba root differs).

## Run
```bash
# dep-free smoke test (no smp/geo stack needed): set model.arch=miniunet in config
python -m src.train --config config/config.yaml

# guaranteed baseline (needs full env): model.arch=smp (default)
python -m src.train --config config/config.yaml
```
Artifacts → `runs/train/<timestamp>/` : `best.pt`, `metrics.csv`,
`figures/{loss_curve,prediction}.{pdf,png}`.

## Models (`cfg.model.arch`)
| arch | what | when |
|---|---|---|
| `miniunet` | dep-free U-Net (+ optional `center: dblock`) | smoke/CI |
| `smp` | UNet++/Linknet, stem-inflated 5-ch | **paper baseline** |
| `dinov3` | DINOv3 SAT-493M (timm, non-gated) + aux | hero (stretch) |
| `clay` | Clay v1.5 (G/R/NIR native) | stretch (stub) |

## Layout
```
config/config.yaml   # single source of truth (search "USER INPUT")
environment.yml      # micromamba env
src/
  data/      synthetic, dataset, augment, indices, sources/ (bhoonidhi,planetary,osm)
  preprocess/ degrade, coregister, pipeline (real LISS-IV -> .npz tiles)
  models/    factory (miniunet | smp | dinov3 | clay; Dblock; stem inflation)
  losses/    BCE + Dice + clDice (+ canopy-weighted)
  metrics/   IoU, Dice, Occlusion-Recall (global), relaxed IoU/F1
  canopy/    OCOI (Treepedia sampling + per-segment occlusion index)
  viz/       publication plots + prediction panel
  train.py   entrypoint
METHODOLOGY.md       # step-by-step pipeline + data-flow graph
REFERENCES.md        # peer-reviewed citations
```

## Real-data path (next milestone)
1. Fill `cfg.preprocess` (AOI bbox, dates) + `.env` (Bhoonidhi creds).
2. `python -m src.preprocess.pipeline --config config/config.yaml --dry-run`.
3. Set `cfg.data.source: tiles`, `cfg.data.root`, and `cfg.data.norm.{mean,std}`.

See `METHODOLOGY.md` for the full pipeline and experiment plan.
