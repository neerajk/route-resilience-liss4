"""Dynamic, arm-aware artifact naming — the single source of truth.

WHY THIS MODULE
---------------
Phase 1 has TWO road-extraction "arms" that share the same infrastructure:

  - **VISTA** (`arm.name: vista`) — VIsible-Surface road segmenTAtion: the smp
    UNet++/SegFormer baseline that segments the road surface the sensor can see.
  - **GROVE** (`arm.name: grove`) — the occlusion-completion arm: topology/
    continuity-aware recovery of roads hidden under tree canopy (the *grove*).

Both write run dirs, checkpoints and Phase-2 hand-off masks. If those names are
hardcoded, the two arms silently overwrite each other's `pred_mask.tif`. So every
artifact name is *derived from config here*, never hardcoded at the call site.
Change `arm.name` in the config and all outputs re-label consistently.

NAMING SCHEME (stable, greppable, glob-friendly)
------------------------------------------------
  run dir      runs/<kind>/<arm>__<model_tag>__<stage>__<stamp>
               e.g. runs/train/vista__unetplusplus-resnet34__liss4__20260628_140000
                    runs/train/grove__ha_roadformer__liss4__20260628_150000
  pred mask    <out_dir>/<arm>__pred_mask.tif          (probability; Phase-2 input)
               <out_dir>/<arm>__pred_mask_bin.tif      (binary, optional)
  orientation  <out_dir>/<arm>__orientation.tif        (GROVE only; 2-band sin2θ/cos2θ)

The `__` (double underscore) separator makes the arm visually obvious and lets
Phase 2 glob `*__pred_mask.tif`. `arm.name` is lower-kebab-cased defensively.

BACKWARD COMPATIBILITY
----------------------
A config WITHOUT an `arm:` block resolves to ``default`` (caller passes "vista"),
so the pre-existing single-arm pipeline keeps working unchanged.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict

# default arm when a config predates the `arm:` block (keeps old runs working)
DEFAULT_ARM = "vista"


def _kebab(s: str) -> str:
    """Lower-case, collapse non-alnum to single hyphens (safe for paths/globs)."""
    return re.sub(r"[^a-z0-9]+", "-", str(s).strip().lower()).strip("-") or DEFAULT_ARM


def arm_name(cfg: Dict[str, Any], default: str = DEFAULT_ARM) -> str:
    """Resolve the arm identity from ``cfg.arm.name`` (kebab-cased)."""
    return _kebab((cfg.get("arm") or {}).get("name", default))


def model_tag(cfg: Dict[str, Any]) -> str:
    """Architecture tag: smp -> ``<decoder>-<encoder>``; else the arch/backbone name.

    Mirrors the historical train.py tag so checkpoints stay self-documenting; for
    GROVE (Stage 2) the backbone name (e.g. ``ha_roadformer``) is used directly.
    """
    m = cfg.get("model", {}) or {}
    arch = str(m.get("arch", "miniunet")).lower()
    if arch == "smp":
        return f"{m.get('decoder', 'unet')}-{m.get('encoder', 'enc')}"
    if arch == "grove":                       # Stage 2: GROVE backbone family
        return str((cfg.get("grove") or {}).get("backbone", "ha_roadformer"))
    if arch == "dinov3":
        return "dinov3"
    return arch


def stage_tag(cfg: Dict[str, Any]) -> str:
    """Map ``cfg.data.source`` to a human role: pretrain | liss4 | synth | <src>."""
    src = cfg.get("data", {}).get("source", "data")
    return {"deepglobe": "pretrain", "tiles": "liss4", "synthetic": "synth"}.get(src, str(src))


def run_name(cfg: Dict[str, Any], stamp: str, kind: str = "train") -> str:
    """``<arm>__<model_tag>__<stage>__<stamp>`` — the self-documenting run-dir name."""
    return f"{arm_name(cfg)}__{model_tag(cfg)}__{stage_tag(cfg)}__{stamp}"


def run_dir(cfg: Dict[str, Any], stamp: str, kind: str = "train") -> Path:
    """``runs/<kind>/<run_name>`` Path (parent of best.pt, metrics, figures)."""
    return Path(cfg.get("paths", {}).get("runs", "runs")) / kind / run_name(cfg, stamp, kind)


def pred_mask_path(cfg: Dict[str, Any], out_dir: str = "data", binary: bool = False) -> Path:
    """Phase-1→2 contract mask path: ``<out_dir>/<arm>__pred_mask[_bin].tif``."""
    suffix = "_bin" if binary else ""
    return Path(out_dir) / f"{arm_name(cfg)}__pred_mask{suffix}.tif"


def orientation_path(cfg: Dict[str, Any], out_dir: str = "data") -> Path:
    """GROVE orientation raster (2-band sin2θ/cos2θ): ``<out_dir>/<arm>__orientation.tif``."""
    return Path(out_dir) / f"{arm_name(cfg)}__orientation.tif"
