"""Shared YAML config loader with `extends:` inheritance (deep-merge).

A config may start with `extends: <sibling.yaml>` to inherit a base config and
override only a few keys (e.g. config_gpu.yaml extends config.yaml). Used by
phase1 (train/eda/ingest) and phase2 (graph)."""
from __future__ import annotations

from pathlib import Path


def deep_merge(base: dict, over: dict) -> dict:
    """Recursively merge `over` onto `base` (override wins; dicts merged)."""
    out = dict(base)
    for k, v in over.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str) -> dict:
    """Load YAML, resolving an `extends:` base (relative to this file's dir).

    `extends` is resolved RECURSIVELY, so multi-level chains work: e.g.
    config_gpu.yaml extends config.yaml (and a child could extend config_gpu.yaml in
    turn) — each level deep-merges over the fully-resolved one below it (nearest
    override wins)."""
    import yaml
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("extends", None)
    if base:
        base_cfg = load_config(str(Path(path).parent / base))   # resolve base's own extends too
        cfg = deep_merge(base_cfg, cfg)
    return cfg
