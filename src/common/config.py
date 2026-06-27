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
    """Load YAML, resolving a `extends:` base (relative to this file's dir)."""
    import yaml
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("extends", None)
    if base:
        with open(Path(path).parent / base) as bf:
            cfg = deep_merge(yaml.safe_load(bf) or {}, cfg)
    return cfg
