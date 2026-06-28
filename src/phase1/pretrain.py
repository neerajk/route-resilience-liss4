"""Phase-I Stage-A pretraining on DeepGlobe (0.5 m → 5.8 m) — thin entrypoint.

DeepGlobe pretraining IS ordinary training on the ``deepglobe`` data source: this
wrapper just runs ``train.run`` with ``config/phase1/pretrain.yaml`` so the whole
machinery (loss, LR schedule, early-stop, metrics, checkpointing) is reused. The
resulting ``runs/train/<ts>/best.pt`` warm-starts the LISS-IV model via
``train.init_from`` (3-ch RGB → [G,R,NIR,NDVI] stem inflation).

RUN:
    python -m src.phase1.pretrain --config config/phase1/pretrain.yaml
Then set in config/phase1/config.yaml:
    train:
      init_from: runs/train/<ts>/best.pt
"""
from __future__ import annotations

import argparse

from ..common.config import load_config
from .train import run


def main() -> None:
    ap = argparse.ArgumentParser(description="DeepGlobe pretraining (0.5 m -> 5.8 m)")
    ap.add_argument("--config", default="config/phase1/pretrain.yaml")
    args = ap.parse_args()
    out = run(load_config(args.config))
    print("\n[pretrain] DONE. Warm-start the LISS-IV model by setting in "
          "config/phase1/config.yaml:")
    print(f"           train.init_from: {out / 'best.pt'}")
    print("           (init_inflate_stem: true maps the 3-ch RGB stem -> [G,R,NIR,NDVI])")


if __name__ == "__main__":
    main()
