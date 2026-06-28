"""VISTA — VIsible-Surface road segmenTAtion (Arm A).

The baseline perception arm: an smp UNet++/SegFormer that segments the road
surface visible in the LISS-IV optical bands. Entrypoints (config-driven; the arm
is selected by `cfg.arm.name`, artifact names by src/common/naming.py):

  train.py     python -m src.phase1.vista.train    --config config/phase1/config.yaml
  pretrain.py  python -m src.phase1.vista.pretrain  --config config/phase1/pretrain.yaml
  predict.py   python -m src.phase1.vista.predict   --ckpt <best.pt>   (-> data/vista__pred_mask.tif)

Shared infrastructure (data/models/losses/metrics/preprocess) lives in
src/phase1/shared/ and is imported by BOTH arms. GROVE (Arm B) is src/phase1/grove/.
"""
