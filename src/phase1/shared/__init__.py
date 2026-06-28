"""Phase 1 shared core — arm-agnostic infrastructure used by BOTH arms.

  data/        datasets, augmentation, spectral indices, DeepGlobe, OSM/source adapters
  models/      model factory (smp / miniunet / dinov3 / clay; GROVE backbone in Stage 2)
  losses/      CombinedRoadLoss (BCE + Dice + clDice, canopy-weighted)
  metrics/     IoU / Dice / Occlusion-Recall (relaxed, buffered)
  preprocess/  LISS-IV ingest, degrade, coregister, pipeline
  eda/         exploratory data analysis
  canopy/      OCOI (per-segment occlusion index; analysis bridge, currently unused)

Kept here (not inside vista/ or grove/) so neither arm owns the other's code; both
import from `..shared`. See METHODOLOGY "Two arms: VISTA & GROVE".
"""
