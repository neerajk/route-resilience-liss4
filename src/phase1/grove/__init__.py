"""GROVE — the occlusion-completion arm (Arm B).

WHAT IT IS
----------
VISTA (Arm A, the smp UNet++/SegFormer baseline) segments the road surface the
sensor can SEE. GROVE is the second arm: it recovers roads HIDDEN under tree
canopy — the *grove* — by exploiting that roads are linear, continuous features.
Where VISTA stops at the canopy edge, GROVE infers the road's continuation.

WHY A SECOND ARM (research gap)
-------------------------------
Under canopy occlusion the road's spectral signature is gone from the optical
LISS-IV bands (the model sees foliage, not asphalt). A per-pixel segmenter has no
notion that an occluded pixel "should continue the line", so it breaks the road
into fragments. Zhang et al. (2022, HA-RoadFormer) name this exact gap in their
conclusion: *"Roads are continuous in space ... how to fuse this topology
information into neural networks [is] a very promising job."* GROVE is that fusion.

THE MECHANISM (how the sin/cos idea enters, at two distinct points)
-------------------------------------------------------------------
  1. INPUT: sinusoidal positional encoding on transformer tokens — gives long-range
     attention the spatial reasoning to connect collinear road pixels across a gap.
  2. TARGET: a per-pixel road-orientation field encoded sin(2θ)/cos(2θ) (axial,
     mod-180°; Batra et al. 2019) — the continuity *carrier* that propagates road
     direction THROUGH the canopy gap.
Plus a topology-preserving loss (clDice; Shit et al. 2021) and supervision focused
on under-canopy road pixels (OSM road ∩ NDVI canopy).

BUILD STAGES (see METHODOLOGY "Two arms: VISTA & GROVE")
--------------------------------------------------------
  Stage 0  scaffold + arm-aware naming + Phase-2 mask contract      <- this package
  Stage 1  supervision: under-canopy-road mask + orientation GT     <- supervision.py
  Stage 2  backbone: HA-RoadFormer (overlapping multi-scale patch +
           hybrid attention) + sinusoidal positional encoding       (next)
  Stage 3  orientation head (sin2θ/cos2θ)
  Stage 4  topology loss (clDice) + under-canopy focal weighting
  Stage 5  strip-conv + connectivity attention (CoANet) — optional
  Stage 6  long-gap graph bridge (reuse Phase-2 heal OR Sat2Graph/RNGDet)
  Stage 7  integration + ablation (VISTA vs GROVE on Occlusion-Recall)

Modules
-------
  supervision.py        per-tile under-canopy-road mask + sin2θ/cos2θ orientation GT
  build_supervision.py  CLI: augment existing data/tiles/*.npz with the GROVE targets

References (see REFERENCES.md): Zhang et al. (2022) HA-RoadFormer; Batra et al.
(2019) joint orientation+segmentation; Mei et al. (2021) CoANet; Shit et al.
(2021) clDice; Dong et al. (2022) CSWin.
"""
