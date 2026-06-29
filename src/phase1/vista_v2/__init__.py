"""VISTA-v2 — ResNet-101 + UNet++ with a pluggable positional encoding.

A single PE-pluggable model (one class, PE chosen by `cfg.model.pe.type`) so every
variant runs with the SAME command, only the config differs:

  botnet  (default) : relative PE in an attention BOTTLENECK   (Srinivas 2021)
  rope              : 2-D RoPE in the attention bottleneck      (Su 2021)
  sincos            : sinusoidal PE concatenated at the INPUT   (Vaswani 2017)
  nope              : no PE  (control row for the benchmark)

Input stack = [Green, Red, NGRDI] (NIR-free, domain-invariant across DeepGlobe and
LISS-IV). Returns plain road logits → trains/predicts via the existing VISTA pipeline.
See docs/vista_v2.md for a from-scratch explanation.
"""
