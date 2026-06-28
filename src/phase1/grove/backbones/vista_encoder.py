"""VISTA-encoder backbone — wraps an smp encoder as a GROVE backbone.

This is the control arm of the backbone benchmark: it puts GROVE's heads + loss +
supervision on the SAME encoder VISTA uses (MiT-B2 / ResNet34), so any GROVE gain
over VISTA on this backbone is attributable to the heads/loss, not the encoder.
It also inherits ImageNet weights for free (the transformer backbones don't), and
can be DeepGlobe-warm-started via the usual train.init_from.
"""
from __future__ import annotations

import torch.nn as nn


class VistaEncoderBackbone(nn.Module):
    """smp encoder → feature pyramid (fine→coarse), with channels/reductions exposed."""

    def __init__(self, in_channels: int = 4, encoder_name: str = "mit_b2",
                 weights: str | None = "imagenet", depth: int = 5) -> None:
        super().__init__()
        try:
            from segmentation_models_pytorch.encoders import get_encoder
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "grove.backbone=vista_* needs segmentation_models_pytorch "
                "(same dep as VISTA arch=smp)."
            ) from e
        self.encoder = get_encoder(encoder_name, in_channels=in_channels,
                                   depth=depth, weights=weights)
        # smp encoders return depth+1 maps at strides [1,2,4,8,16,32]; drop the
        # full-res identity map (index 0) — the decoder upsamples from stride 4.
        chans = list(self.encoder.out_channels)
        reds = [1, 2, 4, 8, 16, 32][: len(chans)]
        self._keep = [i for i, r in enumerate(reds) if r >= 4]   # stride>=4 stages
        self.feature_channels = [chans[i] for i in self._keep]
        self.feature_reductions = [reds[i] for i in self._keep]

    def forward(self, x):
        feats = self.encoder(x)                 # list, stride [1,2,4,8,16,32]
        return [feats[i] for i in self._keep]   # fine→coarse from stride 4
