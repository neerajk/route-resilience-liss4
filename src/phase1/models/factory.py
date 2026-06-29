"""Model factory — resolves an architecture from config.

FOUR TIERS, selected by cfg.model.arch:

  "miniunet"  -> small pure-PyTorch U-Net. ZERO extra deps; runs on M1 with stock
                 torch TODAY. Optional D-LinkNet `center: dblock`. Smoke/CI only.

  "smp"       -> segmentation_models_pytorch (UNet++/DeepLabV3+/Linknet). The
                 GUARANTEED paper BASELINE. For the 5-channel G/R/NIR/NDVI/CHM
                 stack, `stem_init: inflate` copies pretrained RGB conv1 weights
                 onto the new channels (Carreira & Zisserman 2017, I3D inflation)
                 instead of smp's default random re-init.
                 (Ronneberger 2015 U-Net; Zhou 2018 UNet++; Chen 2018 DeepLabV3+;
                  Zhou 2018 D-LinkNet, CVPRW.)

  "dinov3"    -> HERO (stretch): frozen DINOv3-ViT SAT-493M backbone via timm
                 (NON-gated `vit_large_patch16_dinov3.sat493m`) + NDVI/CHM aux
                 branch + light decoder. SAT normalization (NOT ImageNet) set in
                 config (resolves DINOv3 GitHub issue #61).
                 (Oquab/Siméoni et al. 2025, DINOv3, arXiv:2508.10104.)

  "clay"      -> STRETCH: Clay v1.5 (GSD/wavelength-aware, ingests G/R/NIR
                 natively). Guarded stub — wire the encoder load when used.

All models map [B, in_channels, H, W] -> logits [B, 1, H, W].
"""
from __future__ import annotations

import warnings
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Building blocks                                                              #
# --------------------------------------------------------------------------- #
class _DoubleConv(nn.Module):
    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class DBlock(nn.Module):
    """D-LinkNet center block (Zhou et al., 2018, CVPRW).

    Four cascaded dilated 3x3 convs (dilation 1,2,4,8) summed residually with the
    input. Multiplies the receptive field WITHOUT losing resolution, so the model
    can "see across" tree-canopy gaps to infer the hidden road continuation —
    directly targeting Occlusion-Recall. Borrowed pattern (MIT) from
    zlckanata/DeepGlobe-Road-Extraction-Challenge networks/dinknet.py.
    """

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.d1 = nn.Conv2d(ch, ch, 3, padding=1, dilation=1)
        self.d2 = nn.Conv2d(ch, ch, 3, padding=2, dilation=2)
        self.d4 = nn.Conv2d(ch, ch, 3, padding=4, dilation=4)
        self.d8 = nn.Conv2d(ch, ch, 3, padding=8, dilation=8)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.relu(self.d1(x))
        x2 = self.relu(self.d2(x1))
        x3 = self.relu(self.d4(x2))
        x4 = self.relu(self.d8(x3))
        return x + x1 + x2 + x3 + x4


def _resolve_parent(model: nn.Module, dotted: str):
    """Return (parent_module, attr_name) for a dotted module path."""
    parts = dotted.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def inflate_first_conv(model: nn.Module, in_channels: int) -> bool:
    """Inflate the first 3-channel Conv2d to `in_channels` (I3D trick).

    Copies pretrained RGB conv1 weights onto the first 3 input channels
    (G,R,NIR), mean-inits the extra channels (NDVI,CHM), and rescales by
    3/in_channels to preserve the summed response magnitude (Carreira &
    Zisserman, 2017, CVPR). Returns True if a stem was found and replaced.
    """
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d) and mod.in_channels == 3:
            w = mod.weight.data                          # [out, 3, kh, kw]
            out_c, _, kh, kw = w.shape
            new = nn.Conv2d(in_channels, out_c, (kh, kw), stride=mod.stride,
                            padding=mod.padding, dilation=mod.dilation,
                            groups=mod.groups, bias=mod.bias is not None)
            with torch.no_grad():
                nw = new.weight.data
                nw.zero_()
                nw[:, :3] = w                            # G,R,NIR <- pretrained RGB
                if in_channels > 3:                      # NDVI,CHM <- channel mean
                    nw[:, 3:] = w.mean(dim=1, keepdim=True).repeat(1, in_channels - 3, 1, 1)
                nw.mul_(3.0 / in_channels)               # preserve magnitude
                if mod.bias is not None:
                    new.bias.data = mod.bias.data.clone()
            parent, attr = _resolve_parent(model, name)
            setattr(parent, attr, new)
            return True
    return False


# --------------------------------------------------------------------------- #
# Pure-PyTorch fallback U-Net (no third-party deps)                            #
# --------------------------------------------------------------------------- #
class MiniUNet(nn.Module):
    """Compact U-Net (Ronneberger et al., 2015). For dev/smoke tests only.

    Optional D-LinkNet center block (`center='dblock'`) at the bottleneck for the
    occlusion-bridging receptive field.
    """

    def __init__(self, in_channels: int = 5, classes: int = 1, base: int = 32,
                 center: str = "none") -> None:
        super().__init__()
        self.d1 = _DoubleConv(in_channels, base)
        self.d2 = _DoubleConv(base, base * 2)
        self.d3 = _DoubleConv(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.bott = _DoubleConv(base * 4, base * 8)
        self.center = DBlock(base * 8) if center == "dblock" else None
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.u3 = _DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.u2 = _DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.u1 = _DoubleConv(base * 2, base)
        self.head = nn.Conv2d(base, classes, 1)

    def forward(self, x):
        c1 = self.d1(x)
        c2 = self.d2(self.pool(c1))
        c3 = self.d3(self.pool(c2))
        b = self.bott(self.pool(c3))
        if self.center is not None:
            b = self.center(b)
        x = self.u3(torch.cat([self.up3(b), c3], 1))
        x = self.u2(torch.cat([self.up2(x), c2], 1))
        x = self.u1(torch.cat([self.up1(x), c1], 1))
        return self.head(x)


# --------------------------------------------------------------------------- #
# DINOv3 hero model (NON-gated SAT-493M via timm)                              #
# --------------------------------------------------------------------------- #
class DINOv3SegModel(nn.Module):
    """Occlusion-robust segmentation head on a (frozen) DINOv3 backbone.

      optical (G,R,NIR)  ─► DINOv3-ViT SAT-493M (frozen) ─► patch tokens ─► [B,D,h,w]
      aux (NDVI,CHM)     ─► small CNN branch ─────────────► [B,Da,h,w]
                                          concat ─► light decoder ─► logits

    The 3 optical bands feed the frozen ViT (SAT normalization applied here); the
    physical priors (NDVI, CHM) enter via a PARALLEL branch so the pretrained
    patch-embedding is never disturbed. Backbone via timm — NON-gated:
        timm.create_model('vit_large_patch16_dinov3.sat493m', pretrained=True)
    SAT-493M uses non-ImageNet normalization (config model.dinov3.norm; issue #61).
    """

    def __init__(self, in_channels: int = 5, classes: int = 1,
                 timm_model: str = "vit_large_patch16_dinov3.sat493m",
                 weights_path: str | None = None, embed_dim: int = 1024,
                 patch: int = 16, freeze: bool = True, aux_dim: int = 64,
                 norm_mean=(0.430, 0.411, 0.296), norm_std=(0.213, 0.156, 0.143)) -> None:
        super().__init__()
        self.patch = patch
        self.embed_dim = embed_dim
        self.n_aux = max(0, in_channels - 3)
        self.register_buffer("norm_mean", torch.tensor(norm_mean).view(1, 3, 1, 1))
        self.register_buffer("norm_std", torch.tensor(norm_std).view(1, 3, 1, 1))

        if self.n_aux > 0:
            self.aux = nn.Sequential(_DoubleConv(self.n_aux, aux_dim), nn.MaxPool2d(patch))
        else:
            self.aux = None

        dec_in = embed_dim + (aux_dim if self.aux is not None else 0)
        self.decoder = nn.Sequential(
            _DoubleConv(dec_in, 256), nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            _DoubleConv(256, 128), nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            _DoubleConv(128, 64), nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )
        self.head = nn.Conv2d(64, classes, 1)
        self.backbone, self.n_prefix = self._load_backbone(timm_model, weights_path, freeze)

    def _load_backbone(self, timm_model: str, weights_path: str | None, freeze: bool):
        try:
            import timm
        except ImportError as e:
            raise ImportError(
                "arch='dinov3' needs timm (`pip install timm>=1.0.15`). The model id "
                f"'{timm_model}' is NON-gated (no huggingface-cli login)."
            ) from e
        bb = timm.create_model(timm_model, pretrained=(weights_path is None), num_classes=0)
        if weights_path:                                  # optional local override
            sd = torch.load(weights_path, map_location="cpu")
            bb.load_state_dict(sd.get("model", sd), strict=False)
        bb.eval()
        if freeze:
            for p in bb.parameters():
                p.requires_grad_(False)
        n_prefix = int(getattr(bb, "num_prefix_tokens", 1))
        return bb, n_prefix

    def _tokens_to_map(self, x_opt: torch.Tensor) -> torch.Tensor:
        feats = self.backbone.forward_features(x_opt)     # [B, N, D] (ViT)
        if feats.ndim == 3:
            tok = feats[:, self.n_prefix:, :]             # drop cls/register tokens
            b, n, d = tok.shape
            h = w = int(round(n ** 0.5))
            return tok.transpose(1, 2).reshape(b, d, h, w)
        return feats                                      # already [B, D, h, w]

    def forward(self, x):
        x_opt = (x[:, :3] - self.norm_mean) / self.norm_std    # SAT normalization
        fmap = self._tokens_to_map(x_opt)
        if self.aux is not None:
            a = self.aux(x[:, 3:])
            if a.shape[-2:] != fmap.shape[-2:]:
                a = F.interpolate(a, size=fmap.shape[-2:], mode="bilinear", align_corners=False)
            fmap = torch.cat([fmap, a], dim=1)
        return self.head(self.decoder(fmap))


def build_model(cfg: Dict[str, Any]) -> nn.Module:
    """Instantiate a model from cfg.model. See module docstring for tiers."""
    m = cfg["model"]
    arch = str(m.get("arch", "miniunet")).lower()
    in_ch = int(m.get("in_channels", 5))
    classes = int(m.get("classes", 1))

    if arch == "miniunet":
        return MiniUNet(in_channels=in_ch, classes=classes,
                        base=int(m.get("base", 32)), center=str(m.get("center", "none")).lower())

    if arch == "smp":
        try:
            import segmentation_models_pytorch as smp
        except ImportError as e:
            raise ImportError(
                "arch='smp' needs segmentation_models_pytorch. Install the full "
                "env (see README) or set model.arch='miniunet' for a dep-free run."
            ) from e
        decoder = str(m.get("decoder", "unetplusplus")).lower()
        encoder = str(m.get("encoder", "efficientnet-b0"))
        weights = m.get("encoder_weights", "imagenet")
        stem_init = str(m.get("stem_init", "inflate")).lower()
        builders = {
            "unetplusplus": smp.UnetPlusPlus,
            "unet": smp.Unet,
            "deeplabv3plus": smp.DeepLabV3Plus,
            "linknet": smp.Linknet,          # D-LinkNet base decoder
        }
        if hasattr(smp, "Segformer"):        # Transformer/attention (advanced); needs recent smp
            builders["segformer"] = smp.Segformer
        if decoder not in builders:
            raise ValueError(f"Unknown smp decoder '{decoder}'. Options: {list(builders)}")

        # stem adaptation: 'inflate' keeps pretrained RGB filters (build with 3ch,
        # then inflate to in_ch); 'smp_default' lets smp randomly re-init the stem.
        if weights and in_ch != 3 and stem_init == "inflate":
            model = builders[decoder](encoder_name=encoder, encoder_weights=weights,
                                      in_channels=3, classes=classes)
            if not inflate_first_conv(model, in_ch):
                warnings.warn("Could not locate a 3-ch stem to inflate; rebuilding "
                              "with smp default stem re-init.")
                model = builders[decoder](encoder_name=encoder, encoder_weights=weights,
                                          in_channels=in_ch, classes=classes)
        else:
            model = builders[decoder](encoder_name=encoder, encoder_weights=weights,
                                      in_channels=in_ch, classes=classes)
        if str(m.get("center", "none")).lower() == "dblock":
            warnings.warn("center='dblock' is wired for miniunet; for smp insert it "
                          "between encoder and decoder (documented in METHODOLOGY).")
        return model

    if arch == "dinov3":
        dv = m.get("dinov3", {})
        nz = dv.get("norm", {}) or {}
        return DINOv3SegModel(
            in_channels=in_ch, classes=classes,
            timm_model=str(dv.get("timm_model", "vit_large_patch16_dinov3.sat493m")),
            weights_path=dv.get("weights_path"),
            embed_dim=int(dv.get("embed_dim", 1024)),
            patch=int(dv.get("patch", 16)),
            freeze=bool(dv.get("freeze", True)),
            aux_dim=int(dv.get("aux_dim", 64)),
            norm_mean=tuple(nz.get("mean", (0.430, 0.411, 0.296))),
            norm_std=tuple(nz.get("std", (0.213, 0.156, 0.143))),
        )

    if arch == "clay":
        raise NotImplementedError(
            "arch='clay' (Clay v1.5, STRETCH path) stub. Wire the encoder load: "
            "huggingface_hub.hf_hub_download(cfg.model.clay.repo, cfg.model.clay.checkpoint), "
            "build the GSD/wavelength-aware encoder, attach a decoder. Clay ingests "
            "G/R/NIR natively via per-band wavelength + gsd. See METHODOLOGY."
        )

    if arch == "vista_v2":
        # VISTA-v2: ResNet-101 + UNet++ (smp) with a PE-pluggable attention
        # bottleneck (botnet | rope) or input sinusoidal PE (sincos) — selected by
        # cfg.model.pe.type. Lazy import keeps smp optional. Returns plain logits,
        # so it trains/predicts through the existing VISTA pipeline unchanged.
        from ..vista_v2.model import build_vista_v2
        return build_vista_v2(cfg)

    raise ValueError(f"Unknown model.arch='{arch}'. "
                     "Options: miniunet | smp | dinov3 | clay | vista_v2.")
