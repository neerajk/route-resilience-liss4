"""Render the rr Phase-I methodology as a labelled flowchart (PNG + PDF).

Color key: INPUT (green) -> PROCESS (blue) -> MODEL (purple) -> LOSS (pink) ->
METRIC (teal) -> OUTPUT (orange); LOGIC/rationale notes (grey, dashed) on the right.
Run:  python docs/make_methodology_fig.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

C = {  # category fills
    "in": "#bfe3bf", "proc": "#bcd6f0", "model": "#d9c4ef",
    "loss": "#f6cfdd", "metric": "#bfe8e8", "out": "#f8cf95", "logic": "#ececec",
}


def box(ax, x, y, w, h, text, fc, fs=8.5, bold=False, italic=False, ec="#3a3a3a"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15,rounding_size=0.9",
                                fc=fc, ec=ec, lw=1.2, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal",
            fontstyle="italic" if italic else "normal", zorder=3)


def lbox(ax, x, y, w, h, text):  # logic / rationale note
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.6",
                                fc=C["logic"], ec="#999", lw=0.9, ls="--", zorder=2))
    ax.text(x + 0.6, y + h / 2, text, ha="left", va="center", fontsize=7.0,
            fontstyle="italic", color="#333", zorder=3)


def arrow(ax, x1, y1, x2, y2, ls="-", color="#555", lw=1.6):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), zorder=1,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls,
                                shrinkA=1, shrinkB=1))


fig, ax = plt.subplots(figsize=(13, 16))
ax.set_xlim(0, 100); ax.set_ylim(8, 150); ax.axis("off")

ax.text(33, 147, "rr — Phase I Methodology", ha="center", fontsize=16, fontweight="bold")
ax.text(33, 143.5, "Occlusion-robust road extraction from LISS-IV (5.8 m, G/R/NIR)",
        ha="center", fontsize=10, color="#444")

# legend
leg = [("INPUT", "in"), ("PROCESS", "proc"), ("MODEL", "model"),
       ("LOSS", "loss"), ("METRIC", "metric"), ("OUTPUT", "out"), ("LOGIC", "logic")]
for i, (name, k) in enumerate(leg):
    lx = 4 + i * 13.5
    ax.add_patch(FancyBboxPatch((lx, 137), 2.2, 2.2, boxstyle="round,pad=0.05",
                                fc=C[k], ec="#777", lw=0.8))
    ax.text(lx + 3, 138.1, name, fontsize=7.5, va="center")

# ---- INPUTS row ----
inputs = [
    (5, "LISS-IV\nG / R / NIR\n(.tif)"),
    (19, "CHM\n(canopy height)"),
    (33, "OSM roads\n(labels*)"),
    (47, "Sentinel-2\n(optional)"),
]
for ix, t in inputs:
    box(ax, ix, 128, 12, 7, t, C["in"], fs=8)

# ---- spine boxes (x centred 33, width 52 -> 7..59) ----
SX, SW = 7, 52
spine = [
    (119, 8, C["proc"], "[INGEST]  reproject + align all layers to the LISS-IV 5.8 m grid (WarpedVRT)"),
    (108, 8, C["proc"], "[DERIVE]  NDVI=(NIR−Red)/(NIR+Red)   ·   canopy = CHM > thr   ·   OSM → mask (rasterize)"),
    (97, 8, C["proc"], "[STACK]  image = [ G , R , NIR , NDVI , CHM ]   (+ mask, + canopy)"),
    (86, 8, C["proc"], "[TILE]  256×256  +  spatial-block split  →  train / val / test  (.npz)"),
    (75, 8, C["proc"], "[PREP]  normalize DN→z   ·   augment (occlusion, scale)  — train only"),
    (60, 12, C["model"], "[MODEL]\noptical(3) → encoder { smp stem-inflated  |  DINOv3 SAT-493M frozen }\naux(NDVI,CHM) → CNN   →  [Dblock]  →  decoder  →  logits [1,H,W]"),
    (47, 8, C["loss"], "[LOSS]  0.3·BCE + 0.4·Dice + 0.3·clDice  (+ canopy-weight)  →  AdamW backprop"),
    (37, 8, C["metric"], "[VALIDATE]  sigmoid→thr→mask   ·   IoU · Dice · Occlusion-Recall (global) · relaxed F1"),
    (27, 8, C["out"], "[OUTPUT]  runs/<ts>/ :  best.pt · metrics.csv · loss_curve · prediction panel"),
    (17, 8, C["out"], "[NEXT]  inference → road mask → Phase II graph → Phase III resilience"),
]
ys = []
for (y, h, fc, t) in spine:
    box(ax, SX, y, SW, h, t, fc, fs=8.2, bold=(fc in (C["model"], C["out"]) and "MODEL" in t))
    ys.append((y, h))

# inputs -> ingest
for ix, _ in inputs:
    arrow(ax, ix + 6, 128, 33, 127.2)
# spine arrows (top edge of next from bottom edge of prev)
for i in range(len(spine) - 1):
    y_top, h_top = ys[i]
    y_bot, _ = ys[i + 1]
    arrow(ax, 33, y_top, 33, y_bot + ys[i + 1][1])

# ---- LOGIC notes (right column x=62, w=36) ----
LX, LW = 62, 36
notes = [
    (128, 7, "G/R/NIR only — NO Blue/SWIR. CHM from openCHm.\n*OSM = noisy labels (not provided yet)."),
    (119, 6.5, "Memory-safe: CHM resampled per-tile, never\nloads the whole scene."),
    (108, 6.5, "NDVI = canopy discriminator (Rouse'74).\nCHM = occlusion prior. all_touched keeps thin roads."),
    (97, 6, "Physical priors as extra channels → not\n'spectrally blind' under canopy."),
    (86, 6, "Spatial blocks → no autocorrelation leakage\n(Roberts 2017) → honest metrics."),
    (75, 6, "Occlusion aug teaches in-painting;\nscale aug = GSD robustness."),
    (60, 12, "Stem inflation keeps pretrained RGB filters on\nG/R/NIR (I3D). DINOv3 frozen + light head.\nDblock dilation 1/2/4/8 bridges canopy gaps."),
    (47, 6.5, "clDice = topology / connectivity (Shit'21).\ncanopy-weight penalises occluded misses."),
    (37, 6.5, "Occlusion-Recall = HEADLINE metric.\nGlobal pooling = unbiased; relaxed = OSM tolerance."),
]
for (y, h, t) in notes:
    lbox(ax, LX, y, LW, h, t)
    arrow(ax, 59, y + h / 2, LX, y + h / 2, ls=":", color="#999", lw=1.0)

fig.tight_layout()
out = Path(__file__).resolve().parent
fig.savefig(out / "rr_methodology.png", dpi=200, bbox_inches="tight")
fig.savefig(out / "rr_methodology.pdf", bbox_inches="tight")
print(f"wrote {out/'rr_methodology.png'}")
