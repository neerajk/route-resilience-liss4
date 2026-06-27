PHASE I — Occlusion-Robust Segmentation

Goal: turn satellite pixels into a road map, even where trees/shadows hide roads.

Step 1 — Data Preprocessing

What: prepare raw satellite scenes into training-ready patches.
Four sub-parts:
- Tile — a Bangalore LISS-4 scene is huge (~11000×12000 px). Models train on small fixed squares, so we cut the scene into a grid of 256×256 tiles. Why: GPU memory + batching.
- Normalize / enhance contrast — raw pixel values are Digital Numbers (DN, 0–1023 for 10-bit), not real reflectance. Networks learn best on standardized inputs, so we subtract each band's mean and divide by its std (→ ~mean 0). Contrast-stretch makes faint roads visible. Why: stable training, consistent across scenes/dates.
- Simulate occlusions — deliberately hide road pixels (paste canopy-like patches over roads) during training. Why: if the model only ever sees clean roads, it never learns to fill gaps. This teaches gap-inference — the core skill.
- Balance the dataset — roads are ~1–2% of pixels, occluded roads even rarer. We oversample/weight tiles containing occluded roads. Why: otherwise the model ignores the rare-but-critical case.

In: LISS-4 (+ optional S2) scenes + OSM. Out: normalized, tiled, augmented patches + masks.

Step 2 — Baseline Model Development

What: train a standard, reliable segmentation model first (U-Net or DeepLabV3+).
How: a U-Net is an encoder–decoder: the encoder compresses the image into features (what's here), the decoder expands back to a per-pixel road/not-road mask, with "skip connections" that preserve fine detail. DeepLabV3+ adds dilated convolutions (wider view without losing resolution).
Why baseline first: get a working number fast, a reference to beat, and — crucially — find where it fails (gaps under canopy) to guide the advanced model. Don't start fancy.
In: training patches. Out: a baseline model + a failure map.

Step 3 — Advanced Model Design (context-aware)

What: a Transformer / attention model that reasons about far-away context.
How: a Vision Transformer (ViT, e.g. DINOv3) uses self-attention — every patch of the image can "look at" every other patch. Concretely for occlusion: the model sees a road enter one side of a tree clump and exit the other side, and infers it continues underneath. CNNs only see a small local window; Transformers see globally — that's the "see through occlusion" mechanism.
- Spatial attention = which locations matter; channel attention = which feature-types matter.
In: baseline + data. Out: the occlusion-robust context-aware model.

Step 4 — Loss Function Engineering

What: the loss is the score of "how wrong" a prediction is; training shrinks it. We combine four:
- Dice / IoU loss — measure overlap of predicted vs. true roads; robust to the road-rarity imbalance.
- Boundary-aware loss — extra penalty at road edges → sharper outlines.
- Connectivity loss (clDice) — rewards keeping roads connected (topology), not just overlapping → directly helps gap-filling, which Phase II depends on.

Why combine: each targets a different failure (imbalance / blurry edges / broken connectivity). In: prediction + mask. Out: one number that steers learning.

Step 5 — Occlusion Handling Strategy

What: the explicit tactics for hidden roads.
- Context-based inference — lean on the Transformer's global attention (Step 3) to infer hidden roads from surroundings.
- Multi-scale feature fusion — combine fine detail (thin roads) with coarse context (large gaps); done with dilated convs / feature pyramids.
- Inpainting (optional) — a generative "paint-in-the-gap" step, like photo restoration. Advanced/optional.

In: model + occluded inputs. Out: a complete mask despite occlusion.

➡ Phase I output: a per-pixel road probability map → threshold → binary road mask, saved as a georeferenced GeoTIFF (carries real-world coordinates — essential for Phase II).

---
PHASE II — Graph Skeletonization & Healing

Goal: turn the road picture into a routable network.

Thinning (skeletonization)

What/How: the mask is a thick blob; skeletonize shrinks each road to a 1-pixel centerline (its spine). Then nodes = junctions/endpoints, edges = the segments between them → a graph. (Tools: skimage.skeletonize → sknw → NetworkX.)

Topological Healing

What: reconnect roads broken by occlusion.
How (three ideas):
- Disjoint Set (Union-Find) — bookkeeping that knows which road pieces are already connected, so you only bridge genuinely separate broken ends (no accidental loops).
- MST (Minimum Spanning Tree) — connect the broken pieces with minimal total added length.
- Euclidean distance + angular alignment — only join ends that are close and pointing the same way, so the healed road follows a natural trajectory, not a zigzag.

In: binary road mask (GeoTIFF). Out: a connected, routable, weighted graph.

---
PHASE III — Network Analysis & Stress Testing

Goal: find the city's weak points and simulate failure.

- Betweenness centrality — for every pair of places there's a shortest route; a node's betweenness = how many of all those routes pass through it. High = bottleneck / "Gatekeeper Node."
- Node ablation — remove the highest-betweenness nodes one by one (simulate a flooded/closed junction) and watch the network fall apart. Compare vs. random/degree removal.
- Resilience Index — R = (avg shortest-path normal) / (avg shortest-path after damage). Lower R = more fragile (travel times explode when key nodes die).

In: the graph. Out: a criticality heatmap + resilience decay curves.

---
PHASE IV — Interactive Dashboard

- Heatmap overlay — color each road by criticality (weakest links at a glance).
- Simulation toggle — click a junction to "disable" → instantly see rerouting + travel-time increase.
- Built with Streamlit + Leaflet/folium. In: graph + metrics. Out: a planner-friendly web tool.

---
How you can make the build faster/better


✻ Baked for 1m 34s

❯ where are we using Data readiness is fully secured through a zero-manual-effort automation pipeline that pairs open-source ground truth with multi-resolution satellite feeds. Participants may utilise:

  SpaceNet Roads Dataset - for model development and pre-training
  DeepGlobe Road Extraction Dataset - for model development and pre-training
  OpenSatMap - for model development and pre-training
  OpenStreetMap (OSM) road vector layers - ground-truth road masks generated automatically and used as reference annotations for training, validation, and performance assessment these datasets?

Good question — these four datasets play two different roles, and it's worth being crisp about which goes where. Here's the mapping into our methodology.

The two roles

DeepGlobe ─┐
SpaceNet  ─┼─► PRE-TRAINING  (teach the model "what roads look like" before LISS-4)
OpenSatMap─┘

OSM ─────────► GROUND TRUTH  (the actual labels for Bangalore LISS-4: train + validate + score)

Dataset-by-dataset

┌──────────────────┬─────────────────────────┬─────────────────────────────────────────────┬──────────────────────────────────────────┬─────────────────────┐
│     Dataset      │          Role           │              Step in our plan               │               Format / res               │       Status        │
├──────────────────┼─────────────────────────┼─────────────────────────────────────────────┼──────────────────────────────────────────┼─────────────────────┤
│ OSM road vectors │ Labels / ground truth   │ Step 1 (rasterize → masks) + all evaluation │ vectors → burned to 5.8 m masks          │ ⛔ the open blocker │
├──────────────────┼─────────────────────────┼─────────────────────────────────────────────┼──────────────────────────────────────────┼─────────────────────┤
│ DeepGlobe        │ Pre-training (primary)  │ Step 3                                      │ 6,226 imgs, 1024², 0.5 m RGB, road masks │ ⛔ not wired        │
├──────────────────┼─────────────────────────┼─────────────────────────────────────────────┼──────────────────────────────────────────┼─────────────────────┤
│ SpaceNet         │ Pre-training (optional) │ Step 3                                      │ 0.3 m, road vectors (4 cities)           │ ⛔ not wired        │
├──────────────────┼─────────────────────────┼─────────────────────────────────────────────┼──────────────────────────────────────────┼─────────────────────┤
│ OpenSatMap       │ Pre-training (optional) │ Step 3                                      │ fine-grained hi-res, large-scale         │ ⛔ not wired        │
└──────────────────┴─────────────────────────┴─────────────────────────────────────────────┴──────────────────────────────────────────┴─────────────────────┘

How they're actually used (the two-stage strategy)

Stage A — Pre-train on benchmarks (DeepGlobe etc.):
These are big, richly-labeled road datasets. We downsample them from 0.3–0.5 m → 5.8 m (to match LISS-4's scale), then pre-train the encoder so it already "knows roads" before it ever sees LISS-4. Why: your Bangalore LISS-4+OSM labeled set is small; pre-training is the head-start.

Stage B — Fine-tune + evaluate on LISS-4 + OSM:
The real target. OSM road vectors → automatically rasterized to masks on the LISS-4 grid = the ground truth for training, validation, and the headline metrics (Occlusion-Recall, relaxed IoU).

DeepGlobe/SpaceNet/OpenSatMap → downsample to 5.8m → PRETRAIN encoder
                                                          │
LISS-4 (Bangalore) + OSM masks ──────────────────────► FINE-TUNE + EVALUATE

The one caveat that matters

Benchmarks are 0.5 m RGB; LISS-4 is 5.8 m G/R/NIR (no Blue). So pretraining has two gaps:
1. Resolution → fixed by downsampling benchmarks to 5.8 m.
2. Spectral (RGB ≠ G/R/NIR) → the model's input layer must be re-adapted (the "stem inflation" we discussed). The pretrained road-shape knowledge transfers; the input stem is re-fit for G/R/NIR.

Two things to note
How (three ideas):
- Disjoint Set (Union-Find) — bookkeeping that knows which road pieces are already connected, so you only bridge genuinely separate broken ends (no accidental loops).
- MST (Minimum Spanning Tree) — connect the broken pieces with minimal total added length.
- Euclidean distance + angular alignment — only join ends that are close and pointing the same way, so the healed road follows a natural trajectory, not a zigzag.

In: binary road mask (GeoTIFF). Out: a connected, routable, weighted graph.

---
PHASE III — Network Analysis & Stress Testing

Goal: find the city's weak points and simulate failure.

- Betweenness centrality — for every pair of places there's a shortest route; a node's betweenness = how many of all those routes pass through it. High = bottleneck / "Gatekeeper Node."
- Node ablation — remove the highest-betweenness nodes one by one (simulate a flooded/closed junction) and watch the network fall apart. Compare vs. random/degree removal.
- Resilience Index — R = (avg shortest-path normal) / (avg shortest-path after damage). Lower R = more fragile (travel times explode when key nodes die).

In: the graph. Out: a criticality heatmap + resilience decay curves.

---
PHASE IV — Interactive Dashboard

- Heatmap overlay — color each road by criticality (weakest links at a glance).
- Simulation toggle — click a junction to "disable" → instantly see rerouting + travel-time increase.
- Built with Streamlit + Leaflet/folium. In: graph + metrics. Out: a planner-friendly web tool.