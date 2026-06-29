# References (Phase I)

Peer-reviewed / authoritative sources, with how each is used. `[NPR]` = not
peer-reviewed (preprint/blog/docs) — flagged per project rules.

## Sensor / data
- **NRSC/ISRO (2011).** *Resourcesat-2 Data Users Handbook.* — LISS-IV facts:
  5.8 m GSD, B2/B3/B4 = G/R/NIR, ~70 km Mono / 23.5 km Mx swath, 10-bit.
  https://earth.esa.int/eogateway/documents/20142/37627/ResourceSat-2-Data-User-Handbook.pdf
- **eoPortal — ResourceSat-2.** (corroborating) https://www.eoportal.org/satellite-missions/resourcesat-2
- **Boeing, G. (2017).** *OSMnx.* Computers, Environment and Urban Systems 65:126–139. — OSM road graphs → labels.

## Segmentation models
- **Ronneberger, Fischer, Brox (2015).** *U-Net.* MICCAI. — baseline decoder.
- **Zhou et al. (2018).** *UNet++.* DLMIA. — smp baseline decoder.
- **Chen et al. (2018).** *DeepLabV3+.* ECCV. — smp decoder option.
- **Zhou, Zhang, Wu (2018).** *D-LinkNet.* CVPRW (DeepGlobe winner). — Dblock dilated center (occlusion gap-bridging). https://github.com/zlckanata/DeepGlobe-Road-Extraction-Challenge
- **Oquab, Siméoni et al. (2025).** *DINOv3.* `[NPR]` arXiv:2508.10104. — SAT-493M backbone (B1). timm id `vit_large_patch16_dinov3.sat493m` (non-gated); SAT norm mean=[0.430,0.411,0.296], std=[0.213,0.156,0.143].
- **Clay Foundation (2024).** *Clay v1.5 model.* `[NPR]` Apache-2.0. — GSD/wavelength-aware EO foundation model (B2). https://clay-foundation.github.io/model/

## Transfer learning / resolution gap
- **Carreira & Zisserman (2017).** *Quo Vadis (I3D).* CVPR. — conv1 weight **inflation** for the G/R/NIR + NDVI/CHM stem.
- **Bastani et al. (2023).** *SatlasPretrain.* ICCV. — large-scale RS pretraining; downsample-before-pretrain rationale.

## Loss / metrics
- **Milletari, Navab, Ahmadi (2016).** *V-Net (Dice loss).* 3DV. arXiv:1606.04797.
- **Shit et al. (2021).** *clDice — Topology-Preserving Loss.* CVPR, pp. 16560–16569. — connectivity term. https://github.com/jocpae/clDice
- **Wiedemann, Heipke, Mayer, Jamet (1998).** *Empirical evaluation of automatically extracted road axes.* — buffered completeness/correctness (relaxed metrics).
- **Demir et al. (2018).** *DeepGlobe 2018.* CVPRW. — relaxed road F1 protocol.

## GROVE arm — occlusion completion / roads under canopy (Arm B)
- **Zhang, Miao, Liu, Tian, Zhou (2022).** *HA-RoadFormer: Hybrid Attention Transformer with Multi-Branch for Large-Scale High-Resolution Dense Road Segmentation.* Mathematics 10(11):1915. — GROVE backbone: **overlapping multi-scale patch embedding** + **linear-complexity hybrid attention** (adjacent 5-diagonal + random); U-Net fusion. Its conclusion names the open gap GROVE fills: *"fuse [road] topology information into neural networks."* https://www.mdpi.com/2227-7390/10/11/1915
- **Batra, Singh, Reddy, Khan, Chandra, Jawahar (2019).** *Improved Road Connectivity by Joint Learning of Orientation and Segmentation.* CVPR, pp. 10385–10393. — **axial sin(2θ)/cos(2θ) orientation** auxiliary task → connectivity under occlusion (GROVE orientation head).
- **Mei, Li, Ye, Zhao, Liang, Yang (2021).** *CoANet: Connectivity Attention Network for Road Extraction From Satellite Imagery.* IEEE T-IP 30:8540–8552. — **strip convolution** + **connectivity attention**, explicitly targets **tree/building occlusion** (GROVE Stage 5). https://ieeexplore.ieee.org/document/9563125 · code https://github.com/mj129/CoANet
- **Dong et al. (2022).** *CSWin Transformer: Cross-Shaped Windows.* CVPR. `[NPR]` arXiv:2107.00652. — optional stripe-attention backbone (matches road geometry). Road use: **DCS-TransUperNet** (Zhang et al., Appl. Sci. 2022, 12(7):3511).
- **Liu et al. (2021).** *Swin Transformer.* ICCV, pp. 10012–10022. — alternative hierarchical shifted-window backbone.
- **He et al. (2020).** *Sat2Graph: Road Graph Extraction Through Graph-Tensor Encoding.* ECCV. — graph-level long-gap recovery (GROVE Stage 6 option).
- **Xu et al. (2022).** *RNGDet: Road Network Graph Detection by Transformer.* IEEE T-GRS. `[NPR]` arXiv:2202.07824. — transformer vertex-by-vertex tracing (long-gap bridge option).

## Noisy labels / evaluation discipline  (CITATION FIX — two distinct papers)
- **Mnih & Hinton (2010).** *Learning to Detect Roads in High-Resolution Aerial Images.* ECCV, LNCS pp. 210–223. — road-detection CNN lineage.
- **Mnih & Hinton (2012).** *Learning to Label Aerial Images from Noisy Data.* ICML. — **noisy OSM-label** justification (omission + registration noise → motivates relaxed/buffered metrics + robust loss). https://icml.cc/2012/papers/318.pdf
- **Roberts et al. (2017).** *Cross-validation strategies for data with … spatial structure.* Ecography 40(8):913–929. — **spatial-block CV** (no random-split leakage).

## Canopy / OCOI (bridge to Phase II/III)
- **Rouse et al. (1974).** *NDVI.* — vegetation index.
- **Tolan et al. (2024).** *Canopy height from imagery.* Remote Sensing of Environment. — CHM prior.
- **Li et al. (2015).** *Treepedia / Green View Index.* Urban Forestry & Urban Greening 14(3):675–685. — point-sampling design for OCOI.

## Phase 2 — Road Network Extraction (code referenced)
- **Image-Py/sknw** (MIT) — skeleton→NetworkX graph (`build_sknw`). *Dependency.* https://github.com/Image-Py/sknw
- **avanetten/cresi** (Apache-2.0) — mask→sknw→graph→speeds pipeline. *Pattern reference.* https://github.com/avanetten/cresi
- **CosmiQ/apls** (Apache-2.0) — APLS road-graph similarity metric. *Reference for topo-accuracy.* https://github.com/CosmiQ/apls
- **Van Etten et al.** *City-Scale Road Extraction from Satellite Imagery (CRESI).* arXiv:1908.09715.
- **PaRK-Detect (2023)** arXiv:2302.13263 — occlusion-broken endpoints are usually truly connected (justifies healing).
- **scikit-image** — `skeletonize`, morphology. **NetworkX** — graph + centrality.

## Phase 3 — Resilience (forward-looking)
- **Freeman (1977).** *Betweenness centrality.* Sociometry.
- **Albert, Jeong, Barabási (2000).** *Error and attack tolerance of complex networks.* Nature 406.
- **Latora & Marchiori (2001).** *Efficient behavior of small-world networks.* Phys. Rev. Lett.
