"""Microsoft Planetary Computer adapter — Sentinel-2 L2A (10 m), multi-temporal.

WHY (MASTER_PLAN §1.4, research-grounded): Sentinel-2's ~5-day revisit lets us
build a CLOUD-FREE, leaf-aware TEMPORAL COMPOSITE. A median over a season removes
transient occluders (cloud, shadow) and a leaf-off window reveals roads hidden
under deciduous canopy in any single date. Multiple revisits are shown to improve
EO segmentation (arXiv:2409.17363). S2 is CONTEXT only — never road geometry at
10 m (a 6-12 m road is sub-pixel).

Access: PC STAC search is anonymous; asset hrefs must be SIGNED with the
`planetary-computer` package before reading. Docs:
  https://planetarycomputer.microsoft.com/docs/quickstarts/reading-stac/
Sentinel-2 L2A collection id: 'sentinel-2-l2a'
  https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a

NO credentials required (anonymous, throttled). Cloud masking uses the L2A SCL
(Scene Classification Layer) band.
"""
from __future__ import annotations

from typing import Dict, List, Optional

STAC_URL = "https://planetarycomputer.microsoft.com/api/v1/stac"
COLLECTION = "sentinel-2-l2a"
# SCL classes to mask out: 3 cloud-shadow, 8 cloud-medium, 9 cloud-high, 10 cirrus
SCL_MASK_CLASSES = (3, 8, 9, 10)


def search_s2(bbox: List[float], date_range: str, max_cloud: float = 30.0,
              limit: int = 200) -> List:
    """Search S2 L2A items over bbox/date with a scene cloud-cover ceiling.

    `date_range` RFC3339, e.g. '2023-11-01/2024-03-31'. Returns signed pystac
    Items (ready to read).
    """
    import planetary_computer as pc          # lazy
    from pystac_client import Client          # lazy
    catalog = Client.open(STAC_URL, modifier=pc.sign_inplace)
    search = catalog.search(
        collections=[COLLECTION], bbox=bbox, datetime=date_range,
        query={"eo:cloud_cover": {"lt": max_cloud}}, max_items=limit,
    )
    return list(search.items())


def build_s2_composite(items: List, bbox: List[float],
                       bands: Optional[List[str]] = None,
                       resolution: int = 10, crs: str = "EPSG:32643"):
    """Build a cloud-masked MEDIAN temporal composite over `items`.

    Returns an xarray.DataArray [band, y, x] on the target CRS/resolution.
    Median over time is robust to residual clouds and is the standard, defensible
    compositing choice. Bands default to the optical + SWIR set LISS-IV LACKS
    (blue, red-edge, SWIR) so S2 genuinely *adds* spectral information.
    """
    import odc.stac                            # lazy
    import numpy as np
    bands = bands or ["B02", "B03", "B04", "B05", "B08", "B11", "B12", "SCL"]
    ds = odc.stac.load(items, bands=bands, bbox=bbox, crs=crs,
                       resolution=resolution, chunks={})
    scl = ds["SCL"]
    cloud = scl.isin(list(SCL_MASK_CLASSES))
    optical = [b for b in bands if b != "SCL"]
    masked = ds[optical].where(~cloud)         # drop cloudy pixels before median
    composite = masked.to_array().median(dim="time", skipna=True)
    return composite  # [band, y, x]
