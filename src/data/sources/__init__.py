"""Data-source adapters for Phase I.

Each adapter isolates ONE source behind a small function-level API. Heavy
geospatial dependencies (rasterio, pystac_client, planetary_computer, osmnx,
requests) are imported *inside functions* (lazy), so importing this package on a
minimal interpreter never fails — only calling a fetcher requires its deps.

  bhoonidhi.py  -> Resourcesat LISS-IV (5.8 m) via the Bhoonidhi STAC API (JWT)
  planetary.py  -> Sentinel-2 L2A (10 m) via Microsoft Planetary Computer STAC
  osm.py        -> OpenStreetMap roads (osmnx) -> NetworkX graph + raster labels
"""
