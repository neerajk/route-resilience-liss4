# data/ — where to drop your inputs

```
data/
├── raw/
│   ├── liss4/   ← PLACE LISS-IV band GeoTIFFs here
│   │              B2.tif (Green), B3.tif (Red), B4.tif (NIR)
│   │              (or one 3-band stack; confirm band order = G,R,NIR)
│   └── aoi/     ← PLACE the AOI shapefile here
│                  blore_urban.shp  + .shx .dbf .prj .cpg
├── tiles/       ← GENERATED .npz training tiles (do not edit; gitignored)
```

After placing files, point `config/phase1/config.yaml → data.liss4`:
```yaml
  liss4:
    green: data/raw/liss4/B2.tif
    red:   data/raw/liss4/B3.tif
    nir:   data/raw/liss4/B4.tif
    aoi:   data/raw/aoi/blore_urban.shp   # clips imagery + defines OSM fetch area
    out_dir: data/tiles
```
Then run Step 1 (OSM rasterize + tile). OSM roads are auto-pulled via osmnx
(no credentials) for the AOI bounds — that becomes the road label mask.
