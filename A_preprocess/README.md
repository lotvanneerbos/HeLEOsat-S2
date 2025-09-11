## A — Collect Data & Fix Grids

### A.1 — Decide what to download from Sentinel-2
- Choose **SITE**, **START_DATE**, **END_DATE**.
- Download **Sentinel-2 L1C .SAFE** scenes for your AOI/time window

### A.2 — Establish the Sentinel-2 **reference grid** via Band-2 Min/Max
- Build:
- **min_reflectance_B02_{site}.tif**  → this is your **reference GeoTIFF** (CRS, transform, resolution, width/height).
- **max_reflectance_B02_{site}.txt** → robust per-scene upper bound (scalar, e.g., P90).
- **All subsequent rasters must copy** this reference’s CRS/transform/shape to stay **pixel-aligned**.

### A.3 — Define MSG grid relative to S2 ( for downloading CAMS C.2 )
- If combining MSG/SEVIRI, resample MSG → **S2 reference grid** so everything lands on the same raster geometry.

### A.4 — Download **CAMS** minute data (for clear-sky GHI B7 and Low Resolution scenes C)

- Will be matched by rounding each scene’s SENSING_TIME to the nearest **minute**.

### A.5 —  Sentinel-2 pixel grid in coordinates  ( for clear-sky GHI B7 and Generate low-res SSI/GHI maps C.3 )
- You can:
- compute pixel-centre coordinates **on-the-fly** from the reference GeoTIFF, **or**
- generate a CSV once (e.g. `sentinel2_pixel_coords_{site}.csv`).
- If you **hard-code** the grid, replace placeholders before running:
- `ULX = xxxxx`, `ULY = xxxxx`, `UTM_EPSG = "EPSG:xxxxx"`, `WIDTH_M = xxxxx`, `HEIGHT_M = xxxxx`, `RESOLUTION = xxxxx`, `SITE = "{location}"`.

---
