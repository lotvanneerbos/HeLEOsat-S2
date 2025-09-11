# HeLEOsat Solar Maps — Full Workflow (High-Res + Low-Res)

This document describes the end-to-end workflow to produce **high-resolution** and **low-resolution** solar surface irradiance maps from Sentinel-2 data, using the HeLEOsat method plus CAMS radiation. All scripts are simple: edit the **inputs at the top** of each script and run.

---

## Prerequisites

**Python 3.9+** and (typical) packages:

### Core
numpy>=1.23
pandas>=1.5
scipy>=1.10
rasterio>=1.3
pyproj>=3.4

### Geo / utils
shapely>=2.0
geojson>=3.0

### Vision / ML backends
opencv-python-headless>=4.8
torch>=2.0

### Download helpers (S2 on Creodias)
creodias-finder

### --- Low-res (MSG/SEVIRI + CAMS) — only if you run section C ---
satpy
cdsapi



### External repos (install and make importable):

- **Asterisk / cloud_height_prototype_main** (cloud heights)
- **SEnSeIv2** (cloud segmentation)

---

---

## A — Collect Data & Fix Grids

### A.1 — Decide what to download from Sentinel-2
- Choose **SITE**, **START_DATE**, **END_DATE**.
- Download **Sentinel-2 L1C .SAFE** scenes for your AOI/time window

### A.2 — Establish the Sentinel-2 **reference grid** via Band-2 Min/Max (for Cloud Index B.1 and Filtered Cloud Height B.4 and Low Resolution scenes C)
- Build:
- **min_reflectance_B02_{site}.tif**  → this is your **reference GeoTIFF** (CRS, transform, resolution, width/height).
- **max_reflectance_B02_{site}.txt** → robust per-scene upper bound (scalar, e.g., P90).
- **All subsequent rasters must copy** this reference’s CRS/transform/shape to stay **pixel-aligned**.

### A.3 — Define MSG grid relative to S2 ( for downloading CAMS A.4 )
- If combining MSG/SEVIRI, resample MSG → **S2 reference grid** so everything lands on the same raster geometry.

### A.4 — Download **CAMS** minute data (for clear-sky GHI B6 and Low Resolution scenes C)

- Will be matched by rounding each scene’s SENSING_TIME to the nearest **minute**.

### A.5 —  Sentinel-2 pixel grid in coordinates  ( for clear-sky GHI B6 and Generate low-res SSI/GHI maps C )
- You can:
- compute pixel-centre coordinates **on-the-fly** from the reference GeoTIFF, **or**
- generate a CSV once (e.g. `sentinel2_pixel_coords_{site}.csv`).
- If you **hard-code** the grid, replace placeholders before running:
- `ULX = xxxxx`, `ULY = xxxxx`, `UTM_EPSG = "EPSG:xxxxx"`, `WIDTH_M = xxxxx`, `HEIGHT_M = xxxxx`, `RESOLUTION = xxxxx`, `SITE = "{location}"`.

---

## B — Apply the HeLEOsat Method (High-Resolution)

> **Important (clarification):** From here on, **all GeoTIFF outputs** are written on the **Sentinel-2 reference grid** (copy *CRS/transform/shape* from `min_reflectance_B02_{site}.tif`).  
> Intermediate products that stay in native array formats — e.g., **Asterisk** (`.npz`) and **SenSeIv2** (`.npy`) — are **not GeoTIFFs yet**. They are **mapped to the reference grid** at the *Filtered Cloud Height* step and in all subsequent GeoTIFF-writing steps.

### File format & grid per step (quick map)
- **B.1 — Cloud Index** → **GeoTIFF** *(on reference grid)*
- **B.2 — Asterisk Cloud Heights** → **`.npz`** *(points/arrays; **not** on grid yet)*
- **B.3 — SenSeIv2 Segmentation** → **`.npy`** *(model output; **not** a GeoTIFF yet)*
- **B.4 — Filtered Cloud Height** → **GeoTIFF** *(this is where Asterisk + SenSeIv2 are **mapped to the reference grid**)*
- **B.5 — Shadow Index** → **GeoTIFF** *(reference grid)*
- **B.6 — Clear-Sky GHI (CAMS→S2)** → **GeoTIFF** *(reference grid)*
- **B.7 — HeLEOsat High-Res GHI** → **GeoTIFF** *(reference grid)*


### B.1 — Cloud Index per scene
- `cloud_index = (p_scene − p_min) / (p_max − p_min)` → clip to `[0, 1]`.
- Output: `cloud_index_{YYYYMMDDTHHMMSS}.tif` (aligned to reference).

### B.2 — Asterisk Cloud Heights
- Run Asterisk → outputs `.npz` with `heights`, `coords`, `times`


### B.3 — Cloud Segmentation (SEnSeIv2)
- Run SenSeIv2 on each .SAFE → `<SCENE>_mask.npy` (classes incl. thin/thick cloud).

### B.4 — Filtered Cloud Height
- Interpolate Asterisk heights to **10 m reference grid** (NN).
- Keep only **thin/thick** cloud pixels (others → 0 or NaN).
- Fill missing cloud heights (`<= 0`) from nearest valid neighbours.
- Output: `filtered_cloud_heigth_data_{START}_{END}/*.tif`.

### B.5 — Shadow Index (Parallax shift of Cloud Index)
- Per scene, read **Mean Viewing Angle** (satellite) and **Mean Sun Angle** from `MTD_TL.xml`.
- Shift **Cloud Index** per pixel by parallax (**height × tan(zenith)** in azimuth direction(s)); accumulate to a continuous **Shadow Index**; fill holes by NN.
- Output: `shadow_index_{START}_{END}/*.tif`.

### B.6 — Clear-Sky GHI (CAMS → S2 grid)
- Read scene **SENSING_TIME** (rounded to minute).
- Extract **GHI_clear_sky** at that minute from CAMS CSVs; map CAMS points → S2 pixels (NN).
- Output: `ghi_clear_sky_{YYYYMMDD}.tif`.

### B.7 — HeLEOsat High-Res SSI/GHI
- Map **Shadow Index → kc** using your piecewise function:
- `CI < −0.2 → kc = 1.2`  
- `−0.2 ≤ CI < 0.8 → kc = 1 − CI`  
- `0.8 ≤ CI < 1.1 → kc = 2.0667 − 3.6667·CI + 1.6667·CI²`  
- `CI ≥ 1.1 → kc = 0.05`
- Fill NaNs in `GHI_clear_sky` locally (small mean window).
- Compute **`GHI = kc × GHI_clear_sky`** and write GeoTIFF on the reference grid.
- Output: `heleo_ghi_{SCENE_ID}.tif`.

---
## C — Low-Resolution SSI (MSG/SEVIRI + CAMS)

**Purpose**  
Produce coarse/low-resolution SSI/GHI maps that are **aligned** to the Sentinel-2 reference grid. Useful as a baseline and for super-resolution experiments.

**This section reuses outputs from A.3 and A.4**  
- From **A.3** (MSG grid relative to S2): `msg_pixel_coords_{site}.csv`  
- From **A.4** (CAMS minute data): `cams_{start}to{end}_{lat}_{lon}.csv` files

**Path conventions (edit at the top of your scripts)**  
- `SITE = "{location}"`  
- `EPSG = "EPSG:xxxxx"` (CRS of the S2 tile)  
- `S2_COORDS_CSV = "/path_to_grid_csv/.../sentinel2_pixel_coords_{SITE}.csv"`  
- `CAMS_FOLDER = "/path_to_cams/.../cams_downloads_{SITE}_minute/"`  
- `REFERENCE_TIFF_PATH = "/path_to_ref_tif/.../min_reflectance_B02_{SITE}.tif"`  
- `OUT_FOLDER = "/path_to_output/..."`

---

### C.1 — Generate low-res SSI/GHI maps (consume A.3/A.4 outputs)
- For each S2 scene, read **SENSING_TIME** (round to minute).  
- Extract **GHI** from the CAMS CSVs at that minute using the MSG pixel list (`msg_pixel_coords_{site}.csv`).  
- **Map CAMS → S2 pixels** using:
  - `euclidean` (KD-tree NN), `manhattan` (L1 NN), or `linear` (bilinear with NN fallback).  
- Write the raster using the **CRS/transform/shape** from the S2 reference GeoTIFF.  
- Log NaN borders and fill gaps (NN).

**Outputs**  
- `low_res_ghi_YYYYMMDD.tif`  
- `nan_logs/nan_edges_{scene_id}.csv`

**Notes**  
- **Alignment:** always write using `min_reflectance_B02_{SITE}.tif` metadata to match the high-res pipeline pixel-perfectly.  
- **Timing:** CAMS is minute-resolved → always round S2 SENSING_TIME to the **nearest minute**.  
- **Performance:** `linear` yields smoother fields; `euclidean` (NN) is faster.
