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
