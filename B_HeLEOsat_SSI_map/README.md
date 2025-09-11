## B — Apply the HeLEOsat Method (High-Resolution)

> **Important (clarification):** From here on, **all GeoTIFF outputs** are written on the **Sentinel-2 reference grid** (copy *CRS/transform/shape* from `min_reflectance_B02_{site}.tif`).  
> Intermediate products that stay in native array formats — e.g., **Asterisk** (`.npz`) and **SenSeIv2** (`.npy`) — are **not GeoTIFFs yet**. They are **mapped to the reference grid** at the *Filtered Cloud Height* step and in all subsequent GeoTIFF-writing steps.

### File format & grid per step (quick map)
- **B.2 — Cloud Index** → **GeoTIFF** *(on reference grid)*
- **B.3 — Asterisk Cloud Heights** → **`.npz`** *(points/arrays; **not** on grid yet)*
- **B.4 — SenSeIv2 Segmentation** → **`.npy`** *(model output; **not** a GeoTIFF yet)*
- **B.5 — Filtered Cloud Height** → **GeoTIFF** *(this is where Asterisk + SenSeIv2 are **mapped to the reference grid**)*
- **B.6 — Shadow Index** → **GeoTIFF** *(reference grid)*
- **B.7 — Clear-Sky GHI (CAMS→S2)** → **GeoTIFF** *(reference grid)*
- **B.8 — HeLEOsat High-Res GHI** → **GeoTIFF** *(reference grid)*


### B.1 — Min/Max (B02)
- Inputs: all B02 bands from .SAFE scenes.
- Outputs:
- `min_reflectance_B02_{site}.tif`  (**reference grid**)
- `max_reflectance_B02_{site}.txt`  (robust max scalar)

### B.2 — Cloud Index per scene
- `cloud_index = (p_scene − p_min) / (p_max − p_min)` → clip to `[0, 1]`.
- Output: `cloud_index_{YYYYMMDDTHHMMSS}.tif` (aligned to reference).

### B.3 — Asterisk Cloud Heights
- Run Asterisk → outputs `.npz` with `heights`, `coords`, `times`


### B.4 — Cloud Segmentation (SEnSeIv2)
- Run SenSeIv2 on each .SAFE → `<SCENE>_mask.npy` (classes incl. thin/thick cloud).

### B.5 — Filtered Cloud Height
- Interpolate Asterisk heights to **10 m reference grid** (NN).
- Keep only **thin/thick** cloud pixels (others → 0 or NaN).
- Fill missing cloud heights (`<= 0`) from nearest valid neighbours.
- Output: `filtered_cloud_heigth_data_{START}_{END}/*.tif`.

### B.6 — Shadow Index (Parallax shift of Cloud Index)
- Per scene, read **Mean Viewing Angle** (satellite) and **Mean Sun Angle** from `MTD_TL.xml`.
- Shift **Cloud Index** per pixel by parallax (**height × tan(zenith)** in azimuth direction(s)); accumulate to a continuous **Shadow Index**; fill holes by NN.
- Output: `shadow_index_{START}_{END}/*.tif`.

### B.7 — Clear-Sky GHI (CAMS → S2 grid)
- Read scene **SENSING_TIME** (rounded to minute).
- Extract **GHI_clear_sky** at that minute from CAMS CSVs; map CAMS points → S2 pixels (NN).
- Output: `ghi_clear_sky_{YYYYMMDD}.tif`.

### B.8 — HeLEOsat High-Res SSI/GHI
- Map **Shadow Index → kc** using your piecewise function:
- `CI < −0.2 → kc = 1.2`  
- `−0.2 ≤ CI < 0.8 → kc = 1 − CI`  
- `0.8 ≤ CI < 1.1 → kc = 2.0667 − 3.6667·CI + 1.6667·CI²`  
- `CI ≥ 1.1 → kc = 0.05`
- Fill NaNs in `GHI_clear_sky` locally (small mean window).
- Compute **`GHI = kc × GHI_clear_sky`** and write GeoTIFF on the reference grid.
- Output: `heleo_ghi_{SCENE_ID}.tif`.
