#!/usr/bin/env python3
"""
Sentinel-2 Band 2 (Blue) — Site Reference Grid + Min/Max Stats

What this script does
- Scans all Sentinel-2 L1C .SAFE scenes for a site/date window.
- Reads Band 2 (B02) reflectance (scaled by 1/10000).
- Builds:
  (1) A per-pixel MIN reflectance GeoTIFF → used as the site’s canonical *reference grid*
      (same CRS, transform, resolution, width/height) for all later products.
  (2) A global MAX reflectance *scalar* (by default the 90th percentile) saved as a .txt.
      This is handy as a robust “bright limit” for normalisation or QC.

Why use the MIN-B02 GeoTIFF as the reference?
- It guarantees a *stable, single file* with the exact target grid (CRS, transform, 10 m resolution,
  raster shape) derived from your actual Sentinel-2 scenes at this site.
- All subsequent products (e.g., cloud index, filtered cloud height, masks) can be written to
  this **same grid** by copying its metadata ⇒ perfect pixel alignment, no resampling surprises.

Outputs
- <OUTPUT_DIR>/min_reflectance_B02_<SITE>.tif   (float32, nodata=NODATA_VALUE)
- <OUTPUT_DIR>/max_reflectance_B02_<SITE>.txt   (single float: P-th percentile of per-pixel max)

Requirements
- Python 3.9+
- pip install: numpy, rasterio, pandas
"""

# ========= EDIT YOUR INPUTS HERE =========
START_DATE = "2023-01-01"               # "YYYY-MM-DD"
END_DATE   = "2023-12-31"               # "YYYY-MM-DD"
SITE       = "{location}"       # used in output path

INPUT_DIR  = f"/path_to_sentinel_data/{SITE}_{START_DATE}_{END_DATE}"
OUTPUT_DIR = f"/path_to_min_max_cloud_index/{SITE}_min_max_cloud_index"

PERCENTILE_FOR_MAX = 90.0               # robust max (percentile over the per-pixel maxima)
NODATA_VALUE       = 1000.0             # nodata marker in the min-ref GeoTIFF
# =========================================

import os
import glob
import numpy as np
import rasterio


def get_band2_jp2_path(safe_dir):
    """Return the first B02 .jp2 path inside a .SAFE; raise if not found."""
    granule_path = os.path.join(safe_dir, "GRANULE")
    if not os.path.isdir(granule_path):
        raise FileNotFoundError(f"GRANULE folder not found in {safe_dir}")

    for sub in os.listdir(granule_path):
        img_data_path = os.path.join(granule_path, sub, "IMG_DATA")
        if os.path.exists(img_data_path):
            matches = glob.glob(os.path.join(img_data_path, "*_B02*.jp2"))
            if matches:
                return matches[0]
    raise FileNotFoundError(f"B02 .jp2 not found in {safe_dir}")


def extract_scene_dirs(base_path):
    """List .SAFE scenes (L1C) under base_path (names containing 'MSIL1C')."""
    return [d for d in os.listdir(base_path) if d.endswith(".SAFE") and "MSIL1C" in d]


def create_max_min_band2_map(site):
    """
    Build per-pixel MIN(B02) GeoTIFF (reference grid) and a global MAX(B02) scalar (P-th percentile).
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    out_max_txt = os.path.join(OUTPUT_DIR, f"max_reflectance_B02_{site}.txt")
    out_min_tif = os.path.join(OUTPUT_DIR, f"min_reflectance_B02_{site}.tif")

    scene_dirs = extract_scene_dirs(INPUT_DIR)
    full_paths = [os.path.join(INPUT_DIR, s) for s in sorted(scene_dirs)]

    max_reflectance = None
    min_reflectance = None
    meta = None

    for idx, safe_dir in enumerate(full_paths):
        try:
            b02_path = get_band2_jp2_path(safe_dir)
            with rasterio.open(b02_path) as src:
                band = src.read(1).astype(np.float32) / 10000.0
                band[band <= 0] = np.nan

                if max_reflectance is None:
                    max_reflectance = np.nan_to_num(band, nan=-np.inf).copy()
                    min_reflectance = np.full_like(band, NODATA_VALUE, dtype=np.float32)
                    valid = np.isfinite(band)
                    min_reflectance[valid] = band[valid]

                    meta = src.meta.copy()
                    meta.update(dtype=rasterio.float32, driver="GTiff", nodata=NODATA_VALUE)
                else:
                    max_reflectance = np.fmax(max_reflectance, np.nan_to_num(band, nan=-np.inf))
                    valid = np.isfinite(band)
                    min_reflectance[valid] = np.fmin(min_reflectance[valid], band[valid])

            print(f"[{idx+1}/{len(full_paths)}] Processed {os.path.basename(safe_dir)}")
        except Exception as e:
            print(f"[{idx+1}/{len(full_paths)}] Skipped {os.path.basename(safe_dir)}: {e}")

    if max_reflectance is not None:
        max_scalar = np.nanpercentile(max_reflectance, PERCENTILE_FOR_MAX)
        with open(out_max_txt, "w") as f:
            f.write(f"{max_scalar}")
        print(f"\nSaved robust max reflectance (P{PERCENTILE_FOR_MAX}) to: {out_max_txt}")

    if min_reflectance is not None and meta is not None:
        with rasterio.open(out_min_tif, "w", **meta) as dst:
            dst.write(min_reflectance.astype(np.float32), 1)
        print(f"Saved min reflectance map (reference grid) to: {out_min_tif}")


if __name__ == "__main__":
    create_max_min_band2_map(SITE)
