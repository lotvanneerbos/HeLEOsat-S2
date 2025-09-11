#!/usr/bin/env python3
"""
High-Resolution GHI (HeLEOsat2-style)

What this script does
- For each scene:
  1) Reads the parallax-shifted Cloud/Shadow Index (CI→SI) GeoTIFF.
  2) Reads the clear-sky GHI GeoTIFF for the same date.
  3) Resamples clear-sky to the SI grid if needed.
  4) Fills NaNs in clear-sky with a local mean filter.
  5) Converts SI→kc (clearness index) with a piecewise mapping.
  6) Computes GHI = kc * GHI_clear_sky and saves as GeoTIFF (same grid as SI).

Inputs expected on disk
- Shadow Index: /scratch/.../shadow_index_v3/<SITE>/shadow_index_<START>_<END>/*.tif
- Clear-sky GHI: /scratch/.../ghi_clear_sky/<SITE>/ghi_clear_sky_<YYYYMMDD>.tif

Outputs
- /scratch/.../high_res_ghi_v3/<SITE>/high_res_ghi_<START>_<END>/heliosat2/heleo_ghi_<SCENE_ID>.tif

Requirements
- Python 3.9+
- pip install: numpy, rasterio, scipy
"""

# ========= EDIT YOUR INPUTS HERE =========
START_DATE = "2023-01-01"   # "YYYY-MM-DD"
END_DATE   = "2023-12-31"   # "YYYY-MM-DD"
SITE       = "{location}"    

SHADOW_INDEX_DIR = f"/path_to_shadow_index/{SITE}_{START_DATE}_{END_DATE}"
CLEAR_SKY_DIR    = f"/path_to_ghi_clear_sky/{SITE}"
OUTPUT_DIR       = f"/path_to_heleosat_output/{SITE}"

MAX_WORKERS = 8
FILL_WINDOW = 5   # neighbourhood size for NaN filling (odd integer)
# =========================================

import re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy.ndimage import generic_filter


def fill_nan_with_nearest(data, size=5):
    """Fill NaNs by local mean of available neighbours (simple, robust)."""
    def nanmean_filter(x):
        vals = x[~np.isnan(x)]
        return np.nanmean(vals) if len(vals) > 0 else np.nan
    return generic_filter(data, nanmean_filter, size=size)


def compute_kc_from_cloud_index(cloud_index):
    """
    Piecewise mapping CI→kc (HeLEOsat2-style).
    Uses the same ranges you provided.
    """
    kc = np.zeros_like(cloud_index, dtype=np.float32)

    m1 = cloud_index < -0.2
    m2 = (cloud_index >= -0.2) & (cloud_index < 0.8)
    m3 = (cloud_index >= 0.8) & (cloud_index < 1.1)
    m4 = cloud_index >= 1.1

    kc[m1] = 1.2
    kc[m2] = 1 - cloud_index[m2]
    kc[m3] = 2.0667 - 3.6667 * cloud_index[m3] + 1.6667 * (cloud_index[m3] ** 2)
    kc[m4] = 0.05

    return kc


def process_scene(shadow_index_path, clear_sky_dir, output_dir):
    scene_id = Path(shadow_index_path).stem
    m = re.search(r"\d{8}", scene_id)
    if m is None:
        print(f"[WARN] No YYYYMMDD in: {scene_id}")
        return
    date_str = m.group(0)

    clear_sky_path = clear_sky_dir / f"ghi_clear_sky_{date_str}.tif"
    out_path = output_dir / f"heleo_ghi_{scene_id}.tif"

    if out_path.exists():
        print(f"[SKIP] {out_path.name}")
        return
    if not clear_sky_path.exists():
        print(f"[MISS] Clear-sky: {clear_sky_path.name}")
        return

    try:
        # Read Shadow Index (CI shifted by parallax)
        with rasterio.open(shadow_index_path) as src_si:
            shadow_index = src_si.read(1).astype(np.float32)
            si_transform = src_si.transform
            si_crs = src_si.crs
            si_shape = shadow_index.shape
            meta = src_si.meta.copy()

        # Read clear-sky GHI
        with rasterio.open(clear_sky_path) as src_cs:
            clear_sky = src_cs.read(1, masked=True).filled(np.nan).astype(np.float32)
            cs_transform = src_cs.transform
            cs_crs = src_cs.crs
            cs_shape = clear_sky.shape

        # Align clear-sky to SI grid if needed
        if cs_shape != si_shape or cs_transform != si_transform or cs_crs != si_crs:
            resampled = np.empty(si_shape, dtype=np.float32)
            reproject(
                source=clear_sky,
                destination=resampled,
                src_transform=cs_transform,
                src_crs=cs_crs,
                dst_transform=si_transform,
                dst_crs=si_crs,
                resampling=Resampling.bilinear
            )
            clear_sky = resampled

        clear_sky_filled = fill_nan_with_nearest(clear_sky, size=FILL_WINDOW)
        kc = compute_kc_from_cloud_index(shadow_index)
        ghi = kc * clear_sky_filled

        # Save
        meta.update(dtype="float32", count=1, compress="deflate")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(ghi.astype(np.float32), 1)
        print(f"[OK] {out_path.name}")

    except Exception as e:
        print(f"[FAIL] {scene_id}: {e}")


def main():
    shadow_dir = Path(SHADOW_INDEX_DIR)
    clear_sky_dir = Path(CLEAR_SKY_DIR)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    shadow_paths = sorted(shadow_dir.glob("*.tif"))
    print(f"[INFO] Scenes to process: {len(shadow_paths)}")

    args = [(p, clear_sky_dir, out_dir) for p in shadow_paths]

    with ProcessPoolExecutor(MAX_WORKERS) as ex:
        futures = [ex.submit(process_scene, *a) for a in args]
        for f in futures:
            f.result()


if __name__ == "__main__":
    main()
