#!/usr/bin/env python3
"""
Cloud Index (Sentinel-2, Band 2)

What this script does
- Scans all Sentinel-2 L1C .SAFE scenes for a site/date window.
- Reads Band 2 (B02) reflectance (scaled by 1/10000) for each scene.
- Computes per-pixel cloud index using precomputed reference stats:
      cloud_index = (p_scene - p_min) / (p_max - p_min)
  with clipping to [0, 1] and NaN where the denominator is zero.

Why it needs the reference files
- p_min GeoTIFF (min_reflectance_B02_<SITE>.tif) provides the canonical grid
  (CRS, transform, shape) and the per-pixel lower bound.
- p_max TXT (max_reflectance_B02_<SITE>.txt, robust max scalar) provides the
  upper bound used for normalisation.

Outputs
- <OUTPUT_DIR>/cloud_index_<SCENE_TIMESTAMP>.tif (float32, aligned to p_min grid)

Requirements
- Python 3.9+
- pip install: numpy, rasterio
- Reference files created beforehand by your Band-2 min/max script
"""

# ========= EDIT YOUR INPUTS HERE =========
START_DATE = "2023-01-01"                         # "YYYY-MM-DD"
END_DATE   = "2023-12-31"                         # "YYYY-MM-DD"
SITE       = "{location}" 

INPUT_DIR  = f"/path_to_sentinel_data/{SITE}_{START_DATE}_{END_DATE}"
OUTPUT_DIR = f"/path_to_cloud_index/{SITE}_{START_DATE}_{END_DATE}"

# Paths to reference stats (produced by your min/max script)
REF_DIR    = f"/path_to_min_max_cloud_index/{SITE}_min_max_cloud_index"
PMIN_TIF   = f"{REF_DIR}/min_reflectance_B02_{SITE}.tif"
PMAX_TXT   = f"{REF_DIR}/max_reflectance_B02_{SITE}.txt"
# =========================================

import os
import glob
import numpy as np
import rasterio


def find_band2_paths(base_dir):
    """
    Find all B02 .jp2 files inside .SAFE scenes under base_dir.
    """
    band2_paths = []
    for safe_dir in sorted(glob.glob(os.path.join(base_dir, "*.SAFE"))):
        granule_dir = os.path.join(safe_dir, "GRANULE")
        if not os.path.isdir(granule_dir):
            continue
        for sub in sorted(os.listdir(granule_dir)):
            img_data_dir = os.path.join(granule_dir, sub, "IMG_DATA")
            if not os.path.isdir(img_data_dir):
                continue
            matches = sorted(glob.glob(os.path.join(img_data_dir, "*_B02*.jp2")))
            band2_paths.extend(matches)
    return band2_paths


def compute_cloud_index(site):
    """
    Compute and save cloud index GeoTIFFs for all scenes.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Read p_min (reference grid) and metadata
    with rasterio.open(PMIN_TIF) as src_min:
        p_min = src_min.read(1).astype(np.float32)
        meta = src_min.meta.copy()
        meta.update(dtype="float32", count=1)

    # Read robust p_max scalar
    p_max = np.loadtxt(PMAX_TXT, dtype=np.float32)

    band2_paths = find_band2_paths(INPUT_DIR)
    print(f"[INFO] Found {len(band2_paths)} B02 files.")

    for path in band2_paths:
        try:
            with rasterio.open(path) as src:
                p_scene = src.read(1).astype(np.float32) / 10000.0

            # cloud_index = (p_scene - p_min) / (p_max - p_min)
            denominator = p_max - p_min
            denominator[denominator == 0] = np.nan
            numerator = np.maximum(0, p_scene - p_min)
            cloud_index = numerator / denominator
            cloud_index = np.clip(cloud_index, 0, 1)

            # Scene timestamp from filename (e.g., *_20230915T104031_*)
            name_parts = os.path.basename(path).split("_")
            if len(name_parts) > 2:
                scene_id = name_parts[1]
            else:
                scene_id = os.path.splitext(os.path.basename(path))[0]

            out_file = os.path.join(OUTPUT_DIR, f"cloud_index_{scene_id}.tif")
            with rasterio.open(out_file, "w", **meta) as dst:
                dst.write(cloud_index.astype(np.float32), 1)

            print(f"[OK] {out_file}")

        except Exception as e:
            print(f"[FAIL] {path}: {e}")


if __name__ == "__main__":
    compute_cloud_index(SITE)
