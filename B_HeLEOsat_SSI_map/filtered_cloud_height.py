#!/usr/bin/env python3
"""
Filtered Cloud Height Map Generator (Sentinel-2)

This script combines Asterisk cloud height outputs with SenSeIv2 cloud masks
to produce filtered cloud height GeoTIFFs:
- Only "thin" and "thick" cloud pixels (classes 1 and 2) retain heights.
- Non-cloud pixels are set to 0.
- Missing cloud heights are filled from nearby valid cloud pixels using k-NN.

Pipeline:
1. Load Asterisk .npz file (coords + heights).
2. Load SenSeIv2 classification mask (.npy).
3. Interpolate heights onto 10 m Sentinel-2 grid.
4. Resize mask to 10 m resolution and filter valid clouds.
5. Fill missing cloud heights (<= 0) from nearest valid cloud pixels.
6. Save result as GeoTIFF aligned to a reference .tif (same CRS + transform).

Outputs
- <OUTPUT_DIR>/<SCENE_ID>.tif (float32, units: metres)

Requirements
- Python 3.9+
- pip install: numpy, rasterio, scipy, pyproj
- Asterisk outputs (.npz) and SenSeIv2 outputs (.npy) must be available
"""

# ========= EDIT YOUR INPUTS HERE =========
# ========= EDIT YOUR INPUTS HERE =========
START_DATE = "2023-01-01"       # "YYYY-MM-DD"
END_DATE   = "2023-12-31"       # "YYYY-MM-DD"
SITE       = "{location}"       # used in output path
EPSG_CODE  = "EPSG:xxxxx"
MAX_WORKERS = 8

# Paths (adapt for your system)
CLOUD_HEIGHT_DIR   = f"/path_to_cloud_height/{SITE}_{START_DATE}_{END_DATE}"
CLASSIFICATION_DIR = f"/path_to_cloud_segmentation/{SITE}_{START_DATE}_{END_DATE}"
REFERENCE_TIF_PATH = f"/path_to_min_cloud_index/min_reflectance_B02_{SITE}.tif"
OUTPUT_DIR         = f"/path_to_filtered_cloud_height/{SITE}_{START_DATE}_{END_DATE}"
# =========================================

import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import rasterio
from scipy.ndimage import zoom
from scipy.spatial import cKDTree


def save_filtered_cloud_height_tif(npz_path, classification_npy_path, reference_tif_path, output_path, epsg, pixel_size=10):
    """
    Save a GeoTIFF with cloud heights only for thin/thick cloud pixels.
    Other pixels are set to 0.
    """
    # Load Asterisk data
    data = np.load(npz_path)
    coords = data["coords"]
    heights = data["heights"]

    # Load classification mask
    classification_mask = np.load(classification_npy_path)

    # Sentinel-2 10 m grid
    rgb_shape = (10980, 10980)
    nrows, ncols = rgb_shape

    # Interpolate Asterisk heights onto full grid (k=3 nearest-neighbour MEAN)
    grid_y, grid_x = np.meshgrid(np.arange(nrows), np.arange(ncols), indexing="ij")
    grid_coords = np.stack([grid_x.ravel() * pixel_size, grid_y.ravel() * pixel_size], axis=1)
    tree_pts = cKDTree(coords)
    k = int(min(3, coords.shape[0]))
    dist, idx = tree_pts.query(grid_coords, k=k)
    if k == 1:
        # ensure 2D for consistent mean over axis=1
        idx = idx[:, None]
    neighbor_vals = heights[idx]  # shape: (n_pixels, k)
    cloud_height_raster = neighbor_vals.mean(axis=1).astype(heights.dtype).reshape(rgb_shape)

    # Resize classification mask to 10 m
    factor_y = nrows / classification_mask.shape[0]
    factor_x = ncols / classification_mask.shape[1]
    classification_resized = zoom(classification_mask, (factor_y, factor_x), order=0)
    if classification_resized.shape != rgb_shape:
        raise ValueError("Resized classification mask does not match Sentinel-2 grid shape.")

    # Valid cloud pixels = thin (1) or thick (2)
    valid_cloud = np.isin(classification_resized, [1, 2])

    # Fill missing cloud heights (<= 0) from nearest neighbours
    src_r, src_c = np.where(valid_cloud & (cloud_height_raster > 0))
    tgt_r, tgt_c = np.where(valid_cloud & (cloud_height_raster <= 0))

    if src_r.size > 0 and tgt_r.size > 0:
        src_pts = np.column_stack([src_r, src_c])
        tgt_pts = np.column_stack([tgt_r, tgt_c])
        kdt = cKDTree(src_pts)
        k2 = int(min(3, src_pts.shape[0]))
        dist2, idx2 = kdt.query(tgt_pts, k=k2)

        if k2 == 1:
            idx2 = idx2[:, None]

        src_heights_1d = cloud_height_raster[src_r, src_c]
        neighbor_vals2 = src_heights_1d[idx2]
        filled_vals = neighbor_vals2.mean(axis=1).astype(cloud_height_raster.dtype)

        cloud_height_raster[tgt_r, tgt_c] = filled_vals

    # Apply mask: only keep cloud pixels
    filtered_heights = np.where(valid_cloud, cloud_height_raster, 0)

    # Save GeoTIFF
    with rasterio.open(reference_tif_path) as ref:
        meta = ref.meta.copy()
        meta.update(dtype="float32", count=1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(filtered_heights.astype(np.float32), 1)

    print(f"[OK] Saved: {output_path}")


def process_pair(npz_path, mask_path, reference_tif_path, output_path, epsg):
    scene_id = os.path.basename(npz_path).replace(".npz", ".tif")
    out_file = os.path.join(output_path, scene_id)

    if os.path.exists(out_file):
        print(f"[SKIP] {scene_id} already exists")
        return

    try:
        save_filtered_cloud_height_tif(
            npz_path=npz_path,
            classification_npy_path=mask_path,
            reference_tif_path=reference_tif_path,
            output_path=out_file,
            epsg=epsg,
        )
    except Exception as e:
        print(f"[FAIL] {npz_path}: {e}")


def main():
    cloud_height_dir = Path(CLOUD_HEIGHT_DIR)
    classification_dir = Path(CLASSIFICATION_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_pairs = []
    for npz_file in sorted(cloud_height_dir.glob("*.npz")):
        base = npz_file.stem
        mask_file = classification_dir / f"{base}_mask.npy"
        if mask_file.exists():
            file_pairs.append((str(npz_file), str(mask_file)))

    print(f"[INFO] Found {len(file_pairs)} matching pairs.")

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(process_pair, npz, mask, REFERENCE_TIF_PATH, str(output_dir), EPSG_CODE)
            for npz, mask in file_pairs
        ]
        for f in futures:
            f.result()


if __name__ == "__main__":
    main()
