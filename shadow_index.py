#!/usr/bin/env python3
"""
Shadow Index (parallax-shifted cloud index, Sentinel-2)

What this script does
- Reads per-scene Cloud Index (CI) GeoTIFFs and filtered cloud-height (CH) GeoTIFFs.
- Extracts mean satellite viewing and mean sun angles from each scene’s MTD_TL.xml.
- Shifts CI per pixel by a parallax vector derived from CH and the angles (sat + sun).
- Accumulates contributions into a continuous Shadow Index on the same grid as CI.
- Fills any holes by nearest-neighbour copy from the closest valid shadow value.

Outputs
- <OUTPUT_DIR>/<SCENE_ID>.tif  (float32, same CRS/transform/shape as the Cloud Index)

Assumptions / notes
- CI and CH rasters have identical shape/geo-transform (aligned to your site reference grid).
- Angles are taken from MTD_TL.xml (mean viewing incidence per band, and mean sun angle).
- This implementation is deliberately simple and readable; pixel loops are slow on full 10 m grids.
  If needed, vectorise or process in tiles/chunks later.

Requirements
- Python 3.9+
- pip install: numpy, rasterio, scipy
- Sentinel-2 .SAFE directories present to read MTD_TL.xml
"""

# ========= EDIT YOUR INPUTS HERE =========
START_DATE = "2023-01-01"   # "YYYY-MM-DD"
END_DATE   = "2023-12-31"   # "YYYY-MM-DD"
SITE       = "{location}" 

# Paths used by the pipeline
FILTERED_CLOUD_HEIGHT_DIR = f"/path_to_filtered_cloud_height/{SITE}_{START_DATE}_{END_DATE}"
CLOUD_INDEX_DIR           = f"/path_to_cloud_index/{SITE}_{START_DATE}_{END_DATE}"
OUTPUT_DIR                = f"/path_to_shadow_index/{SITE}_{START_DATE}_{END_DATE}"
SENTINEL_SCENES_DIR       = f"/path_to_sentinel_data/{SITE}_{START_DATE}_{END_DATE}"

MAX_WORKERS = 8
# =========================================

import os
import re
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
import rasterio
from rasterio.transform import rowcol
from glob import glob
from concurrent.futures import ProcessPoolExecutor
from scipy.spatial import cKDTree


def get_mtd_tl_path(safe_dir):
    """Return full path to MTD_TL.xml inside a .SAFE directory."""
    granule_path = os.path.join(safe_dir, "GRANULE")
    if not os.path.isdir(granule_path):
        raise FileNotFoundError(f"GRANULE folder not found in {safe_dir}")

    subdirs = os.listdir(granule_path)
    if not subdirs:
        raise FileNotFoundError("No subdirectories in GRANULE folder")

    matching = [d for d in subdirs if "L1C" in d]
    if not matching:
        raise FileNotFoundError("No L1C GRANULE subdirectory found")
    mtd_tl_path = os.path.join(granule_path, matching[0], "MTD_TL.xml")
    if not os.path.exists(mtd_tl_path):
        raise FileNotFoundError(f"MTD_TL.xml not found at {mtd_tl_path}")
    return mtd_tl_path


def extract_mean_viewing_angles(mtd_tl_path):
    """Return (mean_zenith_deg, mean_azimuth_deg) averaged over bands 0..12."""
    tree = ET.parse(mtd_tl_path)
    root = tree.getroot()
    zeniths = []
    azimuths = []
    for angle in root.iter("Mean_Viewing_Incidence_Angle"):
        band_id = angle.attrib.get("bandId")
        if band_id is not None and band_id.isdigit() and 0 <= int(band_id) <= 12:
            try:
                zeniths.append(float(angle.find("ZENITH_ANGLE").text))
                azimuths.append(float(angle.find("AZIMUTH_ANGLE").text))
            except Exception:
                pass
    if not zeniths or not azimuths:
        raise ValueError("No valid viewing angles found for bands 0..12.")
    return np.mean(zeniths), np.mean(azimuths)


def extract_sun_angles(mtd_tl_path):
    """Return (sun_zenith_deg, sun_azimuth_deg) from Mean_Sun_Angle."""
    tree = ET.parse(mtd_tl_path)
    root = tree.getroot()
    for sun_angle in root.iter("Mean_Sun_Angle"):
        return float(sun_angle.find("ZENITH_ANGLE").text), float(sun_angle.find("AZIMUTH_ANGLE").text)
    raise ValueError("Mean_Sun_Angle not found in MTD_TL.xml")


def shift_cloud_index_by_parallax(cloud_index_path, cloud_height_path, output_path, scene_path):
    """
    Shift Cloud Index per pixel using cloud height and parallax from sat + sun.
    Writes a continuous Shadow Index on the same grid as the Cloud Index.
    """
    mtd_path = get_mtd_tl_path(scene_path)
    sat_zenith_deg, sat_azimuth_deg = extract_mean_viewing_angles(mtd_path)
    sun_zenith_deg, sun_azimuth_deg = extract_sun_angles(mtd_path)

    # Read rasters
    with rasterio.open(cloud_index_path) as src_ci:
        cloud_index = src_ci.read(1)
        profile = src_ci.profile
        transform = src_ci.transform
        width = src_ci.width
        height = src_ci.height

    with rasterio.open(cloud_height_path) as src_h:
        cloud_height = src_h.read(1)

    if cloud_index.shape != cloud_height.shape:
        raise ValueError("Cloud Index and Cloud Height shapes do not match.")

    # Angles (radians)
    sat_zenith_rad = np.radians(sat_zenith_deg)
    sat_azimuth_rad = np.radians(sat_azimuth_deg)
    sun_zenith_rad = np.radians(sun_zenith_deg)
    sun_azimuth_rad = np.radians(sun_azimuth_deg)

    shadow_index = np.zeros_like(cloud_index, dtype=np.float32)
    weight_map = np.zeros_like(cloud_index, dtype=np.float32)

    # Naive per-pixel loop (simple & readable; may be slow on full-res)
    for row in range(height):
        for col in range(width):
            ci = cloud_index[row, col]
            ch = cloud_height[row, col]
            if ci == 0 or ch == 0 or np.isnan(ci) or np.isnan(ch):
                continue

            x, y = transform * (col, row)

            dx_sat = ch * np.tan(sat_zenith_rad) * np.sin(sat_azimuth_rad)
            dy_sat = ch * np.tan(sat_zenith_rad) * np.cos(sat_azimuth_rad)
            dx_sun = -ch * np.tan(sun_zenith_rad) * np.sin(sun_azimuth_rad)
            dy_sun = -ch * np.tan(sun_zenith_rad) * np.cos(sun_azimuth_rad)

            dx_total = dx_sat + dx_sun
            dy_total = dy_sat + dy_sun

            x_shifted = x + dx_total
            y_shifted = y + dy_total

            new_row, new_col = rowcol(transform, x_shifted, y_shifted)

            if 0 <= new_row < height and 0 <= new_col < width:
                shadow_index[new_row, new_col] += ci
                weight_map[new_row, new_col] += 1

    # Fill holes by nearest neighbour copy
    holes = weight_map == 0
    if np.any(holes):
        valid_coords = np.column_stack(np.where(weight_map > 0))
        if valid_coords.size > 0:
            target_coords = np.column_stack(np.where(holes))
            tree = cKDTree(valid_coords)
            _, nearest_idxs = tree.query(target_coords)
            for i, (r, c) in enumerate(target_coords):
                nr, nc = valid_coords[nearest_idxs[i]]
                shadow_index[r, c] = shadow_index[nr, nc]

    # Save GeoTIFF
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    profile.update(dtype="float32", count=1, compress="deflate")
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(shadow_index.astype(np.float32), 1)

    print(f"[OK] Shadow index → {output_path}")


def process_scene(filtered_path, cloud_index_dir, output_dir, scene_dir):
    scene_id = Path(filtered_path).stem
    m = re.search(r"\d{8}T\d{6}", scene_id)
    if m is None:
        print(f"[WARN] Could not parse timestamp from: {scene_id}")
        return
    timestamp = m.group(0)

    cloud_index_path = cloud_index_dir / f"cloud_index_{timestamp}.tif"
    out_path = output_dir / f"{scene_id}.tif"
    scene_matches = glob(str(scene_dir / f"*{scene_id}*.SAFE"))

    if out_path.exists():
        print(f"[SKIP] {out_path.name}")
        return
    if not cloud_index_path.exists():
        print(f"[MISS] Cloud Index: {cloud_index_path}")
        return
    if not scene_matches:
        print(f"[MISS] No .SAFE for {scene_id}")
        return

    try:
        shift_cloud_index_by_parallax(
            cloud_index_path=cloud_index_path,
            cloud_height_path=filtered_path,
            output_path=out_path,
            scene_path=scene_matches[0],
        )
    except Exception as e:
        print(f"[FAIL] {scene_id}: {e}")


def main():
    filtered_dir = Path(FILTERED_CLOUD_HEIGHT_DIR)
    ci_dir = Path(CLOUD_INDEX_DIR)
    out_dir = Path(OUTPUT_DIR)
    scene_dir = Path(SENTINEL_SCENES_DIR)

    out_dir.mkdir(parents=True, exist_ok=True)

    filtered_paths = sorted(filtered_dir.glob("*.tif"))
    print(f"[INFO] Scenes to process: {len(filtered_paths)}")

    args = [(fp, ci_dir, out_dir, scene_dir) for fp in filtered_paths]

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(process_scene, *a) for a in args]
        for f in futures:
            f.result()


if __name__ == "__main__":
    main()
