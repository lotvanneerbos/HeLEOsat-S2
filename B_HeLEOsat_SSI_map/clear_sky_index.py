#!/usr/bin/env python3
"""
Clear-Sky GHI (CAMS) → Sentinel-2 Grid

What this script does
- For every Sentinel-2 .SAFE scene of a site/date window:
  1) Reads the scene's SENSING_TIME from MTD_TL.xml (rounded to the nearest minute).
  2) Looks up the matching CAMS minute record (GHI_clear_sky) across all downloaded CAMS CSVs.
  3) Nearest-neighbour matches CAMS points to precomputed Sentinel-2 pixel coordinates.
  4) Writes a GeoTIFF on the *site reference grid* (copying metadata from your min-B02 GeoTIFF).

Inputs expected on disk
- Sentinel-2 scenes: {SENTINEL_BASE}/*.SAFE
- CAMS CSVs (minute): {CAMS_FOLDER}/cams_*_<lat>_<lon>.csv
- Sentinel-2 pixel coords CSV: {COORDS_FILE} with columns Latitude, Longitude
- Reference GeoTIFF: {REFERENCE_TIF_PATH} (your min_reflectance_B02_<SITE>.tif)

Outputs
- /scratch/.../ghi_clear_sky/<SITE>/ghi_clear_sky_<YYYYMMDD>.tif

Requirements
- Python 3.9+
- pip install: numpy, pandas, rasterio, scipy, pyproj
"""

# ========= EDIT YOUR INPUTS HERE =========
START_DATE = "2023-01-01"  # "YYYY-MM-DD"
END_DATE   = "2023-12-31"  # "YYYY-MM-DD"
SITE       = "{location}"
EPSG_CODE  = "EPSG:xxxxx"

SENTINEL_BASE       = f"/path_to_sentinel_data/{SITE}_{START_DATE}_{END_DATE}"
CAMS_FOLDER         = f"/output_path/.../cams_downloads_{SITE}/"
COORDS_FILE         = f"/path_to_sentinel2_pixel_coords/{SITE}.csv"
REFERENCE_TIF_PATH  = f"path_to_min_max_cloud_index/{SITE}_min_max_cloud_index/min_reflectance_B02_{SITE}.tif"

OUTPUT_DIR          = f"/path_to_ghi_clear_sky/{SITE}"
MAX_WORKERS         = 10
# =========================================

import os
import xml.etree.ElementTree as ET
import pandas as pd
from scipy.spatial import cKDTree
import rasterio
from rasterio.transform import rowcol
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from pyproj import Transformer


def get_sensing_time(safe_dir):
    """Return SENSING_TIME ('YYYY-MM-DDTHH:MM:SS') from the scene's MTD_TL.xml."""
    granule_path = os.path.join(safe_dir, "GRANULE")
    subdirs = os.listdir(granule_path)
    matching = [d for d in subdirs if "L1C" in d]
    if not matching:
        raise ValueError(f"No GRANULE subdir containing 'L1C' found in {granule_path}")

    mtd_path = os.path.join(granule_path, matching[0], "MTD_TL.xml")
    root = ET.parse(mtd_path).getroot()
    for elem in root.iter():
        if "SENSING_TIME" in elem.tag:
            t = elem.text
            if t is None:
                raise ValueError(f"SENSING_TIME tag empty in {mtd_path}")
            t = t.strip().replace("Z", "")
            t = t.split(".")[0]  # drop fractional seconds
            return t
    raise ValueError(f"SENSING_TIME not found in {mtd_path}")


def extract_ghi_clear_sky(target_time, cams_folder):
    """
    Read CAMS minute CSVs and return rows matching target_time (pd.Timestamp).
    Expected CSV col 0 like 'YYYY-MM-DDTHH:MM:SS.sss/...' ; GHI_clear_sky in col 2.
    """
    rows = []
    for fname in os.listdir(cams_folder):
        if not fname.endswith(".csv"):
            continue
        parts = fname[:-4].split("_")
        lat = parts[-2]
        lon = parts[-1]
        path = os.path.join(cams_folder, fname)

        with open(path, "r") as f:
            lines = f.readlines()
        data_start = next(i for i, line in enumerate(lines) if not line.startswith("#"))
        df = pd.read_csv(path, skiprows=data_start, header=None, sep=";")
        start_str = df[0].str.extract(r"(^[^/]+)")[0]
        df["start_time"] = pd.to_datetime(start_str, errors="coerce", format="%Y-%m-%dT%H:%M:%S.%f")

        row = df[df["start_time"] == target_time]
        if not row.empty:
            ghi_cs = row.iloc[0, 2]
            rows.append({"latitude": lat, "longitude": lon, "ghi_clear_sky": ghi_cs})
    return pd.DataFrame(rows)


def match_ghi_clear_sky_to_pixels(ghi_df, sentinel_coords_file):
    """
    Nearest-neighbour match CAMS points to Sentinel-2 pixel centres.
    Returns matched GHI array and (lat,lon) coordinates used for mapping.
    """
    s2 = pd.read_csv(sentinel_coords_file)
    s2_coords = s2[["Latitude", "Longitude"]].values
    cams_coords = ghi_df[["latitude", "longitude"]].astype(float).values
    ghi_vals = ghi_df["ghi_clear_sky"].values
    tree = cKDTree(cams_coords)
    _, idx = tree.query(s2_coords)
    return ghi_vals[idx], s2_coords


def save_ghi_clear_sky_tiff(matched_ghi, sentinel_coords, target_time_short, reference_tiff_path, epsg):
    """
    Write a GeoTIFF on the reference grid; fills only pixels for which a matched GHI is available.
    """
    transformer = Transformer.from_crs("EPSG:4326", epsg, always_xy=True)
    with rasterio.open(reference_tiff_path) as ref:
        meta = ref.meta.copy()
        transform = ref.transform
        height = ref.height
        width = ref.width
        meta.update(dtype="float32", count=1)

    out = np.full((height, width), np.nan, dtype=np.float32)

    for (lat, lon), ghi_val in zip(sentinel_coords, matched_ghi):
        try:
            x, y = transformer.transform(lon, lat)
            r, c = rowcol(transform, x, y)
            if 0 <= r < height and 0 <= c < width:
                out[r, c] = ghi_val
        except Exception:
            pass

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"ghi_clear_sky_{target_time_short}.tif")
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(out, 1)
    print(f"[OK] {out_path}")


def process_scene(scene, site, base_folder, cams_folder, coords_file, reference_tiff_path, epsg):
    try:
        scene_path = os.path.join(base_folder, scene)

        sensing_time_raw = get_sensing_time(scene_path)
        sensing_time = pd.to_datetime(sensing_time_raw).round("1min")
        target_time_short = sensing_time.strftime("%Y%m%d")

        out_path = os.path.join(OUTPUT_DIR, f"ghi_clear_sky_{target_time_short}.tif")
        if os.path.exists(out_path):
            print(f"[SKIP] {out_path}")
            return

        ghi_df = extract_ghi_clear_sky(sensing_time, cams_folder)
        if ghi_df.empty:
            print(f"[MISS] No CAMS match for {scene}")
            return

        matched_ghi, s2_coords = match_ghi_clear_sky_to_pixels(ghi_df, coords_file)
        save_ghi_clear_sky_tiff(matched_ghi, s2_coords, target_time_short, reference_tiff_path, epsg)

    except Exception as e:
        print(f"[FAIL] {scene}: {e}")


def process_all_scenes_parallel(site, epsg):
    base_folder = SENTINEL_BASE
    cams_folder = CAMS_FOLDER
    coords_file = COORDS_FILE
    reference_tiff_path = REFERENCE_TIF_PATH

    scenes = [d for d in os.listdir(base_folder) if d.endswith(".SAFE")]
    with ProcessPoolExecutor(MAX_WORKERS) as ex:
        futures = [
            ex.submit(
                process_scene, scene, site, base_folder, cams_folder, coords_file, reference_tiff_path, epsg
            )
            for scene in scenes
        ]
        for f in futures:
            f.result()


if __name__ == "__main__":
    process_all_scenes_parallel(SITE, EPSG_CODE)

