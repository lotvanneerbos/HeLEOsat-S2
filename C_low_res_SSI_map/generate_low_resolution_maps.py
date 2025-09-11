#!/usr/bin/env python3
"""
Low-resolution GHI map generator with NaN-edge logging

Flow
1) For each Sentinel-2 scene: read sensing time (rounded to minute).
2) Extract CAMS GHI at that time (from per-point CAMS CSVs).
3) Map CAMS -> Sentinel-2 pixels via method: "euclidean", "manhattan", or "linear".
4) Build a raster on the Sentinel-2 reference grid.
5) BEFORE filling: count NaN margins (top/bottom/left/right) and write a per-scene CSV log.
6) Fill NaNs by nearest and save the final GeoTIFF.
"""

import os
import xml.etree.ElementTree as ET
import pandas as pd
from scipy.spatial import cKDTree
import rasterio
from rasterio.transform import rowcol
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from pyproj import Transformer
from scipy.interpolate import LinearNDInterpolator
from pathlib import Path
from scipy.ndimage import distance_transform_edt

# ======= USER INPUTS =======
SITE                 = "{location}"
EPSG                 = "EPSG:32631"   # CRS of the S2 tile
MATCH_METHOD         = "linear"       # "euclidean" | "manhattan" | "linear"
MAX_WORKERS          = 5              # this can be computational expensive

BASE_FOLDER          = "path_to_sentinel_data/.../Sentinel2_Data_2023-01-01_2023-12-31/"
CAMS_FOLDER          = "/path_to_cams_data/.../cams_downloads_folder/"
S2_COORDS_CSV        = "/path_to_sentinel_data/.../sentinel2_pixel_coords_{SITE}.csv"
REFERENCE_TIFF_PATH  = "/path_to_sentinel_data_in_tiff_file/..../your_reference.tif"
OUT_FOLDER           = "/path_to_output_folder/.../"
# ===========================

NAN_LOG_DIR          = f"{OUT_FOLDER}/nan_logs"

def fill_nan_with_nearest(data):
    mask = np.isnan(data)
    if not np.any(mask):
        return data
    idx = distance_transform_edt(mask, return_distances=False, return_indices=True)
    return data[tuple(idx)]

def count_nan_margins(arr):
    nan_mask = np.isnan(arr)
    nrows, ncols = arr.shape

    rows_all_nan = nan_mask.all(axis=1)
    cols_all_nan = nan_mask.all(axis=0)

    top_idx = np.where(~rows_all_nan)[0]
    remove_top = int(top_idx[0]) if top_idx.size else nrows

    bottom_idx = np.where(~rows_all_nan)[0]
    remove_bottom = int((nrows - 1) - bottom_idx[-1]) if bottom_idx.size else nrows

    left_idx = np.where(~cols_all_nan)[0]
    remove_left = int(left_idx[0]) if left_idx.size else ncols

    right_idx = np.where(~cols_all_nan)[0]
    remove_right = int((ncols - 1) - right_idx[-1]) if right_idx.size else ncols

    frac_nan = float(np.isnan(arr).sum()) / float(nrows * ncols)
    all_nan = bool(frac_nan == 1.0)

    return {
        "remove_top_rows": remove_top,
        "remove_bottom_rows": remove_bottom,
        "remove_left_cols": remove_left,
        "remove_right_cols": remove_right,
        "nrows": nrows,
        "ncols": ncols,
        "frac_nan": frac_nan,
        "all_nan": all_nan,
    }

def get_sensing_time(safe_file):
    granule_path = os.path.join(safe_file, "GRANULE")
    subdirs = os.listdir(granule_path)
    matching_dirs = [d for d in subdirs if "L1C" in d]
    mtd_path = os.path.join(granule_path, matching_dirs[0], "MTD_TL.xml")
    root = ET.parse(mtd_path).getroot()
    for elem in root.iter():
        if "SENSING_TIME" in elem.tag:
            return elem.text.strip().split(".")[0]

def extract_ghi(target_time_iso, cams_folder):
    out = []
    for fname in os.listdir(cams_folder):
        if not fname.endswith(".csv"):
            continue
        parts = fname[:-4].split("_")
        try:
            lat = float(parts[-2]); lon = float(parts[-1])
        except:
            continue
        file_path = os.path.join(cams_folder, fname)
        with open(file_path, "r") as f:
            lines = f.readlines()
        try:
            data_start = next(i for i, line in enumerate(lines) if not line.startswith("#"))
        except StopIteration:
            continue
        df = pd.read_csv(file_path, skiprows=data_start, header=None, sep=";")
        df["start_time"] = df[0].str.extract(r"(^[^/]+)").str.rstrip("Z")
        df["start_time"] = pd.to_datetime(df["start_time"], format="%Y-%m-%dT%H:%M:%S.%f", errors="coerce")
        df = df.dropna(subset=["start_time"])
        row = df[df["start_time"] == pd.to_datetime(target_time_iso)]
        if not row.empty:
            ghi = float(row.iloc[0, 6])
            out.append({"latitude": lat, "longitude": lon, "ghi": ghi})
    return pd.DataFrame(out)

def match_ghi_to_pixels(ghi_df, sentinel_coords_file, method):
    s2 = pd.read_csv(sentinel_coords_file)
    sentinel_coords = s2[["Latitude", "Longitude"]].values
    cams_coords = ghi_df[["latitude", "longitude"]].values
    ghi_values = ghi_df["ghi"].values

    if method == "euclidean":
        tree = cKDTree(cams_coords)
        _, idx = tree.query(sentinel_coords)
        matched = ghi_values[idx]
    elif method == "manhattan":
        matched = []
        for s_lat, s_lon in sentinel_coords:
            d = np.abs(cams_coords[:, 0] - s_lat) + np.abs(cams_coords[:, 1] - s_lon)
            matched.append(ghi_values[np.argmin(d)])
        matched = np.array(matched, dtype=float)
    elif method == "linear":
        interp = LinearNDInterpolator(cams_coords, ghi_values)
        matched = interp(sentinel_coords)
        if np.any(np.isnan(matched)):
            nan_mask = np.isnan(matched)
            tree = cKDTree(cams_coords)
            _, idx = tree.query(sentinel_coords[nan_mask])
            matched[nan_mask] = ghi_values[idx]
    else:
        raise ValueError("MATCH_METHOD must be 'euclidean', 'manhattan', or 'linear'")

    return matched, sentinel_coords

def build_raster_from_points(matched_ghi, sentinel_coords, reference_tiff_path, epsg):
    transformer = Transformer.from_crs("EPSG:4326", epsg, always_xy=True)
    with rasterio.open(reference_tiff_path) as ref:
        meta = ref.meta.copy()
        transform = ref.transform
        width, height = ref.width, ref.height
        meta.update(dtype="float32", count=1)
    arr = np.full((height, width), np.nan, dtype=np.float32)
    for (lat, lon), val in zip(sentinel_coords, matched_ghi):
        try:
            x, y = transformer.transform(lon, lat)
            row, col = rowcol(transform, x, y)
            if 0 <= row < height and 0 <= col < width:
                arr[row, col] = val
        except:
            continue
    return arr, meta

def save_tiff(array, meta, out_path):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(array, 1)

def write_nan_log(nan_stats, out_dir, scene_id, site, method, target_time_iso):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    rec = dict(nan_stats)
    rec.update({
        "site": site,
        "scene_id": scene_id,
        "method": method,
        "time_iso": target_time_iso,
    })
    pd.DataFrame([rec]).to_csv(Path(out_dir) / f"nan_edges_{scene_id}.csv", index=False)

def process_scene(scene):
    try:
        print(f"[INFO] Processing: {scene}")
        scene_path = os.path.join(BASE_FOLDER, scene)
        scene_id = scene.replace(".SAFE", "")
        sensing_time_min = pd.to_datetime(get_sensing_time(scene_path)).round("1min")
        time_short = sensing_time_min.strftime("%Y%m%d")
        time_iso = sensing_time_min.strftime("%Y-%m-%dT%H:%M:%S")

        out_filled = Path(OUT_FOLDER) / f"low_res_ghi_{time_short}.tif"
        if out_filled.exists():
            print(f"[SKIP] Exists: {out_filled}")
            return

        ghi_df = extract_ghi(time_iso, CAMS_FOLDER)
        if ghi_df.empty:
            print(f"[NOTE] No CAMS GHI at {time_iso} for {scene}")
            return

        matched, s2_coords = match_ghi_to_pixels(ghi_df, S2_COORDS_CSV, MATCH_METHOD)
        arr, meta = build_raster_from_points(matched, s2_coords, REFERENCE_TIFF_PATH, EPSG)

        nan_stats = count_nan_margins(arr)
        write_nan_log(nan_stats, NAN_LOG_DIR, scene_id, SITE, MATCH_METHOD, time_iso)

        arr_filled = fill_nan_with_nearest(arr)
        save_tiff(arr_filled, meta, out_filled)
        print(f"[OK ] Filled GeoTIFF: {out_filled}")

    except Exception as e:
        print(f"[ERR] {scene} -> {e}")

def process_all_scenes_parallel():
    scenes = [f for f in os.listdir(BASE_FOLDER) if f.endswith(".SAFE")]
    with ProcessPoolExecutor(MAX_WORKERS) as ex:
        for _ in ex.map(process_scene, scenes):
            pass

if __name__ == "__main__":
    process_all_scenes_parallel()
