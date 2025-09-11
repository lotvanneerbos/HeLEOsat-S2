#!/usr/bin/env python3
"""
Asterisk Cloud Height Estimation (Sentinel-2)

This script runs the Asterisk algorithm (cloud_height_prototype_main) on Sentinel-2
scenes to estimate cloud top heights. Each processed scene is saved as a compressed
NumPy file (.npz) containing:
    - heights: estimated cloud heights (metres above ground)
    - coords:  pixel coordinates (in Sentinel-2 grid space)
    - times:   acquisition times (as byte strings)

Steps:
1. Find Sentinel-2 .SAFE scenes for the given site/date range.
2. Run Asterisk on each scene (parallelised with ProcessPoolExecutor).
3. Save results to `/scratch/.../cloud_height/{site}/`.

Requirements:
- Python 3.9+
- numpy, opencv-python (cv2), shapely
- cloud_height_prototype_main (with CloudHeightConfig + processScene)
"""

# ========= EDIT YOUR INPUTS HERE =========
START_DATE = "2023-01-01"       # "YYYY-MM-DD"
END_DATE   = "2023-12-31"       # "YYYY-MM-DD"
SITE       = "{location}"       # used in output path

SENTINEL_BASE = f"/path_to_sentinel_data/{SITE}_{START_DATE}_{END_DATE}" 
OUTPUT_DIR    = f"/path_to_cloud_height/{SITE}_{START_DATE}_{END_DATE}" 

MAX_WORKERS = 6   # number of parallel processes
# =========================================

import os
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from cloud_height_prototype_main.src.config import CloudHeightConfig
from cloud_height_prototype_main.src.main import processScene
import argparse
import cv2


def get_qidata_path(safe_dir):
    granule_path = os.path.join(safe_dir, "GRANULE")
    if not os.path.isdir(granule_path):
        raise FileNotFoundError(f"GRANULE folder not found in {safe_dir}")

    subdirs = [d for d in os.listdir(granule_path) if "L1C" in d]
    if not subdirs:
        raise FileNotFoundError("No L1C subdirectories found in GRANULE folder")

    qidata_path = os.path.join(granule_path, subdirs[0], "QI_DATA")
    if not os.path.exists(qidata_path):
        raise FileNotFoundError(f"QI_DATA not found at expected path: {qidata_path}")

    return qidata_path


def estimate_dominant_detector_azimuth(qi_data_dir, band_id=4):
    jp2_path = os.path.join(qi_data_dir, f"MSK_DETFOO_B{band_id:02d}.jp2")
    if not os.path.exists(jp2_path):
        raise FileNotFoundError(f"Detector mask not found: {jp2_path}")

    img = cv2.imread(jp2_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {jp2_path}")

    img = img.astype(np.float32)
    img -= img.min()
    img /= (img.max() if img.max() > 0 else 1)
    img = (img * 255).astype(np.uint8)

    edges = cv2.Canny(img, 100, 200)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=300)
    if lines is None:
        raise ValueError("No lines detected in detector mask")

    angles_deg = [np.rad2deg(line[0][1]) % 180 for line in lines]
    dominant_angle_deg = np.median(angles_deg)
    return np.deg2rad(-dominant_angle_deg)


def run_scene(scene_dir, output_dir):
    """Run Asterisk on one scene and save cloud heights as .npz."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, help="Path to the configuration file")
    args = parser.parse_args([])  # ignore CLI, we set args manually

    try:
        scene_id = Path(scene_dir).stem.replace(".SAFE", "")
        output_file = os.path.join(output_dir, f"{scene_id}.npz")

        # Config with override
        config = CloudHeightConfig(args.config, override_scene_dir=scene_dir)

        qi_path = get_qidata_path(scene_dir)
        hack_azimuth = estimate_dominant_detector_azimuth(qi_path)
        config.HACK_IMAGE_AZIMUTH = hack_azimuth

        print(f"[RUN] {scene_id}")

        # Run Asterisk algorithm
        final_heights, final_coords, times = processScene(config)

        # Save as .npz
        np.savez_compressed(
            output_file,
            heights=final_heights,
            coords=final_coords,
            times=np.array(times, dtype="S"),
        )

        print(f"[OK] {scene_id} → {output_file}")
    except Exception as e:
        print(f"[FAIL] {scene_dir}: {e}")


def run_all_asterisk():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    scene_dirs = [
        os.path.join(SENTINEL_BASE, d)
        for d in os.listdir(SENTINEL_BASE)
        if d.endswith(".SAFE")
    ]

    # Skip already processed
    filtered_scenes = []
    for scene_dir in scene_dirs:
        scene_id = Path(scene_dir).stem.replace(".SAFE", "")
        output_file = os.path.join(OUTPUT_DIR, f"{scene_id}.npz")
        if not os.path.exists(output_file):
            filtered_scenes.append(scene_dir)
        else:
            print(f"[SKIP] {scene_id} already processed")

    # Run parallel
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(run_scene, scene_dir, OUTPUT_DIR) for scene_dir in filtered_scenes]
        for f in futures:
            f.result()


if __name__ == "__main__":
    run_all_asterisk()
