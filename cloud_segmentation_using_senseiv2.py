#!/usr/bin/env python3
"""
SenSeIv2 Cloud Classification (Sentinel-2)

This script runs the SenSeIv2 model on Sentinel-2 .SAFE scenes to classify clouds.
For each scene, it produces a per-pixel cloud mask and saves it as a NumPy .npy file.

Outputs
- <OUTPUT_DIR>/<SCENE_ID>_mask.npy  (uint8, class index per pixel)

Requirements
- Python 3.9+
- pip install: torch, numpy
- SenSeIv2 repo installed (CloudMask, SENTINEL2_DESCRIPTORS, sentinel2_loader)
"""

# ========= EDIT YOUR INPUTS HERE =========
START_DATE = "2023-01-01"       # "YYYY-MM-DD"
END_DATE   = "2023-12-31"       # "YYYY-MM-DD"
SITE       = "{location}"       # used in output path

INPUT_DIR  = f"/path_to_sentinel_data/{SITE}_{START_DATE}_{END_DATE}" 
OUTPUT_DIR = f"/path_to_cloud_segmentation/{SITE}_{START_DATE}_{END_DATE}"

MODEL_NAME = "SEnSeIv2-SegFormerB2-alldata-ambiguous"
DEVICE     = "cpu"              # keep simple: only cpu
# =========================================

import numpy as np
from pathlib import Path
from SEnSeIv2.senseiv2.inference import CloudMask, sentinel2_loader
from SEnSeIv2.senseiv2.constants import SENTINEL2_DESCRIPTORS
from SEnSeIv2.senseiv2.utils import get_model_files


def run_senseiv2_on_folder(start_date, end_date, site, device="cpu"):
    """
    Run SenSeIv2 on all .SAFE scenes in INPUT_DIR and save masks to OUTPUT_DIR.
    """
    input_dir = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    config, weights = get_model_files(MODEL_NAME)
    cm = CloudMask(config, weights, verbose=True, device=device)

    for scene_path in sorted(input_dir.glob("*.SAFE")):
        scene_id = scene_path.stem
        out_path = output_dir / f"{scene_id}_mask.npy"

        if out_path.exists():
            print(f"[SKIP] {out_path.name} already exists")
            continue

        try:
            print(f"[LOAD] {scene_path}")
            data = sentinel2_loader(str(scene_path), verbose=True)

            print("[INFER] Running SenSeIv2 …")
            mask = cm(data, descriptors=SENTINEL2_DESCRIPTORS)

            # Downsample 10 m → ~100 m (assumes 10980 px input)
            mask = mask.reshape(4, 1098, 10, 1098, 10).mean(axis=(2, 4))
            mask = np.argmax(mask, axis=0).astype(np.uint8)

            np.save(out_path, mask)
            print(f"[OK] Saved {out_path}")

        except Exception as e:
            print(f"[FAIL] {scene_id}: {e}")


if __name__ == "__main__":
    run_senseiv2_on_folder(START_DATE, END_DATE, SITE, device=DEVICE)


