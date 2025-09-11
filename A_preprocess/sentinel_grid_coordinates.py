#!/usr/bin/env python3
"""
Sentinel-2 pixel-centre grid (lat/lon) — HARD-CODED TEMPLATE

What this script does
- Builds a 10 m grid from a hard-coded UTM upper-left (ULX, ULY), extent and EPSG.
- Converts pixel centres to WGS84 lon/lat.
- Saves a CSV with columns: Latitude, Longitude.

IMPORTANT
- Replace ALL placeholders below:
    ULX, ULY, WIDTH_M, HEIGHT_M, RESOLUTION → numbers (in metres)
    UTM_EPSG → "EPSG:xxxxx"
    SITE → "{location}"
- The script refuses to run until placeholders are replaced.
"""

# ========= EDIT YOUR INPUTS HERE (REPLACE PLACEHOLDERS) =========
SITE       = "{location}"     
ULX        = "xxxxx"          # e.g. hardcoded for now, you can find it in *.SAFE/GRANULE/L1C*/MTD_TL.XML <Geoposition resolution="10"> 
ULY        = "xxxxx"          # e.g. hardcoded for now, you can find it in *.SAFE/GRANULE/L1C*/MTD_TL.XML <Geoposition resolution="10"> 
WIDTH_M    = "109800"          # e.g. 109800 (REPLACE)
HEIGHT_M   = "109800"          # e.g. 109800 (REPLACE)
RESOLUTION = "10"          # e.g. 10 (REPLACE)
UTM_EPSG   = "EPSG:xxxxx"     # e.g. "EPSG:32630" (REPLACE)

OUT_DIR    = "/path_to_sentinel_grid"  # e.g. "/scratch/…/cams" (REPLACE)
OUT_CSV    = f"{OUT_DIR}/sentinel2_pixel_coords_{SITE}.csv"
# ================================================================

import os
import numpy as np
import pandas as pd
from pyproj import Transformer


def ensure_placeholders_replaced():
    placeholders = [
        (SITE, "{location}"),
        (ULX, "xxxxx"),
        (ULY, "xxxxx"),
        (WIDTH_M, "xxxxx"),
        (HEIGHT_M, "xxxxx"),
        (RESOLUTION, "xxxxx"),
        (UTM_EPSG, "EPSG:xxxxx"),
        (OUT_DIR, "/path_to_cams"),
    ]
    for value, placeholder in placeholders:
        if value == placeholder:
            raise RuntimeError(f"Please replace placeholder: {placeholder}")


def generate_sentinel2_latlon_grid(ulx, uly, width, height, resolution, utm_epsg):
    x_coords = np.arange(ulx, ulx + width, resolution)       # left → right
    y_coords = np.arange(uly, uly - height, -resolution)     # top → bottom
    xx, yy = np.meshgrid(x_coords, y_coords)

    transformer = Transformer.from_crs(utm_epsg, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(xx, yy)

    latlon = np.column_stack((lat.ravel(), lon.ravel()))
    return latlon, xx.shape


def main():
    ensure_placeholders_replaced()

    # Use the (now replaced) values directly
    ulx = ULX
    uly = ULY
    width = WIDTH_M
    height = HEIGHT_M
    resolution = RESOLUTION
    utm_epsg = UTM_EPSG

    latlon_coords, shape = generate_sentinel2_latlon_grid(
        ulx=ulx, uly=uly, width=width, height=height, resolution=resolution, utm_epsg=utm_epsg
    )

    print(f"[INFO] Generated grid: {shape[0]} x {shape[1]} ({len(latlon_coords)} points)")

    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.DataFrame(latlon_coords, columns=["Latitude", "Longitude"])
    df.to_csv(OUT_CSV, index=False)
    print(f"[OK] Saved: {OUT_CSV}")


if __name__ == "__main__":
    main()
