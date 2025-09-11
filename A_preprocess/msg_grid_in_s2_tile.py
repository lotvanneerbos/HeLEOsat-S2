#!/usr/bin/env python3
"""
SEVIRI/MSG ↔ Sentinel-2 overlap (grid extractor)

Purpose
-------
1) Read one SEVIRI Level-1b *native* (.nat) file via Satpy and extract pixel-centre
   longitude/latitude arrays (this defines the MSG grid).
   NOTE: `CHANNEL` refers to the SEVIRI instrument channel on MSG (e.g., "VIS006", "HRV").
         For geometry, any channel present in the file will yield the same lon/lat grid.
2) Convert a Sentinel-2 tile bounding box from UTM (metres) to EPSG:4326 (WGS84 lon/lat).
   EPSG:4326 means geographic coordinates: longitude and latitude in degrees (WGS84 datum).
3) Select MSG pixel centres that fall inside that Sentinel-2 tile.
4) Save the selected (lat, lon) pairs to CSV.

Inputs to set below
-------------------
- NAT_FILE: path to a SEVIRI L1b native *.nat file (any time; geometry is constant per platform).
- CHANNEL:  SEVIRI instrument channel name (e.g., "VIS006", "HRV", "IR_108").
- ULX, ULY, WIDTH, HEIGHT: Sentinel-2 tile extent in UTM metres (upper-left corner + size).
- UTM_EPSG: EPSG string for the UTM zone of the S2 tile (e.g., "EPSG:32631" for UTM 31N).
- OUT_CSV:  output CSV path.

Dependencies
------------
satpy, pyproj, numpy, pandas
"""

from pathlib import Path
import numpy as np
import pandas as pd
from pyproj import Transformer
from satpy import Scene

# ======= USER INPUTS =======
NAT_FILE   = "/path_to_msg_nat_file/*.nat"
CHANNEL    = "VIS006"          # SEVIRI instrument channel on MSG (e.g., "VIS006", "HRV", "IR_108")
ULX        = 600000.0          # S2 tile UTM X (upper-left), metres
ULY        = 5800020.0         # S2 tile UTM Y (upper-left), metres
WIDTH      = 109800.0          # tile width (metres) e.g., 10980 px × 10 m
HEIGHT     = 109800.0          # tile height (metres)
UTM_EPSG   = "EPSG:32631"      # UTM zone of the S2 tile (Cabauw: UTM 31N)
OUT_CSV    = "/output_path/.../msg_pixel_coords_{location}.csv"
# ===========================

def load_seviri_lonlat(nat_file, channel):
    """Load SEVIRI lon/lat pixel-centre arrays from a .nat file for the given instrument channel."""
    scn = Scene(reader="seviri_l1b_native", filenames=[nat_file])
    scn.load([channel])
    lons, lats = scn[channel].area.get_lonlats()  # 2D arrays (same shape as the data grid)
    return lats, lons

def s2_bbox_latlon(ulx, uly, width, height, utm_epsg):
    """
    Convert the Sentinel-2 tile UTM box to EPSG:4326 (WGS84 lon/lat).
    Returns (lat_min, lat_max, lon_min, lon_max).
    """
    # EPSG:4326 = WGS84 geographic coordinates (longitude, latitude) in degrees
    tr = Transformer.from_crs(utm_epsg, "EPSG:4326", always_xy=True)
    # Upper-left (UL):
    lon1, lat1 = tr.transform(ulx, uly)
    # Lower-right (LR):
    lon2, lat2 = tr.transform(ulx + width, uly - height)
    lat_min, lat_max = min(lat1, lat2), max(lat1, lat2)
    lon_min, lon_max = min(lon1, lon2), max(lon1, lon2)
    return lat_min, lat_max, lon_min, lon_max

def select_in_bbox(lats, lons, lat_min, lat_max, lon_min, lon_max):
    """Return flattened arrays of lat/lon values within the bounding box."""
    mask = (lats >= lat_min) & (lats <= lat_max) & (lons >= lon_min) & (lons <= lon_max)
    return lats[mask], lons[mask]

def main():
    lats, lons = load_seviri_lonlat(NAT_FILE, CHANNEL)
    lat_min, lat_max, lon_min, lon_max = s2_bbox_latlon(ULX, ULY, WIDTH, HEIGHT, UTM_EPSG)
    sel_lats, sel_lons = select_in_bbox(lats, lons, lat_min, lat_max, lon_min, lon_max)
    print(f"Selected {sel_lats.size} MSG pixel centres inside the Sentinel-2 tile.")

    out = Path(OUT_CSV)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Latitude": sel_lats.ravel(), "Longitude": sel_lons.ravel()}).to_csv(out, index=False)
    print(f"Saved CSV: {out}")

if __name__ == "__main__":
    main()
