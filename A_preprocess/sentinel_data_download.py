#!/usr/bin/env python3
"""
Sentinel-2 L1C downloader (CREODIAS) — simple, editable-at-the-top version.

What this script does
- Builds a square AOI around (LAT, LON) with a small degree buffer.
- Queries CREODIAS for Sentinel-2 L1C scenes between START_DATE and END_DATE.
- Downloads each product as {id}.zip, extracts the .SAFE folder, then deletes the .zip.
- Skips scenes if the .SAFE folder already exists.

Requirements
- Python 3.9+
- Internet connection 
- pip install: creodias_finder shapely geojson
"""

# ========= EDIT YOUR INPUTS HERE =========
START_DATE = "2023-01-01"       # "YYYY-MM-DD"
END_DATE   = "2023-12-31"       # "YYYY-MM-DD"
LAT        = {location_lat}     # centre latitude (deg)
LON        = {location_lon}     # centre longitude (deg)
SITE       = "{location}"       # used in output path

# Output base directory (both options kept; pick one)
OUTPUT_DIR = f"/path_to_sentinel_data/{SITE}_{START_DATE}_{END_DATE}"   # DelftBlue
# OUTPUT_DIR = "/Users/Lot/Downloads"                # Local

BUFFER_DEG = 0.10  # half-width of square AOI in degrees

# CREODIAS login (fill in your own)
CREDENTIALS = {
    "username": "your.email@example.com",
    "password": "your-password-here"
}
# ========================================

import os
import zipfile
from pathlib import Path
from shapely.geometry import Polygon
import geojson
from datetime import datetime
from creodias_finder import query, download


def download_sentinel2_data(start_date, end_date, lat, lon, site):
    """
    Download Sentinel-2 L1C data from CREODIAS for a small AOI around (lat, lon).
    Saves each product as {id}.zip, extracts the .SAFE folder, then deletes the .zip.
    """
    # Output folder
    output_dir = Path(OUTPUT_DIR) / site / f"Sentinel2_Data_{start_date}_{end_date}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Square AOI around point
    de = BUFFER_DEG
    coordinates = [
        (lon - de, lat + de),
        (lon + de, lat + de),
        (lon + de, lat - de),
        (lon - de, lat - de),
        (lon - de, lat + de)
    ]
    polygon = Polygon(coordinates)
    geometry = geojson.Feature(geometry=polygon)

    # Parse dates
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date   = datetime.strptime(end_date, "%Y-%m-%d")

    # Query CREODIAS
    results = query.query(
        'Sentinel2',
        start_date=start_date,
        end_date=end_date,
        geometry=geometry,
        productType='L1C'
    )
    print(f"Number of images found: {len(results)}")
    if not results:
        return

    # Download + extract each product
    for result in results.values():
        product_id = result['id']
        safe_folder = f"{product_id}.SAFE"
        safe_path = output_dir / safe_folder

        if safe_path.exists():
            print(f"{safe_folder} already exists. Skipping download.")
            continue

        zip_path = output_dir / f"{product_id}.zip"

        print(f"Downloading {product_id} to {zip_path}...")
        try:
            download.download(product_id, outfile=str(zip_path), **CREDENTIALS)
            print("Download complete.")
        except Exception as e:
            print(f"Error downloading {product_id}: {e}")
            # best effort cleanup of partial zip
            try:
                zip_path.unlink(missing_ok=True)
            except Exception:
                pass
            continue

        # Extract SAFE from ZIP
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                names = zip_ref.namelist()
                if not names:
                    print(f"Empty ZIP: {zip_path}")
                else:
                    safe_folder = names[0].split('/')[0]  # first top-level folder = .SAFE name
                    safe_path = output_dir / safe_folder
                    if safe_path.exists():
                        print(f"{safe_folder} already exists. Skipping extraction.")
                    else:
                        print(f"Extracting {safe_folder}...")
                        zip_ref.extractall(output_dir)
                        print("Extraction complete.")
        except Exception as e:
            print(f"Error processing {zip_path}: {e}")

        # Remove ZIP regardless
        try:
            zip_path.unlink(missing_ok=True)
        except Exception as e:
            print(f"Could not delete {zip_path}: {e}")


if __name__ == "__main__":
    download_sentinel2_data(
        start_date=START_DATE,
        end_date=END_DATE,
        lat=LAT,
        lon=LON,
        site=SITE
    )
