#!/usr/bin/env python3
"""
CAMS Solar Radiation (time-series) downloader

What this script does
---------------------
1) Reads a CSV of coordinates (Latitude, Longitude).
2) For each point, requests CAMS Solar Radiation time-series (observed-cloud, 1-minute)
   for a given date range.
3) Saves each point to a CSV file. Skips files that already exist.
4) Stops after a daily limit of new downloads.
5) Prints a summary of completed, remaining, and already existing files.

Inputs to set
-------------
- CSV_PATH:      path to CSV with Latitude, Longitude columns (e.g. MSG pixels in S2 tile)
- START_DATE:    start date (YYYY-MM-DD)
- END_DATE:      end date   (YYYY-MM-DD)
- OUTPUT_FOLDER: folder where CAMS CSVs are stored
- DAILY_LIMIT:   maximum number of new downloads per run

Requirements
------------
- cdsapi (needs ~/.cdsapirc API key from ECMWF)
        This file includes:
            url: https://ads.atmosphere.copernicus.eu/api
            key: ---your personal key ---
- pandas
"""

import os
import pandas as pd
import cdsapi

# ======= USER INPUTS =======
CSV_PATH      = "/output_path/.../msg_pixel_coords_{location}.csv"
START_DATE    = "2019-01-01"
END_DATE      = "2025-06-06"
OUTPUT_FOLDER = "/output_path/.../cams_downloads_{location}/"
DAILY_LIMIT   = 100
# ===========================

def download_cams_data(lat, lon, start_date, end_date, output_folder):
    """Download CAMS solar radiation for one coordinate (lat, lon)."""
    output_filename = f"{output_folder}/cams_{start_date}_to_{end_date}_{lat:.4f}_{lon:.4f}.csv"
    if os.path.exists(output_filename):
        print(f"[SKIP] Already exists: {output_filename}")
        return False

    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    client = cdsapi.Client()
    dataset = "cams-solar-radiation-timeseries"
    request = {
        "sky_type": "observed_cloud",
        "location": {"longitude": lon, "latitude": lat},
        "altitude": 0,
        "date": f"{start_date}/{end_date}",
        "time_step": "1minute",
        "time_reference": "universal_time",
        "format": "csv"
    }

    print(f"[REQ] Requesting lat={lat:.4f}, lon={lon:.4f}...")
    try:
        client.retrieve(dataset, request).download(target=output_filename)
        print(f"[OK ] Saved: {output_filename}")
        return True
    except Exception as e:
        print(f"[ERR] Failed for lat={lat:.4f}, lon={lon:.4f}: {e}")
        return False


def batch_download_cams(csv_path, start_date, end_date, output_folder, daily_limit=100):
    """Loop over all coordinates in CSV and download CAMS data with a daily limit."""
    os.makedirs(output_folder, exist_ok=True)
    df = pd.read_csv(csv_path)

    count = 0
    for _, row in df.iterrows():
        if count >= daily_limit:
            print(f"[STOP] Reached daily limit of {daily_limit}.")
            break
        success = download_cams_data(row["Latitude"], row["Longitude"], start_date, end_date, output_folder)
        if success:
            count += 1

    print(f"[INFO] Completed {count} new downloads this run.")

    total = len(df)
    downloaded_files = sum(
        os.path.exists(
            f"{output_folder}/cams_{start_date}_to_{end_date}_{row['Latitude']:.4f}_{row['Longitude']:.4f}.csv"
        )
        for _, row in df.iterrows()
    )
    remaining = total - downloaded_files

    if count == 0:
        print("[NOTE] No new files were downloaded (all already complete).")
    elif remaining == 0:
        print(f"[DONE] All CAMS requests are now completed. Total: {total} files.")
    else:
        print(f"[NEXT] {remaining} requests remain. Run again tomorrow to continue.")


if __name__ == "__main__":
    batch_download_cams(
        csv_path=CSV_PATH,
        start_date=START_DATE,
        end_date=END_DATE,
        output_folder=OUTPUT_FOLDER,
        daily_limit=DAILY_LIMIT
    )
