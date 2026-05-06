"""
Binarize annual capacity-factor anomaly matrices using predefined percentile thresholds and exclude high-latitude grid cells to produce low-generation event masks.

This script is part of the reproducible workflow for wind-power low-generation event analysis. Hard-coded paths and thresholds are intentionally preserved to maintain consistency with the original experiments.
"""

import time
from pathlib import Path
import pandas as pd
import re
from multiprocessing import Pool


# Manually adjustable parameters
region = 'land'
flag = 3

if region == 'land':
    if flag == 1:
        THRESHOLD = -0.3745731711387634
        INPUT_DIR = Path(r"E:\change_from_scratch\final_optimization\air_density_calculation\output\land_cf_anomaly-1-13")
        OUTPUT_DIR = Path(r"E:\change_from_scratch\final_optimization\air_density_calculation\output\onshore_wind_drought_1pct")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    elif flag == 2:
        THRESHOLD = -0.4250289797782898
        INPUT_DIR = Path(r"E:\change_from_scratch\final_optimization\air_density_calculation\output\land_cf_anomaly-1-13")
        OUTPUT_DIR = Path(r"E:\change_from_scratch\final_optimization\air_density_calculation\output\onshore_wind_drought_0_5pct")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    elif flag == 3:
        THRESHOLD = -0.5233547797203064
        INPUT_DIR = Path(r"E:\change_from_scratch\final_optimization\air_density_calculation\output\land_cf_anomaly-1-13")
        OUTPUT_DIR = Path(r"E:\change_from_scratch\final_optimization\air_density_calculation\output\onshore_wind_drought_0_1pct")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
elif region == 'ocean':
    if flag == 1:
        THRESHOLD = -0.5747506618499756
        INPUT_DIR = Path(r"E:\change_from_scratch\final_optimization\air_density_calculation\output\offshore_cf_anomaly-1-13")
        OUTPUT_DIR = Path(r"E:\change_from_scratch\final_optimization\air_density_calculation\output\offshore_wind_drought_1pct")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    elif flag == 2:
        THRESHOLD = -0.6191235211491585
        INPUT_DIR = Path(r"E:\change_from_scratch\final_optimization\air_density_calculation\output\offshore_cf_anomaly-1-13")
        OUTPUT_DIR = Path(r"E:\change_from_scratch\final_optimization\air_density_calculation\output\offshore_wind_drought_0_5pct")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    elif flag == 3:
        THRESHOLD = -0.6993754511475563
        INPUT_DIR = Path(r"E:\change_from_scratch\final_optimization\air_density_calculation\output\offshore_cf_anomaly-1-13")
        OUTPUT_DIR = Path(r"E:\change_from_scratch\final_optimization\air_density_calculation\output\offshore_wind_drought_0_1pct")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
else:
    raise ValueError('region must be either "land" or "ocean"')


LAT_THRESHOLD = 70


def filter_lowlat_columns(df):
    """
    Retain grid-cell columns below the latitude threshold using coordinates encoded in column names.
    """
    columns_to_keep = []
    pattern = r'\(([-+]?\d*\.\d+)\s*,\s*([-+]?\d*\.\d+)\)'

    for col in df.columns.tolist():
        match = re.match(pattern, str(col))
        if not match:
            columns_to_keep.append(col)
            continue

        try:
            lat = float(match.group(2))
        except ValueError:
            columns_to_keep.append(col)
            continue

        if lat < LAT_THRESHOLD:
            columns_to_keep.append(col)

    return df[columns_to_keep]


def process_one_year(year: int) -> tuple[int, float, int, int]:
    """
    Read one annual anomaly matrix, apply latitude filtering and threshold binarization, and export compressed Parquet output.
    """
    t0 = time.time()
    pq_in = INPUT_DIR / f'{year}{region}-cf.parquet'
    pq_out = OUTPUT_DIR / f"{year}{region}_wind_binary_drought_1pct.parquet"

    df = pd.read_parquet(pq_in)
    print(f"[{year}] Original column count: {len(df.columns)}")

    df_lowlat = filter_lowlat_columns(df)
    print(
        f"[{year}] Low-latitude column count: {len(df_lowlat.columns)} "
        f"(excluded {len(df.columns) - len(df_lowlat.columns)} high-latitude columns)"
    )

    df_bin = (df_lowlat < THRESHOLD).astype("uint8")
    df_bin.to_parquet(pq_out, engine="pyarrow", compression="snappy")

    return year, time.time() - t0, len(df.columns), len(df_lowlat.columns)


if __name__ == "__main__":
    years = list(range(1950, 2025))
    n_worker = 48

    print(f"Launching {n_worker} worker processes for {len(years)} years...")
    print(f"Excluding regions with latitude >= {LAT_THRESHOLD} degrees")
    print(f"Threshold: {THRESHOLD}")
    print("=" * 60)

    T0 = time.time()
    total_original_cols = 0
    total_lowlat_cols = 0

    with Pool(processes=n_worker) as pool:
        for year, cost, orig_cols, lowlat_cols in pool.imap_unordered(process_one_year, years):
            print(
                f"Completed: {year}; elapsed time: {cost:.3f}s; "
                f"retained columns: {lowlat_cols}/{orig_cols}"
            )
            total_original_cols += orig_cols
            total_lowlat_cols += lowlat_cols

    total_time = time.time() - T0
    print("=" * 60)
    print("All tasks completed")
    print(f"Total elapsed time: {total_time:.1f} s")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("Column-count summary:")
    print(f"  Original total column count: {total_original_cols}")
    print(f"  Low-latitude column count: {total_lowlat_cols} (latitude < {LAT_THRESHOLD} degrees)")
    print(f"  Excluded column count: {total_original_cols - total_lowlat_cols} (latitude >= {LAT_THRESHOLD} degrees)")
    print(f"  Retention ratio: {total_lowlat_cols / total_original_cols * 100:.1f}%")
