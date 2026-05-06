"""
Read multi-year capacity-factor anomaly data in parallel and estimate the global lower-tail percentile threshold over a specified baseline period.

This script is part of the reproducible workflow for wind-power low-generation event analysis. Hard-coded paths and thresholds are intentionally preserved to maintain consistency with the original experiments.
"""

import pyarrow.parquet as pq
import numpy as np
import glob
import time
from concurrent.futures import ProcessPoolExecutor
import re


def _read_all_values(args):



    """
    Read and flatten all values from a single Parquet file by row group to reduce peak memory use during parallel threshold estimation.
    """
    path, start_time = args
    t0 = time.time()

    pf = pq.ParquetFile(path)
    all_values = []

    for rg in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg)

        col_vals = tbl.to_pandas().to_numpy(dtype='float32').ravel()
        all_values.append(col_vals)


    file_values = np.concatenate(all_values) if all_values else np.array([], dtype='float32')

    elapsed_proc = time.time() - t0
    elapsed_total = time.time() - start_time
    print(
        f"[{path}] read completed; elapsed time: {elapsed_proc:.2f}s; "
        f"total elapsed time: {elapsed_total:.2f}s; value count: {len(file_values):,}"
    )

    return file_values



def global_q1_from_parquets(pattern, n_workers=None, year_range=None):



    """
    Pool all anomaly values within the baseline period and compute the specified lower-tail percentile as the extreme-event threshold.
    """
    files = glob.glob(pattern)
    assert files, "No Parquet files found."


    if year_range is not None:
        y0, y1 = year_range

        def _extract_year(p):
            """
            Parse a four-digit year from a filename for baseline-period file filtering.
            """
            m = re.search(r'(\d{4})', p)
            return int(m.group(1)) if m else None

        files = [f for f in files
                 if (_extract_year(f) is not None) and (y0 <= _extract_year(f) <= y1)]
        assert files, "No files remain after year filtering."


    print(f"Reading {len(files)} Parquet files...")

    start_time = time.time()
    tasks = [(f, start_time) for f in files]

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        arrays = list(pool.map(_read_all_values, tasks))


    all_vals = np.concatenate(arrays)


    q1 = float(np.percentile(all_vals, 0.5))

    total_time = time.time() - start_time
    print(f"Processing completed; total elapsed time {total_time:.1f}s")
    print(f"Global lower-tail percentile = {q1:.6f}")
    print(f"Total value count: {len(all_vals):,}")
    print(f"Data range: [{all_vals.min():.6f}, {all_vals.max():.6f}]")

    return q1



if __name__ == "__main__":
    pattern = r"E:\change_from_scratch\final_optimization\air_density_calculation\output\land_cf_anomaly-1-13\*.parquet"
    q1 = global_q1_from_parquets(pattern, year_range=(1950, 1980))
    print(q1)
