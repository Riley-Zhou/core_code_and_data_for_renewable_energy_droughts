"""
Map multi-year binary event masks onto a global latitude-longitude grid and identify spatiotemporally connected event trajectories using 3-D connected-component analysis.

This script is part of the reproducible workflow for wind-power low-generation event analysis. Hard-coded paths and thresholds are intentionally preserved to maintain consistency with the original experiments.
"""

import time, os, gc
import pandas as pd
import numpy as np
import cc3d
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import xarray as xr


# Manually adjustable parameters
m_days = 2
k_cells = 2

# Parallel execution configuration
N_JOB = 48



lat_grid = np.arange(-90, 90.25, 0.25)
lon_grid = np.arange(0, 360.25, 0.25)
ny, nx = len(lat_grid), len(lon_grid)


t0 = time.time()
extreme_dir = r'E:\change_from_scratch\final_optimization\air_density_calculation\output\onshore_wind_drought_1pct'


def read_single_year(year):

    """
    Read one annual binary extreme-event file and construct the corresponding daily time index from the number of rows.
    """
    fn = os.path.join(extreme_dir, f'{year}land_wind_binary_drought_1pct.parquet')
    df = pd.read_parquet(fn)


    start_date = pd.Timestamp(f'{year}-01-01')

    dates = pd.date_range(start=start_date, periods=len(df), freq='D')
    df.index = dates

    return df


def main():
    """
    Configure input paths, years, domain, and site indices, then orchestrate the complete annual capacity-factor workflow.
    """
    years = list(range(1950, 2025))
    print('Reading drought-event files...')
    dfs = []
    with ProcessPoolExecutor(max_workers=N_JOB) as exe:
        fut_map = {exe.submit(read_single_year, y): y for y in years}
        for fut in as_completed(fut_map):
            df = fut.result()
            dfs.append(df)
            print(f'{fut_map[fut]} read completed; rows: {len(df)}')


    big_df = pd.concat(dfs, axis=0).sort_index()
    times = big_df.index
    nt = len(times)
    print(f'Concatenation completed; total time steps: {nt}; time range: {times[0]} to {times[-1]}; elapsed time: {time.time() - t0:.2f} s')
    del dfs
    gc.collect()


    t1 = time.time()
    cube = np.zeros((nt, nx, ny), dtype=np.uint8)


    total_cols = len(big_df.columns)
    processed_cols = 0

    for col in big_df.columns:


        try:

            lon_str, lat_str = col.strip('()').split(',')
            lon = float(lon_str)
            lat = float(lat_str)


            if lon < 0:
                lon += 360


            iy = int(round((lat + 90) / 0.25))
            ix = int(round(lon / 0.25))

            if 0 <= iy < ny and 0 <= ix < nx:
                cube[:, ix, iy] = big_df[col].values.astype(np.uint8)

        except Exception as e:
            print(f"Warning: failed to parse column '{col}': {e}")

        processed_cols += 1
        if processed_cols % 10000 == 0:
            print(f'Processed {processed_cols}/{total_cols} columns...')

    print(f'Global grid filling completed; elapsed time: {time.time() - t1:.3f} s')
    print(f'cube shape: {cube.shape}, data type: {cube.dtype}')
    print(f'Total drought-event grid-cell count: {np.sum(cube)}')

    del big_df
    gc.collect()


    t2 = time.time()

    labels = cc3d.connected_components(cube, connectivity=6)
    print(f'cc3d connected-component analysis completed; elapsed time: {time.time() - t2:.2f} s')
    print(f'Connected-component count: {len(np.unique(labels)) - 1}')


    ds = xr.DataArray(
        labels,
        dims=['time', 'lon', 'lat'],
        coords={'time': times, 'lon': lon_grid, 'lat': lat_grid},
        name='event_label',
        attrs={
            'description': 'Connected-component labels for drought events, unfiltered',
            'connectivity': 6,
            'source_data': 'drought-event files',
            'note': 'Includes all connected components before temporal and spatial filtering'
        }
    )

    output_file = r'E:\change_from_scratch\final_optimization\whole-cc3d\cc3d-wind-land-1pct.nc'
    ds.to_netcdf(output_file)
    print(f'Export completed: {output_file}')
    print(f'Total elapsed time: {time.time() - t0:.2f} s')


    final_labels = labels[labels > 0]
    if len(final_labels) > 0:
        print(f'Connected-component count: {len(np.unique(final_labels))}')
        print(f'Total drought-event grid-cell count: {len(final_labels)}')
    else:
        print('No drought events found')


if __name__ == '__main__':
    main()
