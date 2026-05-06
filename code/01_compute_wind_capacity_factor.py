"""
Compute annual site-level wind-power capacity-factor time series from hourly ERA5 wind, near-surface temperature, and elevation inputs for onshore or offshore domains.

This script is part of the reproducible workflow for wind-power low-generation event analysis. Hard-coded paths and thresholds are intentionally preserved to maintain consistency with the original experiments.
"""

import os
import numpy as np
import xarray as xr
import datetime
import netCDF4 as nc
import time
import pandas as pd
from pathlib import Path

start_time = time.time()

'''Function definitions'''


def list_files_in_folder(folder_path):
    """
    Return all NetCDF filenames in the specified directory to construct the candidate annual meteorological data inventory.
    """
    try:

        all_items = os.listdir(folder_path)

        files = [item for item in all_items if os.path.isfile(os.path.join(folder_path, item))]

        nc_files = [file for file in files if file.endswith('.nc')]
        return nc_files
    except FileNotFoundError:
        return "The specified folder path does not exist. Please verify the path."
    except Exception as e:
        return f"Error: {e}"


def year_selected(folder_path,files_list,wanted_year):
    """
    Scan candidate NetCDF files and return the dataset corresponding to the target year.
    """
    for filename in files_list:
        file_path = os.path.join(folder_path, filename)
        ds = xr.open_dataset(file_path, engine="netcdf4")
        years = ds['valid_time'].dt.year.values
        unique_years = sorted(set(years))
        year = unique_years[0] if unique_years else None
        if year == wanted_year:
            print(year)
            print(f"{year} file opened.")
            end_time = time.time()
            print(f"elapsed time: {end_time - start_time:.2f} seconds\n")
            return ds


def calculate_wind_power_land(speed):

    """
    Convert standardized wind speeds to rated power output using the onshore turbine power curve in a vectorized form.
    """
    mask_out = (speed <= 3) | (speed > 25)

    power = np.zeros_like(speed, dtype=np.float64)


    intervals = [
        (3, 19.2851, 0, 69.1787, 17),
        (3.5, -0.4257, 28.9277, 83.6426, 54),
        (4, 6.4176, 28.2892, 112.2510, 103),
        (4.5, -1.2447, 37.9156, 145.3534, 167),
        (5, -1.4387, 36.0485, 182.3354, 249),
        (5.5, 6.9995, 33.8905, 217.3049, 349),
        (6, -2.5592, 44.3897, 256.4450, 467),
        (6.5, 11.2373, 40.5509, 298.9152, 606),
        (7, -10.3899, 57.4068, 347.8941, 767),
        (7.5, 6.3223, 41.8219, 397.5084, 954),
        (8, -6.8994, 51.3054, 444.0721, 1164),
        (8.5, -42.7246, 40.9563, 490.2030, 1398),
        (9, 1.7977, -23.1306, 499.1159, 1648),
        (9.5, -84.4664, -20.4340, 477.3336, 1892),
        (10, 56.0677, -147.1335, 393.5498, 2115),
        (10.5, -43.8046, -63.0319, 288.4671, 2282),
        (11, 55.1505, -128.7388, 192.5818, 2405),
        (11.5, -8.7974, -46.0130, 105.2059, 2476),
        (12, 20.0391, -59.2091, 52.5948, 2516),
        (12.5, 2530)
    ]


    speed_125_mask = (speed > 12.5) & (speed <= 25)
    power[speed_125_mask] = 2530


    for interval in intervals[:-1]:
        x, a, b, c, d = interval
        upper = x + 0.5
        mask = (speed > x) & (speed <= upper) & ~mask_out

        if not np.any(mask):
            continue


        s = speed[mask]
        delta = s - x
        power[mask] = a * delta ** 3 + b * delta ** 2 + c * delta + d

    return power


def calculate_wind_power_ocean(speed):

    """
    Convert hub-height offshore wind speeds to rated power output using the offshore turbine power curve in a vectorized form.
    """
    mask_out = (speed <= 3) | (speed > 34)

    power = np.zeros_like(speed, dtype=np.float64)


    intervals = [
        (3, 19.2851, 0, 69.1787, 17),
        (3.5, -0.4257, 28.9277, 83.6426, 54),
        (4, 6.4176, 28.2892, 112.2510, 103),
        (4.5, -1.2447, 37.9156, 145.3534, 167),
        (5, -1.4387, 36.0485, 182.3354, 249),
        (5.5, 6.9995, 33.8905, 217.3049, 349),
        (6, -2.5592, 44.3897, 256.4450, 467),
        (6.5, 11.2373, 40.5509, 298.9152, 606),
        (7, -10.3899, 57.4068, 347.8941, 767),
        (7.5, 6.3223, 41.8219, 397.5084, 954),
        (8, -6.8994, 51.3054, 444.0721, 1164),
        (8.5, -42.7246, 40.9563, 490.2030, 1398),
        (9, 1.7977, -23.1306, 499.1159, 1648),
        (9.5, -84.4664, -20.4340, 477.3336, 1892),
        (10, 56.0677, -147.1335, 393.5498, 2115),
        (10.5, -43.8046, -63.0319, 288.4671, 2282),
        (11, 55.1505, -128.7388, 192.5818, 2405),
        (11.5, -8.7974, -46.0130, 105.2059, 2476),
        (12, 20.0391, -59.2091, 52.5948, 2516),
        (12.5, 2530)
    ]


    speed_125_mask = (speed > 12.5) & (speed <= 34)
    power[speed_125_mask] = 2530


    for interval in intervals[:-1]:
        x, a, b, c, d = interval
        upper = x + 0.5
        mask = (speed > x) & (speed <= upper) & ~mask_out

        if not np.any(mask):
            continue


        s = speed[mask]
        delta = s - x
        power[mask] = a * delta ** 3 + b * delta ** 2 + c * delta + d

    return power


def choose_t2m_of_wanted_year(wanted_year):

    """
    Locate the target-year 2 m air-temperature NetCDF file from a precomputed file inventory.
    """
    txt_file_path = r'E:\change_from_scratch\final_optimization\air_density_calculation\code\all_t2m_nc_files.txt'


    with open(txt_file_path, 'r') as file:
        file_names = file.readlines()

        for file_name in file_names:
            file_name = file_name.strip()
            if file_name.endswith(f"{wanted_year}.nc"):
                print(f'{wanted_year} t2m file opened')
                return file_name


def select_region(ds,
                  wanted_year,
                  lat_idx,
                  lon_idx,
                  lat_vals,
                  lon_vals,
                  H_i,
                  t2m=None,
                  region='land'
                  ):

    """
    Extract wind and temperature fields for selected grid cells, apply air-density correction, compute capacity factors, and export Parquet output.
    """
    sfcWind = np.sqrt(ds['u100'].values[:, lat_idx, lon_idx]**2 + ds['v100'].values[:, lat_idx, lon_idx]**2)

    print("Actual wind speed calculation completed")


    if region == 'ocean':

        sfcWind = sfcWind * (150/100)**0.1

    T_0 = 288.15
    time_steps, num_points = sfcWind.shape
    H_i_expanded = np.tile(H_i, (time_steps, 1))

    if t2m is not None:
        if region == 'ocean':
            sfcWind = sfcWind * np.power(np.power(1 - 0.0065 * (150) / T_0, 5.256) * T_0 / t2m, 1 / 3)

        else:
            sfcWind = sfcWind * np.power(np.power(1 - 0.0065 * (H_i_expanded + 100) / T_0, 5.256) * T_0 / t2m, 1 / 3)

        print("Standardized wind speed calculation completed with t2m and elevation correction")
    else:
        print("Standardized wind speed calculation skipped without t2m and elevation correction")



    if region == 'land':
        cf = calculate_wind_power_land(sfcWind) / 2530
    elif region == 'ocean':
        cf = calculate_wind_power_ocean(sfcWind) / 2530
    else:
        raise ValueError("region must be 'land' or 'ocean'")

    valid_time = ds['valid_time'].values.astype(str)

    columns = [f'({lon:.2f},{lat:.2f})' for lon, lat in zip(lon_vals, lat_vals)]
    df = pd.DataFrame(cf, index=valid_time, columns=columns)
    df = df.astype('float32')

    if region == 'land':
        base_dir = Path(rf'E:\change_from_scratch\final_optimization\air_density_calculation\output\land_cf')
    elif region == 'ocean':
        base_dir = Path(rf'E:\change_from_scratch\final_optimization\air_density_calculation\output\offshore_cf')
    file_name = f'{wanted_year}{region}-wind_speed_cf.parquet'
    parquet_file_path = base_dir / file_name

    df.to_parquet(parquet_file_path, engine='pyarrow', index=True)

    print("Export completed")
    end_time = time.time()
    print(f"elapsed time: {end_time - start_time:.2f} seconds\n")


def main():

    """
    Configure input paths, years, domain, and site indices, then orchestrate the complete annual capacity-factor workflow.
    """
    folder_path = r'G:\data_from_1940_2024'
    files_list = list_files_in_folder(folder_path)
    wanted_years = [1950,2024]

    region = 'land'


    if region == 'land':
        points_df = pd.read_csv(r'E:\change_from_scratch\final_optimization\air_density_calculation\point_information\onshore_points-1-10.csv')
    elif region == 'ocean':
        points_df = pd.read_csv(r'E:\change_from_scratch\final_optimization\air_density_calculation\point_information\offshore_points-1-10.csv')
    lat_vals = points_df['lat'].values
    lon_vals = points_df['lon'].values
    H_i = points_df['MEAN'].values



    lat_idx = ((90.0 - lat_vals) / 0.25).astype(np.int32)
    lon_idx = ((lon_vals + 360) % 360 / 0.25).astype(np.int32)
    print("Point extraction completed")

    while len(wanted_years) != 0:
        for wanted_year in wanted_years:
            ds = year_selected(folder_path,files_list, wanted_year)

            try:

                t2m_path = choose_t2m_of_wanted_year(wanted_year)
                t2m_nc = xr.open_dataset(t2m_path, engine="netcdf4")
                t2m = t2m_nc['t2m'].values[:, lat_idx, lon_idx]
                print('HDF input opened successfully')


                if region == 'ocean':
                    t2m = t2m - 0.9
                else:
                    t2m = t2m - 0.6

                select_region(ds,
                              wanted_year,
                              lat_idx,
                              lon_idx,
                              lat_vals,
                              lon_vals,
                              H_i,
                              t2m=t2m,
                              region=region
                              )

            except Exception as e:
                print(f"Error: {e}")
                continue

    print(
        'All calculations completed\n'
        'Output directory: '
        + rf'E:\change_from_scratch\final_optimization\air_density_calculation\output\{region}_cf'
        + f'\nOutput filename pattern: {{wanted_year}}{region}-wind_speed_cf.parquet'
    )

if __name__ == '__main__':
    main()
