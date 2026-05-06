"""
Estimate monthly expected capacity factors over the baseline period to provide the climatological reference for low-generation event detection.

This script is part of the reproducible workflow for wind-power low-generation event analysis. Hard-coded paths and thresholds are intentionally preserved to maintain consistency with the original experiments.
"""

import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor

start_time = time.time()


df_list = []


def read_parquet_file(year):
    """
    Read the monthly capacity-factor file for a specified year and normalize the index for interannual aggregation.
    """
    parquet_file_path = fr'E:\change_from_scratch\final_optimization\air_density_calculation\output\land_cf_monthly_mean-1-13\{year}land-cfmonth.parquet'
    df = pd.read_parquet(parquet_file_path, engine="pyarrow")
    df.reset_index(drop=True, inplace=True)
    return df


with ThreadPoolExecutor() as executor:
    futures = {executor.submit(read_parquet_file, year): year for year in range(1950, 1981)}
    for future in futures:
        year = futures[future]
        try:
            df = future.result()
            df_list.append(df)
            end_time = time.time()
            print(f'{year} data loaded; elapsed time: {end_time - start_time:.2f} seconds')
        except Exception as e:
            print(f"Failed to read {year} data: {e}")


first_df = df_list[0]
for df in df_list[1:]:
    if not df.index.equals(first_df.index) or not df.columns.equals(first_df.columns):
        print("DataFrame indices or columns are inconsistent; inspect the data files")
        exit()


result_df = pd.DataFrame(0, index=df_list[0].index, columns=df_list[0].columns)


for df in df_list:
    result_df += df
print(f'Summation completed; elapsed time: {time.time() - start_time:.2f} seconds')


result_df /= 31


print(result_df)

result_df.to_parquet(r'E:\change_from_scratch\final_optimization\air_density_calculation\output\land_expected_cf-1-13.parquet', index=True)
