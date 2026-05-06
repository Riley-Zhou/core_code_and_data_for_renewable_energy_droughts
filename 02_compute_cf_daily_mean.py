"""
Aggregate hourly capacity-factor series into daily and monthly means for subsequent climatological baseline and anomaly calculations.

This script is part of the reproducible workflow for wind-power low-generation event analysis. Hard-coded paths and thresholds are intentionally preserved to maintain consistency with the original experiments.
"""

import pandas as pd
from pathlib import Path
import time

def calculate_mean(df,wanted_year):

    """
    Aggregate one annual hourly capacity-factor matrix into daily and monthly mean matrices.
    """
    df.index = pd.to_datetime(df.index)
    print("Index conversion completed")


    daily_means = df.groupby(df.index.date).mean()
    print("Mean calculation completed")




    print(daily_means.head())

    base_dir = Path(r'E:\change_from_scratch\final_optimization\air_density_calculation\output\land_cf_daily_mean-1-13')
    file_name = f'{wanted_year}land-cf.parquet'
    parquet_file_path = base_dir / file_name
    daily_means.to_parquet(parquet_file_path)
    print(f'{wanted_year} daily mean exported')


    daily_means.index = pd.to_datetime(daily_means.index)

    monthly_means = daily_means.resample('ME').mean()

    monthly_means.index = monthly_means.index.strftime('%Y-%m')
    print(monthly_means)

    base_dir = Path(r'E:\change_from_scratch\final_optimization\air_density_calculation\output\land_cf_monthly_mean-1-13')
    file_name = f'{wanted_year}land-cfmonth.parquet'
    parquet_file_path = base_dir / file_name
    monthly_means.to_parquet(parquet_file_path)
    print(f'{wanted_year} monthly mean exported')


time0 = time.time()
for wanted_year in range(2021,2025):
    try:
        print("Processing started")
        parquet_file_path = fr'E:\change_from_scratch\final_optimization\air_density_calculation\output\land_cf\{wanted_year}land-wind_speed_cf.parquet'
        df = pd.read_parquet(
            parquet_file_path,
            engine='pyarrow',
            use_threads=True
        )
        print("success read")
        calculate_mean(df,wanted_year)
        print(f'{wanted_year}processing completed,elapsed time: {time.time() - time0:.4f} seconds')
    except:
        print(f'{wanted_year}data missing')
