# Code Reproducibility Guide

This directory contains the executable workflow used to reproduce the renewable energy drought analysis. The scripts follow the original numerical sequence, and filenames have been standardized for GitHub release. Capacity-factor generation is now separated by energy technology, with dedicated scripts for wind and solar power.

## Execution Order

The two `01_*` scripts are technology-specific first-stage branches. Run the wind or solar capacity-factor script according to the energy technology being reproduced, then continue with the common aggregation, baseline, thresholding, event-labeling, and lifecycle-identification workflow.

`requirements.py` centralizes imports for the libraries used by the workflow. Run `python requirements.py` before the main workflow to check whether core dependencies can be imported; GPU-related dependencies are reported as optional.

1. `01_compute_wind_capacity_factor.py`: computes hourly wind-power capacity factors from ERA5 wind fields, near-surface temperature, elevation, and onshore/offshore point-location inputs.
2. `01_compute_solar_capacity_factor.py`: computes site-level solar-power capacity factors from surface solar radiation, ultraviolet-visible albedo, and optional near-surface meteorological inputs for thermal and wind-speed corrections.
3. `02_compute_cf_daily_mean.py`: aggregates hourly capacity-factor matrices to daily and monthly mean capacity factors.
4. `03_compute_expected_cf_baseline.py`: estimates baseline expected monthly capacity factors for the 1950-1980 reference period.
5. `04_compute_drought_threshold.py`: estimates lower-tail capacity-factor anomaly thresholds used to define renewable energy droughts.
6. `05_label_drought_events.py`: binarizes annual anomaly matrices into renewable energy drought masks.
7. `06_cc3d_event_lifecycle.py`: applies a 3-D connected-components framework to identify complete spatiotemporal drought-event lifecycles.

## Inputs, Outputs, and Manual Parameters

| Script | Main inputs | Main outputs | Manual parameters |
| --- | --- | --- | --- |
| `01_compute_wind_capacity_factor.py` | ERA5 wind NetCDF files, 2 m temperature NetCDF inventory, onshore/offshore point-location CSV files | Annual wind-power capacity-factor Parquet matrices | `folder_path`, `wanted_years`, `region`, point-file paths, output directories |
| `01_compute_solar_capacity_factor.py` | Site-matching CSV table, surface solar radiation NetCDF files, albedo NetCDF files, optional near-surface temperature and wind-speed NetCDF files | Annual solar-power capacity-factor Parquet matrices named as `cf_{year}.parquet` | `land_site_path`, `ssrd_folder`, `albedo_folder`, `wind_temp_folder`, `save_folder`, `years_to_process`, `num_processes`, GPU/cache environment variables |
| `02_compute_cf_daily_mean.py` | Annual hourly capacity-factor Parquet files | Daily and monthly capacity-factor Parquet files | Input/output directories and year range |
| `03_compute_expected_cf_baseline.py` | Monthly mean capacity-factor Parquet files | Baseline expected capacity-factor Parquet file | Baseline years and output path |
| `04_compute_drought_threshold.py` | Capacity-factor anomaly Parquet files | Printed percentile threshold | `pattern`, `year_range`, `n_workers`, percentile argument inside `np.percentile` |
| `05_label_drought_events.py` | Annual capacity-factor anomaly Parquet files | Annual binary renewable energy drought Parquet files | `region`, `flag`, `THRESHOLD`, `INPUT_DIR`, `OUTPUT_DIR`, `LAT_THRESHOLD`, `n_worker` |
| `06_cc3d_event_lifecycle.py` | Annual binary drought-event Parquet files | NetCDF connected-component label cube | `extreme_dir`, `N_JOB`, `lat_grid`, `lon_grid`, `output_file` |

## Reproducibility Notes

The scripts retain path placeholders adapted from the original computing environment. Before reuse on another machine, update only the documented path variables and preserve numerical thresholds unless intentionally repeating the threshold calibration step.
