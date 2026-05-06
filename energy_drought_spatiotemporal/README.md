# Data Dictionary: Spatiotemporal Renewable Energy Drought Statistics

This directory contains gridded and domain-aggregated statistics for renewable energy droughts from 1950 to 2024. The released archive contains the 1% lower-tail threshold products used for the primary analysis. The datasets summarize wind-power droughts, solar-power droughts, and coincident wind-solar droughts across onshore and offshore domains. Grid-cell annual-frequency products are not included in the current release.

## Filename Semantics

Filenames encode three dimensions:

- Energy-event type: `wind`, `solar`, or `coincident`.
- Spatial domain: `Onshore`, `Offshore`, or paired land/sea columns.
- Threshold: `1pct`, referring to the 1% lower-tail anomaly percentile used to label drought events in the primary analysis.

Files within the same group have the same schema. To avoid redundant documentation, each group is described once below; files differing only by energy type or domain should be interpreted analogously.

## Dataset Groups

| Group | Files | Row meaning |
| --- | --- | --- |
| Yearly domain totals | `extreme_events_yearly_all_wind_1pct.csv`; `extreme_events_yearly_all_solar_1pct.csv`; `extreme_events_yearly_all_coincident_1pct.csv` | One row per year, with land and sea sample totals, event counts, and event frequencies. |
| Monthly domain statistics | `monthly_statistics_wind_1pct.csv`; `monthly_statistics_solar_1pct.csv`; `monthly_statistics_coincident_1pct.csv` | One row per year-month-domain combination. |
| Grid-cell summary statistics | `point_statistics_wind_1pct.csv`; `point_statistics_solar_1pct.csv`; `point_statistics_coincident_1pct.csv` | One row per grid cell, with geographic coordinates, land/sea class, event count, sample count, and frequency. |
| Grid-cell monthly climatological frequency | `points_monthly_frequency_*_1pct.csv` | One row per grid cell; month columns store climatological monthly drought frequency. |

## Common Spatial and Temporal Variables

| Column | Meaning |
| --- | --- |
| `lon` | Longitude of the grid-cell centre. |
| `lat` | Latitude of the grid-cell centre. |
| `land_sea` | Land-sea class of the grid cell, encoded as `Onshore` or `Offshore`. |
| `GridCell` | Grid-cell centre encoded as `(longitude, latitude)`. |
| `Year` / `year` | Calendar year. |
| `Month` | Calendar month, encoded as 1-12. |
| `Jan`-`Dec` | Calendar-month columns; each value is the climatological drought frequency for the corresponding month. |
| `OnOffshoreType` | Domain label, encoded as `Onshore` or `Offshore`. |

## Yearly Domain Totals: `extreme_events_yearly_all_*.csv`

These files report yearly aggregate drought statistics for land and sea domains. The `wind`, `solar`, and `coincident` files follow the same conceptual structure: domain-specific sample totals, event counts, and event frequencies. Some files may include additional column-classification diagnostics generated during land-sea assignment.

| Column | Meaning |
| --- | --- |
| `Year` | Calendar year. |
| `sea_total` | Total number of offshore grid-time samples evaluated in the year. |
| `sea_extreme` | Number of offshore grid-time samples classified as drought events. |
| `sea_ratio` | Offshore drought-event frequency, computed as `sea_extreme / sea_total`. |
| `land_total` | Total number of onshore grid-time samples evaluated in the year. |
| `land_extreme` | Number of onshore grid-time samples classified as drought events. |
| `land_ratio` | Onshore drought-event frequency, computed as `land_extreme / land_total`. |
| `original_columns` | Number of grid-cell columns before land-sea classification. |
| `sea_columns` | Number of grid-cell columns classified as offshore. |
| `land_columns` | Number of grid-cell columns classified as onshore. |
| `unmatched_columns` | Number of grid-cell columns not assigned to either onshore or offshore classes. |

## Monthly Domain Statistics: `monthly_statistics_*.csv`

These files provide monthly drought-event counts and frequencies by domain. Files differ only by energy-event type.

| Column | Meaning |
| --- | --- |
| `Year` | Calendar year. |
| `Month` | Calendar month, encoded as 1-12. |
| `OnOffshoreType` | Spatial domain, encoded as `Onshore` or `Offshore`. |
| `extreme_count` | Number of grid-time samples classified as drought events during the month. |
| `total_count` | Total number of grid-time samples evaluated during the month. |
| `Frequency` | Monthly drought-event frequency, computed as `extreme_count / total_count`. |

## Grid-Cell Summary Statistics: `point_statistics_*.csv`

These files summarize drought occurrence at each grid cell over the full study period. Files differ only by energy-event type.

| Column | Meaning |
| --- | --- |
| `lon` | Longitude of the grid-cell centre. |
| `lat` | Latitude of the grid-cell centre. |
| `land_sea` | Land-sea class of the grid cell. |
| `extreme_count` | Number of drought-event samples at the grid cell over the study period. |
| `total_count` | Total number of valid samples evaluated at the grid cell. |
| `frequency` | Drought-event frequency at the grid cell. In these files, values are expressed in percent. |

## Grid-Cell Monthly Climatological Frequency: `points_monthly_frequency_*.csv`

These files contain climatological monthly drought frequencies for each grid cell. Files differ by energy-event type and domain.

| Column | Meaning |
| --- | --- |
| `GridCell` | Grid-cell centre encoded as `(longitude, latitude)`. |
| `Jan`-`Dec` | Drought frequency for the corresponding calendar month, aggregated across the study period. |
