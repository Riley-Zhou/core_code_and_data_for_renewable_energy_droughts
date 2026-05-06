# Data Dictionary: Renewable Energy Drought Event Lifecycles

This directory contains event-level lifecycle tables for renewable energy droughts identified using a three-dimensional connected-components framework. Each row represents one spatiotemporally connected drought event. The released archive contains tail-event subsets derived from the 1% lower-tail threshold products used for the primary analysis. Tail events are defined as the upper 5% of events ranked by `total_points`, i.e., the largest 5% of event volumes within each corresponding energy-domain lifecycle table.

## Files

| File group | Files |
| --- | --- |
| Wind drought tail-event lifecycles | `event-wind-Onshore-tailevent-1pct.csv`; `event-wind-Offshore-tailevent-1pct.csv` |
| Solar drought tail-event lifecycles | `event-solar-Onshore-tailevent-1pct.csv`; `event-solar-Offshore-tailevent-1pct.csv` |
| Coincident wind-solar drought tail-event lifecycles | `event-coincident-Onshore-tailevent-1pct.csv`; `event-coincident-Offshore-tailevent-1pct.csv` |

## Filename Semantics

- `event-wind-Onshore-tailevent-1pct.csv`: onshore wind-power drought tail events selected from 1% lower-tail drought events.
- `event-wind-Offshore-tailevent-1pct.csv`: offshore wind-power drought tail events selected from 1% lower-tail drought events.
- `event-solar-Onshore-tailevent-1pct.csv`: onshore solar-power drought tail events selected from 1% lower-tail drought events.
- `event-solar-Offshore-tailevent-1pct.csv`: offshore solar-power drought tail events selected from 1% lower-tail drought events.
- `event-coincident-Onshore-tailevent-1pct.csv`: onshore coincident wind-solar drought tail events selected from 1% lower-tail drought events.
- `event-coincident-Offshore-tailevent-1pct.csv`: offshore coincident wind-solar drought tail events selected from 1% lower-tail drought events.

All tabular files in this directory are stored as CSV and share the same schema. The variable definitions below apply to every file in the directory.

## Column Definitions

| Column                    | Meaning                                                                                                                    |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `event_id`              | Unique identifier of the renewable energy drought event within the corresponding file.                                     |
| `duration_days`         | Event duration in days.                                                                                                    |
| `total_points`          | Total event volume, measured as the cumulative number of affected grid-cell days across the full event lifecycle.          |
| `max_daily`             | Maximum daily spatial footprint, measured as the largest number of affected grid cells on any single day during the event. |
| `mean_daily`            | Mean daily spatial footprint, measured as the average number of affected grid cells per day across the event duration.     |
| `start_time_idx`        | Time-index position of the event start.                                                                                    |
| `end_time_idx`          | Time-index position of the event end.                                                                                      |
| `time_span`             | Event duration in days; this column is equivalent to `duration_days`.                                                    |
| `start_date`            | Calendar date on which the event starts.                                                                                   |
| `end_date`              | Calendar date on which the event ends.                                                                                     |
| `is_duration_qualified` | Boolean flag indicating whether the event duration exceeds 2 days.                                                         |
| `is_spatial_qualified`  | Boolean flag indicating whether the event volume exceeds 2 grid-cell days.                                                 |
| `is_qualified`          | Boolean flag indicating whether the event satisfies either `is_duration_qualified` or `is_spatial_qualified`.          |

## Interpretation Notes

The lifecycle metrics support analyses of event persistence, spatial extent, and integrated severity. `total_points` should be interpreted as event volume rather than area at one instant, whereas `max_daily` and `mean_daily` summarize the daily spatial footprint. The current tail-event files retain only the largest 5% of connected events by `total_points`, so they emphasize the most spatially extensive and persistent drought-event lifecycles rather than the full event population. The qualification flags are screening indicators and do not change the raw lifecycle metrics.
