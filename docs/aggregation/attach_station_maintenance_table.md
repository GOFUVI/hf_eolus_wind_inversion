# Station Maintenance Tables

## Purpose
`scripts/aggregation/attach_station_maintenance_table.sh` materialises
maintenance-aware versions of the pivoted HF-radar station tables. For each
station it reads `calibrations.csv`, identifies the latest maintenance event
preceding every observation timestamp, and writes a Parquet-backed Athena table
(CTAS) with interval identifiers and recency metrics. The enriched tables feed
the downstream joins so that maintenance context propagates through every
GeoParquet and STAC artifact.

## Input Data Preparation
Keep `calibrations.csv` in the repository root updated with the columns
`station_id`, `event_type` (interpreted as maintenance type), and
`effective_start` (ISO-8601 timestamp in UTC). The helper treats
`maintenance_type` and `maintenance_start` as synonyms for those headers, so the
CSV can reuse whichever naming convention is most convenient. It reads the CSV
verbatim, normalises station identifiers to lowercase, converts timestamps via
`from_iso8601_timestamp`, and discards rows that lack an intervention start.
Because the dataset is embedded into the CTAS statement, there is no need to
stage the file elsewhere.

## Execution Overview
The script accepts the source pivot table (`--source`), the CTAS destination
(`--out database.table@s3://prefix/`), the CSV path, and the station metadata
(`--station-id`, `--prefix`). It also requires an Athena results bucket
(`--results-s3`) used to host the query execution logs; the helper aborts if the
parameter is omitted. For the requested station it builds an inline `VALUES`
clause with all maintenance events and runs a single Athena CTAS:

1. Drop any existing Glue table with the specified name.
2. Remove the target S3 prefix to avoid stale files.
3. Execute a `CREATE TABLE ... AS SELECT` that left-joins the pivot table with
a lateral subquery returning the most recent maintenance interval whose start is
less than or equal to the observation timestamp.

The resulting table appends the columns
`<prefix>_maintenance_interval_id`, `<prefix>_maintenance_type`,
`<prefix>_maintenance_start`, and `<prefix>_hours_since_last_calibration`.

## Output Schema
- `<prefix>_maintenance_interval_id`: Deterministic identifier of the form
  `<prefix>_YYYYMMDDTHHMMSSZ`, enabling traceability back to the maintenance
  catalogue.
- `<prefix>_maintenance_type`: Operation category captured in the maintenance
  log (e.g., calibration, repair).
- `<prefix>_maintenance_start`: ISO-8601 timestamp marking the moment the radar
  returned to service.
- `<prefix>_hours_since_last_calibration`: Hours elapsed between the
  maintenance start and the observation timestamp, providing a continuous
  indicator of sensor age.

## Integration Notes
Typical workflows invoke the helper immediately after generating the
station-specific pivot views, producing maintenance-enriched variants (for
example, `<prefix>_PIVOT_FEATURES_MAINT`). These tables can be fed directly into
`join_pivoted_tables.sh` or any custom union, ensuring that maintenance
attributes flow through the joined datasets, GeoParquet consolidation, and STAC
publication steps without additional wiring. When the CSV lacks interventions
for a station, the appended columns remain `NULL`, signalling the absence of
recorded maintenance events.


## Troubleshooting
- Ensure `calibrations.csv` is present and contains the required headers; the
  script fails fast if the file is missing.
- If the CTAS execution fails, check permissions for the supplied `--results-s3`
  bucket and confirm that the Athena profile has Glue and S3 access rights.
- Null maintenance metrics typically indicate that no intervention was recorded
  before the observation timestamp; update the CSV if additional events become
  available.

## File Reference
- `scripts/aggregation/attach_station_maintenance_table.sh`
