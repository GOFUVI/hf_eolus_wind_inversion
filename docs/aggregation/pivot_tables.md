# Pivot Tables (Athena to Parquet)

## Purpose
`pivot_tables.sh` reshapes one or more Athena tables that contain a `pos_bragg`
column into a station-level wide format. Each input table becomes its own
Parquet-backed Athena table, preserving the timestamp, node identifier, and
geometry columns while duplicating the remaining measures into `_0` and `_1`
suffixes.

## Prerequisites
- AWS CLI v2 configured with an account that can run Athena CTAS queries, read
  Glue metadata, and write to the target S3 locations.
- `jq` available in `$PATH`.
- Source tables must expose the columns referenced by `--ts-col`, `--node-col`,
  `--geom-col`, and `--posbragg-col` (default names: `timestamp`, `node_id`,
  `geometry`, `pos_bragg`).

## Key CLI Options
- `--pivot in_db.in_table=out_db.out_table@s3://path/` (repeatable): defines the
  input/output pair and the S3 prefix where Parquet files land.
- `--prefix-mode table|none`: controls whether the script prefixes generated
  column names with the sanitized input table (default `table`).
- `--require-geometry-unique`: fails the run if any `node_id` maps to more than
  one geometry.
- `--results-s3 s3://bucket/path/`: overrides the Athena results bucket (defaults
  to `<first_pivot_s3>/athena-results/`).
- `--ts-col`, `--node-col`, `--geom-col`, `--posbragg-col`: customise the key and
  pivot columns when upstream schemas differ from the defaults.

## Typical Usage
```bash
scripts/aggregation/pivot_tables.sh \
  --pivot analytics_db.site1_AGGREGATED=analytics_db.site1_PIVOT@s3://bucket/pivots/site1/ \
  --pivot analytics_db.site2_AGGREGATED=analytics_db.site2_PIVOT@s3://bucket/pivots/site2/ \
  --profile your_profile \
  --region us-east-1 \
  --require-geometry-unique \
  --results-s3 s3://bucket/athena-results/
```

## Behaviour Notes
- The script enforces that both `pos_bragg=0` and `pos_bragg=1` observations are
  present for each `(timestamp, node_id, geometry)` group. Rows missing either
  half are filtered out via the `HAVING` clause.
- Column prefixes derive from the sanitized input table name when
  `--prefix-mode table`. Use `--prefix-mode none` to drop that table prefix;
  the `_0` and `_1` suffixes attached during the pivot remain regardless.
- Geometry columns remain binary WKB values. Run the GeoParquet utilities after
  the pivot to inject CRS metadata if required.
- The Athena Glue database defined in each output specification is created on
  the fly when absent.
- Existing Glue tables and S3 prefixes matching the requested outputs are
  deleted ahead of each CTAS execution to prevent stale artefacts.

## Post-processing
After the pivot completes you can optionally compact the Parquet files and add
GeoParquet metadata via `scripts/geo_utils/finalize_geoparquet.sh`

## Troubleshooting
- *Missing key columns*: confirm the source view/table exposes every column
  referenced by the `--*-col` parameters.
- *Geometry uniqueness failures*: either fix the upstream data, or rerun without
  `--require-geometry-unique` to downgrade the check into warnings.
- *Empty output tables*: inspect the source data to verify that both
  `pos_bragg` polarities exist for the same `(timestamp, node_id, geometry)`
  triplets.

## File Reference
- `scripts/aggregation/pivot_tables.sh`
