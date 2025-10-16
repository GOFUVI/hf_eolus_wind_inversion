# Join Pivoted Tables (Athena to Parquet)

## Purpose
`join_pivoted_tables.sh` performs a multi-table inner join of pivoted station
tables on the shared `timestamp`, `node_id`, and `geometry` keys, then
optionally enriches the result with SAR and buoy data. The script materialises
each stage as a Parquet-backed Athena table so downstream tooling can consume
the combined features efficiently.

## Prerequisites
- AWS CLI v2 with permissions to execute Athena CTAS queries, manage Glue tables,
  and write to the target S3 prefixes.
- `jq` installed and available.
- Each `--source` table must expose the key columns configured through
  `--ts-col`, `--node-col`, and `--geom-col` (defaults: `timestamp`, `node_id`,
  `geometry`). Auxiliary SAR and buoy tables must include at least the
  timestamp/node columns; geometry is optional but recommended.

## Key CLI Options
- `--source db.table` (repeatable): pivoted tables to include in the join chain.
- `--union-out db.table@s3://path/`: destination for the joined result (required).
- `--sar db.table` and `--sar-out db.table@s3://path/`: optional SAR join source
  and output specification.
- `--buoy db.table` and `--buoy-out db.table@s3://path/`: optional buoy join
  source and output specification.
- `--prefix-mode table|none`: controls how columns contributed by auxiliary
  tables are prefixed (`table` by default, e.g. `sar__speed_over_ground`).
- `--results-s3 s3://bucket/path/`: overrides the Athena scratch bucket
  (defaults to `<union_out_s3>/athena-results/`).
- `--ts-col`, `--node-col`, `--geom-col`: customise the join keys if your schema
  names differ.

## Typical Usage
```bash
scripts/aggregation/join_pivoted_tables.sh \
  --source <db.station_a_pivot> \
  --source <db.station_b_pivot> \
  --union-out <db.joined_table>@s3://<bucket>/<joined_prefix>/ \
  --sar <db.sar_table> \
  --sar-out <db.joined_sar>@s3://<bucket>/<sar_prefix>/ \
  --buoy <db.reference_table> \
  --buoy-out <db.joined_reference>@s3://<bucket>/<reference_prefix>/ \
  --profile <aws_profile> \
  --region <aws_region> \
  --results-s3 s3://<bucket>/<athena_results_prefix>/
```

## Behaviour Notes
- The first `--source` establishes the join keys and column order; subsequent
  sources are inner-joined on the same key set.
- Auxiliary joins automatically include the geometry key when the right-hand
  table exposes a geometry column. If geometry is missing a warning is emitted
  and the join falls back to timestamp and node only.
- Glue databases referenced in any output specification are created when they do
  not yet exist, and existing Parquet data at the destination prefix is cleared.
- When `--prefix-mode table` (the default) is enabled, SAR fields are prefixed with `sar__` and buoy fields with `buoy__`; select `--prefix-mode none` to retain the auxiliary column names untouched.

## GeoParquet Follow-up
All results keep the geometry column as raw WKB bytes. After generating each
output table you can run `scripts/geo_utils/finalize_geoparquet.sh` (or the
component steps) to produce GeoParquet compliant datasets.

## Troubleshooting
- *Missing join keys*: verify that every source table exposes the configured key
  columns and that they share the same data types.
- *Unexpected row drops*: inner joins only retain rows present across all
  sources. Investigate upstream pivots if rows disappear.
- *Auxiliary join failures*: ensure that you pass `--sar` together with
  `--sar-out` (same for buoy). The script will exit early if only one flag from a
  pair is provided.

## File Reference
- `scripts/aggregation/join_pivoted_tables.sh`
