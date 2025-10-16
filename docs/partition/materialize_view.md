# Materialise View to Parquet

## Purpose
`scripts/partition/materialize_view.sh` issues an Athena `CREATE TABLE AS SELECT` (CTAS) to persist the rows of any view or table into a dedicated Parquet-backed dataset under a chosen S3 prefix. The script deletes any existing Glue table metadata and objects at that prefix, ensuring the resulting table reflects only the supplied source view. It is designed to bridge lightweight view-based harmonisation (e.g., via `concat_tables_view.sh`) with downstream components that expect fully materialised Parquet artefacts.

## Prerequisites
- AWS CLI v2 on the execution host, authenticated with permissions to run Athena queries, drop Glue tables, and modify the target S3 prefix.
- `jq` available on the `PATH` for response parsing.
- The input view must reside in Athena and be readable under the provided AWS profile.
- An S3 location for Athena query spill outputs (`--results-s3`).

## Key Options
- `--source <db.view>`: Fully-qualified Athena view (or table) to materialise.
- `--target-table <db.table>` / `--target <db.table>`: Destination Glue/Athena table name created by the CTAS statement.
- `--s3-output s3://.../`: S3 prefix that will receive the Parquet files produced by the CTAS query. Existing contents are removed before writing.
- `--results-s3 s3://.../`: Scratch bucket/prefix for Athena query execution metadata.
- `--profile <name>` / `--region <code>`: Optional overrides for AWS profile and region resolution.
- `--log-dir <path>`: Directory where the script writes an execution log (default: current working directory).

## Behaviour Notes
- The script first drops the existing Glue table (if any) and waits for the operation to succeed before proceeding. This removes stale schema definitions.
- All objects beneath `--s3-output` are deleted with `aws s3 rm --recursive` to avoid mixing historical files with the new export.
- The generated CTAS statement enforces `PARQUET` format with `SNAPPY` compression, mirroring the conventions used elsewhere in the project.
- Athena query execution IDs and status payloads are appended to the log so that failures can be diagnosed post-mortem without re-running the job.
- Because Athena CTAS writes directly to S3, the script completes once the query reports `SUCCEEDED`; no local artefacts are left behind besides the log and temporary SQL file.

## Typical Usage
Invoke `scripts/partition/materialize_view.sh` with the harmonised view as `--source`, the desired Glue table name via `--target-table`, and the destination bucket supplied through `--s3-output`. For example, the combined SAR+stationX training split is materialised by pointing the script to `analytics_db.PIVOTS_SAR_BUOY_WIND_TRAIN_VIEW` and writing into `s3://<your-bucket>/analytics_db/training/wind_combined/train/`, while the test split reuses the same options with the `_TEST_VIEW` identifiers. In both cases `--results-s3` should reference `s3://<your-bucket>/analytics_db/training/wind_combined/athena-results/`, with the `your_profile` profile and the `us-east-1` region mirroring the rest of the data-preparation pipeline. The `--log-dir` flag (e.g., `artifacts_root/sar_stationX/logs/partition`) keeps the output logs co-located with the legacy partition runbooks.

## Troubleshooting
- *`DROP TABLE` failures*: Confirm the caller has Glue permissions and that the target table name is valid; review `materialize_view_<table>.log` for the Athena error message.
- *Residual data in S3*: Check that the AWS principal has `s3:DeleteObject` privileges on the prefix supplied via `--s3-output`.
- *CTAS query errors*: Inspect the log for the full SQL and execution status. Typical causes include mismatched column types in the underlying view or insufficient permissions on the source tables.

## File Reference
- `scripts/partition/materialize_view.sh`
