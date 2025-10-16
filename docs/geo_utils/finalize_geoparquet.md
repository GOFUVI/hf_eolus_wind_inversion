# GeoParquet Finalisation Utility

## Purpose
`scripts/geo_utils/finalize_geoparquet.sh` standardises Athena CTAS outputs into GeoParquet datasets suitable for downstream geospatial tooling. The script downloads the Parquet keys for a table, merges small part files into a single file per partition, injects GeoParquet metadata via `merge_parquet.py`, `add_geoparquet_metadata.py`, and `build_glue_table_input.py`, and synchronises the enhanced artefacts back to S3. When partition columns are provided it also runs `MSCK REPAIR TABLE` so Athena reflects the updated layout.

## Prerequisites
- AWS CLI v2 with S3 and Athena permissions for the target prefixes and tables.
- Docker available locally; the script launches a Python container that installs `pyarrow` and `shapely` on the fly.
- `jq` for inspecting Athena responses.
- Sufficient local disk space to stage the downloaded dataset.

## Key Options
- `--db-name <name>`: Glue/Athena database containing the table.
- `--bucket-name <name>` and `--output-prefix <prefix>`: Identify the S3 location of the Parquet dataset (e.g., `analytics_db/pivots/site1/`).
- `--output-table <table>`: Table name used for logging and optional repair.
- `--partition-cols col1,col2`: Optional comma-separated list of partition columns; triggers an Athena `MSCK REPAIR TABLE` after uploading.
- `--geometry-column <name>`: Column to annotate as the GeoParquet primary geometry (defaults to `geometry`).
- `--profile <profile>`: AWS CLI profile. The script honours the region configured on the profile (or the global AWS CLI configuration) and falls back to `us-east-1` only when no region is discoverable.
- `--log-dir <dir>`: Directory for log files and temporary staging (defaults to the current directory).
- `--register-table`: Registers or creates the target table in Glue using the schema captured by `build_glue_table_input.py` while preserving any existing table definition.

## Typical Usage
```bash
bash scripts/geo_utils/finalize_geoparquet.sh \
  --db-name <source_db> \
  --bucket-name <s3_bucket> \
  --output-prefix <dataset_prefix>/ \
  --output-table <table_name> \
  --profile <aws_profile>
```
This template mirrors the consolidation step executed for any Athena CTAS output before cataloguing or model training.

## Behaviour Notes
- The script stages files into a temporary working tree under `--log-dir`, deleting it on exit. Logs are persisted as `<log-dir>/finalize_geoparquet_<table>.log`.
- During the Docker run each partition is merged into a single Parquet file to reduce Athena query overhead and to simplify STAC packaging. The container also materialises the Glue table description so downstream registration can operate without additional metadata passes.
- GeoParquet metadata encodes CRS84, geometry type, bounding boxes, and orientation, ensuring the outputs remain compatible with GIS clients.
- When `--register-table` is supplied the script creates the Glue database if absent, checks whether the destination table already exists, and only issues `glue create-table` when the table is missing. The generated `__glue_table_input.json` lives in the temporary staging tree and is removed when cleanup runs at the end of the script.
- When `--partition-cols` is provided, `MSCK REPAIR TABLE` writes its scratch data under `<output-prefix>_athena_results/`; make sure this location is writable.

## Troubleshooting
- *Empty dataset*: The script aborts if no Parquet files are found after syncing. Verify upstream Athena jobs completed successfully.
- *Docker missing*: Install Docker locally; the Python environment inside the container provides the required dependencies.
- *Large datasets*: Ensure local disk capacity can temporarily hold the dataset; consider running from an EC2 instance close to the S3 bucket to reduce transfer times.

## File Reference
- `scripts/geo_utils/finalize_geoparquet.sh`
