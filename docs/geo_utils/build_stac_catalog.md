# STAC Catalog Builder

## Purpose
`scripts/geo_utils/build_stac_catalog.sh` packages GeoParquet datasets into lightweight SpatioTemporal Asset Catalogs (STAC). It stages Parquet assets from S3 or a local directory, builds a Docker image with the STAC generator, and emits a collection containing items, assets, and optional metadata enrichments. The resulting folder contains `collection.json`, per-partition item JSON files, and an `assets/` directory ready for publication or local inspection.

## Prerequisites
- Docker available locally; the script builds and runs the bundled STAC builder image.
- AWS CLI v2 when sourcing data from S3 (`--s3-uri`). Appropriate read permissions for the dataset prefix are required.
- Optional JSON files describing additional collection or item properties to attach to the payload.

## Key Options
- `--collection <id>`: Identifier written into `collection.json`.
- **Source selection** (choose one):
  - `--s3-uri s3://bucket/prefix`: Sync Parquet assets from S3 (requires `--profile`).
  - `--local-source-dir <dir>`: Copy assets from an existing local directory.
- `--profile` / `--region`: AWS CLI configuration for S3 sync operations.
- `--output-dir <dir>`: Destination directory for the generated catalog (default: `scripts/geo_utils/catalog_output`). Existing contents are removed unless `--keep-output` is set.
- `--stac-item-properties-json <file>` / `--stac-collection-properties-json <file>`: Inject additional metadata blocks into item or collection JSON, matching the theoretical context outlined in `README.md`.
- `--keep-incoming`: Preserve the staged `incoming/` directory after the build.
- `--zip-file <path>`: Optionally emit a zip archive of the catalog.
- `--build-opts <opts>`: Pass extra flags to `docker build` (e.g., `--no-cache`).
- `--verbose`: Enable step-by-step logging for troubleshooting.

## Typical Usage
```bash
bash scripts/geo_utils/build_stac_catalog.sh \
  --collection <collection_id> \
  --s3-uri s3://<bucket>/<dataset_prefix>/ \
  --stac-collection-properties-json <collection_properties.json> \
  --stac-item-properties-json <item_properties.json> \
  --output-dir catalogs/<catalog_name> \
  --profile <aws_profile> \
  --region <aws_region>
```
This template packages any GeoParquet export into a STAC catalog; supply the identifiers and property files that best describe your dataset.

## Behaviour Notes
- The script cleans the output directory unless `--keep-output` is specified, ensuring catalog contents reflect the latest dataset state.
- When sourcing from S3, common Athena scratch directories (`query_results`, `_athena_results`) are excluded automatically.
- Additional metadata files are mounted read-only into the Docker container and appended to the generated JSON. Use the templates in `catalogs/` as references.
- If the parent directory of the emitted `collection.json` already hosts a `catalog.json`, the collection is linked under that catalog and inherits it as both parent and root, preserving hierarchical navigation across catalog generations.
- The Docker image embeds `scripts/geo_utils/build_geo_catalog.py`, which performs the item generation, applies the STAC Table extension (column schemas, row counts, primary geometries), and materialises partition-aware subcatalogs. Inspect that module whenever you need to adjust how assets, temporal ranges, or metadata enrichments are emitted.
- After the Docker run, the staged `incoming/` directory is removed by default to keep the workspace tidy.

## Post-generation Catalog Repair
Occasionally a collection may be generated before its enclosing `catalog.json` is available, leaving the STAC graph without parent or root references. The helper script `scripts/geo_utils/repair_stac_links.sh` revisits an existing catalog tree and restores those relationships. It mounts the repository inside a lightweight Python 3.11 container, iterates through every `collection.json`, and rewrites their `parent` and `root` links to target the shared catalog. While doing so it also refreshes the catalog's `child` entries so that the hierarchy mirrors the layout on disk. Invoke it after adding a new `catalog.json` or when moving collections between directories with `bash scripts/geo_utils/repair_stac_links.sh --catalog-dir catalogs`. If Docker access is unavailable, the underlying Python module can be executed directly via `python scripts/geo_utils/repair_stac_links.py catalogs`, yielding the same result.

## Troubleshooting
- *Docker build failures*: Re-run with `--build-opts "--no-cache"` to invalidate stale layers or confirm Docker permissions.
- *Missing Parquet files*: Ensure the source prefix or directory contains the GeoParquet outputs produced by `finalize_geoparquet.sh`.
- *AWS credential errors*: Provide a profile with read access to the S3 prefix and confirm the region matches the dataset location.

## File Reference
- `scripts/geo_utils/build_stac_catalog.sh`
