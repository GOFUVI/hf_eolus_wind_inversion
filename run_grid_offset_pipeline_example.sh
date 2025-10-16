#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Author: Example Team <team@example.org>
# Created: 2025-10-16
# Disclaimer: This obfuscated sample mirrors the grid-offset workflow and must be adapted before any deployment.
# -----------------------------------------------------------------------------

# Exit immediately on error, undefined variable, or pipeline failure
set -euo pipefail

# Example grid-offset pipeline (obfuscated).
# Replace placeholders such as example-profile, example-region-1, project-bucket-placeholder, and wind_training.* before running.

# Toggle blocks (set to 0 to skip specific sections)
RUN_GRID_OFFSET_DATA_PREPARATION=1
RUN_SAR_FINAL_INFERENCE=1
RUN_SAR_FINETUNED_INFERENCE=1
RUN_SAR_FINETUNED_L2SP_INFERENCE=1
RUN_SAR_FINETUNED_L2SP_KD_INFERENCE=1
RUN_BUOY_FINAL_INFERENCE=1
RUN_BUOY_FINETUNED_INFERENCE=1
RUN_BUOY_FINETUNED_L2SP_INFERENCE=1
RUN_BUOY_FINETUNED_L2SP_KD_INFERENCE=1
RUN_COMBINED_FINAL_INFERENCE=1

echo "=== Grid-offset evaluation pipeline started at $(date) ==="

mkdir -p \
  artifacts_root/grid_offset \
  artifacts_root/grid_offset/reports \
  artifacts_root/grid_offset/logs \
  artifacts_root/grid_offset/logs/materialize \
  artifacts_root/grid_offset/logs/finalize_geoparquet \
  artifacts_root/grid_offset/logs/finalize_geoparquet_inference \
  artifacts_root/grid_offset/inference_metrics

if [[ "${RUN_GRID_OFFSET_DATA_PREPARATION}" == "1" ]]; then
  echo ">>> Pivoting grid-offset aggregated tables at $(date) <<<"
  ./scripts/aggregation/pivot_tables.sh \
    --pivot wind_training.RADAR_A_AGGREGATED_OFFSET=wind_training.RADAR_A_PIVOT_OFFSET@s3://project-bucket-placeholder/wind_training/pivots_offset/radar_a/ \
    --pivot wind_training.RADAR_B_AGGREGATED_OFFSET=wind_training.RADAR_B_PIVOT_OFFSET@s3://project-bucket-placeholder/wind_training/pivots_offset/radar_b/ \
    --profile example-profile \
    --region example-region-1 \
    --require-geometry-unique \
    --results-s3 s3://project-bucket-placeholder/wind_training/pivots_offset/athena-results/
  echo ">>> Completed grid-offset pivoting at $(date) <<<"

  echo ">>> Adding station bearing and distance features (grid-offset) at $(date) <<<"
  ./scripts/geo_utils/add_station_bearing_distance_view.sh \
    --source wind_training.RADAR_A_PIVOT_OFFSET \
    --view wind_training.RADAR_A_PIVOT_OFFSET_FEATURES \
    --station radar_a:12.345678:-45.678912 \
    --results-s3 s3://project-bucket-placeholder/wind_training/pivots_offset/athena-results/ \
    --profile example-profile \
    --region example-region-1
  ./scripts/geo_utils/add_station_bearing_distance_view.sh \
    --source wind_training.RADAR_B_PIVOT_OFFSET \
    --view wind_training.RADAR_B_PIVOT_OFFSET_FEATURES \
    --station radar_b:11.223344:-44.556677 \
    --results-s3 s3://project-bucket-placeholder/wind_training/pivots_offset/athena-results/ \
    --profile example-profile \
    --region example-region-1
  echo ">>> Completed station feature enrichment (grid-offset) at $(date) <<<"

  echo ">>> Attaching maintenance intervals (grid-offset) at $(date) <<<"
  ./scripts/aggregation/attach_station_maintenance_table.sh \
    --source wind_training.RADAR_A_PIVOT_OFFSET_FEATURES \
    --out wind_training.RADAR_A_PIVOT_OFFSET_FEATURES_MAINT@s3://project-bucket-placeholder/wind_training/pivots_offset/radar_a_with_maintenance/ \
    --maintenance-csv maintenance_events.csv \
    --station-id radar_a \
    --prefix radar_a \
    --timestamp-col timestamp \
    --results-s3 s3://project-bucket-placeholder/wind_training/pivots_offset/athena-results/ \
    --profile example-profile \
    --region example-region-1
  ./scripts/aggregation/attach_station_maintenance_table.sh \
    --source wind_training.RADAR_B_PIVOT_OFFSET_FEATURES \
    --out wind_training.RADAR_B_PIVOT_OFFSET_FEATURES_MAINT@s3://project-bucket-placeholder/wind_training/pivots_offset/radar_b_with_maintenance/ \
    --maintenance-csv maintenance_events.csv \
    --station-id radar_b \
    --prefix radar_b \
    --timestamp-col timestamp \
    --results-s3 s3://project-bucket-placeholder/wind_training/pivots_offset/athena-results/ \
    --profile example-profile \
    --region example-region-1
  echo ">>> Completed maintenance enrichment (grid-offset) at $(date) <<<"

  echo ">>> Joining grid-offset pivots and SAR aggregates at $(date) <<<"
  ./scripts/aggregation/join_pivoted_tables.sh \
    --source wind_training.RADAR_A_PIVOT_OFFSET_FEATURES_MAINT \
    --source wind_training.RADAR_B_PIVOT_OFFSET_FEATURES_MAINT \
    --union-out wind_training.PIVOTS_OFFSET_JOINED@s3://project-bucket-placeholder/wind_training/pivots_offset/joined/ \
    --sar wind_training.SAR_AGGREGATED_OFFSET \
    --sar-out wind_training.PIVOTS_OFFSET_SAR@s3://project-bucket-placeholder/wind_training/pivots_offset/with_sar/ \
    --profile example-profile \
    --region example-region-1 \
    --results-s3 s3://project-bucket-placeholder/wind_training/pivots_offset/athena-results/
  echo ">>> Completed grid-offset join at $(date) <<<"

  echo ">>> Canonicalizing grid-offset SAR view column names at $(date) <<<"
  python3 - <<'PY'
import json
import pathlib
import subprocess
import shlex

PROFILE = "example-profile"
REGION = "example-region-1"
DB = "wind_training"
SOURCE_TABLE = "PIVOTS_OFFSET_SAR"
SQL_PATH = pathlib.Path("artifacts_root/pivot_and_join/sql/pivots_offset_sar_canonical.sql")

cmd = [
    "aws",
    "--profile",
    PROFILE,
    "--region",
    REGION,
    "glue",
    "get-table",
    "--database-name",
    DB,
    "--name",
    SOURCE_TABLE,
]
result = subprocess.run(cmd, check=True, capture_output=True, text=True)
table = json.loads(result.stdout)["Table"]
columns = table["StorageDescriptor"]["Columns"]

select_parts = []
for col in columns:
    name = col["Name"]
    if name in {"timestamp", "node_id", "geometry"}:
        select_parts.append(name)
    elif name.startswith("radar_a_aggregated_offset__"):
        suffix = name.split("__", 1)[1]
        select_parts.append(f"{name} AS radar_a_aggregated__{suffix}")
    elif name.startswith("radar_b_aggregated_offset__"):
        suffix = name.split("__", 1)[1]
        select_parts.append(f"{name} AS radar_b_aggregated__{suffix}")
    else:
        select_parts.append(name)

sql = "SELECT\n  " + ",\n  ".join(select_parts) + f"\nFROM {DB}.{SOURCE_TABLE}"
SQL_PATH.write_text(sql + "\n", encoding="utf-8")
PY

  ./scripts/aggregation/create_filtered_view.sh \
    --source wind_training.PIVOTS_OFFSET_SAR \
    --view wind_training.PIVOTS_OFFSET_SAR_CANONICAL \
    --sql-file artifacts_root/pivot_and_join/sql/pivots_offset_sar_canonical.sql \
    --results-s3 s3://project-bucket-placeholder/wind_training/pivots_offset/athena-results/ \
    --profile example-profile \
    --region example-region-1
  echo ">>> Completed canonical view creation at $(date) <<<"

  echo ">>> Filtering SAR-valid samples for grid-offset dataset at $(date) <<<"
  ./scripts/aggregation/create_filtered_view.sh \
    --source wind_training.PIVOTS_OFFSET_SAR_CANONICAL \
    --view wind_training.PIVOTS_OFFSET_SAR_VALID \
    --sql-file artifacts_root/pivot_and_join/sql/pivots_sar_valid_wind.sql \
    --results-s3 s3://project-bucket-placeholder/wind_training/pivots_offset/athena-results/ \
    --profile example-profile \
    --region example-region-1
  echo ">>> Completed SAR-valid filtering (grid-offset) at $(date) <<<"

  echo ">>> Annotating grid-offset SAR view with source metadata at $(date) <<<"
  ./scripts/aggregation/create_filtered_view.sh \
    --source wind_training.PIVOTS_OFFSET_SAR_VALID \
    --view wind_training.PIVOTS_OFFSET_SAR_VALID_SOURCE \
    --sql-file artifacts_root/pivot_and_join/sql/pivots_sar_valid_with_source.sql \
    --results-s3 s3://project-bucket-placeholder/wind_training/pivots_offset/athena-results/ \
    --profile example-profile \
    --region example-region-1
  echo ">>> Completed source annotation for grid-offset SAR view at $(date) <<<"

  echo ">>> Materialising grid-offset SAR view at $(date) <<<"
  ./scripts/partition/materialize_view.sh \
    --source wind_training.PIVOTS_OFFSET_SAR_VALID_SOURCE \
    --target-table wind_training.PIVOTS_OFFSET_SAR_VALID_SOURCE_CTAS \
    --s3-output s3://project-bucket-placeholder/wind_training/pivots_offset/materialized/sar_valid_source/ \
    --results-s3 s3://project-bucket-placeholder/wind_training/pivots_offset/materialized/athena-results/ \
    --profile example-profile \
    --region example-region-1 \
    --log-dir artifacts_root/grid_offset/logs/materialize
  echo ">>> Completed grid-offset materialisation at $(date) <<<"

  echo ">>> Documenting grid-offset dataset lineage at $(date) <<<"
  python3 - <<'PY'
from datetime import datetime, timezone

timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
note = f"""# Grid-offset evaluation dataset lineage

Generated on {timestamp}.

This dataset mirrors the pivot-join pipeline used for the nominal grid but
operates on the offset aggregations (`*_AGGREGATED_OFFSET`) that shift the grid
10 km eastward and northward. The preparation keeps all timestamps available in
the joined SAR–HF radar corpus without applying train/test partitioning so that
every model variant can be scored on an identical spatial footprint.

Pipeline stages:
1. Pivoted `wind_training.RADAR_A_AGGREGATED_OFFSET` and `wind_training.RADAR_B_AGGREGATED_OFFSET`,
   enforcing unique geometries per node.
2. Added per-station bearing and distance features relative to the reference buoy and
  Coastal radars and attached maintenance-interval tags via `maintenance_events.csv`.
3. Joined the station pivots and merged the result with
   `wind_training.SAR_AGGREGATED_OFFSET`, producing `wind_training.PIVOTS_OFFSET_SAR`.
4. Harmonised field names by stripping the `_offset` infix so that all HF-Radar
   aggregates reuse the canonical `radar_a_aggregated__*` and `radar_b_aggregated__*`
   identifiers expected by the trained models, while preserving the SAR wind
   targets under their canonical `sar__*` identifiers for metric evaluation.
5. Filtered records to retain valid SAR winds and annotated the view with the
   deterministic `wind_source='sar'` metadata.
6. Materialised `wind_training.PIVOTS_OFFSET_SAR_VALID_SOURCE` into the S3 prefix
   `s3://project-bucket-placeholder/wind_training/pivots_offset/materialized/sar_valid_source/`
   for downstream inference jobs.
"""

with open("artifacts_root/grid_offset/reports/grid_offset_dataset_report.md", "w", encoding="utf-8") as handle:
    handle.write(note)
PY
  echo ">>> Completed grid-offset dataset documentation at $(date) <<<"

  echo ">>> Finalising GeoParquet assets for grid-offset intermediates at $(date) <<<"
  bash scripts/geo_utils/finalize_geoparquet.sh \
    --db-name wind_training \
    --bucket-name project-bucket-placeholder \
    --output-prefix wind_training/pivots_offset/radar_a/ \
    --output-table RADAR_A_PIVOT_OFFSET \
    --register-table \
    --profile example-profile \
    --region example-region-1 \
    --log-dir artifacts_root/grid_offset/logs/finalize_geoparquet
  bash scripts/geo_utils/finalize_geoparquet.sh \
    --db-name wind_training \
    --bucket-name project-bucket-placeholder \
    --output-prefix wind_training/pivots_offset/radar_b/ \
    --output-table RADAR_B_PIVOT_OFFSET \
    --register-table \
    --profile example-profile \
    --region example-region-1 \
    --log-dir artifacts_root/grid_offset/logs/finalize_geoparquet
  bash scripts/geo_utils/finalize_geoparquet.sh \
    --db-name wind_training \
    --bucket-name project-bucket-placeholder \
    --output-prefix wind_training/pivots_offset/radar_a_with_maintenance/ \
    --output-table RADAR_A_PIVOT_OFFSET_FEATURES_MAINT \
    --register-table \
    --profile example-profile \
    --region example-region-1 \
    --log-dir artifacts_root/grid_offset/logs/finalize_geoparquet
  bash scripts/geo_utils/finalize_geoparquet.sh \
    --db-name wind_training \
    --bucket-name project-bucket-placeholder \
    --output-prefix wind_training/pivots_offset/radar_b_with_maintenance/ \
    --output-table RADAR_B_PIVOT_OFFSET_FEATURES_MAINT \
    --register-table \
    --profile example-profile \
    --region example-region-1 \
    --log-dir artifacts_root/grid_offset/logs/finalize_geoparquet
  bash scripts/geo_utils/finalize_geoparquet.sh \
    --db-name wind_training \
    --bucket-name project-bucket-placeholder \
    --output-prefix wind_training/pivots_offset/joined/ \
    --output-table PIVOTS_OFFSET_JOINED \
    --register-table \
    --profile example-profile \
    --region example-region-1 \
    --log-dir artifacts_root/grid_offset/logs/finalize_geoparquet
  bash scripts/geo_utils/finalize_geoparquet.sh \
    --db-name wind_training \
    --bucket-name project-bucket-placeholder \
    --output-prefix wind_training/pivots_offset/with_sar/ \
    --output-table PIVOTS_OFFSET_SAR \
    --register-table \
    --profile example-profile \
    --region example-region-1 \
    --log-dir artifacts_root/grid_offset/logs/finalize_geoparquet
  bash scripts/geo_utils/finalize_geoparquet.sh \
    --db-name wind_training \
    --bucket-name project-bucket-placeholder \
    --output-prefix wind_training/pivots_offset/materialized/sar_valid_source/ \
    --output-table PIVOTS_OFFSET_SAR_VALID_SOURCE \
    --register-table \
    --profile example-profile \
    --region example-region-1 \
    --log-dir artifacts_root/grid_offset/logs/finalize_geoparquet
  echo ">>> Completed GeoParquet finalisation for grid-offset intermediates at $(date) <<<"

  echo ">>> Building STAC catalogs for grid-offset intermediates at $(date) <<<"
  bash scripts/geo_utils/build_stac_catalog.sh \
    --collection RADAR_A_PIVOT_OFFSET \
    --s3-uri s3://project-bucket-placeholder/wind_training/pivots_offset/radar_a/ \
    --output-dir catalogs/grid_offset_pipeline/radar_a_pivot_offset \
    --profile example-profile \
    --region example-region-1
  bash scripts/geo_utils/build_stac_catalog.sh \
    --collection RADAR_B_PIVOT_OFFSET \
    --s3-uri s3://project-bucket-placeholder/wind_training/pivots_offset/radar_b/ \
    --output-dir catalogs/grid_offset_pipeline/radar_b_pivot_offset \
    --profile example-profile \
    --region example-region-1
  bash scripts/geo_utils/build_stac_catalog.sh \
    --collection RADAR_A_PIVOT_OFFSET_FEATURES_MAINT \
    --s3-uri s3://project-bucket-placeholder/wind_training/pivots_offset/radar_a_with_maintenance/ \
    --output-dir catalogs/grid_offset_pipeline/radar_a_pivot_offset_features_maint \
    --profile example-profile \
    --region example-region-1
  bash scripts/geo_utils/build_stac_catalog.sh \
    --collection RADAR_B_PIVOT_OFFSET_FEATURES_MAINT \
    --s3-uri s3://project-bucket-placeholder/wind_training/pivots_offset/radar_b_with_maintenance/ \
    --output-dir catalogs/grid_offset_pipeline/radar_b_pivot_offset_features_maint \
    --profile example-profile \
    --region example-region-1
  bash scripts/geo_utils/build_stac_catalog.sh \
    --collection PIVOTS_OFFSET_JOINED \
    --s3-uri s3://project-bucket-placeholder/wind_training/pivots_offset/joined/ \
    --output-dir catalogs/grid_offset_pipeline/pivots_offset_joined \
    --profile example-profile \
    --region example-region-1
  bash scripts/geo_utils/build_stac_catalog.sh \
    --collection PIVOTS_OFFSET_SAR \
    --s3-uri s3://project-bucket-placeholder/wind_training/pivots_offset/with_sar/ \
    --output-dir catalogs/grid_offset_pipeline/pivots_offset_sar \
    --profile example-profile \
    --region example-region-1
  bash scripts/geo_utils/build_stac_catalog.sh \
    --collection PIVOTS_OFFSET_SAR_VALID_SOURCE \
    --s3-uri s3://project-bucket-placeholder/wind_training/pivots_offset/materialized/sar_valid_source/ \
    --output-dir catalogs/grid_offset_pipeline/pivots_offset_sar_valid_source \
    --profile example-profile \
    --region example-region-1
  echo ">>> Completed STAC catalog build for grid-offset intermediates at $(date) <<<"
else
  echo ">>> Skipping grid-offset data preparation per configuration <<<"
fi

# -----------------------------------------------------------------------------
# Inference blocks: SAR models
# -----------------------------------------------------------------------------
if [[ "${RUN_SAR_FINAL_INFERENCE}" == "1" ]]; then
  if [ -f artifacts_root/sar/config/final_model.txt ]; then
    echo ">>> Running grid-offset inference for SAR final model at $(date) <<<"
    ./scripts/inference/run_inference.sh \
      --model-artifact "s3://project-bucket-placeholder/wind_training/models/sar/final/$(awk '{sub(/#.*/,"");gsub(/^[ \t]+|[ \t]+$/,"");if(length($0)){print;exit}}' artifacts_root/sar/config/final_model.txt)/output/model.tar.gz" \
      --input-data s3://project-bucket-placeholder/wind_training/pivots_offset/materialized/sar_valid_source/ \
      --output-s3-uri "s3://project-bucket-placeholder/wind_training/inference/grid_offset/sar_final" \
      --profile example-profile \
      --region example-region-1 \
      --output-format parquet
    bash scripts/geo_utils/finalize_geoparquet.sh \
      --db-name wind_training \
      --bucket-name project-bucket-placeholder \
      --output-prefix wind_training/inference/grid_offset/sar_final \
      --output-table GRID_OFFSET_SAR_FINAL_INFERENCE \
      --register-table \
      --profile example-profile \
      --region example-region-1 \
      --log-dir artifacts_root/grid_offset/logs/finalize_geoparquet_inference
    mkdir -p example-profile/grid_offset/inference_metrics/sar_final
    ./scripts/inference/compute_inference_metrics.sh \
      --predictions-s3 s3://project-bucket-placeholder/wind_training/inference/grid_offset/sar_final/data.parquet \
      --metadata-s3 s3://project-bucket-placeholder/wind_training/inference/grid_offset/sar_final/inference_metadata.json \
      --output-dir artifacts_root/grid_offset/inference_metrics/sar_final \
      --profile example-profile \
      --region example-region-1 \
      --wind-bin-column wind_bin \
      --truth-speed-col sar__owiwindspeed_mean \
      --truth-dir-col sar__owiwinddirection_mean
    bash scripts/geo_utils/build_stac_catalog.sh \
      --collection GRID_OFFSET_SAR_FINAL_INFERENCE \
      --s3-uri s3://project-bucket-placeholder/wind_training/inference/grid_offset/sar_final/ \
      --output-dir catalogs/grid_offset_pipeline/grid_offset_sar_final_inference \
      --profile example-profile \
      --region example-region-1
    echo ">>> Completed grid-offset inference for SAR final model at $(date) <<<"
  else
    echo ">>> Skipping SAR final inference: artifacts_root/sar/config/final_model.txt not found <<<"
  fi
else
  echo ">>> Skipping SAR final inference per configuration <<<"
fi

if [[ "${RUN_SAR_FINETUNED_INFERENCE}" == "1" ]]; then
  if [ -f artifacts_root/sar/config/final_model_finetuned.txt ]; then
    echo ">>> Running grid-offset inference for SAR fine-tuned model at $(date) <<<"
    ./scripts/inference/run_inference.sh \
      --model-artifact "s3://project-bucket-placeholder/wind_training/models/sar/finetuned/$(awk '{sub(/#.*/,"");gsub(/^[ \t]+|[ \t]+$/,"");if(length($0)){print;exit}}' artifacts_root/sar/config/final_model_finetuned.txt)/output/model.tar.gz" \
      --input-data s3://project-bucket-placeholder/wind_training/pivots_offset/materialized/sar_valid_source/ \
      --output-s3-uri "s3://project-bucket-placeholder/wind_training/inference/grid_offset/sar_finetuned" \
      --profile example-profile \
      --region example-region-1 \
      --output-format parquet
    bash scripts/geo_utils/finalize_geoparquet.sh \
      --db-name wind_training \
      --bucket-name project-bucket-placeholder \
      --output-prefix wind_training/inference/grid_offset/sar_finetuned \
      --output-table GRID_OFFSET_SAR_FINETUNED_INFERENCE \
      --register-table \
      --profile example-profile \
      --region example-region-1 \
      --log-dir artifacts_root/grid_offset/logs/finalize_geoparquet_inference
    mkdir -p example-profile/grid_offset/inference_metrics/sar_finetuned
    ./scripts/inference/compute_inference_metrics.sh \
      --predictions-s3 s3://project-bucket-placeholder/wind_training/inference/grid_offset/sar_finetuned/data.parquet \
      --metadata-s3 s3://project-bucket-placeholder/wind_training/inference/grid_offset/sar_finetuned/inference_metadata.json \
      --output-dir artifacts_root/grid_offset/inference_metrics/sar_finetuned \
      --profile example-profile \
      --region example-region-1 \
      --wind-bin-column wind_bin \
      --truth-speed-col sar__owiwindspeed_mean \
      --truth-dir-col sar__owiwinddirection_mean
    bash scripts/geo_utils/build_stac_catalog.sh \
      --collection GRID_OFFSET_SAR_FINETUNED_INFERENCE \
      --s3-uri s3://project-bucket-placeholder/wind_training/inference/grid_offset/sar_finetuned/ \
      --output-dir catalogs/grid_offset_pipeline/grid_offset_sar_finetuned_inference \
      --profile example-profile \
      --region example-region-1
    echo ">>> Completed grid-offset inference for SAR fine-tuned model at $(date) <<<"
  else
    echo ">>> Skipping SAR fine-tuned inference: artifacts_root/sar/config/final_model_finetuned.txt not found <<<"
  fi
else
  echo ">>> Skipping SAR fine-tuned inference per configuration <<<"
fi

if [[ "${RUN_SAR_FINETUNED_L2SP_INFERENCE}" == "1" ]]; then
  if [ -f artifacts_root/sar/config/final_model_finetuned_l2sp.txt ]; then
    echo ">>> Running grid-offset inference for SAR L2-SP model at $(date) <<<"
    ./scripts/inference/run_inference.sh \
      --model-artifact "s3://project-bucket-placeholder/wind_training/models/sar/finetuned_l2sp/$(awk '{sub(/#.*/,"");gsub(/^[ \t]+|[ \t]+$/,"");if(length($0)){print;exit}}' artifacts_root/sar/config/final_model_finetuned_l2sp.txt)/output/model.tar.gz" \
      --input-data s3://project-bucket-placeholder/wind_training/pivots_offset/materialized/sar_valid_source/ \
      --output-s3-uri "s3://project-bucket-placeholder/wind_training/inference/grid_offset/sar_finetuned_l2sp" \
      --profile example-profile \
      --region example-region-1 \
      --output-format parquet
    bash scripts/geo_utils/finalize_geoparquet.sh \
      --db-name wind_training \
      --bucket-name project-bucket-placeholder \
      --output-prefix wind_training/inference/grid_offset/sar_finetuned_l2sp \
      --output-table GRID_OFFSET_SAR_FINETUNED_L2SP_INFERENCE \
      --register-table \
      --profile example-profile \
      --region example-region-1 \
      --log-dir artifacts_root/grid_offset/logs/finalize_geoparquet_inference
    mkdir -p example-profile/grid_offset/inference_metrics/sar_finetuned_l2sp
    ./scripts/inference/compute_inference_metrics.sh \
      --predictions-s3 s3://project-bucket-placeholder/wind_training/inference/grid_offset/sar_finetuned_l2sp/data.parquet \
      --metadata-s3 s3://project-bucket-placeholder/wind_training/inference/grid_offset/sar_finetuned_l2sp/inference_metadata.json \
      --output-dir artifacts_root/grid_offset/inference_metrics/sar_finetuned_l2sp \
      --profile example-profile \
      --region example-region-1 \
      --wind-bin-column wind_bin \
      --truth-speed-col sar__owiwindspeed_mean \
      --truth-dir-col sar__owiwinddirection_mean
    bash scripts/geo_utils/build_stac_catalog.sh \
      --collection GRID_OFFSET_SAR_FINETUNED_L2SP_INFERENCE \
      --s3-uri s3://project-bucket-placeholder/wind_training/inference/grid_offset/sar_finetuned_l2sp/ \
      --output-dir catalogs/grid_offset_pipeline/grid_offset_sar_finetuned_l2sp_inference \
      --profile example-profile \
      --region example-region-1
    echo ">>> Completed grid-offset inference for SAR L2-SP model at $(date) <<<"
  else
    echo ">>> Skipping SAR L2-SP inference: artifacts_root/sar/config/final_model_finetuned_l2sp.txt not found <<<"
  fi
else
  echo ">>> Skipping SAR L2-SP inference per configuration <<<"
fi

if [[ "${RUN_SAR_FINETUNED_L2SP_KD_INFERENCE}" == "1" ]]; then
  if [ -f artifacts_root/sar/config/final_model_finetuned_l2sp_kd.txt ]; then
    echo ">>> Running grid-offset inference for SAR L2-SP+KD model at $(date) <<<"
    ./scripts/inference/run_inference.sh \
      --model-artifact "s3://project-bucket-placeholder/wind_training/models/sar/finetuned_l2sp_kd/$(awk '{sub(/#.*/,"");gsub(/^[ \t]+|[ \t]+$/,"");if(length($0)){print;exit}}' artifacts_root/sar/config/final_model_finetuned_l2sp_kd.txt)/output/model.tar.gz" \
      --input-data s3://project-bucket-placeholder/wind_training/pivots_offset/materialized/sar_valid_source/ \
      --output-s3-uri "s3://project-bucket-placeholder/wind_training/inference/grid_offset/sar_finetuned_l2sp_kd" \
      --profile example-profile \
      --region example-region-1 \
      --output-format parquet
    bash scripts/geo_utils/finalize_geoparquet.sh \
      --db-name wind_training \
      --bucket-name project-bucket-placeholder \
      --output-prefix wind_training/inference/grid_offset/sar_finetuned_l2sp_kd \
      --output-table GRID_OFFSET_SAR_FINETUNED_L2SP_KD_INFERENCE \
      --register-table \
      --profile example-profile \
      --region example-region-1 \
      --log-dir artifacts_root/grid_offset/logs/finalize_geoparquet_inference
    mkdir -p example-profile/grid_offset/inference_metrics/sar_finetuned_l2sp_kd
    ./scripts/inference/compute_inference_metrics.sh \
      --predictions-s3 s3://project-bucket-placeholder/wind_training/inference/grid_offset/sar_finetuned_l2sp_kd/data.parquet \
      --metadata-s3 s3://project-bucket-placeholder/wind_training/inference/grid_offset/sar_finetuned_l2sp_kd/inference_metadata.json \
      --output-dir artifacts_root/grid_offset/inference_metrics/sar_finetuned_l2sp_kd \
      --profile example-profile \
      --region example-region-1 \
      --wind-bin-column wind_bin \
      --truth-speed-col sar__owiwindspeed_mean \
      --truth-dir-col sar__owiwinddirection_mean
    bash scripts/geo_utils/build_stac_catalog.sh \
      --collection GRID_OFFSET_SAR_FINETUNED_L2SP_KD_INFERENCE \
      --s3-uri s3://project-bucket-placeholder/wind_training/inference/grid_offset/sar_finetuned_l2sp_kd/ \
      --output-dir catalogs/grid_offset_pipeline/grid_offset_sar_finetuned_l2sp_kd_inference \
      --profile example-profile \
      --region example-region-1
    echo ">>> Completed grid-offset inference for SAR L2-SP+KD model at $(date) <<<"
  else
    echo ">>> Skipping SAR L2-SP+KD inference: artifacts_root/sar/config/final_model_finetuned_l2sp_kd.txt not found <<<"
  fi
else
  echo ">>> Skipping SAR L2-SP+KD inference per configuration <<<"
fi

# -----------------------------------------------------------------------------
# Inference blocks: reference buoy models
# -----------------------------------------------------------------------------
if [[ "${RUN_BUOY_FINAL_INFERENCE}" == "1" ]]; then
  if [ -f artifacts_root/buoy/config/final_model.txt ]; then
    echo ">>> Running grid-offset inference for reference buoy final model at $(date) <<<"
    ./scripts/inference/run_inference.sh \
      --model-artifact "s3://project-bucket-placeholder/wind_training/models/reference_buoy/final/$(awk '{sub(/#.*/,"");gsub(/^[ \t]+|[ \t]+$/,"");if(length($0)){print;exit}}' artifacts_root/buoy/config/final_model.txt)/output/model.tar.gz" \
      --input-data s3://project-bucket-placeholder/wind_training/pivots_offset/materialized/sar_valid_source/ \
      --output-s3-uri "s3://project-bucket-placeholder/wind_training/inference/grid_offset/buoy_final" \
      --profile example-profile \
      --region example-region-1 \
      --output-format parquet
    bash scripts/geo_utils/finalize_geoparquet.sh \
      --db-name wind_training \
      --bucket-name project-bucket-placeholder \
      --output-prefix wind_training/inference/grid_offset/buoy_final \
      --output-table GRID_OFFSET_BUOY_FINAL_INFERENCE \
      --register-table \
      --profile example-profile \
      --region example-region-1 \
      --log-dir artifacts_root/grid_offset/logs/finalize_geoparquet_inference
    mkdir -p example-profile/grid_offset/inference_metrics/buoy_final
    ./scripts/inference/compute_inference_metrics.sh \
      --predictions-s3 s3://project-bucket-placeholder/wind_training/inference/grid_offset/buoy_final/data.parquet \
      --metadata-s3 s3://project-bucket-placeholder/wind_training/inference/grid_offset/buoy_final/inference_metadata.json \
      --output-dir artifacts_root/grid_offset/inference_metrics/buoy_final \
      --profile example-profile \
      --region example-region-1 \
      --wind-bin-column wind_bin \
      --truth-speed-col sar__owiwindspeed_mean \
      --truth-dir-col sar__owiwinddirection_mean
    bash scripts/geo_utils/build_stac_catalog.sh \
      --collection GRID_OFFSET_BUOY_FINAL_INFERENCE \
      --s3-uri s3://project-bucket-placeholder/wind_training/inference/grid_offset/buoy_final/ \
      --output-dir catalogs/grid_offset_pipeline/grid_offset_buoy_final_inference \
      --profile example-profile \
      --region example-region-1
    echo ">>> Completed grid-offset inference for reference buoy final model at $(date) <<<"
  else
    echo ">>> Skipping reference buoy final inference: artifacts_root/buoy/config/final_model.txt not found <<<"
  fi
else
  echo ">>> Skipping reference buoy final inference per configuration <<<"
fi

if [[ "${RUN_BUOY_FINETUNED_INFERENCE}" == "1" ]]; then
  if [ -f artifacts_root/buoy/config/final_model_finetuned.txt ]; then
    echo ">>> Running grid-offset inference for reference buoy fine-tuned model at $(date) <<<"
    ./scripts/inference/run_inference.sh \
      --model-artifact "s3://project-bucket-placeholder/wind_training/models/reference_buoy/finetuned/$(awk '{sub(/#.*/,"");gsub(/^[ \t]+|[ \t]+$/,"");if(length($0)){print;exit}}' artifacts_root/buoy/config/final_model_finetuned.txt)/output/model.tar.gz" \
      --input-data s3://project-bucket-placeholder/wind_training/pivots_offset/materialized/sar_valid_source/ \
      --output-s3-uri "s3://project-bucket-placeholder/wind_training/inference/grid_offset/buoy_finetuned" \
      --profile example-profile \
      --region example-region-1 \
      --output-format parquet
    bash scripts/geo_utils/finalize_geoparquet.sh \
      --db-name wind_training \
      --bucket-name project-bucket-placeholder \
      --output-prefix wind_training/inference/grid_offset/buoy_finetuned \
      --output-table GRID_OFFSET_BUOY_FINETUNED_INFERENCE \
      --register-table \
      --profile example-profile \
      --region example-region-1 \
      --log-dir artifacts_root/grid_offset/logs/finalize_geoparquet_inference
    mkdir -p example-profile/grid_offset/inference_metrics/buoy_finetuned
    ./scripts/inference/compute_inference_metrics.sh \
      --predictions-s3 s3://project-bucket-placeholder/wind_training/inference/grid_offset/buoy_finetuned/data.parquet \
      --metadata-s3 s3://project-bucket-placeholder/wind_training/inference/grid_offset/buoy_finetuned/inference_metadata.json \
      --output-dir artifacts_root/grid_offset/inference_metrics/buoy_finetuned \
      --profile example-profile \
      --region example-region-1 \
      --wind-bin-column wind_bin \
      --truth-speed-col sar__owiwindspeed_mean \
      --truth-dir-col sar__owiwinddirection_mean
    bash scripts/geo_utils/build_stac_catalog.sh \
      --collection GRID_OFFSET_BUOY_FINETUNED_INFERENCE \
      --s3-uri s3://project-bucket-placeholder/wind_training/inference/grid_offset/buoy_finetuned/ \
      --output-dir catalogs/grid_offset_pipeline/grid_offset_buoy_finetuned_inference \
      --profile example-profile \
      --region example-region-1
    echo ">>> Completed grid-offset inference for reference buoy fine-tuned model at $(date) <<<"
  else
    echo ">>> Skipping reference buoy fine-tuned inference: artifacts_root/buoy/config/final_model_finetuned.txt not found <<<"
  fi
else
  echo ">>> Skipping reference buoy fine-tuned inference per configuration <<<"
fi

if [[ "${RUN_BUOY_FINETUNED_L2SP_INFERENCE}" == "1" ]]; then
  if [ -f artifacts_root/buoy/config/final_model_finetuned_l2sp.txt ]; then
    echo ">>> Running grid-offset inference for reference buoy L2-SP model at $(date) <<<"
    ./scripts/inference/run_inference.sh \
      --model-artifact "s3://project-bucket-placeholder/wind_training/models/reference_buoy/finetuned_l2sp/$(awk '{sub(/#.*/,"");gsub(/^[ \t]+|[ \t]+$/,"");if(length($0)){print;exit}}' artifacts_root/buoy/config/final_model_finetuned_l2sp.txt)/output/model.tar.gz" \
      --input-data s3://project-bucket-placeholder/wind_training/pivots_offset/materialized/sar_valid_source/ \
      --output-s3-uri "s3://project-bucket-placeholder/wind_training/inference/grid_offset/buoy_finetuned_l2sp" \
      --profile example-profile \
      --region example-region-1 \
      --output-format parquet
    bash scripts/geo_utils/finalize_geoparquet.sh \
      --db-name wind_training \
      --bucket-name project-bucket-placeholder \
      --output-prefix wind_training/inference/grid_offset/buoy_finetuned_l2sp \
      --output-table GRID_OFFSET_BUOY_FINETUNED_L2SP_INFERENCE \
      --register-table \
      --profile example-profile \
      --region example-region-1 \
      --log-dir artifacts_root/grid_offset/logs/finalize_geoparquet_inference
    mkdir -p example-profile/grid_offset/inference_metrics/buoy_finetuned_l2sp
    ./scripts/inference/compute_inference_metrics.sh \
      --predictions-s3 s3://project-bucket-placeholder/wind_training/inference/grid_offset/buoy_finetuned_l2sp/data.parquet \
      --metadata-s3 s3://project-bucket-placeholder/wind_training/inference/grid_offset/buoy_finetuned_l2sp/inference_metadata.json \
      --output-dir artifacts_root/grid_offset/inference_metrics/buoy_finetuned_l2sp \
      --profile example-profile \
      --region example-region-1 \
      --wind-bin-column wind_bin \
      --truth-speed-col sar__owiwindspeed_mean \
      --truth-dir-col sar__owiwinddirection_mean
    bash scripts/geo_utils/build_stac_catalog.sh \
      --collection GRID_OFFSET_BUOY_FINETUNED_L2SP_INFERENCE \
      --s3-uri s3://project-bucket-placeholder/wind_training/inference/grid_offset/buoy_finetuned_l2sp/ \
      --output-dir catalogs/grid_offset_pipeline/grid_offset_buoy_finetuned_l2sp_inference \
      --profile example-profile \
      --region example-region-1
    echo ">>> Completed grid-offset inference for reference buoy L2-SP model at $(date) <<<"
  else
    echo ">>> Skipping reference buoy L2-SP inference: artifacts_root/buoy/config/final_model_finetuned_l2sp.txt not found <<<"
  fi
else
  echo ">>> Skipping reference buoy L2-SP inference per configuration <<<"
fi

if [[ "${RUN_BUOY_FINETUNED_L2SP_KD_INFERENCE}" == "1" ]]; then
  if [ -f artifacts_root/buoy/config/final_model_finetuned_l2sp_kd.txt ]; then
    echo ">>> Running grid-offset inference for reference buoy L2-SP+KD model at $(date) <<<"
    ./scripts/inference/run_inference.sh \
      --model-artifact "s3://project-bucket-placeholder/wind_training/models/reference_buoy/finetuned_l2sp_kd/$(awk '{sub(/#.*/,"");gsub(/^[ \t]+|[ \t]+$/,"");if(length($0)){print;exit}}' artifacts_root/buoy/config/final_model_finetuned_l2sp_kd.txt)/output/model.tar.gz" \
      --input-data s3://project-bucket-placeholder/wind_training/pivots_offset/materialized/sar_valid_source/ \
      --output-s3-uri "s3://project-bucket-placeholder/wind_training/inference/grid_offset/buoy_finetuned_l2sp_kd" \
      --profile example-profile \
      --region example-region-1 \
      --output-format parquet
    bash scripts/geo_utils/finalize_geoparquet.sh \
      --db-name wind_training \
      --bucket-name project-bucket-placeholder \
      --output-prefix wind_training/inference/grid_offset/buoy_finetuned_l2sp_kd \
      --output-table GRID_OFFSET_BUOY_FINETUNED_L2SP_KD_INFERENCE \
      --register-table \
      --profile example-profile \
      --region example-region-1 \
      --log-dir artifacts_root/grid_offset/logs/finalize_geoparquet_inference
    mkdir -p example-profile/grid_offset/inference_metrics/buoy_finetuned_l2sp_kd
    ./scripts/inference/compute_inference_metrics.sh \
      --predictions-s3 s3://project-bucket-placeholder/wind_training/inference/grid_offset/buoy_finetuned_l2sp_kd/data.parquet \
      --metadata-s3 s3://project-bucket-placeholder/wind_training/inference/grid_offset/buoy_finetuned_l2sp_kd/inference_metadata.json \
      --output-dir artifacts_root/grid_offset/inference_metrics/buoy_finetuned_l2sp_kd \
      --profile example-profile \
      --region example-region-1 \
      --wind-bin-column wind_bin \
      --truth-speed-col sar__owiwindspeed_mean \
      --truth-dir-col sar__owiwinddirection_mean
    bash scripts/geo_utils/build_stac_catalog.sh \
      --collection GRID_OFFSET_BUOY_FINETUNED_L2SP_KD_INFERENCE \
      --s3-uri s3://project-bucket-placeholder/wind_training/inference/grid_offset/buoy_finetuned_l2sp_kd/ \
      --output-dir catalogs/grid_offset_pipeline/grid_offset_buoy_finetuned_l2sp_kd_inference \
      --profile example-profile \
      --region example-region-1
    echo ">>> Completed grid-offset inference for reference buoy L2-SP+KD model at $(date) <<<"
  else
    echo ">>> Skipping reference buoy L2-SP+KD inference: artifacts_root/buoy/config/final_model_finetuned_l2sp_kd.txt not found <<<"
  fi
else
  echo ">>> Skipping reference buoy L2-SP+KD inference per configuration <<<"
fi

if [[ "${RUN_COMBINED_FINAL_INFERENCE}" == "1" ]]; then
  if [ -f artifacts_root/sar_buoy/config/final_model.txt ]; then
    echo ">>> Running grid-offset inference for SAR+reference buoy combined model at $(date) <<<"
    ./scripts/inference/run_inference.sh \
      --model-artifact "s3://project-bucket-placeholder/wind_training/models/sar_buoy/final/$(awk '{sub(/#.*/,"");gsub(/^[ \t]+|[ \t]+$/,"");if(length($0)){print;exit}}' artifacts_root/sar_buoy/config/final_model.txt)/output/model.tar.gz" \
      --input-data s3://project-bucket-placeholder/wind_training/pivots_offset/materialized/sar_valid_source/ \
      --output-s3-uri "s3://project-bucket-placeholder/wind_training/inference/grid_offset/combined_final" \
      --profile example-profile \
      --region example-region-1 \
      --output-format parquet
    bash scripts/geo_utils/finalize_geoparquet.sh \
      --db-name wind_training \
      --bucket-name project-bucket-placeholder \
      --output-prefix wind_training/inference/grid_offset/combined_final \
      --output-table GRID_OFFSET_COMBINED_FINAL_INFERENCE \
      --register-table \
      --profile example-profile \
      --region example-region-1 \
      --log-dir artifacts_root/grid_offset/logs/finalize_geoparquet_inference
    mkdir -p example-profile/grid_offset/inference_metrics/combined_final
    ./scripts/inference/compute_inference_metrics.sh \
      --predictions-s3 s3://project-bucket-placeholder/wind_training/inference/grid_offset/combined_final/data.parquet \
      --metadata-s3 s3://project-bucket-placeholder/wind_training/inference/grid_offset/combined_final/inference_metadata.json \
      --output-dir artifacts_root/grid_offset/inference_metrics/combined_final \
      --profile example-profile \
      --region example-region-1 \
      --wind-bin-column wind_bin \
      --truth-speed-col sar__owiwindspeed_mean \
      --truth-dir-col sar__owiwinddirection_mean \
      --group-column wind_source
    bash scripts/geo_utils/build_stac_catalog.sh \
      --collection GRID_OFFSET_COMBINED_FINAL_INFERENCE \
      --s3-uri s3://project-bucket-placeholder/wind_training/inference/grid_offset/combined_final/ \
      --output-dir catalogs/grid_offset_pipeline/grid_offset_combined_final_inference \
      --profile example-profile \
      --region example-region-1
    echo ">>> Completed grid-offset inference for SAR+reference buoy combined model at $(date) <<<"
  else
    echo ">>> Skipping SAR+reference buoy combined inference: artifacts_root/sar_buoy/config/final_model.txt not found <<<"
  fi
else
  echo ">>> Skipping SAR+reference buoy combined inference per configuration <<<"
fi

echo "=== Grid-offset evaluation pipeline completed at $(date) ==="
