#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Author: Example Team <team@example.org>
# Created: 2025-10-16
# Disclaimer: This obfuscated sample mirrors the data preparation workflow and must be adapted before any deployment.
# -----------------------------------------------------------------------------

# Exit immediately on error, undefined variable, or pipeline failure
set -euo pipefail

# Example data-preparation pipeline (obfuscated).
# Replace placeholders such as example-profile, example-region-1, project-bucket-placeholder, and wind_training.* with deployment-specific values before running.

echo "=== Pipeline started at $(date) ==="

mkdir -p \
  artifacts_root/pivot_and_join \
  artifacts_root/pivot_and_join/logs \
  artifacts_root/pivot_and_join/stac_config \
  artifacts_root/pivot_and_join/sql \
  artifacts_root/pivot_and_join/stac_catalogs \
  artifacts_root/sar/reports/partition \
  artifacts_root/sar/logs/partition \
  artifacts_root/sar/logs/finalize_geoparquet_train \
  artifacts_root/sar/logs/finalize_geoparquet_test \
  artifacts_root/buoy/reports/partition \
  artifacts_root/buoy/logs/partition \
  artifacts_root/buoy/logs/finalize_geoparquet_train \
  artifacts_root/buoy/logs/finalize_geoparquet_test \
  artifacts_root/sar_buoy/reports/partition \
  artifacts_root/sar_buoy/logs/partition \
  artifacts_root/sar_buoy/logs/finalize_geoparquet_train \
  artifacts_root/sar_buoy/logs/finalize_geoparquet_test

  # ---------------------------------------------------------------------------
  # Pivot and join aggregated tables (customise paths as needed)
  echo ">>> Pivot aggregated tables at $(date) <<<"
  ./scripts/aggregation/pivot_tables.sh \
    --pivot wind_training.RADAR_A_AGGREGATED=wind_training.RADAR_A_PIVOT@s3://project-bucket-placeholder/training/pivots/radar_a/ \
    --pivot wind_training.RADAR_B_AGGREGATED=wind_training.RADAR_B_PIVOT@s3://project-bucket-placeholder/training/pivots/radar_b/ \
    --profile example-profile \
    --region example-region-1 \
    --require-geometry-unique \
    --results-s3 s3://project-bucket-placeholder/athena-query-results/
  echo ">>> Completed pivot aggregated tables at $(date) <<<"

  echo ">>> Adding station bearing and distance features to pivot tables at $(date) <<<"
  ./scripts/geo_utils/add_station_bearing_distance_view.sh \
    --source wind_training.RADAR_A_PIVOT \
    --view wind_training.RADAR_A_PIVOT_FEATURES \
    --station radar_a:12.345678:-45.678912 \
    --results-s3 s3://project-bucket-placeholder/athena-query-results/ \
    --profile example-profile \
    --region example-region-1
  ./scripts/geo_utils/add_station_bearing_distance_view.sh \
    --source wind_training.RADAR_B_PIVOT \
    --view wind_training.RADAR_B_PIVOT_FEATURES \
    --station radar_b:11.223344:-44.556677 \
    --results-s3 s3://project-bucket-placeholder/athena-query-results/ \
    --profile example-profile \
    --region example-region-1
  echo ">>> Completed station bearing and distance feature views at $(date) <<<"

  echo ">>> Creating maintenance-enriched tables at $(date) <<<"
  ./scripts/aggregation/attach_station_maintenance_table.sh \
    --source wind_training.RADAR_A_PIVOT_FEATURES \
    --out wind_training.RADAR_A_PIVOT_FEATURES_MAINT@s3://project-bucket-placeholder/training/pivots/radar_a_with_maintenance/ \
    --maintenance-csv maintenance_events.csv \
    --station-id radar_a \
    --prefix radar_a \
    --timestamp-col timestamp \
    --results-s3 s3://project-bucket-placeholder/athena-query-results/ \
    --profile example-profile \
    --region example-region-1
  ./scripts/aggregation/attach_station_maintenance_table.sh \
    --source wind_training.RADAR_B_PIVOT_FEATURES \
    --out wind_training.RADAR_B_PIVOT_FEATURES_MAINT@s3://project-bucket-placeholder/training/pivots/radar_b_with_maintenance/ \
    --maintenance-csv maintenance_events.csv \
    --station-id radar_b \
    --prefix radar_b \
    --timestamp-col timestamp \
    --results-s3 s3://project-bucket-placeholder/athena-query-results/ \
    --profile example-profile \
    --region example-region-1
  echo ">>> Completed maintenance tables at $(date) <<<"

  echo ">>> Applying 10 m wind speed correction to raw reference-buoy measurements at $(date) <<<"
  bash ./scripts/aggregation/apply_buoy_wind_height_correction.sh \
    --source wind_training.REFERENCE_BUOY \
    --target-table wind_training.REFERENCE_BUOY_HEIGHT10M \
    --s3-output s3://project-bucket-placeholder/training/reference_buoy_height10m/ \
    --results-s3 s3://project-bucket-placeholder/athena-query-results/ \
    --source-height 3 \
    --target-height 10 \
    --roughness-length 0.0002 \
    --profile example-profile \
    --region example-region-1 \
    --log-dir artifacts_root/pivot_and_join/logs
  echo ">>> Completed raw reference-buoy correction at $(date) <<<"

  echo ">>> Joining pivoted tables at $(date) <<<"
  ./scripts/aggregation/join_pivoted_tables.sh \
    --source wind_training.RADAR_A_PIVOT_FEATURES_MAINT \
    --source wind_training.RADAR_B_PIVOT_FEATURES_MAINT \
    --union-out wind_training.PIVOTS_JOINED@s3://project-bucket-placeholder/training/pivots/joined/ \
    --sar wind_training.SAR_AGGREGATED \
    --sar-out wind_training.PIVOTS_SAR@s3://project-bucket-placeholder/training/pivots/with_sar/ \
    --buoy wind_training.REFERENCE_BUOY_HEIGHT10M \
    --buoy-out wind_training.PIVOTS_REFERENCE_BUOY@s3://project-bucket-placeholder/training/pivots/with_reference_buoy/ \
    --profile example-profile \
    --region example-region-1 \
    --results-s3 s3://project-bucket-placeholder/athena-query-results/
  echo ">>> Completed join of pivoted tables at $(date) <<<"

  echo ">>> Creating filtered view for valid reference-buoy wind data at $(date) <<<"
  ./scripts/aggregation/create_filtered_view.sh \
    --source wind_training.PIVOTS_REFERENCE_BUOY \
    --view wind_training.PIVOTS_REFERENCE_BUOY_VALID \
    --sql-file artifacts_root/pivot_and_join/sql/pivots_reference_buoy_valid_wind.sql \
    --results-s3 s3://project-bucket-placeholder/athena-query-results/ \
    --profile example-profile \
    --region example-region-1
  echo ">>> Completed filtered view creation at $(date) <<<"

  echo ">>> Creating filtered view for valid SAR wind aggregates at $(date) <<<"
  ./scripts/aggregation/create_filtered_view.sh \
    --source wind_training.PIVOTS_SAR \
    --view wind_training.PIVOTS_SAR_VALID \
    --sql-file artifacts_root/pivot_and_join/sql/pivots_sar_valid_wind.sql \
    --results-s3 s3://project-bucket-placeholder/athena-query-results/ \
    --profile example-profile \
    --region example-region-1
  echo ">>> Completed SAR filtered view creation at $(date) <<<"

  echo ">>> Annotating SAR and reference-buoy views with source labels at $(date) <<<"
  ./scripts/aggregation/create_filtered_view.sh \
    --source wind_training.PIVOTS_SAR_VALID \
    --view wind_training.PIVOTS_SAR_VALID_SOURCE \
    --sql-file artifacts_root/pivot_and_join/sql/pivots_sar_valid_with_source.sql \
    --results-s3 s3://project-bucket-placeholder/athena-query-results/ \
    --profile example-profile \
    --region example-region-1
  ./scripts/aggregation/create_filtered_view.sh \
    --source wind_training.PIVOTS_REFERENCE_BUOY_VALID \
    --view wind_training.PIVOTS_REFERENCE_BUOY_VALID_SOURCE \
    --sql-file artifacts_root/pivot_and_join/sql/pivots_reference_buoy_valid_with_source.sql \
    --results-s3 s3://project-bucket-placeholder/athena-query-results/ \
    --profile example-profile \
    --region example-region-1
  echo ">>> Completed source annotation at $(date) <<<"

  echo ">>> Annotating SAR and reference-buoy partitions with source metadata at $(date) <<<"
  ./scripts/aggregation/create_filtered_view.sh \
    --source wind_training.PIVOTS_SAR_VALID_train \
    --view wind_training.PIVOTS_SAR_VALID_TRAIN_SOURCE \
    --sql-file artifacts_root/pivot_and_join/sql/pivots_sar_valid_with_source.sql \
    --results-s3 s3://project-bucket-placeholder/athena-query-results/ \
    --profile example-profile \
    --region example-region-1
  ./scripts/aggregation/create_filtered_view.sh \
    --source wind_training.PIVOTS_SAR_VALID_test \
    --view wind_training.PIVOTS_SAR_VALID_TEST_SOURCE \
    --sql-file artifacts_root/pivot_and_join/sql/pivots_sar_valid_with_source.sql \
    --results-s3 s3://project-bucket-placeholder/athena-query-results/ \
    --profile example-profile \
    --region example-region-1
  ./scripts/aggregation/create_filtered_view.sh \
    --source wind_training.PIVOTS_REFERENCE_BUOY_VALID_train \
    --view wind_training.PIVOTS_REFERENCE_BUOY_VALID_TRAIN_SOURCE \
    --sql-file artifacts_root/pivot_and_join/sql/pivots_reference_buoy_valid_with_source.sql \
    --results-s3 s3://project-bucket-placeholder/athena-query-results/ \
    --profile example-profile \
    --region example-region-1
  ./scripts/aggregation/create_filtered_view.sh \
    --source wind_training.PIVOTS_REFERENCE_BUOY_VALID_test \
    --view wind_training.PIVOTS_REFERENCE_BUOY_VALID_TEST_SOURCE \
    --sql-file artifacts_root/pivot_and_join/sql/pivots_reference_buoy_valid_with_source.sql \
    --results-s3 s3://project-bucket-placeholder/athena-query-results/ \
    --profile example-profile \
    --region example-region-1
  echo ">>> Completed partition source annotation at $(date) <<<"

  echo ">>> Building partition report for SAR training/test at $(date) <<<"
  ./scripts/partition/partition_report.sh \
    --table wind_training.PIVOTS_SAR_VALID \
    --id-column node_id \
    --wind-column wind_speed \
    --wind-bin-column wind_bin \
    --output artifacts_root/sar/reports/partition/pivots_sar_partition_report.md \
    --profile example-profile \
    --region example-region-1
  echo ">>> Completed SAR partition report at $(date) <<<"

  echo ">>> Building partition report for reference-buoy training/test at $(date) <<<"
  ./scripts/partition/partition_report.sh \
    --table wind_training.PIVOTS_REFERENCE_BUOY_VALID \
    --id-column node_id \
    --wind-column wind_speed \
    --wind-bin-column wind_bin \
    --output artifacts_root/buoy/reports/partition/pivots_reference_buoy_partition_report.md \
    --profile example-profile \
    --region example-region-1
  echo ">>> Completed reference-buoy partition report at $(date) <<<"

  echo ">>> Building partition report for combined SAR+reference-buoy training/test at $(date) <<<"
  ./scripts/partition/partition_report.sh \
    --table wind_training.PIVOTS_JOINED \
    --id-column node_id \
    --wind-column wind_speed \
    --wind-bin-column wind_bin \
    --output artifacts_root/sar_buoy/reports/partition/pivots_sar_buoy_partition_report.md \
    --profile example-profile \
    --region example-region-1
  echo ">>> Completed combined partition report at $(date) <<<"

  echo ">>> Finalising GeoParquet for SAR training partition at $(date) <<<"
  ./scripts/geo_utils/finalize_geoparquet.sh \
    --db-name wind_training \
    --bucket-name project-bucket-placeholder \
    --output-prefix training/sar/train \
    --output-table SAR_TRAINING \
    --register-table \
    --profile example-profile \
    --region example-region-1 \
    --log-dir artifacts_root/sar/logs/finalize_geoparquet_train
  echo ">>> Completed GeoParquet finalisation for SAR training at $(date) <<<"

  echo ">>> Finalising GeoParquet for SAR test partition at $(date) <<<"
  ./scripts/geo_utils/finalize_geoparquet.sh \
    --db-name wind_training \
    --bucket-name project-bucket-placeholder \
    --output-prefix training/sar/test \
    --output-table SAR_TEST \
    --register-table \
    --profile example-profile \
    --region example-region-1 \
    --log-dir artifacts_root/sar/logs/finalize_geoparquet_test
  echo ">>> Completed GeoParquet finalisation for SAR test at $(date) <<<"

  echo ">>> Finalising GeoParquet for reference-buoy training partition at $(date) <<<"
  ./scripts/geo_utils/finalize_geoparquet.sh \
    --db-name wind_training \
    --bucket-name project-bucket-placeholder \
    --output-prefix training/buoy/train \
    --output-table BUOY_TRAINING \
    --register-table \
    --profile example-profile \
    --region example-region-1 \
    --log-dir artifacts_root/buoy/logs/finalize_geoparquet_train
  echo ">>> Completed GeoParquet finalisation for reference-buoy training at $(date) <<<"

  echo ">>> Finalising GeoParquet for reference-buoy test partition at $(date) <<<"
  ./scripts/geo_utils/finalize_geoparquet.sh \
    --db-name wind_training \
    --bucket-name project-bucket-placeholder \
    --output-prefix training/buoy/test \
    --output-table BUOY_TEST \
    --register-table \
    --profile example-profile \
    --region example-region-1 \
    --log-dir artifacts_root/buoy/logs/finalize_geoparquet_test
  echo ">>> Completed GeoParquet finalisation for reference-buoy test at $(date) <<<"

  echo ">>> Finalising GeoParquet for combined SAR+reference-buoy training partition at $(date) <<<"
  ./scripts/geo_utils/finalize_geoparquet.sh \
    --db-name wind_training \
    --bucket-name project-bucket-placeholder \
    --output-prefix training/sar_buoy/train \
    --output-table SAR_BUOY_TRAINING \
    --register-table \
    --profile example-profile \
    --region example-region-1 \
    --log-dir artifacts_root/sar_buoy/logs/finalize_geoparquet_train
  echo ">>> Completed GeoParquet finalisation for combined training at $(date) <<<"

  echo ">>> Finalising GeoParquet for combined SAR+reference-buoy test partition at $(date) <<<"
  ./scripts/geo_utils/finalize_geoparquet.sh \
    --db-name wind_training \
    --bucket-name project-bucket-placeholder \
    --output-prefix training/sar_buoy/test \
    --output-table SAR_BUOY_TEST \
    --register-table \
    --profile example-profile \
    --region example-region-1 \
    --log-dir artifacts_root/sar_buoy/logs/finalize_geoparquet_test
  echo ">>> Completed GeoParquet finalisation for combined test at $(date) <<<"

  echo ">>> Building STAC catalogues for partition outputs at $(date) <<<"
  ./scripts/geo_utils/build_stac_catalog.sh \
    --collection SAR_TRAINING \
    --s3-uri s3://project-bucket-placeholder/training/sar/train/ \
    --stac-collection-properties-json artifacts_root/sar/stac_config/stac_properties_collection_SAR_TRAINING.json \
    --stac-item-properties-json artifacts_root/sar/stac_config/stac_properties_item_SAR_TRAINING.json \
    --output-dir catalogs/example/sar_training \
    --profile example-profile \
    --region example-region-1
  ./scripts/geo_utils/build_stac_catalog.sh \
    --collection BUOY_TRAINING \
    --s3-uri s3://project-bucket-placeholder/training/buoy/train/ \
    --stac-collection-properties-json artifacts_root/buoy/stac_config/stac_properties_collection_BUOY_TRAINING.json \
    --stac-item-properties-json artifacts_root/buoy/stac_config/stac_properties_item_BUOY_TRAINING.json \
    --output-dir catalogs/example/buoy_training \
    --profile example-profile \
    --region example-region-1
  echo ">>> Completed STAC catalogues at $(date) <<<"

echo "=== Pipeline finished at $(date) ==="
