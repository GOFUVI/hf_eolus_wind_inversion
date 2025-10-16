#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Author: Example Team <team@example.org>
# Created: 2025-10-16
# Disclaimer: This obfuscated sample mirrors the SAR workflow and must be adapted before any deployment.
# -----------------------------------------------------------------------------

# Exit immediately on error, undefined variable, or pipeline failure
set -euo pipefail

# Example SAR pipeline (obfuscated).
# Replace placeholders such as example-profile, example-region-1, project-bucket-placeholder, and wind_training.* before running.

# Inference block toggles (set to 0 to skip a block)
RUN_SAR_FINAL_TEST_INFERENCE=1
RUN_SAR_FINAL_PIVOTS_INFERENCE=1
RUN_SAR_ON_BUOY_INFERENCE=1
RUN_SAR_FINETUNED_TEST_INFERENCE=1
RUN_SAR_FINETUNED_PIVOTS_INFERENCE=1
RUN_SAR_FINETUNED_ON_BUOY_INFERENCE=1
RUN_SAR_L2SP_TRAINING=1
RUN_SAR_L2SP_TEST_INFERENCE=1
RUN_SAR_L2SP_ON_BUOY_INFERENCE=1
RUN_SAR_L2SP_KD_TRAINING=1
RUN_SAR_L2SP_KD_TEST_INFERENCE=1
RUN_SAR_L2SP_KD_ON_BUOY_INFERENCE=1
RUN_SAR_FEATURE_IMPORTANCE_FINAL=1
RUN_SAR_FEATURE_IMPORTANCE_FINETUNED=1
RUN_SAR_FEATURE_IMPORTANCE_L2SP=1
RUN_SAR_FEATURE_IMPORTANCE_L2SP_KD=1

# Pipeline: partition for buoy training 10km observational range

echo "=== Pipeline started at $(date) ==="

build_analysis_image() {
  if [[ "${ANALYSIS_IMAGE_BUILT:-0}" == "0" ]]; then
    echo ">>> Building analysis Docker image at $(date) <<<"
    docker build -t example-wind-analysis:latest -f scripts/analysis/Dockerfile .
    ANALYSIS_IMAGE_BUILT=1
  fi
}

ensure_analysis_dirs() {
  mkdir -p \
    artifacts_root/analysis/models \
    artifacts_root/analysis/data \
    artifacts_root/analysis/logs
}

mkdir -p \
  artifacts_root/sar \
  artifacts_root/sar/reports/partition \
  artifacts_root/sar/reports/hpo \
  artifacts_root/sar/inference_metrics \
  artifacts_root/sar/train_metrics \
  artifacts_root/sar/normalization_params \
  artifacts_root/sar/bin_metrics \
  artifacts_root/sar/config \
  artifacts_root/sar/stac_config \
  artifacts_root/sar/final_training \
  artifacts_root/sar/fine_tuning \
  artifacts_root/sar/fine_tuning/train_metrics \
  artifacts_root/sar/fine_tuning/normalization_params \
  artifacts_root/sar/fine_tuning/bin_metrics \
  artifacts_root/sar/fine_tuning_l2sp \
  artifacts_root/sar/fine_tuning_l2sp_kd \
  artifacts_root/sar/logs \
  artifacts_root/sar/logs/partition \
  artifacts_root/sar/logs/finalize_geoparquet_train \
  artifacts_root/sar/logs/finalize_geoparquet_test \
  artifacts_root/sar/logs/hpo \
  artifacts_root/sar/logs/train_model \
  artifacts_root/sar/logs/fine_tune_model \
  artifacts_root/sar/logs/finalize_geoparquet_inference_test \
  artifacts_root/sar/logs/finalize_geoparquet_inference_pivots

if [ -f artifacts_root/sar/config/selected_model.txt ]; then
  echo ">>> Selected model file found; skipping SAR HPO <<<"
else
  echo ">>> Starting HPO for sar-hpo at $(date) <<<"
  ./scripts/HPO/run_hpo.sh \
    --profile example-profile \
    --region example-region-1 \
    --job-name sar-hpo-campaign-i-1 \
    --train-data-uri s3://project-bucket-placeholder/wind_training/training/sar/train/ \
    --output-s3-uri s3://project-bucket-placeholder/wind_training/models/sar \
    --model-config artifacts_root/sar/config/sar_model.json \
    --hpo-config artifacts_root/sar/config/sar_hpo.json \
    --log-dir artifacts_root/sar/logs/hpo/sar-hpo-campaign-i-1
  echo ">>> Completed HPO for sar-hpo at $(date) <<<"

  echo ">>> Starting follow-up HPO for sar-hpo-2 (warm start) at $(date) <<<"
  ./scripts/HPO/run_hpo.sh \
    --profile example-profile \
    --region example-region-1 \
    --job-name sar-hpo-campaign-i-2 \
    --parent-jobs sar-hpo-campaign-i-1 \
    --train-data-uri s3://project-bucket-placeholder/wind_training/training/sar/train/ \
    --output-s3-uri s3://project-bucket-placeholder/wind_training/models/sar \
    --model-config artifacts_root/sar/config/sar_model.json \
    --hpo-config artifacts_root/sar/config/sar_hpo.json \
    --log-dir artifacts_root/sar/logs/hpo/sar-hpo-campaign-i-2
  echo ">>> Completed HPO for sar-hpo-2 at $(date) <<<"

  echo ">>> Integrating SAR HPO reports at $(date) <<<"
  ./scripts/HPO/hpo_metrics_report.sh -p example-profile -r example-region-1 -n sar-hpo-campaign-i-1 -o artifacts_root/sar/reports/hpo/sar-hpo-campaign-i-1_hpo_report.md
  ./scripts/HPO/hpo_metrics_report.sh -p example-profile -r example-region-1 -n sar-hpo-campaign-i-2 -o artifacts_root/sar/reports/hpo/sar-hpo-campaign-i-2_hpo_report.md
  ./scripts/HPO/integrate_hpo_reports.sh -i "artifacts_root/sar/reports/hpo/sar-hpo-campaign-i-2_hpo_report.md,artifacts_root/sar/reports/hpo/sar-hpo-campaign-i-1_hpo_report.md" -o artifacts_root/sar/reports/hpo/sar-hpo_all_hpo_report.md
  echo ">>> Completed integrating SAR HPO reports at $(date) <<<"

  ./scripts/HPO/select_best_hpo_job.sh --report artifacts_root/sar/reports/hpo/sar-hpo_all_hpo_report.md --output artifacts_root/sar/config/selected_model.txt
fi

if [ -f artifacts_root/sar/config/final_model.txt ]; then
  echo ">>> Final model file found; skipping SAR training stage <<<"
else
  echo ">>> Generating SAR model config from HPO at $(date) <<<"
  ./scripts/training/generate_model_config_from_hpo.sh \
    --train-job "$(awk '{sub(/#.*/,"");gsub(/^[ \t]+|[ \t]+$/,"");if(length($0)){print;exit}}' artifacts_root/sar/config/selected_model.txt)" \
    --profile example-profile \
    --region example-region-1 \
    --output artifacts_root/sar/config/sar_hpo_final_model.json
  echo ">>> Completed SAR model config generation at $(date) <<<"

  echo ">>> Starting SAR full-training run at $(date) <<<"
  ./scripts/training/train_model.sh \
    --s3-prefix s3://project-bucket-placeholder/wind_training/models/sar/final \
    --profile example-profile \
    --region example-region-1 \
    --train-data-uri s3://project-bucket-placeholder/wind_training/training/sar/train/ \
    --job-base-prefix sar-range-example \
    --model-config artifacts_root/sar/config/sar_hpo_final_model.json \
    --output-dir artifacts_root/sar/final_training \
    --no-cv \
    --seed 42
  echo ">>> Completed SAR full-training run at $(date) <<<"

  ./scripts/training/record_final_job.sh --log artifacts_root/sar/final_training/train_model.log --output artifacts_root/sar/config/final_model.txt
  echo ">>> Stored SAR final training job $(sed -n '1p' artifacts_root/sar/config/final_model.txt) in artifacts_root/sar/config/final_model.txt <<<"
fi

if [ ! -f artifacts_root/sar/config/final_model.txt ]; then
  echo "Error: artifacts_root/sar/config/final_model.txt not found. Remove this file to retrain or ensure training completed successfully." >&2
  exit 1
fi

echo ">>> Using SAR final training job $(awk '{sub(/#.*/,"");gsub(/^[ \t]+|[ \t]+$/,"");if(length($0)){print;exit}}' artifacts_root/sar/config/final_model.txt) for downstream steps <<<"

# Fetch training metrics
  echo ">>> Fetching training metrics at $(date) <<<"
  ./scripts/training/get_train_metrics.sh \
    --aws-profile example-profile \
    --train-job-name "$(awk '{sub(/#.*/,"");gsub(/^[ \t]+|[ \t]+$/,"");if(length($0)){print;exit}}' artifacts_root/sar/config/final_model.txt)" \
    --output-directory artifacts_root/sar/train_metrics
  ./scripts/training/get_norm_params.sh \
    --aws-profile example-profile \
    --train-job-name "$(awk '{sub(/#.*/,"");gsub(/^[ \t]+|[ \t]+$/,"");if(length($0)){print;exit}}' artifacts_root/sar/config/final_model.txt)" \
    --output-directory artifacts_root/sar/normalization_params
  ./scripts/training/get_bin_metrics.sh \
    --aws-profile example-profile \
    --train-job-name "$(awk '{sub(/#.*/,"");gsub(/^[ \t]+|[ \t]+$/,"");if(length($0)){print;exit}}' artifacts_root/sar/config/final_model.txt)" \
    --output-directory artifacts_root/sar/bin_metrics
  echo ">>> Completed fetching training metrics at $(date) <<<"
  
  if [[ "${RUN_SAR_FINAL_TEST_INFERENCE}" == "1" ]]; then
    echo ">>> Running SAR inference on test set at $(date) <<<"
    ./scripts/inference/run_inference.sh \
      --model-artifact "s3://project-bucket-placeholder/wind_training/models/sar/final/$(awk '{sub(/#.*/,"");gsub(/^[ \t]+|[ \t]+$/,"");if(length($0)){print;exit}}' artifacts_root/sar/config/final_model.txt)/output/model.tar.gz" \
      --input-data s3://project-bucket-placeholder/wind_training/training/sar/test/ \
      --output-s3-uri "s3://project-bucket-placeholder/wind_training/inference/sar-range-example/test_set" \
      --profile example-profile \
      --region example-region-1 \
      --output-format parquet
    echo ">>> Completed SAR inference at $(date) <<<"

    echo ">>> Finalizing GeoParquet for SAR inference (test set) at $(date) <<<"
    ./scripts/geo_utils/finalize_geoparquet.sh \
      --db-name wind_training \
      --bucket-name project-bucket-placeholder \
      --output-prefix wind_training/inference/sar-range-example/test_set \
      --output-table SAR_RANGE_EXAMPLE_TEST_INFERENCE \
      --register-table \
      --profile example-profile \
      --region example-region-1 \
      --log-dir artifacts_root/sar/logs/finalize_geoparquet_inference_test
    echo ">>> Completed GeoParquet finalization for SAR inference (test set) at $(date) <<<"

    echo ">>> Computing SAR inference metrics at $(date) <<<"
    mkdir -p "artifacts_root/sar/inference_metrics/$(awk '{sub(/#.*/,"");gsub(/^[ \t]+|[ \t]+$/,"");if(length($0)){print;exit}}' artifacts_root/sar/config/final_model.txt)"
    ./scripts/inference/compute_inference_metrics.sh \
      --predictions-s3 s3://project-bucket-placeholder/wind_training/inference/sar-range-example/test_set/data.parquet \
      --metadata-s3 s3://project-bucket-placeholder/wind_training/inference/sar-range-example/test_set/inference_metadata.json \
      --output-dir "artifacts_root/sar/inference_metrics/$(awk '{sub(/#.*/,"");gsub(/^[ \t]+|[ \t]+$/,"");if(length($0)){print;exit}}' artifacts_root/sar/config/final_model.txt)" \
      --wind-bin-column wind_bin \
      --profile example-profile \
      --region example-region-1
    echo ">>> Completed SAR inference metrics at $(date) <<<"

    echo ">>> Building STAC catalog for SAR inference (test set) at $(date) <<<"
    ./scripts/geo_utils/build_stac_catalog.sh \
      --collection SAR_RANGE_EXAMPLE_TEST_INFERENCE \
      --s3-uri s3://project-bucket-placeholder/wind_training/inference/sar-range-example/test_set/ \
      --stac-collection-properties-json artifacts_root/sar/stac_config/stac_properties_collection_SAR_RANGE_EXAMPLE_TEST_SET.json \
      --stac-item-properties-json artifacts_root/sar/stac_config/stac_properties_item_SAR_RANGE_EXAMPLE_TEST_SET.json \
      --output-dir catalogs/sar_pipeline/sar_range_final_test_set \
      --profile example-profile \
      --region example-region-1
    echo ">>> Completed STAC catalog build for SAR inference (test set) at $(date) <<<"
  else
    echo ">>> Skipping SAR final test-set inference per configuration <<<"
  fi

  if [[ "${RUN_SAR_FINAL_PIVOTS_INFERENCE}" == "1" ]]; then
    echo ">>> Running SAR inference on PIVOTS_JOINED (unlabeled) at $(date) <<<"
    ./scripts/inference/run_inference.sh \
      --model-artifact "s3://project-bucket-placeholder/wind_training/models/sar/final/$(awk '{sub(/#.*/,"");gsub(/^[ \t]+|[ \t]+$/,"");if(length($0)){print;exit}}' artifacts_root/sar/config/final_model.txt)/output/model.tar.gz" \
      --input-data s3://project-bucket-placeholder/wind_training/pivots/joined/ \
      --output-s3-uri "s3://project-bucket-placeholder/wind_training/inference/sar-range-example/pivots_joined" \
      --profile example-profile \
      --region example-region-1 \
      --output-format parquet
    echo ">>> Completed SAR inference on PIVOTS_JOINED at $(date) <<<"

    echo ">>> Finalizing GeoParquet for SAR inference (PIVOTS_JOINED) at $(date) <<<"
    ./scripts/geo_utils/finalize_geoparquet.sh \
      --db-name wind_training \
      --bucket-name project-bucket-placeholder \
      --output-prefix wind_training/inference/sar-range-example/pivots_joined \
      --output-table SAR_RANGE_EXAMPLE_PIVOTS_JOINED \
      --register-table \
      --profile example-profile \
      --region example-region-1 \
      --log-dir artifacts_root/sar/logs/finalize_geoparquet_inference_pivots
    echo ">>> Completed GeoParquet finalization for SAR inference (PIVOTS_JOINED) at $(date) <<<"

    echo ">>> Building STAC catalog for SAR inference (PIVOTS_JOINED) at $(date) <<<"
    ./scripts/geo_utils/build_stac_catalog.sh \
      --collection SAR_RANGE_EXAMPLE_PIVOTS_JOINED \
      --s3-uri s3://project-bucket-placeholder/wind_training/inference/sar-range-example/pivots_joined/ \
      --stac-collection-properties-json artifacts_root/sar/stac_config/stac_properties_collection_SAR_RANGE_EXAMPLE_PIVOTS_JOINED.json \
      --stac-item-properties-json artifacts_root/sar/stac_config/stac_properties_item_SAR_RANGE_EXAMPLE_PIVOTS_JOINED.json \
      --output-dir catalogs/sar_pipeline/sar_range_final_pivots_joined \
      --profile example-profile \
      --region example-region-1
    echo ">>> Completed STAC catalog build for SAR inference (PIVOTS_JOINED) at $(date) <<<"
  else
    echo ">>> Skipping SAR final pivots inference per configuration <<<"
  fi


  if [[ "${RUN_SAR_ON_BUOY_INFERENCE}" == "1" ]]; then
    ./scripts/inference/run_inference.sh \
      --model-artifact "s3://project-bucket-placeholder/wind_training/models/sar/final/$(awk '{sub(/#.*/,"");gsub(/^[ \t]+|[ \t]+$/,"");if(length($0)){print;exit}}' artifacts_root/sar/config/final_model.txt)/output/model.tar.gz" \
      --input-data s3://project-bucket-placeholder/wind_training/training/buoy/test/ \
      --output-s3-uri "s3://project-bucket-placeholder/wind_training/inference/sar-on-buoy/test_set" \
      --profile example-profile \
      --region example-region-1 \
      --output-format parquet

    bash scripts/geo_utils/finalize_geoparquet.sh \
      --db-name wind_training \
      --bucket-name project-bucket-placeholder \
      --output-prefix wind_training/inference/sar-on-buoy/test_set \
      --output-table SAR_ON_BUOY_TEST_INFERENCE \
      --register-table \
      --profile example-profile \
      --region example-region-1 \
      --log-dir artifacts_root/buoy/logs/finalize_geoparquet_inference_test

    mkdir -p "artifacts_root/sar/inference_metrics/$(awk '{sub(/#.*/,"");gsub(/^[ \t]+|[ \t]+$/,"");if(length($0)){print;exit}}' artifacts_root/sar/config/final_model.txt)_on_buoy"
    ./scripts/inference/compute_inference_metrics.sh \
      --predictions-s3 s3://project-bucket-placeholder/wind_training/inference/sar-on-buoy/test_set/data.parquet \
      --metadata-s3  s3://project-bucket-placeholder/wind_training/inference/sar-on-buoy/test_set/inference_metadata.json \
      --output-dir   "artifacts_root/sar/inference_metrics/$(awk '{sub(/#.*/,"");gsub(/^[ \t]+|[ \t]+$/,"");if(length($0)){print;exit}}' artifacts_root/sar/config/final_model.txt)_on_buoy" \
      --profile example-profile \
      --region example-region-1 \
      --wind-bin-column wind_bin \
      --truth-speed-col buoy__wind_speed \
      --truth-dir-col  buoy__wind_dir

    echo ">>> Building STAC catalog for SAR-on-reference buoy inference (test set) at $(date) <<<"
    ./scripts/geo_utils/build_stac_catalog.sh \
      --collection SAR_ON_BUOY_TEST_INFERENCE \
      --s3-uri s3://project-bucket-placeholder/wind_training/inference/sar-on-buoy/test_set/ \
      --stac-collection-properties-json artifacts_root/sar/stac_config/stac_properties_collection_SAR_ON_BUOY_TEST_SET.json \
      --stac-item-properties-json artifacts_root/sar/stac_config/stac_properties_item_SAR_ON_BUOY_TEST_SET.json \
      --output-dir catalogs/sar_pipeline/sar_on_buoy_test_set \
      --profile example-profile \
      --region example-region-1
    echo ">>> Completed STAC catalog build for SAR-on-reference buoy inference (test set) at $(date) <<<"
  else
    echo ">>> Skipping SAR-on-reference buoy inference per configuration <<<"
  fi

if [ -f artifacts_root/sar/config/final_model_finetuned.txt ]; then
  echo ">>> Fine-tuned model file found; skipping SAR fine-tuning stage <<<"
else
  echo ">>> Preparing SAR fine-tuning config at $(date) using existing final model settings <<<"
  ./scripts/training/prepare_finetune_config.sh \
    --base artifacts_root/sar/config/sar_hpo_final_model.json \
    --output artifacts_root/sar/config/sar_finetune_model.json \
    --target-speed-col buoy__wind_speed \
    --target-dir-col buoy__wind_dir

  echo ">>> Starting SAR fine-tuning run on reference buoy training set at $(date) <<<"
  ./scripts/training/train_model.sh \
    --s3-prefix s3://project-bucket-placeholder/wind_training/models/sar/finetuned \
    --profile example-profile \
    --region example-region-1 \
    --train-data-uri s3://project-bucket-placeholder/wind_training/training/buoy/train/ \
    --artifact-uri "s3://project-bucket-placeholder/wind_training/models/sar/final/$(sed -n '1p' artifacts_root/sar/config/final_model.txt)/output/model.tar.gz" \
    --job-base-prefix sar-range-finetuned \
    --model-config artifacts_root/sar/config/sar_finetune_model.json \
    --output-dir artifacts_root/sar/fine_tuning \
    --no-cv \
    --seed 42
  echo ">>> Completed SAR fine-tuning run at $(date) <<<"

  ./scripts/training/record_final_job.sh --log artifacts_root/sar/fine_tuning/train_model.log --output artifacts_root/sar/config/final_model_finetuned.txt
  echo ">>> Stored SAR fine-tuned training job $(sed -n '1p' artifacts_root/sar/config/final_model_finetuned.txt) in artifacts_root/sar/config/final_model_finetuned.txt <<<"
fi

if [ ! -f artifacts_root/sar/config/final_model_finetuned.txt ]; then
  echo "Error: artifacts_root/sar/config/final_model_finetuned.txt not found. Remove this file to re-run fine-tuning or ensure training completed successfully." >&2
  exit 1
fi

echo ">>> Using SAR fine-tuned training job $(awk '{sub(/#.*/,"");gsub(/^[ \t]+|[ \t]+$/,"");if(length($0)){print;exit}}' artifacts_root/sar/config/final_model_finetuned.txt) for fine-tuned evaluation <<<"

echo ">>> Fetching SAR fine-tuned training metrics at $(date) <<<"
./scripts/training/get_train_metrics.sh \
  --aws-profile example-profile \
  --train-job-name "$(awk '{sub(/#.*/,"");gsub(/^[ \t]+|[ \t]+$/,"");if(length($0)){print;exit}}' artifacts_root/sar/config/final_model_finetuned.txt)" \
  --output-directory artifacts_root/sar/fine_tuning/train_metrics
./scripts/training/get_norm_params.sh \
  --aws-profile example-profile \
  --train-job-name "$(awk '{sub(/#.*/,"");gsub(/^[ \t]+|[ \t]+$/,"");if(length($0)){print;exit}}' artifacts_root/sar/config/final_model_finetuned.txt)" \
  --output-directory artifacts_root/sar/fine_tuning/normalization_params
./scripts/training/get_bin_metrics.sh \
  --aws-profile example-profile \
  --train-job-name "$(awk '{sub(/#.*/,"");gsub(/^[ \t]+|[ \t]+$/,"");if(length($0)){print;exit}}' artifacts_root/sar/config/final_model_finetuned.txt)" \
  --output-directory artifacts_root/sar/fine_tuning/bin_metrics
echo ">>> Completed fetching SAR fine-tuned training metrics at $(date) <<<"

  if [[ "${RUN_SAR_FINETUNED_TEST_INFERENCE}" == "1" ]]; then
  echo ">>> Running SAR fine-tuned inference on SAR test set at $(date) <<<"
  ./scripts/inference/run_inference.sh \
    --model-artifact "s3://project-bucket-placeholder/wind_training/models/sar/finetuned/$(awk '{sub(/#.*/,"");gsub(/^[ \	]+|[ \	]+$/,"");if(length($0)){print;exit}}' artifacts_root/sar/config/final_model_finetuned.txt)/output/model.tar.gz" \
    --input-data s3://project-bucket-placeholder/wind_training/training/sar/test/ \
    --output-s3-uri "s3://project-bucket-placeholder/wind_training/inference/sar-range-finetuned/test_set" \
    --profile example-profile \
    --region example-region-1 \
    --output-format parquet
  echo ">>> Completed SAR fine-tuned inference on SAR test set at $(date) <<<"

  echo ">>> Finalizing GeoParquet for SAR fine-tuned inference (test set) at $(date) <<<"
  ./scripts/geo_utils/finalize_geoparquet.sh \
    --db-name wind_training \
    --bucket-name project-bucket-placeholder \
    --output-prefix wind_training/inference/sar-range-finetuned/test_set \
    --output-table SAR_RANGE_FINETUNED_TEST_INFERENCE \
    --register-table \
    --profile example-profile \
    --region example-region-1 \
    --log-dir artifacts_root/sar/logs/finalize_geoparquet_inference_test
  echo ">>> Completed GeoParquet finalization for SAR fine-tuned inference (test set) at $(date) <<<"

  echo ">>> Computing SAR fine-tuned inference metrics at $(date) <<<"
  mkdir -p "artifacts_root/sar/inference_metrics/$(awk '{sub(/#.*/,"");gsub(/^[ \	]+|[ \	]+$/,"");if(length($0)){print;exit}}' artifacts_root/sar/config/final_model_finetuned.txt)_finetuned"
  ./scripts/inference/compute_inference_metrics.sh \
    --predictions-s3 s3://project-bucket-placeholder/wind_training/inference/sar-range-finetuned/test_set/data.parquet \
    --metadata-s3 s3://project-bucket-placeholder/wind_training/inference/sar-range-finetuned/test_set/inference_metadata.json \
    --output-dir "artifacts_root/sar/inference_metrics/$(awk '{sub(/#.*/,"");gsub(/^[ \	]+|[ \	]+$/,"");if(length($0)){print;exit}}' artifacts_root/sar/config/final_model_finetuned.txt)_finetuned" \
    --profile example-profile \
    --region example-region-1 \
      --wind-bin-column wind_bin \
    --truth-speed-col sar__owiwindspeed_mean \
    --truth-dir-col  sar__owiwinddirection_mean
  echo ">>> Completed SAR fine-tuned inference metrics at $(date) <<<"

  echo ">>> Building STAC catalog for SAR fine-tuned inference (test set) at $(date) <<<"
  ./scripts/geo_utils/build_stac_catalog.sh \
    --collection SAR_RANGE_FINETUNED_TEST_INFERENCE \
    --s3-uri s3://project-bucket-placeholder/wind_training/inference/sar-range-finetuned/test_set/ \
    --stac-collection-properties-json artifacts_root/sar/stac_config/stac_properties_collection_SAR_RANGE_EXAMPLE_TEST_SET.json \
    --stac-item-properties-json artifacts_root/sar/stac_config/stac_properties_item_SAR_RANGE_EXAMPLE_TEST_SET.json \
    --output-dir catalogs/sar_pipeline/sar_range_finetuned_test_set \
    --profile example-profile \
    --region example-region-1
  echo ">>> Completed STAC catalog build for SAR fine-tuned inference (test set) at $(date) <<<"
  else
    echo ">>> Skipping SAR fine-tuned test-set inference per configuration <<<"
  fi

  if [[ "${RUN_SAR_FINETUNED_PIVOTS_INFERENCE}" == "1" ]]; then
  echo ">>> Running SAR fine-tuned inference on PIVOTS_JOINED at $(date) <<<"
  ./scripts/inference/run_inference.sh \
    --model-artifact "s3://project-bucket-placeholder/wind_training/models/sar/finetuned/$(awk '{sub(/#.*/,"");gsub(/^[ \	]+|[ \	]+$/,"");if(length($0)){print;exit}}' artifacts_root/sar/config/final_model_finetuned.txt)/output/model.tar.gz" \
    --input-data s3://project-bucket-placeholder/wind_training/pivots/joined/ \
    --output-s3-uri "s3://project-bucket-placeholder/wind_training/inference/sar-range-finetuned/pivots_joined" \
    --profile example-profile \
    --region example-region-1 \
    --output-format parquet
  echo ">>> Completed SAR fine-tuned inference on PIVOTS_JOINED at $(date) <<<"

  echo ">>> Finalizing GeoParquet for SAR fine-tuned inference (PIVOTS_JOINED) at $(date) <<<"
  ./scripts/geo_utils/finalize_geoparquet.sh \
    --db-name wind_training \
    --bucket-name project-bucket-placeholder \
    --output-prefix wind_training/inference/sar-range-finetuned/pivots_joined \
    --output-table SAR_RANGE_FINETUNED_PIVOTS_JOINED \
    --register-table \
    --profile example-profile \
    --region example-region-1 \
    --log-dir artifacts_root/sar/logs/finalize_geoparquet_inference_pivots
  echo ">>> Completed GeoParquet finalization for SAR fine-tuned inference (PIVOTS_JOINED) at $(date) <<<"

  echo ">>> Building STAC catalog for SAR fine-tuned inference (PIVOTS_JOINED) at $(date) <<<"
  ./scripts/geo_utils/build_stac_catalog.sh \
    --collection SAR_RANGE_FINETUNED_PIVOTS_JOINED \
    --s3-uri s3://project-bucket-placeholder/wind_training/inference/sar-range-finetuned/pivots_joined/ \
    --stac-collection-properties-json artifacts_root/sar/stac_config/stac_properties_collection_SAR_RANGE_EXAMPLE_PIVOTS_JOINED.json \
    --stac-item-properties-json artifacts_root/sar/stac_config/stac_properties_item_SAR_RANGE_EXAMPLE_PIVOTS_JOINED.json \
    --output-dir catalogs/sar_pipeline/sar_range_finetuned_pivots_joined \
    --profile example-profile \
    --region example-region-1
  echo ">>> Completed STAC catalog build for SAR fine-tuned inference (PIVOTS_JOINED) at $(date) <<<"
  else
    echo ">>> Skipping SAR fine-tuned pivots inference per configuration <<<"
  fi

  if [[ "${RUN_SAR_FINETUNED_ON_BUOY_INFERENCE}" == "1" ]]; then
  echo ">>> Running SAR fine-tuned inference on reference buoy test set at $(date) <<<"
  ./scripts/inference/run_inference.sh \
    --model-artifact "s3://project-bucket-placeholder/wind_training/models/sar/finetuned/$(awk '{sub(/#.*/,"");gsub(/^[ \	]+|[ \	]+$/,"");if(length($0)){print;exit}}' artifacts_root/sar/config/final_model_finetuned.txt)/output/model.tar.gz" \
    --input-data s3://project-bucket-placeholder/wind_training/training/buoy/test/ \
    --output-s3-uri "s3://project-bucket-placeholder/wind_training/inference/sar-finetuned-on-buoy/test_set" \
    --profile example-profile \
    --region example-region-1 \
    --output-format parquet
  echo ">>> Completed SAR fine-tuned inference on reference buoy test set at $(date) <<<"

  bash scripts/geo_utils/finalize_geoparquet.sh \
    --db-name wind_training \
    --bucket-name project-bucket-placeholder \
    --output-prefix wind_training/inference/sar-finetuned-on-buoy/test_set \
    --output-table SAR_FINETUNED_ON_BUOY_TEST_INFERENCE \
    --register-table \
    --profile example-profile \
    --region example-region-1 \
    --log-dir artifacts_root/sar/logs/finalize_geoparquet_inference_test

  mkdir -p "artifacts_root/sar/inference_metrics/$(awk '{sub(/#.*/,"");gsub(/^[ \	]+|[ \	]+$/,"");if(length($0)){print;exit}}' artifacts_root/sar/config/final_model_finetuned.txt)_finetuned_on_buoy"
  ./scripts/inference/compute_inference_metrics.sh \
    --predictions-s3 s3://project-bucket-placeholder/wind_training/inference/sar-finetuned-on-buoy/test_set/data.parquet \
    --metadata-s3  s3://project-bucket-placeholder/wind_training/inference/sar-finetuned-on-buoy/test_set/inference_metadata.json \
    --output-dir   "artifacts_root/sar/inference_metrics/$(awk '{sub(/#.*/,"");gsub(/^[ \	]+|[ \	]+$/,"");if(length($0)){print;exit}}' artifacts_root/sar/config/final_model_finetuned.txt)_finetuned_on_buoy" \
    --profile example-profile \
    --region example-region-1 \
      --wind-bin-column wind_bin \
    --truth-speed-col buoy__wind_speed \
    --truth-dir-col  buoy__wind_dir

  echo ">>> Building STAC catalog for SAR fine-tuned on reference buoy inference (test set) at $(date) <<<"
  ./scripts/geo_utils/build_stac_catalog.sh \
    --collection SAR_FINETUNED_ON_BUOY_TEST_INFERENCE \
    --s3-uri s3://project-bucket-placeholder/wind_training/inference/sar-finetuned-on-buoy/test_set/ \
    --stac-collection-properties-json artifacts_root/sar/stac_config/stac_properties_collection_SAR_ON_BUOY_TEST_SET.json \
    --stac-item-properties-json artifacts_root/sar/stac_config/stac_properties_item_SAR_ON_BUOY_TEST_SET.json \
    --output-dir catalogs/sar_pipeline/sar_finetuned_on_buoy_test_set \
    --profile example-profile \
    --region example-region-1
  echo ">>> Completed STAC catalog build for SAR fine-tuned on reference buoy inference (test set) at $(date) <<<"
  else
    echo ">>> Skipping SAR fine-tuned reference buoy inference per configuration <<<"
  fi

if [[ "${RUN_SAR_L2SP_TRAINING}" == "1" ]]; then
  if [ -f artifacts_root/sar/config/final_model_finetuned_l2sp.txt ]; then
    echo ">>> Fine-tuned (L2-SP) model file found; skipping SAR L2-SP fine-tuning stage <<<"
  else
    echo ">>> Preparing SAR L2-SP rehearsal model config at $(date) <<<"
    if [ ! -f artifacts_root/sar/config/sar_finetune_model.json ]; then
      echo ">>> Base fine-tuning config missing; regenerating from final model settings <<<"
      ./scripts/training/prepare_finetune_config.sh \
        --base artifacts_root/sar/config/sar_hpo_final_model.json \
        --output artifacts_root/sar/config/sar_finetune_model.json \
        --target-speed-col buoy__wind_speed \
        --target-dir-col buoy__wind_dir
    fi
    ./scripts/training/prepare_finetune_config.sh \
      --base artifacts_root/sar/config/sar_finetune_model.json \
      --output artifacts_root/sar/config/sar_finetune_l2sp_rehearsal.json \
      --target-speed-col buoy__wind_speed \
      --target-dir-col buoy__wind_dir \
      --force
    python3 - <<'PY'
import json
from pathlib import Path

cfg_path = Path("artifacts_root/sar/config/sar_finetune_l2sp_rehearsal.json")
cfg = json.loads(cfg_path.read_text())
model = cfg.setdefault("model", {})
base_lr = float(model.get("lr", 0.001))
model.update({
    "finetune_heads_lr": base_lr,
    "finetune_backbone_lr": base_lr * 0.25,
    "use_l2sp": 1,
    "l2sp_backbone_lambda": 0.005,
    "l2sp_heads_lambda": 0.002,
    "use_kd": 0,
    "lambda_kd": 0.0,
    "lambda_kd_reg": 0.0,
    "lambda_kd_cls": 0.0,
    "kd_temperature": 1.0,
    "teacher_checkpoint": None
})
cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
PY
    echo ">>> Prepared SAR L2-SP rehearsal model config <<<"

    echo ">>> Starting SAR fine-tuning (L2-SP + partial freeze) at $(date) <<<"
    ./scripts/training/train_model.sh \
      --s3-prefix s3://project-bucket-placeholder/wind_training/models/sar/finetuned_l2sp \
      --profile example-profile \
      --region example-region-1 \
      --train-data-uri s3://project-bucket-placeholder/wind_training/training/buoy/train/ \
      --artifact-uri "s3://project-bucket-placeholder/wind_training/models/sar/final/$(sed -n '1p' artifacts_root/sar/config/final_model.txt)/output/model.tar.gz" \
      --job-base-prefix sar-range-finetuned-l2sp \
      --model-config artifacts_root/sar/config/sar_finetune_l2sp_rehearsal.json \
      --rehearsal-data-uri s3://project-bucket-placeholder/wind_training/training/sar/train/ \
      --rehearsal-target-speed-col sar__owiwindspeed_mean \
      --rehearsal-target-dir-col sar__owiwinddirection_mean \
      --rehearsal-fraction 0.20 \
      --output-dir artifacts_root/sar/fine_tuning_l2sp \
      --no-cv \
      --seed 42
    echo ">>> Completed SAR L2-SP fine-tuning at $(date) <<<"

    ./scripts/training/record_final_job.sh --log artifacts_root/sar/fine_tuning_l2sp/train_model.log --output artifacts_root/sar/config/final_model_finetuned_l2sp.txt
    echo ">>> Stored SAR fine-tuned (L2-SP) job $(sed -n '1p' artifacts_root/sar/config/final_model_finetuned_l2sp.txt) in artifacts_root/sar/config/final_model_finetuned_l2sp.txt <<<"
  fi
else
  echo ">>> Skipping SAR L2-SP fine-tuning stage per configuration <<<"
fi

if [ ! -f artifacts_root/sar/config/final_model_finetuned_l2sp.txt ]; then
  echo "Error: artifacts_root/sar/config/final_model_finetuned_l2sp.txt not found. Remove this file to re-run fine-tuning or ensure training completed successfully." >&2
  exit 1
fi

SAR_L2SP_JOB=$(awk '{sub(/#.*/,"");gsub(/^[ \t]+|[ \t]+$/,"");if(length($0)){print;exit}}' artifacts_root/sar/config/final_model_finetuned_l2sp.txt)
echo ">>> Using SAR fine-tuned (L2-SP) training job ${SAR_L2SP_JOB} for L2-SP evaluation <<<"

if [[ "${RUN_SAR_L2SP_TEST_INFERENCE}" == "1" ]]; then
  echo ">>> Running SAR (L2-SP) inference on SAR test set at $(date) <<<"
  ./scripts/inference/run_inference.sh \
    --model-artifact "s3://project-bucket-placeholder/wind_training/models/sar/finetuned_l2sp/${SAR_L2SP_JOB}/output/model.tar.gz" \
    --input-data s3://project-bucket-placeholder/wind_training/training/sar/test/ \
    --output-s3-uri "s3://project-bucket-placeholder/wind_training/inference/sar-range-finetuned-l2sp/test_set" \
    --profile example-profile \
    --region example-region-1 \
    --output-format parquet
  echo ">>> Completed SAR (L2-SP) inference on SAR test set at $(date) <<<"

  ./scripts/geo_utils/finalize_geoparquet.sh \
    --db-name wind_training \
    --bucket-name project-bucket-placeholder \
    --output-prefix wind_training/inference/sar-range-finetuned-l2sp/test_set \
    --output-table SAR_RANGE_FINETUNED_L2SP_TEST_INFERENCE \
    --register-table \
    --profile example-profile \
    --log-dir artifacts_root/sar/logs/finalize_geoparquet_inference_test
  echo ">>> Completed GeoParquet finalization for SAR (L2-SP) test inference at $(date) <<<"

  echo ">>> Computing SAR (L2-SP) inference metrics at $(date) <<<"
  mkdir -p "artifacts_root/sar/inference_metrics/${SAR_L2SP_JOB}_finetuned_l2sp"
  ./scripts/inference/compute_inference_metrics.sh \
    --predictions-s3 s3://project-bucket-placeholder/wind_training/inference/sar-range-finetuned-l2sp/test_set/data.parquet \
    --metadata-s3 s3://project-bucket-placeholder/wind_training/inference/sar-range-finetuned-l2sp/test_set/inference_metadata.json \
    --output-dir "artifacts_root/sar/inference_metrics/${SAR_L2SP_JOB}_finetuned_l2sp" \
    --profile example-profile \
    --region example-region-1 \
    --wind-bin-column wind_bin \
    --truth-speed-col sar__owiwindspeed_mean \
    --truth-dir-col  sar__owiwinddirection_mean
  echo ">>> Completed SAR (L2-SP) inference metrics at $(date) <<<"
else
  echo ">>> Skipping SAR (L2-SP) test-set inference per configuration <<<"
fi

if [[ "${RUN_SAR_L2SP_ON_BUOY_INFERENCE}" == "1" ]]; then
  echo ">>> Running SAR (L2-SP) inference on reference buoy test set at $(date) <<<"
  ./scripts/inference/run_inference.sh \
    --model-artifact "s3://project-bucket-placeholder/wind_training/models/sar/finetuned_l2sp/${SAR_L2SP_JOB}/output/model.tar.gz" \
    --input-data s3://project-bucket-placeholder/wind_training/training/buoy/test/ \
    --output-s3-uri "s3://project-bucket-placeholder/wind_training/inference/sar-finetuned-l2sp-on-buoy/test_set" \
    --profile example-profile \
    --region example-region-1 \
    --output-format parquet
  echo ">>> Completed SAR (L2-SP) inference on reference buoy test set at $(date) <<<"

  ./scripts/geo_utils/finalize_geoparquet.sh \
    --db-name wind_training \
    --bucket-name project-bucket-placeholder \
    --output-prefix wind_training/inference/sar-finetuned-l2sp-on-buoy/test_set \
    --output-table SAR_FINETUNED_L2SP_ON_BUOY_TEST_INFERENCE \
    --register-table \
    --profile example-profile \
    --log-dir artifacts_root/sar/logs/finalize_geoparquet_inference_test
  echo ">>> Completed GeoParquet finalization for SAR (L2-SP) on reference buoy test at $(date) <<<"

  echo ">>> Computing SAR (L2-SP) on reference buoy inference metrics at $(date) <<<"
  mkdir -p "artifacts_root/sar/inference_metrics/${SAR_L2SP_JOB}_finetuned_l2sp_on_buoy"
  ./scripts/inference/compute_inference_metrics.sh \
    --predictions-s3 s3://project-bucket-placeholder/wind_training/inference/sar-finetuned-l2sp-on-buoy/test_set/data.parquet \
    --metadata-s3  s3://project-bucket-placeholder/wind_training/inference/sar-finetuned-l2sp-on-buoy/test_set/inference_metadata.json \
    --output-dir   "artifacts_root/sar/inference_metrics/${SAR_L2SP_JOB}_finetuned_l2sp_on_buoy" \
    --profile example-profile \
    --region example-region-1 \
    --wind-bin-column wind_bin \
    --truth-speed-col buoy__wind_speed \
    --truth-dir-col  buoy__wind_dir
  echo ">>> Completed SAR (L2-SP) on reference buoy inference metrics at $(date) <<<"
else
  echo ">>> Skipping SAR (L2-SP) on reference buoy inference per configuration <<<"
fi

if [[ "${RUN_SAR_L2SP_KD_TRAINING}" == "1" ]]; then
  if [ -f artifacts_root/sar/config/final_model_finetuned_l2sp_kd.txt ]; then
    echo ">>> Fine-tuned (L2-SP+KD) model file found; skipping SAR L2-SP+KD fine-tuning stage <<<"
  else
    echo ">>> Preparing SAR L2-SP+KD rehearsal model config at $(date) <<<"
    if [ ! -f artifacts_root/sar/config/sar_finetune_model.json ]; then
      echo ">>> Base fine-tuning config missing; regenerating from final model settings <<<"
      ./scripts/training/prepare_finetune_config.sh \
        --base artifacts_root/sar/config/sar_hpo_final_model.json \
        --output artifacts_root/sar/config/sar_finetune_model.json \
        --target-speed-col buoy__wind_speed \
        --target-dir-col buoy__wind_dir
    fi
    ./scripts/training/prepare_finetune_config.sh \
      --base artifacts_root/sar/config/sar_finetune_model.json \
      --output artifacts_root/sar/config/sar_finetune_l2sp_rehearsal_kd.json \
      --target-speed-col buoy__wind_speed \
      --target-dir-col buoy__wind_dir \
      --force
    python3 - <<'PY'
import json
from pathlib import Path

cfg_path = Path("artifacts_root/sar/config/sar_finetune_l2sp_rehearsal_kd.json")
cfg = json.loads(cfg_path.read_text())
model = cfg.setdefault("model", {})
base_lr = float(model.get("lr", 0.001))
model.update({
    "finetune_heads_lr": base_lr,
    "finetune_backbone_lr": base_lr * 0.25,
    "use_l2sp": 1,
    "l2sp_backbone_lambda": 0.005,
    "l2sp_heads_lambda": 0.002,
    "use_kd": 1,
    "lambda_kd": 0.3,
    "lambda_kd_reg": 0.3,
    "lambda_kd_cls": 0.3,
    "kd_temperature": 1.0,
    "teacher_checkpoint": None
})
cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
PY
    echo ">>> Prepared SAR L2-SP+KD rehearsal model config <<<"

    echo ">>> Starting SAR fine-tuning (L2-SP + partial freeze + KD) at $(date) <<<"
    ./scripts/training/train_model.sh \
      --s3-prefix s3://project-bucket-placeholder/wind_training/models/sar/finetuned_l2sp_kd \
      --profile example-profile \
      --region example-region-1 \
      --train-data-uri s3://project-bucket-placeholder/wind_training/training/buoy/train/ \
      --artifact-uri "s3://project-bucket-placeholder/wind_training/models/sar/final/$(sed -n '1p' artifacts_root/sar/config/final_model.txt)/output/model.tar.gz" \
      --job-base-prefix sar-range-finetuned-l2sp-kd \
      --model-config artifacts_root/sar/config/sar_finetune_l2sp_rehearsal_kd.json \
      --rehearsal-data-uri s3://project-bucket-placeholder/wind_training/training/sar/train/ \
      --rehearsal-target-speed-col sar__owiwindspeed_mean \
      --rehearsal-target-dir-col sar__owiwinddirection_mean \
      --rehearsal-fraction 0.20 \
      --output-dir artifacts_root/sar/fine_tuning_l2sp_kd \
      --no-cv \
      --seed 42
    echo ">>> Completed SAR L2-SP+KD fine-tuning at $(date) <<<"

    ./scripts/training/record_final_job.sh --log artifacts_root/sar/fine_tuning_l2sp_kd/train_model.log --output artifacts_root/sar/config/final_model_finetuned_l2sp_kd.txt
    echo ">>> Stored SAR fine-tuned (L2-SP+KD) job $(sed -n '1p' artifacts_root/sar/config/final_model_finetuned_l2sp_kd.txt) in artifacts_root/sar/config/final_model_finetuned_l2sp_kd.txt <<<"
  fi
else
  echo ">>> Skipping SAR L2-SP+KD fine-tuning stage per configuration <<<"
fi

if [ ! -f artifacts_root/sar/config/final_model_finetuned_l2sp_kd.txt ]; then
  echo "Error: artifacts_root/sar/config/final_model_finetuned_l2sp_kd.txt not found. Remove this file to re-run fine-tuning or ensure training completed successfully." >&2
  exit 1
fi

SAR_L2SP_KD_JOB=$(awk '{sub(/#.*/,"");gsub(/^[ \t]+|[ \t]+$/,"");if(length($0)){print;exit}}' artifacts_root/sar/config/final_model_finetuned_l2sp_kd.txt)
cat <<EOF > artifacts_root/sar/fine_tuning_l2sp_kd/finetune_strategy.json
{
  "strategy": "l2sp_rehearsal_freeze_partial_kd",
  "timestamp_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "use_kd": true,
  "lambda_kd": 0.3,
  "teacher": {
    "model_artifact": "s3://project-bucket-placeholder/wind_training/models/sar/final/$(sed -n '1p' artifacts_root/sar/config/final_model.txt)/output/model.tar.gz",
    "checkpoint": "s3://project-bucket-placeholder/wind_training/models/sar/finetuned_l2sp_kd/fine-tuning/${SAR_L2SP_KD_JOB}/checkpoint.pth"
  },
  "student_job": "${SAR_L2SP_KD_JOB}",
  "rehearsal_fraction": 0.20,
  "l2sp_backbone_lambda": 0.005,
  "l2sp_heads_lambda": 0.002
}
EOF

echo ">>> Using SAR fine-tuned (L2-SP+KD) training job ${SAR_L2SP_KD_JOB} for L2-SP+KD evaluation <<<"

if [[ "${RUN_SAR_L2SP_KD_TEST_INFERENCE}" == "1" ]]; then
  echo ">>> Running SAR (L2-SP+KD) inference on SAR test set at $(date) <<<"
  ./scripts/inference/run_inference.sh \
    --model-artifact "s3://project-bucket-placeholder/wind_training/models/sar/finetuned_l2sp_kd/${SAR_L2SP_KD_JOB}/output/model.tar.gz" \
    --input-data s3://project-bucket-placeholder/wind_training/training/sar/test/ \
    --output-s3-uri "s3://project-bucket-placeholder/wind_training/inference/sar-range-finetuned-l2sp-kd/test_set" \
    --profile example-profile \
    --region example-region-1 \
    --output-format parquet
  echo ">>> Completed SAR (L2-SP+KD) inference on SAR test set at $(date) <<<"

  ./scripts/geo_utils/finalize_geoparquet.sh \
    --db-name wind_training \
    --bucket-name project-bucket-placeholder \
    --output-prefix wind_training/inference/sar-range-finetuned-l2sp-kd/test_set \
    --output-table SAR_RANGE_FINETUNED_L2SP_TEST_INFERENCE \
    --register-table \
    --profile example-profile \
    --log-dir artifacts_root/sar/logs/finalize_geoparquet_inference_test
  echo ">>> Completed GeoParquet finalization for SAR (L2-SP+KD) test inference at $(date) <<<"

  echo ">>> Computing SAR (L2-SP+KD) inference metrics at $(date) <<<"
  mkdir -p "artifacts_root/sar/inference_metrics/${SAR_L2SP_KD_JOB}_finetuned_l2sp_kd"
  ./scripts/inference/compute_inference_metrics.sh \
    --predictions-s3 s3://project-bucket-placeholder/wind_training/inference/sar-range-finetuned-l2sp-kd/test_set/data.parquet \
    --metadata-s3 s3://project-bucket-placeholder/wind_training/inference/sar-range-finetuned-l2sp-kd/test_set/inference_metadata.json \
    --output-dir "artifacts_root/sar/inference_metrics/${SAR_L2SP_KD_JOB}_finetuned_l2sp_kd" \
    --profile example-profile \
    --region example-region-1 \
    --wind-bin-column wind_bin \
    --truth-speed-col sar__owiwindspeed_mean \
    --truth-dir-col  sar__owiwinddirection_mean
  echo ">>> Completed SAR (L2-SP+KD) inference metrics at $(date) <<<"
else
  echo ">>> Skipping SAR (L2-SP+KD) test-set inference per configuration <<<"
fi

if [[ "${RUN_SAR_L2SP_KD_ON_BUOY_INFERENCE}" == "1" ]]; then
  echo ">>> Running SAR (L2-SP+KD) inference on reference buoy test set at $(date) <<<"
  ./scripts/inference/run_inference.sh \
    --model-artifact "s3://project-bucket-placeholder/wind_training/models/sar/finetuned_l2sp_kd/${SAR_L2SP_KD_JOB}/output/model.tar.gz" \
    --input-data s3://project-bucket-placeholder/wind_training/training/buoy/test/ \
    --output-s3-uri "s3://project-bucket-placeholder/wind_training/inference/sar-finetuned-l2sp-kd-on-buoy/test_set" \
    --profile example-profile \
    --region example-region-1 \
    --output-format parquet
  echo ">>> Completed SAR (L2-SP+KD) inference on reference buoy test set at $(date) <<<"

  ./scripts/geo_utils/finalize_geoparquet.sh \
    --db-name wind_training \
    --bucket-name project-bucket-placeholder \
    --output-prefix wind_training/inference/sar-finetuned-l2sp-kd-on-buoy/test_set \
    --output-table SAR_FINETUNED_L2SP_KD_ON_BUOY_TEST_INFERENCE \
    --register-table \
    --profile example-profile \
    --log-dir artifacts_root/sar/logs/finalize_geoparquet_inference_test
  echo ">>> Completed GeoParquet finalization for SAR (L2-SP+KD) on reference buoy test at $(date) <<<"

  echo ">>> Computing SAR (L2-SP+KD) on reference buoy inference metrics at $(date) <<<"
  mkdir -p "artifacts_root/sar/inference_metrics/${SAR_L2SP_KD_JOB}_finetuned_l2sp_kd_on_buoy"
  ./scripts/inference/compute_inference_metrics.sh \
    --predictions-s3 s3://project-bucket-placeholder/wind_training/inference/sar-finetuned-l2sp-kd-on-buoy/test_set/data.parquet \
    --metadata-s3  s3://project-bucket-placeholder/wind_training/inference/sar-finetuned-l2sp-kd-on-buoy/test_set/inference_metadata.json \
    --output-dir   "artifacts_root/sar/inference_metrics/${SAR_L2SP_KD_JOB}_finetuned_l2sp_kd_on_buoy" \
    --profile example-profile \
    --region example-region-1 \
    --wind-bin-column wind_bin \
    --truth-speed-col buoy__wind_speed \
    --truth-dir-col  buoy__wind_dir
  echo ">>> Completed SAR (L2-SP+KD) on reference buoy inference metrics at $(date) <<<"
else
  echo ">>> Skipping SAR (L2-SP+KD) on reference buoy inference per configuration <<<"
fi

if [[ "${RUN_SAR_FEATURE_IMPORTANCE_FINAL}" == "1" ]]; then
  if [ -f artifacts_root/sar/config/final_model.txt ]; then
    ensure_analysis_dirs
    build_analysis_image
    echo ">>> SAR FINAL: downloading model artifact and test data at $(date) <<<"
    aws s3 cp \
      "s3://project-bucket-placeholder/wind_training/models/sar/final/$(sed -n '1p' artifacts_root/sar/config/final_model.txt)/output/model.tar.gz" \
      artifacts_root/analysis/models/sar_final_model.tar.gz \
      --profile example-profile --region example-region-1
    aws s3 sync \
      s3://project-bucket-placeholder/wind_training/training/sar/test/ \
      artifacts_root/analysis/data/sar_test \
      --profile example-profile --region example-region-1

    echo ">>> SAR FINAL: running feature-importance analysis at $(date) <<<"
    docker run --rm \
      -v "$(pwd)":/work \
      -w /work \
      example-wind-analysis:latest bash -lc "\
        python scripts/analysis/feature_importance.py \
          --model-artifact artifacts_root/analysis/models/sar_final_model.tar.gz \
          --data-path artifacts_root/analysis/data/sar_test \
          --output-dir artifacts_root/analysis \
          --grouping both \
          --top-k 30" \
      | tee -a artifacts_root/analysis/logs/sar_final_test_feature_importance.log

    docker run --rm -v "$(pwd)":/work -w /work example-wind-analysis:latest bash -lc "\
      python scripts/analysis/integrate_feature_importance_report.py \
        --analysis-dir \"$(ls -dt artifacts_root/analysis/feature_importance_* | head -1)\" \
        --report-file \"$(ls -dt artifacts_root/analysis/feature_importance_* | head -1)/report.md\" \
        --section-title 'Feature Importance — SAR FINAL model on SAR TEST set'"
  else
    echo ">>> Skipping SAR FINAL feature-importance analysis: artifacts_root/sar/config/final_model.txt not found <<<"
  fi
else
  echo ">>> Skipping SAR FINAL feature-importance analysis per configuration <<<"
fi

if [[ "${RUN_SAR_FEATURE_IMPORTANCE_FINETUNED}" == "1" ]]; then
  if [ -f artifacts_root/sar/config/final_model_finetuned.txt ]; then
    ensure_analysis_dirs
    build_analysis_image
    echo ">>> SAR FINETUNED: downloading model artifact at $(date) <<<"
    aws s3 cp \
      "s3://project-bucket-placeholder/wind_training/models/sar/finetuned/$(sed -n '1p' artifacts_root/sar/config/final_model_finetuned.txt)/output/model.tar.gz" \
      artifacts_root/analysis/models/sar_finetuned_model.tar.gz \
      --profile example-profile --region example-region-1
    aws s3 sync \
      s3://project-bucket-placeholder/wind_training/training/buoy/test/ \
      artifacts_root/analysis/data/buoy_test \
      --profile example-profile --region example-region-1
    aws s3 sync \
      s3://project-bucket-placeholder/wind_training/training/sar/test/ \
      artifacts_root/analysis/data/sar_test \
      --profile example-profile --region example-region-1

    echo ">>> SAR FINETUNED: analysis on BUOY TEST at $(date) <<<"
    docker run --rm \
      -v "$(pwd)":/work \
      -w /work \
      example-wind-analysis:latest bash -lc "\
        python scripts/analysis/feature_importance.py \
          --model-artifact artifacts_root/analysis/models/sar_finetuned_model.tar.gz \
          --data-path artifacts_root/analysis/data/buoy_test \
          --output-dir artifacts_root/analysis \
          --grouping both \
          --top-k 30" \
      | tee -a artifacts_root/analysis/logs/sar_finetuned_on_buoy_test_feature_importance.log

    docker run --rm -v "$(pwd)":/work -w /work example-wind-analysis:latest bash -lc "\
      python scripts/analysis/integrate_feature_importance_report.py \
        --analysis-dir \"$(ls -dt artifacts_root/analysis/feature_importance_* | head -1)\" \
        --report-file \"$(ls -dt artifacts_root/analysis/feature_importance_* | head -1)/report.md\" \
        --section-title 'Feature Importance — SAR FINETUNED model on reference buoy TEST set'"

    echo ">>> SAR FINETUNED: analysis on SAR TEST at $(date) <<<"
    docker run --rm \
      -v "$(pwd)":/work \
      -w /work \
      example-wind-analysis:latest bash -lc "\
        python scripts/analysis/feature_importance.py \
          --model-artifact artifacts_root/analysis/models/sar_finetuned_model.tar.gz \
          --data-path artifacts_root/analysis/data/sar_test \
          --output-dir artifacts_root/analysis \
          --grouping both \
          --top-k 30" \
      | tee -a artifacts_root/analysis/logs/sar_finetuned_on_sar_test_feature_importance.log

    docker run --rm -v "$(pwd)":/work -w /work example-wind-analysis:latest bash -lc "\
      python scripts/analysis/integrate_feature_importance_report.py \
        --analysis-dir \"$(ls -dt artifacts_root/analysis/feature_importance_* | head -1)\" \
        --report-file \"$(ls -dt artifacts_root/analysis/feature_importance_* | head -1)/report.md\" \
        --section-title 'Feature Importance — SAR FINETUNED model on SAR TEST set'"
  else
    echo ">>> Skipping SAR FINETUNED feature-importance analysis: artifacts_root/sar/config/final_model_finetuned.txt not found <<<"
  fi
else
  echo ">>> Skipping SAR FINETUNED feature-importance analysis per configuration <<<"
fi

if [[ "${RUN_SAR_FEATURE_IMPORTANCE_L2SP}" == "1" ]]; then
  if [ -f artifacts_root/sar/config/final_model_finetuned_l2sp.txt ]; then
    ensure_analysis_dirs
    build_analysis_image
    echo ">>> SAR L2SP: downloading model artifact at $(date) <<<"
    aws s3 cp \
      "s3://project-bucket-placeholder/wind_training/models/sar/finetuned_l2sp/${SAR_L2SP_JOB}/output/model.tar.gz" \
      artifacts_root/analysis/models/sar_l2sp_model.tar.gz \
      --profile example-profile --region example-region-1
    aws s3 sync \
      s3://project-bucket-placeholder/wind_training/training/buoy/test/ \
      artifacts_root/analysis/data/buoy_test \
      --profile example-profile --region example-region-1
    aws s3 sync \
      s3://project-bucket-placeholder/wind_training/training/sar/test/ \
      artifacts_root/analysis/data/sar_test \
      --profile example-profile --region example-region-1

    echo ">>> SAR L2SP: analysis on BUOY TEST at $(date) <<<"
    docker run --rm \
      -v "$(pwd)":/work \
      -w /work \
      example-wind-analysis:latest bash -lc "\
        python scripts/analysis/feature_importance.py \
          --model-artifact artifacts_root/analysis/models/sar_l2sp_model.tar.gz \
          --data-path artifacts_root/analysis/data/buoy_test \
          --output-dir artifacts_root/analysis \
          --grouping both \
          --top-k 30" \
      | tee -a artifacts_root/analysis/logs/sar_l2sp_on_buoy_test_feature_importance.log

    docker run --rm -v "$(pwd)":/work -w /work example-wind-analysis:latest bash -lc "\
      python scripts/analysis/integrate_feature_importance_report.py \
        --analysis-dir \"$(ls -dt artifacts_root/analysis/feature_importance_* | head -1)\" \
        --report-file \"$(ls -dt artifacts_root/analysis/feature_importance_* | head -1)/report.md\" \
        --section-title 'Feature Importance — SAR L2SP model on reference buoy TEST set'"

    echo ">>> SAR L2SP: analysis on SAR TEST at $(date) <<<"
    docker run --rm \
      -v "$(pwd)":/work \
      -w /work \
      example-wind-analysis:latest bash -lc "\
        python scripts/analysis/feature_importance.py \
          --model-artifact artifacts_root/analysis/models/sar_l2sp_model.tar.gz \
          --data-path artifacts_root/analysis/data/sar_test \
          --output-dir artifacts_root/analysis \
          --grouping both \
          --top-k 30" \
      | tee -a artifacts_root/analysis/logs/sar_l2sp_on_sar_test_feature_importance.log

    docker run --rm -v "$(pwd)":/work -w /work example-wind-analysis:latest bash -lc "\
      python scripts/analysis/integrate_feature_importance_report.py \
        --analysis-dir \"$(ls -dt artifacts_root/analysis/feature_importance_* | head -1)\" \
        --report-file \"$(ls -dt artifacts_root/analysis/feature_importance_* | head -1)/report.md\" \
        --section-title 'Feature Importance — SAR L2SP model on SAR TEST set'"
  else
    echo ">>> Skipping SAR L2SP feature-importance analysis: artifacts_root/sar/config/final_model_finetuned_l2sp.txt not found <<<"
  fi
else
  echo ">>> Skipping SAR L2SP feature-importance analysis per configuration <<<"
fi

if [[ "${RUN_SAR_FEATURE_IMPORTANCE_L2SP_KD}" == "1" ]]; then
  if [ -f artifacts_root/sar/config/final_model_finetuned_l2sp_kd.txt ]; then
    ensure_analysis_dirs
    build_analysis_image
    echo ">>> SAR L2SP+KD: downloading model artifact at $(date) <<<"
    aws s3 cp \
      "s3://project-bucket-placeholder/wind_training/models/sar/finetuned_l2sp_kd/${SAR_L2SP_KD_JOB}/output/model.tar.gz" \
      artifacts_root/analysis/models/sar_l2sp_kd_model.tar.gz \
      --profile example-profile --region example-region-1
    aws s3 sync \
      s3://project-bucket-placeholder/wind_training/training/buoy/test/ \
      artifacts_root/analysis/data/buoy_test \
      --profile example-profile --region example-region-1
    aws s3 sync \
      s3://project-bucket-placeholder/wind_training/training/sar/test/ \
      artifacts_root/analysis/data/sar_test \
      --profile example-profile --region example-region-1

    echo ">>> SAR L2SP+KD: analysis on BUOY TEST at $(date) <<<"
    docker run --rm \
      -v "$(pwd)":/work \
      -w /work \
      example-wind-analysis:latest bash -lc "\
        python scripts/analysis/feature_importance.py \
          --model-artifact artifacts_root/analysis/models/sar_l2sp_kd_model.tar.gz \
          --data-path artifacts_root/analysis/data/buoy_test \
          --output-dir artifacts_root/analysis \
          --grouping both \
          --top-k 30" \
      | tee -a artifacts_root/analysis/logs/sar_l2sp_kd_on_buoy_test_feature_importance.log

    docker run --rm -v "$(pwd)":/work -w /work example-wind-analysis:latest bash -lc "\
      python scripts/analysis/integrate_feature_importance_report.py \
        --analysis-dir \"$(ls -dt artifacts_root/analysis/feature_importance_* | head -1)\" \
        --report-file \"$(ls -dt artifacts_root/analysis/feature_importance_* | head -1)/report.md\" \
        --section-title 'Feature Importance — SAR L2SP+KD model on reference buoy TEST set'"

    echo ">>> SAR L2SP+KD: analysis on SAR TEST at $(date) <<<"
    docker run --rm \
      -v "$(pwd)":/work \
      -w /work \
      example-wind-analysis:latest bash -lc "\
        python scripts/analysis/feature_importance.py \
          --model-artifact artifacts_root/analysis/models/sar_l2sp_kd_model.tar.gz \
          --data-path artifacts_root/analysis/data/sar_test \
          --output-dir artifacts_root/analysis \
          --grouping both \
          --top-k 30" \
      | tee -a artifacts_root/analysis/logs/sar_l2sp_kd_on_sar_test_feature_importance.log

    docker run --rm -v "$(pwd)":/work -w /work example-wind-analysis:latest bash -lc "\
      python scripts/analysis/integrate_feature_importance_report.py \
        --analysis-dir \"$(ls -dt artifacts_root/analysis/feature_importance_* | head -1)\" \
        --report-file \"$(ls -dt artifacts_root/analysis/feature_importance_* | head -1)/report.md\" \
        --section-title 'Feature Importance — SAR L2SP+KD model on SAR TEST set'"
  else
    echo ">>> Skipping SAR L2SP+KD feature-importance analysis: artifacts_root/sar/config/final_model_finetuned_l2sp_kd.txt not found <<<"
  fi
else
  echo ">>> Skipping SAR L2SP+KD feature-importance analysis per configuration <<<"
fi
