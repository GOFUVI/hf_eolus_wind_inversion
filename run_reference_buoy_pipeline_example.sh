#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Author: Example Team <team@example.org>
# Created: 2025-10-16
# Disclaimer: This obfuscated sample mirrors the reference-buoy workflow and must be adapted before any deployment.
# -----------------------------------------------------------------------------

# Exit immediately on error, undefined variable, or pipeline failure
set -euo pipefail

# Example reference-buoy pipeline (obfuscated).
# Replace placeholders such as example-profile, example-region-1, project-bucket-placeholder, and wind_training.* before running.

# Inference block toggle (set to 0 to skip every inference-related step)
RUN_BUOY_INFERENCE=1
RUN_BUOY_FINETUNED_TEST_INFERENCE=1
RUN_BUOY_FINETUNED_PIVOTS_INFERENCE=1
RUN_BUOY_FINETUNED_ON_SAR_INFERENCE=1
RUN_BUOY_L2SP_TRAINING=1
RUN_BUOY_L2SP_TEST_INFERENCE=1
RUN_BUOY_L2SP_ON_SAR_INFERENCE=1
RUN_BUOY_L2SP_KD_TRAINING=1
RUN_BUOY_L2SP_KD_TEST_INFERENCE=1
RUN_BUOY_L2SP_KD_ON_SAR_INFERENCE=1
RUN_BUOY_FEATURE_IMPORTANCE_FINAL=1
RUN_BUOY_FEATURE_IMPORTANCE_FINETUNED=1
RUN_BUOY_FEATURE_IMPORTANCE_L2SP=1
RUN_BUOY_FEATURE_IMPORTANCE_L2SP_KD=1

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
  artifacts_root/buoy \
  artifacts_root/buoy/reports/partition \
  artifacts_root/buoy/reports/hpo \
  artifacts_root/buoy/inference_metrics \
  artifacts_root/buoy/training_metrics \
  artifacts_root/buoy/normalization_params \
  artifacts_root/buoy/bin_metrics \
  artifacts_root/buoy/final_training \
  artifacts_root/buoy/fine_tuning \
  artifacts_root/buoy/fine_tuning/train_metrics \
  artifacts_root/buoy/fine_tuning/normalization_params \
  artifacts_root/buoy/fine_tuning/bin_metrics \
  artifacts_root/buoy/fine_tuning_l2sp \
  artifacts_root/buoy/fine_tuning_l2sp_kd \
  artifacts_root/buoy/config \
  artifacts_root/buoy/logs \
  artifacts_root/buoy/logs/partition \
  artifacts_root/buoy/logs/finalize_geoparquet_train \
  artifacts_root/buoy/logs/finalize_geoparquet_test \
  artifacts_root/buoy/logs/hpo \
  artifacts_root/buoy/logs/train_model \
  artifacts_root/buoy/logs/fine_tune_model \
  artifacts_root/buoy/logs/finalize_geoparquet_inference_test \
  artifacts_root/buoy/logs/finalize_geoparquet_inference_pivots

if [ -f artifacts_root/buoy/config/selected_model.txt ]; then
  echo ">>> Selected model file found; skipping reference buoy HPO <<<"
else
  echo ">>> Starting HPO for buoy-hpo at $(date) <<<"
  ./scripts/HPO/run_hpo.sh \
    --profile example-profile \
    --region example-region-1 \
    --job-name buoy-hpo-campaign-d-1 \
    --train-data-uri s3://project-bucket-placeholder/wind_training/training/buoy/train/ \
    --output-s3-uri s3://project-bucket-placeholder/wind_training/models/reference_buoy \
    --model-config artifacts_root/buoy/config/buoy_model.json \
    --hpo-config artifacts_root/buoy/config/buoy_hpo.json \
    --log-dir artifacts_root/buoy/logs/hpo/buoy-hpo-campaign-d-1
  echo ">>> Completed HPO for buoy-hpo at $(date) <<<"

  echo ">>> Starting follow-up HPO for buoy-hpo-2 (warm start) at $(date) <<<"
  ./scripts/HPO/run_hpo.sh \
    --profile example-profile \
    --region example-region-1 \
    --job-name buoy-hpo-campaign-d-2 \
    --parent-jobs buoy-hpo-campaign-d-1 \
    --train-data-uri s3://project-bucket-placeholder/wind_training/training/buoy/train/ \
    --output-s3-uri s3://project-bucket-placeholder/wind_training/models/reference_buoy \
    --model-config artifacts_root/buoy/config/buoy_model.json \
    --hpo-config artifacts_root/buoy/config/buoy_hpo.json \
    --log-dir artifacts_root/buoy/logs/hpo/buoy-hpo-campaign-d-2
  echo ">>> Completed HPO for buoy-hpo-2 at $(date) <<<"

  echo ">>> Integrating reference buoy HPO reports at $(date) <<<"
  ./scripts/HPO/hpo_metrics_report.sh -p example-profile -r example-region-1 -n buoy-hpo-campaign-d-1 -o artifacts_root/buoy/reports/hpo/buoy-hpo-campaign-d-1_hpo_report.md
  ./scripts/HPO/hpo_metrics_report.sh -p example-profile -r example-region-1 -n buoy-hpo-campaign-d-2 -o artifacts_root/buoy/reports/hpo/buoy-hpo-campaign-d-2_hpo_report.md
  ./scripts/HPO/integrate_hpo_reports.sh -i "artifacts_root/buoy/reports/hpo/buoy-hpo-campaign-d-2_hpo_report.md,artifacts_root/buoy/reports/hpo/buoy-hpo-campaign-d-1_hpo_report.md" -o artifacts_root/buoy/reports/hpo/buoy-hpo_all_hpo_report.md
  echo ">>> Completed integrating reference buoy HPO reports at $(date) <<<"

  ./scripts/HPO/select_best_hpo_job.sh --report artifacts_root/buoy/reports/hpo/buoy-hpo_all_hpo_report.md --output artifacts_root/buoy/config/selected_model.txt
fi

if [ -f artifacts_root/buoy/config/final_model.txt ]; then
  echo ">>> Final model file found; skipping reference buoy training stage <<<"
else
  echo ">>> Generating reference buoy model config from HPO at $(date) <<<"
  ./scripts/training/generate_model_config_from_hpo.sh \
    --train-job "$(awk '{sub(/#.*/,"");gsub(/^[ \t]+|[ \t]+$/,"");if(length($0)){print;exit}}' artifacts_root/buoy/config/selected_model.txt)" \
    --profile example-profile \
    --region example-region-1 \
    --output artifacts_root/buoy/config/buoy_hpo_best_model.json
  echo ">>> Completed reference buoy model config generation at $(date) <<<"

  echo ">>> Starting reference buoy full-training run at $(date) <<<"
  ./scripts/training/train_model.sh \
    --s3-prefix s3://project-bucket-placeholder/wind_training/models/reference_buoy/final \
    --profile example-profile \
    --region example-region-1 \
    --train-data-uri s3://project-bucket-placeholder/wind_training/training/buoy/train/ \
    --job-base-prefix buoy-range-example \
    --model-config artifacts_root/buoy/config/buoy_hpo_best_model.json \
    --output-dir artifacts_root/buoy/final_training \
    --no-cv \
    --seed 42
  echo ">>> Completed reference buoy full-training run at $(date) <<<"

  ./scripts/training/record_final_job.sh --log artifacts_root/buoy/final_training/train_model.log --output artifacts_root/buoy/config/final_model.txt
  echo ">>> Stored reference buoy final training job $(sed -n '1p' artifacts_root/buoy/config/final_model.txt) in artifacts_root/buoy/config/final_model.txt <<<"
fi

if [ ! -f artifacts_root/buoy/config/final_model.txt ]; then
  echo "Error: artifacts_root/buoy/config/final_model.txt not found. Remove this file to retrain or ensure training completed successfully." >&2
  exit 1
fi

echo ">>> Using reference buoy final training job $(awk '{sub(/#.*/,"");gsub(/^[ \t]+|[ \t]+$/,"");if(length($0)){print;exit}}' artifacts_root/buoy/config/final_model.txt) for downstream steps <<<"

  echo ">>> Fetching reference buoy training metrics at $(date) <<<"
  ./scripts/training/get_train_metrics.sh \
    --aws-profile example-profile \
    --train-job-name "$(awk '{sub(/#.*/,"");gsub(/^[ \t]+|[ \t]+$/,"");if(length($0)){print;exit}}' artifacts_root/buoy/config/final_model.txt)" \
    --output-directory artifacts_root/buoy/training_metrics
  ./scripts/training/get_norm_params.sh \
    --aws-profile example-profile \
    --train-job-name "$(awk '{sub(/#.*/,"");gsub(/^[ \t]+|[ \t]+$/,"");if(length($0)){print;exit}}' artifacts_root/buoy/config/final_model.txt)" \
    --output-directory artifacts_root/buoy/normalization_params
  ./scripts/training/get_bin_metrics.sh \
    --aws-profile example-profile \
    --train-job-name "$(awk '{sub(/#.*/,"");gsub(/^[ \t]+|[ \t]+$/,"");if(length($0)){print;exit}}' artifacts_root/buoy/config/final_model.txt)" \
    --output-directory artifacts_root/buoy/bin_metrics
  echo ">>> Completed fetching reference buoy training metrics at $(date) <<<"

  echo ">>> Locating latest reference buoy training job at $(date) <<<"

  if [[ "${RUN_BUOY_INFERENCE}" == "1" ]]; then
    echo ">>> Running reference buoy inference on test set at $(date) <<<"
    ./scripts/inference/run_inference.sh \
      --model-artifact "s3://project-bucket-placeholder/wind_training/models/reference_buoy/final/$(awk '{sub(/#.*/,"");gsub(/^[ \t]+|[ \t]+$/,"");if(length($0)){print;exit}}' artifacts_root/buoy/config/final_model.txt)/output/model.tar.gz" \
      --input-data s3://project-bucket-placeholder/wind_training/training/buoy/test/ \
      --output-s3-uri "s3://project-bucket-placeholder/wind_training/inference/buoy-range-example/test_set" \
      --profile example-profile \
      --region example-region-1 \
      --output-format parquet
    echo ">>> Completed reference buoy inference at $(date) <<<"
    echo ">>> Finalizing GeoParquet for reference buoy inference (test set) at $(date) <<<"
    ./scripts/geo_utils/finalize_geoparquet.sh \
      --db-name wind_training \
      --bucket-name project-bucket-placeholder \
      --output-prefix wind_training/inference/buoy-range-example/test_set \
      --output-table BUOY_RANGE_EXAMPLE_TEST_INFERENCE \
      --register-table \
      --profile example-profile \
      --region example-region-1 \
      --log-dir artifacts_root/buoy/logs/finalize_geoparquet_inference_test
    echo ">>> Completed GeoParquet finalization for reference buoy inference (test set) at $(date) <<<"


    echo ">>> Computing reference buoy inference metrics at $(date) <<<"
    mkdir -p "artifacts_root/buoy/inference_metrics/$(awk '{sub(/#.*/,"");gsub(/^[ \t]+|[ \t]+$/,"");if(length($0)){print;exit}}' artifacts_root/buoy/config/final_model.txt)"
    ./scripts/inference/compute_inference_metrics.sh \
      --predictions-s3 s3://project-bucket-placeholder/wind_training/inference/buoy-range-example/test_set/data.parquet \
      --metadata-s3 s3://project-bucket-placeholder/wind_training/inference/buoy-range-example/test_set/inference_metadata.json \
      --output-dir "artifacts_root/buoy/inference_metrics/$(awk '{sub(/#.*/,"");gsub(/^[ \t]+|[ \t]+$/,"");if(length($0)){print;exit}}' artifacts_root/buoy/config/final_model.txt)" \
      --profile example-profile \
      --region example-region-1 \
      --wind-bin-column wind_bin \
      --truth-speed-col buoy__wind_speed \
      --truth-dir-col  buoy__wind_dir
    echo ">>> Completed reference buoy inference metrics at $(date) <<<"

    echo ">>> Building STAC catalog for reference buoy inference (test set) at $(date) <<<"
    ./scripts/geo_utils/build_stac_catalog.sh \
      --collection BUOY_RANGE_EXAMPLE_TEST_INFERENCE \
      --s3-uri s3://project-bucket-placeholder/wind_training/inference/buoy-range-example/test_set/ \
      --stac-collection-properties-json artifacts_root/buoy/stac_config/stac_properties_collection_BUOY_RANGE_EXAMPLE_TEST_SET.json \
      --stac-item-properties-json artifacts_root/buoy/stac_config/stac_properties_item_BUOY_RANGE_EXAMPLE_TEST_SET.json \
      --output-dir catalogs/buoy_pipeline/buoy_range_final_test_set \
      --profile example-profile \
      --region example-region-1
    echo ">>> Completed STAC catalog build for reference buoy inference (test set) at $(date) <<<"

    echo ">>> Running reference buoy inference on PIVOTS_REFERENCE_BUOY (unlabeled) at $(date) <<<"
    ./scripts/inference/run_inference.sh \
      --model-artifact "s3://project-bucket-placeholder/wind_training/models/reference_buoy/final/$(awk '{sub(/#.*/,"");gsub(/^[ \t]+|[ \t]+$/,"");if(length($0)){print;exit}}' artifacts_root/buoy/config/final_model.txt)/output/model.tar.gz" \
      --input-data s3://project-bucket-placeholder/wind_training/pivots/with_reference_buoy/ \
      --output-s3-uri "s3://project-bucket-placeholder/wind_training/inference/buoy-range-example/pivots_reference_buoy" \
      --profile example-profile \
      --region example-region-1 \
      --output-format parquet
    echo ">>> Completed reference buoy inference on PIVOTS_REFERENCE_BUOY at $(date) <<<"

    echo ">>> Finalizing GeoParquet for reference buoy inference (PIVOTS_REFERENCE_BUOY) at $(date) <<<"
    ./scripts/geo_utils/finalize_geoparquet.sh \
      --db-name wind_training \
      --bucket-name project-bucket-placeholder \
      --output-prefix wind_training/inference/buoy-range-example/pivots_reference_buoy \
      --output-table BUOY_RANGE_EXAMPLE_PIVOTS_REFERENCE_BUOY \
      --register-table \
      --profile example-profile \
      --region example-region-1 \
      --log-dir artifacts_root/buoy/logs/finalize_geoparquet_inference_pivots
    echo ">>> Completed GeoParquet finalization for reference buoy inference (PIVOTS_REFERENCE_BUOY) at $(date) <<<"

    echo ">>> Building STAC catalog for reference buoy inference (PIVOTS_REFERENCE_BUOY) at $(date) <<<"
    ./scripts/geo_utils/build_stac_catalog.sh \
      --collection BUOY_RANGE_EXAMPLE_PIVOTS_REFERENCE_BUOY \
      --s3-uri s3://project-bucket-placeholder/wind_training/inference/buoy-range-example/pivots_reference_buoy/ \
      --stac-collection-properties-json artifacts_root/buoy/stac_config/stac_properties_collection_BUOY_RANGE_EXAMPLE_PIVOTS_REFERENCE_BUOY.json \
      --stac-item-properties-json artifacts_root/buoy/stac_config/stac_properties_item_BUOY_RANGE_EXAMPLE_PIVOTS_REFERENCE_BUOY.json \
      --output-dir catalogs/buoy_pipeline/buoy_range_final_pivots_reference_buoy \
      --profile example-profile \
      --region example-region-1
    echo ">>> Completed STAC catalog build for reference buoy inference (PIVOTS_REFERENCE_BUOY) at $(date) <<<"

    ./scripts/inference/run_inference.sh \
      --model-artifact "s3://project-bucket-placeholder/wind_training/models/reference_buoy/final/$(awk '{sub(/#.*/,"");gsub(/^[ \t]+|[ \t]+$/,"");if(length($0)){print;exit}}' artifacts_root/buoy/config/final_model.txt)/output/model.tar.gz" \
      --input-data s3://project-bucket-placeholder/wind_training/training/sar/test/ \
      --output-s3-uri "s3://project-bucket-placeholder/wind_training/inference/buoy-on-sar/test_set" \
      --profile example-profile \
      --region example-region-1 \
      --output-format parquet
    bash scripts/geo_utils/finalize_geoparquet.sh \
      --db-name wind_training \
      --bucket-name project-bucket-placeholder \
      --output-prefix wind_training/inference/buoy-on-sar/test_set \
      --output-table BUOY_ON_SAR_TEST_INFERENCE \
      --register-table \
      --profile example-profile \
      --region example-region-1 \
      --log-dir artifacts_root/buoy/logs/finalize_geoparquet_inference_test


    mkdir -p "artifacts_root/buoy/inference_metrics/$(awk '{sub(/#.*/,"");gsub(/^[ \t]+|[ \t]+$/,"");if(length($0)){print;exit}}' artifacts_root/buoy/config/final_model.txt)_on_sar"
    ./scripts/inference/compute_inference_metrics.sh \
      --predictions-s3 s3://project-bucket-placeholder/wind_training/inference/buoy-on-sar/test_set/data.parquet \
      --metadata-s3  s3://project-bucket-placeholder/wind_training/inference/buoy-on-sar/test_set/inference_metadata.json \
      --output-dir   "artifacts_root/buoy/inference_metrics/$(awk '{sub(/#.*/,"");gsub(/^[ \t]+|[ \t]+$/,"");if(length($0)){print;exit}}' artifacts_root/buoy/config/final_model.txt)_on_sar" \
      --profile example-profile \
      --region example-region-1 \
      --wind-bin-column wind_bin \
      --truth-speed-col sar__owiwindspeed_mean \
      --truth-dir-col  sar__owiwinddirection_mean

    echo ">>> Building STAC catalog for reference buoy-on-SAR inference (test set) at $(date) <<<"
    ./scripts/geo_utils/build_stac_catalog.sh \
      --collection BUOY_ON_SAR_TEST_INFERENCE \
      --s3-uri s3://project-bucket-placeholder/wind_training/inference/buoy-on-sar/test_set/ \
      --stac-collection-properties-json artifacts_root/buoy/stac_config/stac_properties_collection_BUOY_ON_SAR_TEST_SET.json \
      --stac-item-properties-json artifacts_root/buoy/stac_config/stac_properties_item_BUOY_ON_SAR_TEST_SET.json \
      --output-dir catalogs/buoy_pipeline/buoy_on_sar_test_set \
      --profile example-profile \
      --region example-region-1
    echo ">>> Completed STAC catalog build for reference buoy-on-SAR inference (test set) at $(date) <<<"
  else
    echo ">>> Skipping every reference buoy inference block per configuration <<<"
  fi


if [ -f artifacts_root/buoy/config/final_model_finetuned.txt ]; then
  echo ">>> Fine-tuned model file found; skipping reference buoy fine-tuning stage <<<"
else
  echo ">>> Preparing reference buoy fine-tuning config at $(date) using existing final model settings <<<"
  ./scripts/training/prepare_finetune_config.sh \
    --base artifacts_root/buoy/config/buoy_hpo_best_model.json \
    --output artifacts_root/buoy/config/buoy_finetune_model.json \
    --target-speed-col sar__owiwindspeed_mean \
    --target-dir-col sar__owiwinddirection_mean

  echo ">>> Starting reference buoy fine-tuning run on SAR training set at $(date) <<<"
  ./scripts/training/train_model.sh \
    --s3-prefix s3://project-bucket-placeholder/wind_training/models/reference_buoy/finetuned \
    --profile example-profile \
    --region example-region-1 \
    --train-data-uri s3://project-bucket-placeholder/wind_training/training/sar/train/ \
    --artifact-uri "s3://project-bucket-placeholder/wind_training/models/reference_buoy/final/$(sed -n '1p' artifacts_root/buoy/config/final_model.txt)/output/model.tar.gz" \
    --job-base-prefix buoy-range-finetuned \
    --model-config artifacts_root/buoy/config/buoy_finetune_model.json \
    --output-dir artifacts_root/buoy/fine_tuning \
    --no-cv \
    --seed 42
  echo ">>> Completed reference buoy fine-tuning run at $(date) <<<"

  ./scripts/training/record_final_job.sh --log artifacts_root/buoy/fine_tuning/train_model.log --output artifacts_root/buoy/config/final_model_finetuned.txt
  echo ">>> Stored reference buoy fine-tuned training job $(sed -n '1p' artifacts_root/buoy/config/final_model_finetuned.txt) in artifacts_root/buoy/config/final_model_finetuned.txt <<<"
fi

if [ ! -f artifacts_root/buoy/config/final_model_finetuned.txt ]; then
  echo "Error: artifacts_root/buoy/config/final_model_finetuned.txt not found. Remove this file to re-run fine-tuning or ensure training completed successfully." >&2
  exit 1
fi

echo ">>> Using reference buoy fine-tuned training job $(awk '{sub(/#.*/,"");gsub(/^[ \t]+|[ \t]+$/,"");if(length($0)){print;exit}}' artifacts_root/buoy/config/final_model_finetuned.txt) for fine-tuned evaluation <<<"

echo ">>> Fetching reference buoy fine-tuned training metrics at $(date) <<<"
./scripts/training/get_train_metrics.sh \
  --aws-profile example-profile \
  --train-job-name "$(awk '{sub(/#.*/,"");gsub(/^[ \t]+|[ \t]+$/,"");if(length($0)){print;exit}}' artifacts_root/buoy/config/final_model_finetuned.txt)" \
  --output-directory artifacts_root/buoy/fine_tuning/train_metrics
./scripts/training/get_norm_params.sh \
  --aws-profile example-profile \
  --train-job-name "$(awk '{sub(/#.*/,"");gsub(/^[ \t]+|[ \t]+$/,"");if(length($0)){print;exit}}' artifacts_root/buoy/config/final_model_finetuned.txt)" \
  --output-directory artifacts_root/buoy/fine_tuning/normalization_params
./scripts/training/get_bin_metrics.sh \
  --aws-profile example-profile \
  --train-job-name "$(awk '{sub(/#.*/,"");gsub(/^[ \t]+|[ \t]+$/,"");if(length($0)){print;exit}}' artifacts_root/buoy/config/final_model_finetuned.txt)" \
  --output-directory artifacts_root/buoy/fine_tuning/bin_metrics
echo ">>> Completed fetching reference buoy fine-tuned training metrics at $(date) <<<"

if [[ "${RUN_BUOY_FINETUNED_TEST_INFERENCE}" == "1" ]]; then
  echo ">>> Running reference buoy fine-tuned inference on reference buoy test set at $(date) <<<"
  ./scripts/inference/run_inference.sh \
    --model-artifact "s3://project-bucket-placeholder/wind_training/models/reference_buoy/finetuned/$(awk '{sub(/#.*/,"");gsub(/^[ \t]+|[ \t]+$/,"");if(length($0)){print;exit}}' artifacts_root/buoy/config/final_model_finetuned.txt)/output/model.tar.gz" \
    --input-data s3://project-bucket-placeholder/wind_training/training/buoy/test/ \
    --output-s3-uri "s3://project-bucket-placeholder/wind_training/inference/buoy-range-finetuned/test_set" \
    --profile example-profile \
    --region example-region-1 \
    --output-format parquet
  echo ">>> Completed reference buoy fine-tuned inference on reference buoy test set at $(date) <<<"
  echo ">>> Finalizing GeoParquet for reference buoy fine-tuned inference (test set) at $(date) <<<"
  ./scripts/geo_utils/finalize_geoparquet.sh \
    --db-name wind_training \
    --bucket-name project-bucket-placeholder \
    --output-prefix wind_training/inference/buoy-range-finetuned/test_set \
    --output-table BUOY_RANGE_FINETUNED_TEST_INFERENCE \
    --register-table \
    --profile example-profile \
    --region example-region-1 \
    --log-dir artifacts_root/buoy/logs/finalize_geoparquet_inference_test
  echo ">>> Completed GeoParquet finalization for reference buoy fine-tuned inference (test set) at $(date) <<<"


  echo ">>> Computing reference buoy fine-tuned inference metrics at $(date) <<<"
  mkdir -p "artifacts_root/buoy/inference_metrics/$(awk '{sub(/#.*/,"");gsub(/^[ \t]+|[ \t]+$/,"");if(length($0)){print;exit}}' artifacts_root/buoy/config/final_model_finetuned.txt)_finetuned"
  ./scripts/inference/compute_inference_metrics.sh \
    --predictions-s3 s3://project-bucket-placeholder/wind_training/inference/buoy-range-finetuned/test_set/data.parquet \
    --metadata-s3 s3://project-bucket-placeholder/wind_training/inference/buoy-range-finetuned/test_set/inference_metadata.json \
    --output-dir "artifacts_root/buoy/inference_metrics/$(awk '{sub(/#.*/,"");gsub(/^[ \t]+|[ \t]+$/,"");if(length($0)){print;exit}}' artifacts_root/buoy/config/final_model_finetuned.txt)_finetuned" \
    --profile example-profile \
    --region example-region-1 \
    --wind-bin-column wind_bin \
    --truth-speed-col buoy__wind_speed \
    --truth-dir-col  buoy__wind_dir
  echo ">>> Completed reference buoy fine-tuned inference metrics at $(date) <<<"

  echo ">>> Building STAC catalog for reference buoy fine-tuned inference (test set) at $(date) <<<"
  ./scripts/geo_utils/build_stac_catalog.sh \
    --collection BUOY_RANGE_FINETUNED_TEST_INFERENCE \
    --s3-uri s3://project-bucket-placeholder/wind_training/inference/buoy-range-finetuned/test_set/ \
    --stac-collection-properties-json artifacts_root/buoy/stac_config/stac_properties_collection_BUOY_RANGE_EXAMPLE_TEST_SET.json \
    --stac-item-properties-json artifacts_root/buoy/stac_config/stac_properties_item_BUOY_RANGE_EXAMPLE_TEST_SET.json \
    --output-dir catalogs/buoy_pipeline/buoy_range_finetuned_test_set \
    --profile example-profile \
    --region example-region-1
  echo ">>> Completed STAC catalog build for reference buoy fine-tuned inference (test set) at $(date) <<<"
else
  echo ">>> Skipping reference buoy fine-tuned test-set inference per configuration <<<"
fi

if [[ "${RUN_BUOY_FINETUNED_PIVOTS_INFERENCE}" == "1" ]]; then
  echo ">>> Running reference buoy fine-tuned inference on PIVOTS_REFERENCE_BUOY at $(date) <<<"
  ./scripts/inference/run_inference.sh \
    --model-artifact "s3://project-bucket-placeholder/wind_training/models/reference_buoy/finetuned/$(awk '{sub(/#.*/,"");gsub(/^[ \t]+|[ \t]+$/,"");if(length($0)){print;exit}}' artifacts_root/buoy/config/final_model_finetuned.txt)/output/model.tar.gz" \
    --input-data s3://project-bucket-placeholder/wind_training/pivots/with_reference_buoy/ \
    --output-s3-uri "s3://project-bucket-placeholder/wind_training/inference/buoy-range-finetuned/pivots_reference_buoy" \
    --profile example-profile \
    --region example-region-1 \
    --output-format parquet
  echo ">>> Completed reference buoy fine-tuned inference on PIVOTS_REFERENCE_BUOY at $(date) <<<"

  echo ">>> Finalizing GeoParquet for reference buoy fine-tuned inference (PIVOTS_REFERENCE_BUOY) at $(date) <<<"
  ./scripts/geo_utils/finalize_geoparquet.sh \
    --db-name wind_training \
    --bucket-name project-bucket-placeholder \
    --output-prefix wind_training/inference/buoy-range-finetuned/pivots_reference_buoy \
    --output-table BUOY_RANGE_FINETUNED_PIVOTS_REFERENCE_BUOY \
    --register-table \
    --profile example-profile \
    --region example-region-1 \
    --log-dir artifacts_root/buoy/logs/finalize_geoparquet_inference_pivots
  echo ">>> Completed GeoParquet finalization for reference buoy fine-tuned inference (PIVOTS_REFERENCE_BUOY) at $(date) <<<"

  echo ">>> Building STAC catalog for reference buoy fine-tuned inference (PIVOTS_REFERENCE_BUOY) at $(date) <<<"
  ./scripts/geo_utils/build_stac_catalog.sh \
    --collection BUOY_RANGE_FINETUNED_PIVOTS_REFERENCE_BUOY \
    --s3-uri s3://project-bucket-placeholder/wind_training/inference/buoy-range-finetuned/pivots_reference_buoy/ \
    --stac-collection-properties-json artifacts_root/buoy/stac_config/stac_properties_collection_BUOY_RANGE_EXAMPLE_PIVOTS_REFERENCE_BUOY.json \
    --stac-item-properties-json artifacts_root/buoy/stac_config/stac_properties_item_BUOY_RANGE_EXAMPLE_PIVOTS_REFERENCE_BUOY.json \
    --output-dir catalogs/buoy_pipeline/buoy_range_finetuned_pivots_reference_buoy \
    --profile example-profile \
    --region example-region-1
  echo ">>> Completed STAC catalog build for reference buoy fine-tuned inference (PIVOTS_REFERENCE_BUOY) at $(date) <<<"
else
  echo ">>> Skipping reference buoy fine-tuned pivots inference per configuration <<<"
fi

if [[ "${RUN_BUOY_FINETUNED_ON_SAR_INFERENCE}" == "1" ]]; then
  echo ">>> Running reference buoy fine-tuned inference on SAR test set at $(date) <<<"
  ./scripts/inference/run_inference.sh \
    --model-artifact "s3://project-bucket-placeholder/wind_training/models/reference_buoy/finetuned/$(awk '{sub(/#.*/,"");gsub(/^[ \t]+|[ \t]+$/,"");if(length($0)){print;exit}}' artifacts_root/buoy/config/final_model_finetuned.txt)/output/model.tar.gz" \
    --input-data s3://project-bucket-placeholder/wind_training/training/sar/test/ \
    --output-s3-uri "s3://project-bucket-placeholder/wind_training/inference/buoy-finetuned-on-sar/test_set" \
    --profile example-profile \
    --region example-region-1 \
    --output-format parquet
  echo ">>> Completed reference buoy fine-tuned inference on SAR test set at $(date) <<<"
  bash scripts/geo_utils/finalize_geoparquet.sh \
    --db-name wind_training \
    --bucket-name project-bucket-placeholder \
    --output-prefix wind_training/inference/buoy-finetuned-on-sar/test_set \
    --output-table BUOY_FINETUNED_ON_SAR_TEST_INFERENCE \
    --register-table \
    --profile example-profile \
    --region example-region-1 \
    --log-dir artifacts_root/buoy/logs/finalize_geoparquet_inference_test


  mkdir -p "artifacts_root/buoy/inference_metrics/$(awk '{sub(/#.*/,"");gsub(/^[ \t]+|[ \t]+$/,"");if(length($0)){print;exit}}' artifacts_root/buoy/config/final_model_finetuned.txt)_finetuned_on_sar"
  ./scripts/inference/compute_inference_metrics.sh \
    --predictions-s3 s3://project-bucket-placeholder/wind_training/inference/buoy-finetuned-on-sar/test_set/data.parquet \
    --metadata-s3  s3://project-bucket-placeholder/wind_training/inference/buoy-finetuned-on-sar/test_set/inference_metadata.json \
    --output-dir   "artifacts_root/buoy/inference_metrics/$(awk '{sub(/#.*/,"");gsub(/^[ \t]+|[ \t]+$/,"");if(length($0)){print;exit}}' artifacts_root/buoy/config/final_model_finetuned.txt)_finetuned_on_sar" \
    --profile example-profile \
    --region example-region-1 \
    --wind-bin-column wind_bin \
    --truth-speed-col sar__owiwindspeed_mean \
    --truth-dir-col  sar__owiwinddirection_mean

  echo ">>> Building STAC catalog for reference buoy fine-tuned on SAR inference (test set) at $(date) <<<"
  ./scripts/geo_utils/build_stac_catalog.sh \
    --collection BUOY_FINETUNED_ON_SAR_TEST_INFERENCE \
    --s3-uri s3://project-bucket-placeholder/wind_training/inference/buoy-finetuned-on-sar/test_set/ \
    --stac-collection-properties-json artifacts_root/buoy/stac_config/stac_properties_collection_BUOY_ON_SAR_TEST_SET.json \
    --stac-item-properties-json artifacts_root/buoy/stac_config/stac_properties_item_BUOY_ON_SAR_TEST_SET.json \
    --output-dir catalogs/buoy_pipeline/buoy_finetuned_on_sar_test_set \
    --profile example-profile \
    --region example-region-1
  echo ">>> Completed STAC catalog build for reference buoy fine-tuned on SAR inference (test set) at $(date) <<<"
else
  echo ">>> Skipping reference buoy fine-tuned SAR inference per configuration <<<"
fi

if [[ "${RUN_BUOY_L2SP_TRAINING}" == "1" ]]; then
  if [ -f artifacts_root/buoy/config/final_model_finetuned_l2sp.txt ]; then
    echo ">>> Fine-tuned (L2-SP) model file found; skipping reference buoy L2-SP fine-tuning stage <<<"
  else
    echo ">>> Preparing reference buoy L2-SP rehearsal model config at $(date) <<<"
    if [ ! -f artifacts_root/buoy/config/buoy_finetune_model.json ]; then
      echo ">>> Base fine-tuning config missing; regenerating from final model settings <<<"
      ./scripts/training/prepare_finetune_config.sh \
        --base artifacts_root/buoy/config/buoy_hpo_best_model.json \
        --output artifacts_root/buoy/config/buoy_finetune_model.json \
        --target-speed-col sar__owiwindspeed_mean \
        --target-dir-col sar__owiwinddirection_mean
    fi
    ./scripts/training/prepare_finetune_config.sh \
      --base artifacts_root/buoy/config/buoy_finetune_model.json \
      --output artifacts_root/buoy/config/buoy_finetune_l2sp_rehearsal.json \
      --target-speed-col sar__owiwindspeed_mean \
      --target-dir-col sar__owiwinddirection_mean \
      --force
    python3 - <<'PY'
import json
from pathlib import Path

cfg_path = Path("artifacts_root/buoy/config/buoy_finetune_l2sp_rehearsal.json")
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
    echo ">>> Prepared reference buoy L2-SP rehearsal model config <<<"

    echo ">>> Starting reference buoy fine-tuning (L2-SP + partial freeze) at $(date) <<<"
    ./scripts/training/train_model.sh \
      --s3-prefix s3://project-bucket-placeholder/wind_training/models/reference_buoy/finetuned_l2sp \
      --profile example-profile \
      --region example-region-1 \
      --train-data-uri s3://project-bucket-placeholder/wind_training/training/sar/train/ \
      --artifact-uri "s3://project-bucket-placeholder/wind_training/models/reference_buoy/final/$(sed -n '1p' artifacts_root/buoy/config/final_model.txt)/output/model.tar.gz" \
      --job-base-prefix buoy-range-finetuned-l2sp \
      --model-config artifacts_root/buoy/config/buoy_finetune_l2sp_rehearsal.json \
      --rehearsal-data-uri s3://project-bucket-placeholder/wind_training/training/buoy/train/ \
      --rehearsal-target-speed-col buoy__wind_speed \
      --rehearsal-target-dir-col buoy__wind_dir \
      --rehearsal-fraction 0.15 \
      --output-dir artifacts_root/buoy/fine_tuning_l2sp \
      --no-cv \
      --seed 42
    echo ">>> Completed reference buoy L2-SP fine-tuning at $(date) <<<"

    ./scripts/training/record_final_job.sh --log artifacts_root/buoy/fine_tuning_l2sp/train_model.log --output artifacts_root/buoy/config/final_model_finetuned_l2sp.txt
    echo ">>> Stored reference buoy fine-tuned (L2-SP) job $(sed -n '1p' artifacts_root/buoy/config/final_model_finetuned_l2sp.txt) in artifacts_root/buoy/config/final_model_finetuned_l2sp.txt <<<"
  fi
else
  echo ">>> Skipping reference buoy L2-SP fine-tuning stage per configuration <<<"
fi

if [ ! -f artifacts_root/buoy/config/final_model_finetuned_l2sp.txt ]; then
  echo "Error: artifacts_root/buoy/config/final_model_finetuned_l2sp.txt not found. Remove this file to re-run fine-tuning or ensure training completed successfully." >&2
  exit 1
fi

BUOY_L2SP_JOB=$(awk '{sub(/#.*/,"");gsub(/^[ \t]+|[ \t]+$/,"");if(length($0)){print;exit}}' artifacts_root/buoy/config/final_model_finetuned_l2sp.txt)
echo ">>> Using reference buoy fine-tuned (L2-SP) training job ${BUOY_L2SP_JOB} for L2-SP evaluation <<<"

if [[ "${RUN_BUOY_L2SP_TEST_INFERENCE}" == "1" ]]; then
  echo ">>> Running reference buoy (L2-SP) inference on reference buoy test set at $(date) <<<"
  ./scripts/inference/run_inference.sh \
    --model-artifact "s3://project-bucket-placeholder/wind_training/models/reference_buoy/finetuned_l2sp/$(sed -n '1p' artifacts_root/buoy/config/final_model_finetuned_l2sp.txt)/output/model.tar.gz" \
    --input-data s3://project-bucket-placeholder/wind_training/training/buoy/test/ \
    --output-s3-uri "s3://project-bucket-placeholder/wind_training/inference/buoy-range-finetuned-l2sp/test_set" \
    --profile example-profile \
    --region example-region-1 \
    --output-format parquet
  echo ">>> Completed reference buoy (L2-SP) inference on reference buoy test set at $(date) <<<"

  ./scripts/geo_utils/finalize_geoparquet.sh \
    --db-name wind_training \
    --bucket-name project-bucket-placeholder \
    --output-prefix wind_training/inference/buoy-range-finetuned-l2sp/test_set \
    --output-table BUOY_RANGE_FINETUNED_L2SP_TEST_INFERENCE \
    --register-table \
    --profile example-profile \
    --log-dir artifacts_root/buoy/logs/finalize_geoparquet_inference_test
  echo ">>> Completed GeoParquet finalization for reference buoy (L2-SP) test inference at $(date) <<<"

  echo ">>> Computing reference buoy (L2-SP) inference metrics at $(date) <<<"
  mkdir -p "artifacts_root/buoy/inference_metrics/$(sed -n '1p' artifacts_root/buoy/config/final_model_finetuned_l2sp.txt)_finetuned_l2sp"
  ./scripts/inference/compute_inference_metrics.sh \
    --predictions-s3 s3://project-bucket-placeholder/wind_training/inference/buoy-range-finetuned-l2sp/test_set/data.parquet \
    --metadata-s3 s3://project-bucket-placeholder/wind_training/inference/buoy-range-finetuned-l2sp/test_set/inference_metadata.json \
    --output-dir "artifacts_root/buoy/inference_metrics/$(sed -n '1p' artifacts_root/buoy/config/final_model_finetuned_l2sp.txt)_finetuned_l2sp" \
    --profile example-profile \
    --region example-region-1 \
    --wind-bin-column wind_bin \
    --truth-speed-col buoy__wind_speed \
    --truth-dir-col  buoy__wind_dir
  echo ">>> Completed reference buoy (L2-SP) inference metrics at $(date) <<<"
else
  echo ">>> Skipping reference buoy (L2-SP) test-set inference per configuration <<<"
fi

if [[ "${RUN_BUOY_L2SP_ON_SAR_INFERENCE}" == "1" ]]; then
  echo ">>> Running reference buoy (L2-SP) inference on SAR test set at $(date) <<<"
  ./scripts/inference/run_inference.sh \
    --model-artifact "s3://project-bucket-placeholder/wind_training/models/reference_buoy/finetuned_l2sp/$(sed -n '1p' artifacts_root/buoy/config/final_model_finetuned_l2sp.txt)/output/model.tar.gz" \
    --input-data s3://project-bucket-placeholder/wind_training/training/sar/test/ \
    --output-s3-uri "s3://project-bucket-placeholder/wind_training/inference/buoy-finetuned-l2sp-on-sar/test_set" \
    --profile example-profile \
    --region example-region-1 \
    --output-format parquet
  echo ">>> Completed reference buoy (L2-SP) inference on SAR test set at $(date) <<<"

  ./scripts/geo_utils/finalize_geoparquet.sh \
    --db-name wind_training \
    --bucket-name project-bucket-placeholder \
    --output-prefix wind_training/inference/buoy-finetuned-l2sp-on-sar/test_set \
    --output-table BUOY_FINETUNED_L2SP_ON_SAR_TEST_INFERENCE \
    --register-table \
    --profile example-profile \
    --log-dir artifacts_root/buoy/logs/finalize_geoparquet_inference_test
  echo ">>> Completed GeoParquet finalization for reference buoy (L2-SP) on SAR test at $(date) <<<"

  echo ">>> Computing reference buoy (L2-SP) on SAR inference metrics at $(date) <<<"
  mkdir -p "artifacts_root/buoy/inference_metrics/$(sed -n '1p' artifacts_root/buoy/config/final_model_finetuned_l2sp.txt)_finetuned_l2sp_on_sar"
  ./scripts/inference/compute_inference_metrics.sh \
    --predictions-s3 s3://project-bucket-placeholder/wind_training/inference/buoy-finetuned-l2sp-on-sar/test_set/data.parquet \
    --metadata-s3  s3://project-bucket-placeholder/wind_training/inference/buoy-finetuned-l2sp-on-sar/test_set/inference_metadata.json \
    --output-dir   "artifacts_root/buoy/inference_metrics/$(sed -n '1p' artifacts_root/buoy/config/final_model_finetuned_l2sp.txt)_finetuned_l2sp_on_sar" \
    --profile example-profile \
    --region example-region-1 \
    --wind-bin-column wind_bin \
    --truth-speed-col sar__owiwindspeed_mean \
    --truth-dir-col  sar__owiwinddirection_mean
  echo ">>> Completed reference buoy (L2-SP) on SAR inference metrics at $(date) <<<"
else
  echo ">>> Skipping reference buoy (L2-SP) on SAR inference per configuration <<<"
fi

if [[ "${RUN_BUOY_L2SP_KD_TRAINING}" == "1" ]]; then
  if [ -f artifacts_root/buoy/config/final_model_finetuned_l2sp_kd.txt ]; then
    echo ">>> Fine-tuned (L2-SP+KD) model file found; skipping reference buoy L2-SP+KD fine-tuning stage <<<"
  else
    echo ">>> Preparing reference buoy L2-SP+KD rehearsal model config at $(date) <<<"
    if [ ! -f artifacts_root/buoy/config/buoy_finetune_model.json ]; then
      echo ">>> Base fine-tuning config missing; regenerating from final model settings <<<"
      ./scripts/training/prepare_finetune_config.sh \
        --base artifacts_root/buoy/config/buoy_hpo_best_model.json \
        --output artifacts_root/buoy/config/buoy_finetune_model.json \
        --target-speed-col sar__owiwindspeed_mean \
        --target-dir-col sar__owiwinddirection_mean
    fi
    ./scripts/training/prepare_finetune_config.sh \
      --base artifacts_root/buoy/config/buoy_finetune_model.json \
      --output artifacts_root/buoy/config/buoy_finetune_l2sp_rehearsal_kd.json \
      --target-speed-col sar__owiwindspeed_mean \
      --target-dir-col sar__owiwinddirection_mean \
      --force
    python3 - <<'PY'
import json
from pathlib import Path

cfg_path = Path("artifacts_root/buoy/config/buoy_finetune_l2sp_rehearsal_kd.json")
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
    echo ">>> Prepared reference buoy L2-SP+KD rehearsal model config <<<"

    echo ">>> Starting reference buoy fine-tuning (L2-SP + partial freeze + KD) at $(date) <<<"
    ./scripts/training/train_model.sh \
      --s3-prefix s3://project-bucket-placeholder/wind_training/models/reference_buoy/finetuned_l2sp_kd \
      --profile example-profile \
      --region example-region-1 \
      --train-data-uri s3://project-bucket-placeholder/wind_training/training/sar/train/ \
      --artifact-uri "s3://project-bucket-placeholder/wind_training/models/reference_buoy/final/$(sed -n '1p' artifacts_root/buoy/config/final_model.txt)/output/model.tar.gz" \
      --job-base-prefix buoy-range-finetuned-l2sp-kd \
      --model-config artifacts_root/buoy/config/buoy_finetune_l2sp_rehearsal_kd.json \
      --rehearsal-data-uri s3://project-bucket-placeholder/wind_training/training/buoy/train/ \
      --rehearsal-target-speed-col buoy__wind_speed \
      --rehearsal-target-dir-col buoy__wind_dir \
      --rehearsal-fraction 0.15 \
      --output-dir artifacts_root/buoy/fine_tuning_l2sp_kd \
      --no-cv \
      --seed 42
    echo ">>> Completed reference buoy L2-SP+KD fine-tuning at $(date) <<<"

    ./scripts/training/record_final_job.sh --log artifacts_root/buoy/fine_tuning_l2sp_kd/train_model.log --output artifacts_root/buoy/config/final_model_finetuned_l2sp_kd.txt
    echo ">>> Stored reference buoy fine-tuned (L2-SP+KD) job $(sed -n '1p' artifacts_root/buoy/config/final_model_finetuned_l2sp_kd.txt) in artifacts_root/buoy/config/final_model_finetuned_l2sp_kd.txt <<<"
  fi
else
  echo ">>> Skipping reference buoy L2-SP+KD fine-tuning stage per configuration <<<"
fi

if [ ! -f artifacts_root/buoy/config/final_model_finetuned_l2sp_kd.txt ]; then
  echo "Error: artifacts_root/buoy/config/final_model_finetuned_l2sp_kd.txt not found. Remove this file to re-run fine-tuning or ensure training completed successfully." >&2
  exit 1
fi

BUOY_L2SP_KD_JOB=$(awk '{sub(/#.*/,"");gsub(/^[ \t]+|[ \t]+$/,"");if(length($0)){print;exit}}' artifacts_root/buoy/config/final_model_finetuned_l2sp_kd.txt)
cat <<EOF > artifacts_root/buoy/fine_tuning_l2sp_kd/finetune_strategy.json
{
  "strategy": "l2sp_rehearsal_freeze_partial_kd",
  "timestamp_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "use_kd": true,
  "lambda_kd": 0.3,
  "teacher": {
    "model_artifact": "s3://project-bucket-placeholder/wind_training/models/reference_buoy/final/$(sed -n '1p' artifacts_root/buoy/config/final_model.txt)/output/model.tar.gz",
    "checkpoint": "s3://project-bucket-placeholder/wind_training/models/reference_buoy/finetuned_l2sp_kd/fine-tuning/${BUOY_L2SP_KD_JOB}/checkpoint.pth"
  },
  "student_job": "${BUOY_L2SP_KD_JOB}",
  "rehearsal_fraction": 0.25,
  "l2sp_backbone_lambda": 0.005,
  "l2sp_heads_lambda": 0.002
}
EOF

echo ">>> Using reference buoy fine-tuned (L2-SP+KD) training job ${BUOY_L2SP_KD_JOB} for L2-SP+KD evaluation <<<"

if [[ "${RUN_BUOY_L2SP_KD_TEST_INFERENCE}" == "1" ]]; then
  echo ">>> Running reference buoy (L2-SP+KD) inference on reference buoy test set at $(date) <<<"
  ./scripts/inference/run_inference.sh \
    --model-artifact "s3://project-bucket-placeholder/wind_training/models/reference_buoy/finetuned_l2sp_kd/${BUOY_L2SP_KD_JOB}/output/model.tar.gz" \
    --input-data s3://project-bucket-placeholder/wind_training/training/buoy/test/ \
    --output-s3-uri "s3://project-bucket-placeholder/wind_training/inference/buoy-range-finetuned-l2sp-kd/test_set" \
    --profile example-profile \
    --region example-region-1 \
    --output-format parquet
  echo ">>> Completed reference buoy (L2-SP+KD) inference on reference buoy test set at $(date) <<<"

  ./scripts/geo_utils/finalize_geoparquet.sh \
    --db-name wind_training \
    --bucket-name project-bucket-placeholder \
    --output-prefix wind_training/inference/buoy-range-finetuned-l2sp-kd/test_set \
    --output-table BUOY_RANGE_FINETUNED_L2SP_TEST_INFERENCE \
    --register-table \
    --profile example-profile \
    --log-dir artifacts_root/buoy/logs/finalize_geoparquet_inference_test
  echo ">>> Completed GeoParquet finalization for reference buoy (L2-SP+KD) test inference at $(date) <<<"

  echo ">>> Computing reference buoy (L2-SP+KD) inference metrics at $(date) <<<"
  mkdir -p "artifacts_root/buoy/inference_metrics/${BUOY_L2SP_KD_JOB}_finetuned_l2sp_kd"
  ./scripts/inference/compute_inference_metrics.sh \
    --predictions-s3 s3://project-bucket-placeholder/wind_training/inference/buoy-range-finetuned-l2sp-kd/test_set/data.parquet \
    --metadata-s3 s3://project-bucket-placeholder/wind_training/inference/buoy-range-finetuned-l2sp-kd/test_set/inference_metadata.json \
    --output-dir "artifacts_root/buoy/inference_metrics/${BUOY_L2SP_KD_JOB}_finetuned_l2sp_kd" \
    --profile example-profile \
    --region example-region-1 \
    --wind-bin-column wind_bin \
    --truth-speed-col buoy__wind_speed \
    --truth-dir-col  buoy__wind_dir
  echo ">>> Completed reference buoy (L2-SP+KD) inference metrics at $(date) <<<"
else
  echo ">>> Skipping reference buoy (L2-SP+KD) test-set inference per configuration <<<"
fi

if [[ "${RUN_BUOY_L2SP_KD_ON_SAR_INFERENCE}" == "1" ]]; then
  echo ">>> Running reference buoy (L2-SP+KD) inference on SAR test set at $(date) <<<"
  ./scripts/inference/run_inference.sh \
    --model-artifact "s3://project-bucket-placeholder/wind_training/models/reference_buoy/finetuned_l2sp_kd/${BUOY_L2SP_KD_JOB}/output/model.tar.gz" \
    --input-data s3://project-bucket-placeholder/wind_training/training/sar/test/ \
    --output-s3-uri "s3://project-bucket-placeholder/wind_training/inference/buoy-finetuned-l2sp-kd-on-sar/test_set" \
    --profile example-profile \
    --region example-region-1 \
    --output-format parquet
  echo ">>> Completed reference buoy (L2-SP+KD) inference on SAR test set at $(date) <<<"

  ./scripts/geo_utils/finalize_geoparquet.sh \
    --db-name wind_training \
    --bucket-name project-bucket-placeholder \
    --output-prefix wind_training/inference/buoy-finetuned-l2sp-kd-on-sar/test_set \
    --output-table BUOY_FINETUNED_L2SP_KD_ON_SAR_TEST_INFERENCE \
    --register-table \
    --profile example-profile \
    --log-dir artifacts_root/buoy/logs/finalize_geoparquet_inference_test
  echo ">>> Completed GeoParquet finalization for reference buoy (L2-SP+KD) on SAR test at $(date) <<<"

  echo ">>> Computing reference buoy (L2-SP+KD) on SAR inference metrics at $(date) <<<"
  mkdir -p "artifacts_root/buoy/inference_metrics/${BUOY_L2SP_KD_JOB}_finetuned_l2sp_kd_on_sar"
  ./scripts/inference/compute_inference_metrics.sh \
    --predictions-s3 s3://project-bucket-placeholder/wind_training/inference/buoy-finetuned-l2sp-kd-on-sar/test_set/data.parquet \
    --metadata-s3  s3://project-bucket-placeholder/wind_training/inference/buoy-finetuned-l2sp-kd-on-sar/test_set/inference_metadata.json \
    --output-dir   "artifacts_root/buoy/inference_metrics/${BUOY_L2SP_KD_JOB}_finetuned_l2sp_kd_on_sar" \
    --profile example-profile \
    --region example-region-1 \
    --wind-bin-column wind_bin \
    --truth-speed-col sar__owiwindspeed_mean \
    --truth-dir-col  sar__owiwinddirection_mean
  echo ">>> Completed reference buoy (L2-SP+KD) on SAR inference metrics at $(date) <<<"
else
  echo ">>> Skipping reference buoy (L2-SP+KD) on SAR inference per configuration <<<"
fi

if [[ "${RUN_BUOY_FEATURE_IMPORTANCE_FINAL}" == "1" ]]; then
  if [ -f artifacts_root/buoy/config/final_model.txt ]; then
    ensure_analysis_dirs
    build_analysis_image
    echo ">>> BUOY FINAL: downloading model artifact and test data at $(date) <<<"
    aws s3 cp \
      "s3://project-bucket-placeholder/wind_training/models/reference_buoy/final/$(sed -n '1p' artifacts_root/buoy/config/final_model.txt)/output/model.tar.gz" \
      artifacts_root/analysis/models/buoy_final_model.tar.gz \
      --profile example-profile --region example-region-1
    aws s3 sync \
      s3://project-bucket-placeholder/wind_training/training/buoy/test/ \
      artifacts_root/analysis/data/buoy_test \
      --profile example-profile --region example-region-1
    echo ">>> BUOY FINAL: running feature-importance analysis at $(date) <<<"
    docker run --rm \
      -v "$(pwd)":/work \
      -w /work \
      example-wind-analysis:latest bash -lc "\
        python scripts/analysis/feature_importance.py \
          --model-artifact artifacts_root/analysis/models/buoy_final_model.tar.gz \
          --data-path artifacts_root/analysis/data/buoy_test \
          --output-dir artifacts_root/analysis \
          --grouping both \
          --top-k 30" \
      | tee -a artifacts_root/analysis/logs/buoy_final_test_feature_importance.log

    docker run --rm -v "$(pwd)":/work -w /work example-wind-analysis:latest bash -lc "\
      python scripts/analysis/integrate_feature_importance_report.py \
        --analysis-dir \"$(ls -dt artifacts_root/analysis/feature_importance_* | head -1)\" \
        --report-file \"$(ls -dt artifacts_root/analysis/feature_importance_* | head -1)/report.md\" \
        --section-title 'Feature Importance — reference buoy FINAL model on reference buoy TEST set'"
  else
    echo ">>> Skipping reference buoy FINAL feature-importance analysis: artifacts_root/buoy/config/final_model.txt not found <<<"
  fi
else
  echo ">>> Skipping reference buoy FINAL feature-importance analysis per configuration <<<"
fi

if [[ "${RUN_BUOY_FEATURE_IMPORTANCE_FINETUNED}" == "1" ]]; then
  if [ -f artifacts_root/buoy/config/final_model_finetuned.txt ]; then
    ensure_analysis_dirs
    build_analysis_image
    echo ">>> BUOY FINETUNED: downloading model artifact at $(date) <<<"
    aws s3 cp \
      "s3://project-bucket-placeholder/wind_training/models/reference_buoy/finetuned/$(sed -n '1p' artifacts_root/buoy/config/final_model_finetuned.txt)/output/model.tar.gz" \
      artifacts_root/analysis/models/buoy_finetuned_model.tar.gz \
      --profile example-profile --region example-region-1
    aws s3 sync \
      s3://project-bucket-placeholder/wind_training/training/sar/test/ \
      artifacts_root/analysis/data/sar_test \
      --profile example-profile --region example-region-1
    aws s3 sync \
      s3://project-bucket-placeholder/wind_training/training/buoy/test/ \
      artifacts_root/analysis/data/buoy_test \
      --profile example-profile --region example-region-1

    echo ">>> BUOY FINETUNED: analysis on SAR TEST at $(date) <<<"
    docker run --rm \
      -v "$(pwd)":/work \
      -w /work \
      example-wind-analysis:latest bash -lc "\
        python scripts/analysis/feature_importance.py \
          --model-artifact artifacts_root/analysis/models/buoy_finetuned_model.tar.gz \
          --data-path artifacts_root/analysis/data/sar_test \
          --output-dir artifacts_root/analysis \
          --grouping both \
          --top-k 30" \
      | tee -a artifacts_root/analysis/logs/buoy_finetuned_on_sar_test_feature_importance.log

    docker run --rm -v "$(pwd)":/work -w /work example-wind-analysis:latest bash -lc "\
      python scripts/analysis/integrate_feature_importance_report.py \
        --analysis-dir \"$(ls -dt artifacts_root/analysis/feature_importance_* | head -1)\" \
        --report-file \"$(ls -dt artifacts_root/analysis/feature_importance_* | head -1)/report.md\" \
        --section-title 'Feature Importance — reference buoy FINETUNED model on SAR TEST set'"

    echo ">>> BUOY FINETUNED: analysis on BUOY TEST at $(date) <<<"
    docker run --rm \
      -v "$(pwd)":/work \
      -w /work \
      example-wind-analysis:latest bash -lc "\
        python scripts/analysis/feature_importance.py \
          --model-artifact artifacts_root/analysis/models/buoy_finetuned_model.tar.gz \
          --data-path artifacts_root/analysis/data/buoy_test \
          --output-dir artifacts_root/analysis \
          --grouping both \
          --top-k 30" \
      | tee -a artifacts_root/analysis/logs/buoy_finetuned_on_buoy_test_feature_importance.log

    docker run --rm -v "$(pwd)":/work -w /work example-wind-analysis:latest bash -lc "\
      python scripts/analysis/integrate_feature_importance_report.py \
        --analysis-dir \"$(ls -dt artifacts_root/analysis/feature_importance_* | head -1)\" \
        --report-file \"$(ls -dt artifacts_root/analysis/feature_importance_* | head -1)/report.md\" \
        --section-title 'Feature Importance — reference buoy FINETUNED model on reference buoy TEST set'"
  else
    echo ">>> Skipping reference buoy FINETUNED feature-importance analysis: artifacts_root/buoy/config/final_model_finetuned.txt not found <<<"
  fi
else
  echo ">>> Skipping reference buoy FINETUNED feature-importance analysis per configuration <<<"
fi

if [[ "${RUN_BUOY_FEATURE_IMPORTANCE_L2SP}" == "1" ]]; then
  if [ -f artifacts_root/buoy/config/final_model_finetuned_l2sp.txt ]; then
    ensure_analysis_dirs
    build_analysis_image
    echo ">>> BUOY L2SP: downloading model artifact at $(date) <<<"
    aws s3 cp \
      "s3://project-bucket-placeholder/wind_training/models/reference_buoy/finetuned_l2sp/${BUOY_L2SP_JOB}/output/model.tar.gz" \
      artifacts_root/analysis/models/buoy_l2sp_model.tar.gz \
      --profile example-profile --region example-region-1
    aws s3 sync \
      s3://project-bucket-placeholder/wind_training/training/sar/test/ \
      artifacts_root/analysis/data/sar_test \
      --profile example-profile --region example-region-1
    aws s3 sync \
      s3://project-bucket-placeholder/wind_training/training/buoy/test/ \
      artifacts_root/analysis/data/buoy_test \
      --profile example-profile --region example-region-1

    echo ">>> BUOY L2SP: analysis on SAR TEST at $(date) <<<"
    docker run --rm \
      -v "$(pwd)":/work \
      -w /work \
      example-wind-analysis:latest bash -lc "\
        python scripts/analysis/feature_importance.py \
          --model-artifact artifacts_root/analysis/models/buoy_l2sp_model.tar.gz \
          --data-path artifacts_root/analysis/data/sar_test \
          --output-dir artifacts_root/analysis \
          --grouping both \
          --top-k 30" \
      | tee -a artifacts_root/analysis/logs/buoy_l2sp_on_sar_test_feature_importance.log

    docker run --rm -v "$(pwd)":/work -w /work example-wind-analysis:latest bash -lc "\
      python scripts/analysis/integrate_feature_importance_report.py \
        --analysis-dir \"$(ls -dt artifacts_root/analysis/feature_importance_* | head -1)\" \
        --report-file \"$(ls -dt artifacts_root/analysis/feature_importance_* | head -1)/report.md\" \
        --section-title 'Feature Importance — reference buoy L2SP model on SAR TEST set'"

    echo ">>> BUOY L2SP: analysis on BUOY TEST at $(date) <<<"
    docker run --rm \
      -v "$(pwd)":/work \
      -w /work \
      example-wind-analysis:latest bash -lc "\
        python scripts/analysis/feature_importance.py \
          --model-artifact artifacts_root/analysis/models/buoy_l2sp_model.tar.gz \
          --data-path artifacts_root/analysis/data/buoy_test \
          --output-dir artifacts_root/analysis \
          --grouping both \
          --top-k 30" \
      | tee -a artifacts_root/analysis/logs/buoy_l2sp_on_buoy_test_feature_importance.log

    docker run --rm -v "$(pwd)":/work -w /work example-wind-analysis:latest bash -lc "\
      python scripts/analysis/integrate_feature_importance_report.py \
        --analysis-dir \"$(ls -dt artifacts_root/analysis/feature_importance_* | head -1)\" \
        --report-file \"$(ls -dt artifacts_root/analysis/feature_importance_* | head -1)/report.md\" \
        --section-title 'Feature Importance — reference buoy L2SP model on reference buoy TEST set'"
  else
    echo ">>> Skipping reference buoy L2SP feature-importance analysis: artifacts_root/buoy/config/final_model_finetuned_l2sp.txt not found <<<"
  fi
else
  echo ">>> Skipping reference buoy L2SP feature-importance analysis per configuration <<<"
fi

if [[ "${RUN_BUOY_FEATURE_IMPORTANCE_L2SP_KD}" == "1" ]]; then
  if [ -f artifacts_root/buoy/config/final_model_finetuned_l2sp_kd.txt ]; then
    ensure_analysis_dirs
    build_analysis_image
    echo ">>> BUOY L2SP+KD: downloading model artifact at $(date) <<<"
    aws s3 cp \
      "s3://project-bucket-placeholder/wind_training/models/reference_buoy/finetuned_l2sp_kd/${BUOY_L2SP_KD_JOB}/output/model.tar.gz" \
      artifacts_root/analysis/models/buoy_l2sp_kd_model.tar.gz \
      --profile example-profile --region example-region-1
    aws s3 sync \
      s3://project-bucket-placeholder/wind_training/training/sar/test/ \
      artifacts_root/analysis/data/sar_test \
      --profile example-profile --region example-region-1
    aws s3 sync \
      s3://project-bucket-placeholder/wind_training/training/buoy/test/ \
      artifacts_root/analysis/data/buoy_test \
      --profile example-profile --region example-region-1

    echo ">>> BUOY L2SP+KD: analysis on SAR TEST at $(date) <<<"
    docker run --rm \
      -v "$(pwd)":/work \
      -w /work \
      example-wind-analysis:latest bash -lc "\
        python scripts/analysis/feature_importance.py \
          --model-artifact artifacts_root/analysis/models/buoy_l2sp_kd_model.tar.gz \
          --data-path artifacts_root/analysis/data/sar_test \
          --output-dir artifacts_root/analysis \
          --grouping both \
          --top-k 30" \
      | tee -a artifacts_root/analysis/logs/buoy_l2sp_kd_on_sar_test_feature_importance.log

    docker run --rm -v "$(pwd)":/work -w /work example-wind-analysis:latest bash -lc "\
      python scripts/analysis/integrate_feature_importance_report.py \
        --analysis-dir \"$(ls -dt artifacts_root/analysis/feature_importance_* | head -1)\" \
        --report-file \"$(ls -dt artifacts_root/analysis/feature_importance_* | head -1)/report.md\" \
        --section-title 'Feature Importance — reference buoy L2SP+KD model on SAR TEST set'"

    echo ">>> BUOY L2SP+KD: analysis on BUOY TEST at $(date) <<<"
    docker run --rm \
      -v "$(pwd)":/work \
      -w /work \
      example-wind-analysis:latest bash -lc "\
        python scripts/analysis/feature_importance.py \
          --model-artifact artifacts_root/analysis/models/buoy_l2sp_kd_model.tar.gz \
          --data-path artifacts_root/analysis/data/buoy_test \
          --output-dir artifacts_root/analysis \
          --grouping both \
          --top-k 30" \
      | tee -a artifacts_root/analysis/logs/buoy_l2sp_kd_on_buoy_test_feature_importance.log

    docker run --rm -v "$(pwd)":/work -w /work example-wind-analysis:latest bash -lc "\
      python scripts/analysis/integrate_feature_importance_report.py \
        --analysis-dir \"$(ls -dt artifacts_root/analysis/feature_importance_* | head -1)\" \
        --report-file \"$(ls -dt artifacts_root/analysis/feature_importance_* | head -1)/report.md\" \
        --section-title 'Feature Importance — reference buoy L2SP+KD model on reference buoy TEST set'"
  else
    echo ">>> Skipping reference buoy L2SP+KD feature-importance analysis: artifacts_root/buoy/config/final_model_finetuned_l2sp_kd.txt not found <<<"
  fi
else
  echo ">>> Skipping reference buoy L2SP+KD feature-importance analysis per configuration <<<"
fi
