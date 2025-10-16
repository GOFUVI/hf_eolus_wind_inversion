#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Author: Example Team <team@example.org>
# Created: 2025-10-16
# Disclaimer: This obfuscated sample mirrors the SAR+reference buoy workflow and must be adapted before any deployment.
# -----------------------------------------------------------------------------

# Exit immediately on error, undefined variable, or pipeline failure
set -euo pipefail

# Example SAR+reference buoy pipeline (obfuscated).
# Replace placeholders such as example-profile, example-region-1, project-bucket-placeholder, and wind_training.* before running.

# Inference block toggles (set to 0 to skip a block)
RUN_SAR_BUOY_FINAL_TEST_INFERENCE=1
RUN_SAR_BUOY_FINAL_PIVOTS_INFERENCE=1
RUN_SAR_BUOY_FEATURE_IMPORTANCE_FINAL=1

# Pipeline: combined SAR + reference buoy wind training

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
  artifacts_root/sar_buoy \
  artifacts_root/sar_buoy/reports/hpo \
  artifacts_root/sar_buoy/inference_metrics \
  artifacts_root/sar_buoy/train_metrics \
  artifacts_root/sar_buoy/normalization_params \
  artifacts_root/sar_buoy/bin_metrics \
  artifacts_root/sar_buoy/config \
  artifacts_root/sar_buoy/final_training \
  artifacts_root/sar_buoy/logs \
  artifacts_root/sar_buoy/logs/hpo \
  artifacts_root/sar_buoy/logs/train_model \
  artifacts_root/sar_buoy/logs/finalize_geoparquet_inference_test \
  artifacts_root/sar_buoy/logs/finalize_geoparquet_inference_pivots

if [ -f artifacts_root/sar_buoy/config/selected_model.txt ]; then
  echo ">>> Selected model file found; skipping SAR+reference buoy HPO <<<"
else
  echo ">>> Starting HPO for sar-buoy-hpo at $(date) <<<"
  ./scripts/HPO/run_hpo.sh \
    --profile example-profile \
    --region example-region-1 \
    --job-name sar-buoy-hpo-job-d-1 \
    --train-data-uri s3://project-bucket-placeholder/wind_training/training/wind_combined/train/ \
    --output-s3-uri s3://project-bucket-placeholder/wind_training/models/sar_buoy \
    --model-config artifacts_root/sar_buoy/config/sar_buoy_model.json \
    --hpo-config artifacts_root/sar_buoy/config/sar_buoy_hpo.json \
    --log-dir artifacts_root/sar_buoy/logs/hpo/sar-buoy-hpo-job-d-1
  echo ">>> Completed HPO for sar-buoy-hpo at $(date) <<<"

  echo ">>> Starting follow-up HPO for sar-buoy-hpo-2 (warm start) at $(date) <<<"
  ./scripts/HPO/run_hpo.sh \
    --profile example-profile \
    --region example-region-1 \
    --job-name sar-buoy-hpo-job-d-2 \
    --parent-jobs sar-buoy-hpo-job-d-1 \
    --train-data-uri s3://project-bucket-placeholder/wind_training/training/wind_combined/train/ \
    --output-s3-uri s3://project-bucket-placeholder/wind_training/models/sar_buoy \
    --model-config artifacts_root/sar_buoy/config/sar_buoy_model.json \
    --hpo-config artifacts_root/sar_buoy/config/sar_buoy_hpo.json \
    --log-dir artifacts_root/sar_buoy/logs/hpo/sar-buoy-hpo-job-d-2
  echo ">>> Completed HPO for sar-buoy-hpo-2 at $(date) <<<"

  echo ">>> Integrating SAR+reference buoy HPO reports at $(date) <<<"
  ./scripts/HPO/hpo_metrics_report.sh -p example-profile -r example-region-1 -n sar-buoy-hpo-job-d-1 -o artifacts_root/sar_buoy/reports/hpo/sar-buoy-hpo-job-d-1_hpo_report.md
  ./scripts/HPO/hpo_metrics_report.sh -p example-profile -r example-region-1 -n sar-buoy-hpo-job-d-2 -o artifacts_root/sar_buoy/reports/hpo/sar-buoy-hpo-job-d-2_hpo_report.md
  ./scripts/HPO/integrate_hpo_reports.sh -i "artifacts_root/sar_buoy/reports/hpo/sar-buoy-hpo-job-d-2_hpo_report.md,artifacts_root/sar_buoy/reports/hpo/sar-buoy-hpo-job-d-1_hpo_report.md" -o artifacts_root/sar_buoy/reports/hpo/sar-buoy-hpo_all_hpo_report.md
  echo ">>> Completed integrating SAR+reference buoy HPO reports at $(date) <<<"

  ./scripts/HPO/select_best_hpo_job.sh --report artifacts_root/sar_buoy/reports/hpo/sar-buoy-hpo_all_hpo_report.md --output artifacts_root/sar_buoy/config/selected_model.txt

  echo ">>> Generating SAR+reference buoy model config from HPO at $(date) <<<"
  ./scripts/training/generate_model_config_from_hpo.sh \
    --train-job "$(awk '{sub(/#.*/,"");gsub(/^[ \t]+|[ \t]+$/,"");if(length($0)){print;exit}}' artifacts_root/sar_buoy/config/selected_model.txt)" \
    --profile example-profile \
    --region example-region-1 \
    --output artifacts_root/sar_buoy/config/sar_buoy_hpo_final_model.json
  echo ">>> Completed SAR+reference buoy model config generation at $(date) <<<"
fi

if [ -f artifacts_root/sar_buoy/config/final_model.txt ]; then
  echo ">>> Final model file found; skipping SAR+reference buoy training stage <<<"
else
  

  echo ">>> Starting SAR+reference buoy full-training run at $(date) <<<"
  ./scripts/training/train_model.sh \
    --s3-prefix s3://project-bucket-placeholder/wind_training/models/sar_buoy/final \
    --profile example-profile \
    --region example-region-1 \
    --train-data-uri s3://project-bucket-placeholder/wind_training/training/wind_combined/train/ \
    --job-base-prefix sar-buoy-range-example \
    --model-config artifacts_root/sar_buoy/config/sar_buoy_hpo_final_model.json \
    --max-runtime 18000 \
    --output-dir artifacts_root/sar_buoy/final_training \
    --no-cv \
    --seed 42
  echo ">>> Completed SAR+reference buoy full-training run at $(date) <<<"

  ./scripts/training/record_final_job.sh --log artifacts_root/sar_buoy/final_training/train_model.log --output artifacts_root/sar_buoy/config/final_model.txt
  echo ">>> Stored SAR+reference buoy final training job $(sed -n '1p' artifacts_root/sar_buoy/config/final_model.txt) in artifacts_root/sar_buoy/config/final_model.txt <<<"
fi

if [ ! -f artifacts_root/sar_buoy/config/final_model.txt ]; then
  echo "Error: artifacts_root/sar_buoy/config/final_model.txt not found. Remove this file to retrain or ensure training completed successfully." >&2
  exit 1
fi

echo ">>> Using SAR+reference buoy final training job $(awk '{sub(/#.*/,"");gsub(/^[ \t]+|[ \t]+$/,"");if(length($0)){print;exit}}' artifacts_root/sar_buoy/config/final_model.txt) for downstream steps <<<"

echo ">>> Fetching training diagnostics at $(date) <<<"
./scripts/training/get_train_metrics.sh \
  --aws-profile example-profile \
  --train-job-name "$(awk '{sub(/#.*/,"");gsub(/^[ \t]+|[ \t]+$/,"");if(length($0)){print;exit}}' artifacts_root/sar_buoy/config/final_model.txt)" \
  --output-directory artifacts_root/sar_buoy/train_metrics
./scripts/training/get_norm_params.sh \
  --aws-profile example-profile \
  --train-job-name "$(awk '{sub(/#.*/,"");gsub(/^[ \t]+|[ \t]+$/,"");if(length($0)){print;exit}}' artifacts_root/sar_buoy/config/final_model.txt)" \
  --output-directory artifacts_root/sar_buoy/normalization_params
./scripts/training/get_bin_metrics.sh \
  --aws-profile example-profile \
  --train-job-name "$(awk '{sub(/#.*/,"");gsub(/^[ \t]+|[ \t]+$/,"");if(length($0)){print;exit}}' artifacts_root/sar_buoy/config/final_model.txt)" \
  --output-directory artifacts_root/sar_buoy/bin_metrics

if [[ "${RUN_SAR_BUOY_FINAL_TEST_INFERENCE}" == "1" ]]; then
  echo ">>> Running SAR+reference buoy inference on combined test set at $(date) <<<"
  ./scripts/inference/run_inference.sh \
    --model-artifact "s3://project-bucket-placeholder/wind_training/models/sar_buoy/final/$(awk '{sub(/#.*/,"");gsub(/^[ \t]+|[ \t]+$/,"");if(length($0)){print;exit}}' artifacts_root/sar_buoy/config/final_model.txt)/output/model.tar.gz" \
    --input-data s3://project-bucket-placeholder/wind_training/training/wind_combined/test/ \
    --output-s3-uri "s3://project-bucket-placeholder/wind_training/inference/sar-buoy-range-example/test_set" \
    --profile example-profile \
    --region example-region-1 \
    --output-format parquet
  echo ">>> Completed SAR+reference buoy inference on combined test set at $(date) <<<"
  echo ">>> Finalizing GeoParquet for SAR+reference buoy inference (test set) at $(date) <<<"
  ./scripts/geo_utils/finalize_geoparquet.sh \
    --db-name wind_training \
    --bucket-name project-bucket-placeholder \
    --output-prefix wind_training/inference/sar-buoy-range-example/test_set \
    --output-table SAR_BUOY_RANGE_FINAL_TEST_INFERENCE \
    --register-table \
    --profile example-profile \
    --region example-region-1 \
    --log-dir artifacts_root/sar_buoy/logs/finalize_geoparquet_inference_test
  echo ">>> Completed GeoParquet finalization for SAR+reference buoy inference (test set) at $(date) <<<"


  echo ">>> Computing SAR+reference buoy inference metrics at $(date) <<<"
  mkdir -p "artifacts_root/sar_buoy/inference_metrics/$(awk '{sub(/#.*/,"");gsub(/^[ \t]+|[ \t]+$/,"");if(length($0)){print;exit}}' artifacts_root/sar_buoy/config/final_model.txt)"
  ./scripts/inference/compute_inference_metrics.sh \
    --predictions-s3 s3://project-bucket-placeholder/wind_training/inference/sar-buoy-range-example/test_set/data.parquet \
    --metadata-s3 s3://project-bucket-placeholder/wind_training/inference/sar-buoy-range-example/test_set/inference_metadata.json \
    --output-dir "artifacts_root/sar_buoy/inference_metrics/$(awk '{sub(/#.*/,"");gsub(/^[ \t]+|[ \t]+$/,"");if(length($0)){print;exit}}' artifacts_root/sar_buoy/config/final_model.txt)" \
    --profile example-profile \
    --region example-region-1 \
    --wind-bin-column wind_bin \
    --truth-speed-col wind_speed \
    --truth-dir-col wind_direction \
    --group-column wind_source
  echo ">>> Completed SAR+reference buoy inference metrics at $(date) <<<"

  echo ">>> Building STAC catalog for SAR+reference buoy inference (test set) at $(date) <<<"
  ./scripts/geo_utils/build_stac_catalog.sh \
    --collection SAR_BUOY_RANGE_FINAL_TEST_INFERENCE \
    --s3-uri s3://project-bucket-placeholder/wind_training/inference/sar-buoy-range-example/test_set/ \
    --stac-collection-properties-json artifacts_root/sar_buoy/stac_config/stac_properties_collection_SAR_BUOY_RANGE_FINAL_TEST_SET.json \
    --stac-item-properties-json artifacts_root/sar_buoy/stac_config/stac_properties_item_SAR_BUOY_RANGE_FINAL_TEST_SET.json \
    --output-dir catalogs/sar_buoy_pipeline/sar_buoy_range_example_test_set \
    --profile example-profile \
    --region example-region-1
  echo ">>> Completed STAC catalog build for SAR+reference buoy inference (test set) at $(date) <<<"
else
  echo ">>> Skipping SAR+reference buoy final test-set inference per configuration <<<"
fi

if [[ "${RUN_SAR_BUOY_FEATURE_IMPORTANCE_FINAL}" == "1" ]]; then
  if [ -f artifacts_root/sar_buoy/config/final_model.txt ]; then
    ensure_analysis_dirs
    build_analysis_image
    echo ">>> SAR+BUOY FINAL: downloading model artifact and combined test data at $(date) <<<"
    aws s3 cp \
      "s3://project-bucket-placeholder/wind_training/models/sar_buoy/final/$(sed -n '1p' artifacts_root/sar_buoy/config/final_model.txt)/output/model.tar.gz" \
      artifacts_root/analysis/models/sar_buoy_final_model.tar.gz \
      --profile example-profile --region example-region-1
    aws s3 sync \
      s3://project-bucket-placeholder/wind_training/training/wind_combined/test/ \
      artifacts_root/analysis/data/wind_combined_test \
      --profile example-profile --region example-region-1

    echo ">>> SAR+BUOY FINAL: running feature-importance analysis at $(date) <<<"
    docker run --rm \
      -v "$(pwd)":/work \
      -w /work \
      example-wind-analysis:latest bash -lc "\
        python scripts/analysis/feature_importance.py \
          --model-artifact artifacts_root/analysis/models/sar_buoy_final_model.tar.gz \
          --data-path artifacts_root/analysis/data/wind_combined_test \
          --output-dir artifacts_root/analysis \
          --grouping both \
          --top-k 30" \
      | tee -a artifacts_root/analysis/logs/sar_buoy_final_combined_feature_importance.log

    docker run --rm -v "$(pwd)":/work -w /work example-wind-analysis:latest bash -lc "\
      python scripts/analysis/integrate_feature_importance_report.py \
        --analysis-dir \"$(ls -dt artifacts_root/analysis/feature_importance_* | head -1)\" \
        --report-file \"$(ls -dt artifacts_root/analysis/feature_importance_* | head -1)/report.md\" \
        --section-title 'Feature Importance — SAR+reference buoy FINAL model on combined TEST set'"
  else
    echo ">>> Skipping SAR+BUOY FINAL feature-importance analysis: artifacts_root/sar_buoy/config/final_model.txt not found <<<"
  fi
else
  echo ">>> Skipping SAR+BUOY FINAL feature-importance analysis per configuration <<<"
fi

if [[ "${RUN_SAR_BUOY_FINAL_PIVOTS_INFERENCE}" == "1" ]]; then
  echo ">>> Running SAR+reference buoy inference on PIVOTS_JOINED at $(date) <<<"
  ./scripts/inference/run_inference.sh \
    --model-artifact "s3://project-bucket-placeholder/wind_training/models/sar_buoy/final/$(awk '{sub(/#.*/,"");gsub(/^[ \t]+|[ \t]+$/,"");if(length($0)){print;exit}}' artifacts_root/sar_buoy/config/final_model.txt)/output/model.tar.gz" \
    --input-data s3://project-bucket-placeholder/wind_training/pivots/joined/ \
    --output-s3-uri "s3://project-bucket-placeholder/wind_training/inference/sar-buoy-range-example/pivots_joined" \
    --profile example-profile \
    --region example-region-1 \
    --output-format parquet
  echo ">>> Completed SAR+reference buoy inference on PIVOTS_JOINED at $(date) <<<"

  echo ">>> Finalizing GeoParquet for SAR+reference buoy inference (PIVOTS_JOINED) at $(date) <<<"
  ./scripts/geo_utils/finalize_geoparquet.sh \
    --db-name wind_training \
    --bucket-name project-bucket-placeholder \
    --output-prefix wind_training/inference/sar-buoy-range-example/pivots_joined \
    --output-table SAR_BUOY_RANGE_FINAL_PIVOTS_JOINED \
    --register-table \
    --profile example-profile \
    --region example-region-1 \
    --log-dir artifacts_root/sar_buoy/logs/finalize_geoparquet_inference_pivots
  echo ">>> Completed GeoParquet finalization for SAR+reference buoy inference (PIVOTS_JOINED) at $(date) <<<"

  echo ">>> Building STAC catalog for SAR+reference buoy inference (PIVOTS_JOINED) at $(date) <<<"
  ./scripts/geo_utils/build_stac_catalog.sh \
    --collection SAR_BUOY_RANGE_FINAL_PIVOTS_JOINED \
    --s3-uri s3://project-bucket-placeholder/wind_training/inference/sar-buoy-range-example/pivots_joined/ \
    --stac-collection-properties-json artifacts_root/sar_buoy/stac_config/stac_properties_collection_SAR_BUOY_RANGE_FINAL_PIVOTS_JOINED.json \
    --stac-item-properties-json artifacts_root/sar_buoy/stac_config/stac_properties_item_SAR_BUOY_RANGE_FINAL_PIVOTS_JOINED.json \
    --output-dir catalogs/sar_buoy_pipeline/sar_buoy_range_example_pivots_joined \
    --profile example-profile \
    --region example-region-1
  echo ">>> Completed STAC catalog build for SAR+reference buoy inference (PIVOTS_JOINED) at $(date) <<<"
else
  echo ">>> Skipping SAR+reference buoy PIVOTS_JOINED inference per configuration <<<"
fi

echo "=== Pipeline completed at $(date) ==="
