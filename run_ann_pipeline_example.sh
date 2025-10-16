#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Author: Example Team <team@example.org>
# Created: 2025-10-16
# Disclaimer: This obfuscated sample mirrors the orchestration structure and must be adapted before any deployment.
# -----------------------------------------------------------------------------

# Exit immediately on error, undefined variable, or pipeline failure
set -euo pipefail

# Example holistic ANN pipeline (obfuscated).
# Replace placeholders such as example-profile, example-region-1, project-bucket-placeholder, and wind_training.* before running.

# Pipeline: partition for buoy training 10km observational range

echo "=== Pipeline started at $(date) ==="

source run_data_preparation_pipeline_example.sh
source run_sar_pipeline_example.sh
source run_reference_buoy_pipeline_example.sh
source run_sar_reference_buoy_pipeline_example.sh
source run_grid_offset_pipeline_example.sh
