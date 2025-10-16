#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Author: Juan Luis Herrera Cortijo <juan.luis.herrera.cortijo@gmail.com>
# Created: 2025-10-16
# Disclaimer: This script is provided as-is for research workflows and must be validated before any operational deployment.
# -----------------------------------------------------------------------------

# Compute inference diagnostics inside a Dockerized environment.
#
# This helper mirrors Stage 10 of ``docs/inference.md``: it retrieves freshly
# generated predictions, (re)builds the analytics container, and executes the
# Python post-processing routine so regression/classification metrics, group
# breakdowns, and Markdown reports can be generated locally with bitwise
# reproducibility. Heavy dependencies stay encapsulated inside the container,
# keeping workstation environments clean.
set -euo pipefail

usage() {
  cat <<'USAGE' >&2
Usage: compute_inference_metrics.sh \
  --predictions-s3 S3_URI \
  [--metadata-s3 S3_URI] \
  --output-dir PATH \
  --profile PROFILE \
  --region REGION \
  [--truth-speed-col NAME] \
  [--truth-dir-col NAME] \
  [--group-column COLUMN] \
  [--wind-bin-column COLUMN]

Downloads inference outputs from S3, builds the metrics Docker image if needed,
and runs the metrics container to generate regression/classification reports.
USAGE
  exit 1
}

PREDICTIONS_S3=""
METADATA_S3=""
OUTPUT_DIR=""
PROFILE=""
REGION=""
TRUTH_SPEED_COL=""
TRUTH_DIR_COL=""
GROUP_COLUMNS=()
WIND_BIN_COLUMN="wind_bin"

trim_whitespace() {
  local value="$1"
  # Use awk to trim leading/trailing whitespace while preserving inner spacing.
  value=$(printf '%s' "$value" | awk '{$1=$1;print}')
  printf '%s' "$value"
}

# Parse CLI options, supporting both ``--flag value`` and ``--flag=value`` forms
# so that the script behaves well inside runbooks and manual workflows.
while [[ $# -gt 0 ]]; do
  case "$1" in
    --predictions-s3)
      PREDICTIONS_S3="$2"; shift 2 ;;
    --predictions-s3=*)
      PREDICTIONS_S3="${1#*=}"; shift ;;
    --metadata-s3)
      METADATA_S3="$2"; shift 2 ;;
    --metadata-s3=*)
      METADATA_S3="${1#*=}"; shift ;;
    --output-dir)
      OUTPUT_DIR="$2"; shift 2 ;;
    --output-dir=*)
      OUTPUT_DIR="${1#*=}"; shift ;;
    --profile)
      PROFILE="$2"; shift 2 ;;
    --profile=*)
      PROFILE="${1#*=}"; shift ;;
    --region)
      REGION="$2"; shift 2 ;;
    --region=*)
      REGION="${1#*=}"; shift ;;
    --truth-speed-col)
      TRUTH_SPEED_COL="$2"; shift 2 ;;
    --truth-speed-col=*)
      TRUTH_SPEED_COL="${1#*=}"; shift ;;
    --truth-dir-col)
      TRUTH_DIR_COL="$2"; shift 2 ;;
    --truth-dir-col=*)
      TRUTH_DIR_COL="${1#*=}"; shift ;;
    --group-column)
      value="$2"
      shift 2
      IFS=',' read -ra parts <<< "$value"
      for part in "${parts[@]}"; do
        trimmed=$(trim_whitespace "$part")
        if [[ -n "$trimmed" ]]; then
          GROUP_COLUMNS+=("$trimmed")
        fi
      done
      ;;
    --group-column=*)
      value="${1#*=}"
      shift
      IFS=',' read -ra parts <<< "$value"
      for part in "${parts[@]}"; do
        trimmed=$(trim_whitespace "$part")
        if [[ -n "$trimmed" ]]; then
          GROUP_COLUMNS+=("$trimmed")
        fi
      done
      ;;
    --wind-bin-column)
      WIND_BIN_COLUMN="$2"; shift 2 ;;
    --wind-bin-column=*)
      WIND_BIN_COLUMN="${1#*=}"; shift ;;
    --help)
      usage ;;
    *)
      echo "Error: unknown option '$1'" >&2
      usage ;;
esac
done

if [[ -z "$PREDICTIONS_S3" || -z "$OUTPUT_DIR" || -z "$PROFILE" || -z "$REGION" ]]; then
  usage
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TMP_DIR=$(mktemp -d)
if [[ -z "${KEEP_TMP_DIR:-}" ]]; then
  trap 'rm -rf "$TMP_DIR"' EXIT
else
  echo "Keeping temporary directory: $TMP_DIR"
fi

# Stage 1 — copy remote artefacts locally so the container can mount them as bind volumes.
aws --profile "$PROFILE" --region "$REGION" \
  s3 cp "$PREDICTIONS_S3" "$TMP_DIR/predictions.parquet"

if [[ -n "$METADATA_S3" ]]; then
  # Metadata is optional; missing files are tolerated so the workflow still emits regression metrics.
  if ! aws --profile "$PROFILE" --region "$REGION" s3 cp "$METADATA_S3" "$TMP_DIR/inference_metadata.json"; then
    rm -f "$TMP_DIR/inference_metadata.json"
  fi
fi

ABS_OUTPUT_DIR="$OUTPUT_DIR"
if [[ "$ABS_OUTPUT_DIR" != /* ]]; then
  ABS_OUTPUT_DIR="${PROJECT_ROOT}/${OUTPUT_DIR}"
fi
mkdir -p "$ABS_OUTPUT_DIR"

# Stage 2 — build the analysis image on demand to keep dependencies encapsulated and reproducible
# even when multiple projects share the same workstation.
docker build -t hf_wind_metrics -f "$PROJECT_ROOT/scripts/inference/Dockerfile.metrics" "$PROJECT_ROOT"

# Configure the container command, mirroring the CLI of compute_inference_metrics.py.
CMD=("docker" "run" "--rm" "-v" "${TMP_DIR}:/data" "-v" "${ABS_OUTPUT_DIR}:/output" "hf_wind_metrics" "--predictions" "/data/predictions.parquet" "--output-dir" "/output")

if [[ -f "$TMP_DIR/inference_metadata.json" ]]; then
  CMD+=("--metadata" "/data/inference_metadata.json")
fi

if [[ -n "$TRUTH_SPEED_COL" ]]; then
  CMD+=("--truth-speed-col" "$TRUTH_SPEED_COL")
fi

if [[ -n "$TRUTH_DIR_COL" ]]; then
  CMD+=("--truth-dir-col" "$TRUTH_DIR_COL")
fi

if [[ -n "$WIND_BIN_COLUMN" ]]; then
  CMD+=("--wind-bin-column" "$WIND_BIN_COLUMN")
fi

if [[ ${#GROUP_COLUMNS[@]} -gt 0 ]]; then
  # Forward each requested dimension individually so the Python helper can compute dedicated
  # breakdowns without dealing with CSV splitting logic.
  for column in "${GROUP_COLUMNS[@]}"; do
    [[ -n "$column" ]] || continue
    CMD+=("--group-column" "$column")
  done
fi

# Stage 3 — execute the containerised post-processing step.
"${CMD[@]}"

echo "Inference metrics written to ${ABS_OUTPUT_DIR}"
