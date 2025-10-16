#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Author: Juan Luis Herrera Cortijo <juan.luis.herrera.cortijo@gmail.com>
# Created: 2025-10-16
# Disclaimer: This script is provided as-is for research workflows and must be validated before any operational deployment.
# -----------------------------------------------------------------------------

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: get_norm_params.sh --train-job-name JOB_NAME --aws-profile PROFILE [--output-directory PATH]
Fetch normalization_params.json from the SageMaker job output or model artifact tarball.

  --train-job-name JOB_NAME    Name (or unique prefix) of the SageMaker training job
  --aws-profile PROFILE        AWS CLI profile to use
  --output-directory PATH      Local directory to save normalization parameters (default: current directory)
  --help                       Show this help message
USAGE
  exit 1
}

TRAIN_JOB=""
PROFILE=""
OUTPUT_DIR="."

while [[ $# -gt 0 ]]; do
  case "$1" in
    --train-job-name)
      [[ $# -ge 2 ]] || { echo "Error: --train-job-name requires a value." >&2; usage; }
      TRAIN_JOB="$2"
      shift 2
      ;;
    --train-job-name=*)
      TRAIN_JOB="${1#*=}"
      shift
      ;;
    --aws-profile)
      [[ $# -ge 2 ]] || { echo "Error: --aws-profile requires a value." >&2; usage; }
      PROFILE="$2"
      shift 2
      ;;
    --aws-profile=*)
      PROFILE="${1#*=}"
      shift
      ;;
    --output-directory)
      [[ $# -ge 2 ]] || { echo "Error: --output-directory requires a value." >&2; usage; }
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --output-directory=*)
      OUTPUT_DIR="${1#*=}"
      shift
      ;;
    --help)
      usage
      ;;
    *)
      echo "Error: Unknown option '$1'." >&2
      usage
      ;;
  esac
done

if [[ -z "$TRAIN_JOB" || -z "$PROFILE" ]]; then
  echo "Error: --train-job-name and --aws-profile are required." >&2
  usage
fi

mkdir -p "$OUTPUT_DIR"
NORM_DIR="${OUTPUT_DIR%/}/normalization_params/${TRAIN_JOB}"
mkdir -p "$NORM_DIR"

echo "Describing SageMaker training job '$TRAIN_JOB' (profile '$PROFILE')..."
if ! S3_OUTPUT=$(aws sagemaker describe-training-job \
      --training-job-name "$TRAIN_JOB" \
      --profile "$PROFILE" \
      --output text \
      --query 'OutputDataConfig.S3OutputPath' 2>/dev/null); then
  echo "Training job '$TRAIN_JOB' not found. Searching for completed jobs matching prefix '$TRAIN_JOB'..."
  max_attempts=3
  attempt=1
  JOBS=""
  while true; do
    if list_output=$(aws sagemaker list-training-jobs \
      --profile "$PROFILE" \
      --query "TrainingJobSummaries[?contains(TrainingJobName, '${TRAIN_JOB}') && TrainingJobStatus=='Completed'].TrainingJobName" \
      --output text 2>&1); then
      JOBS="$list_output"
      break
    fi
    err="$list_output"
    if echo "$err" | grep -q 'ThrottlingException'; then
      if [[ $attempt -lt $max_attempts ]]; then
        delay=$((attempt * 2))
        echo "AWS rate limit hit. Retrying in ${delay}s..." >&2
        sleep "$delay"
        attempt=$((attempt + 1))
        continue
      else
        echo "Rate limit exceeded after $max_attempts attempts. Please provide the full job name." >&2
        exit 2
      fi
    else
      echo "Error listing training jobs: $err" >&2
      exit 2
    fi
  done

  if [[ -z "$JOBS" ]]; then
    echo "No completed training jobs found with prefix '$TRAIN_JOB'" >&2
    exit 2
  fi

  if [[ $(echo "$JOBS" | wc -w) -gt 1 ]]; then
    echo "Multiple jobs match the prefix. Please specify one of: $JOBS" >&2
    exit 2
  fi

  TRAIN_JOB="$JOBS"
  echo "Using job name '$TRAIN_JOB'"
  S3_OUTPUT=$(aws sagemaker describe-training-job \
      --training-job-name "$TRAIN_JOB" \
      --profile "$PROFILE" \
      --output text \
      --query 'OutputDataConfig.S3OutputPath')
fi

echo "S3 output path: $S3_OUTPUT"

bucket=$(echo "$S3_OUTPUT" | awk -F/ '{print $3}')
prefix_path=$(echo "$S3_OUTPUT" | sed -e 's|s3://[^/]*/||')
tarball_key="${prefix_path%/}/${TRAIN_JOB}/output/output.tar.gz"

echo "Checking for tarball at s3://$bucket/$tarball_key"
if ! aws s3 ls "s3://$bucket/$tarball_key" --profile "$PROFILE" > /dev/null; then
  echo "Error: output.tar.gz not found at s3://$bucket/$tarball_key" >&2
  exit 3
fi

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

echo "Downloading tarball to $TMPDIR/output.tar.gz"
aws s3 cp "s3://$bucket/$tarball_key" "$TMPDIR/output.tar.gz" --profile "$PROFILE"

echo "Extracting tarball contents..."
tar -xzf "$TMPDIR/output.tar.gz" -C "$TMPDIR"

echo "Searching for normalization_params.json in output archive..."
FILES=$(find "$TMPDIR" -type f -name 'normalization_params.json' || true)

if [[ -z "$FILES" ]]; then
  echo "Not found in output archive; attempting to fetch from model artifacts..."
  MODEL_URI=$(aws sagemaker describe-training-job \
    --training-job-name "$TRAIN_JOB" \
    --profile "$PROFILE" \
    --output text \
    --query 'ModelArtifacts.S3ModelArtifacts')
  if [[ -z "$MODEL_URI" ]]; then
    echo "Error: ModelArtifacts.S3ModelArtifacts not available for job '$TRAIN_JOB'" >&2
    exit 4
  fi
  echo "Model artifact URI: $MODEL_URI"
  echo "Downloading model artifacts to $TMPDIR/model.tar.gz"
  aws s3 cp "$MODEL_URI" "$TMPDIR/model.tar.gz" --profile "$PROFILE"
  echo "Extracting model artifacts..."
  tar -xzf "$TMPDIR/model.tar.gz" -C "$TMPDIR"
  FILES=$(find "$TMPDIR" -type f -name 'normalization_params.json' || true)
  if [[ -z "$FILES" ]]; then
    echo "Error: normalization_params.json not found in output or model artifacts" >&2
    exit 5
  fi
fi

for FILE in $FILES; do
  echo "Moving $FILE -> $NORM_DIR/normalization_params.json"
  mv "$FILE" "$NORM_DIR/normalization_params.json"
done

TARGET_FILE="$NORM_DIR/normalization_params.json"
REPORT_FILE="$NORM_DIR/normalization_params_report.md"
if [[ -f "$TARGET_FILE" ]]; then
  TARGET_FILE="$TARGET_FILE" REPORT_FILE="$REPORT_FILE" TRAIN_JOB="$TRAIN_JOB" python3 <<'PYCODE'
import json
import os
from pathlib import Path

target = Path(os.environ['TARGET_FILE'])
report_path = Path(os.environ['REPORT_FILE'])
train_job = os.environ.get('TRAIN_JOB', 'unknown-job')

data = {}
try:
    with target.open('r', encoding='utf-8') as handle:
        data = json.load(handle)
except FileNotFoundError:
    data = {}
except json.JSONDecodeError as exc:
    report_path.write_text(f"Failed to parse {target.name}: {exc}\n", encoding='utf-8')
else:
    def fmt(value):
        if isinstance(value, (int, float)):
            return f"{value:.6f}"
        if value is None:
            return ''
        return str(value)

    lines = [f"# Normalization Parameters for training job {train_job}", '', f"Source file: {target.name}", '']
    handled_keys = set()

    metadata = []
    mode = data.get('mode')
    if mode is not None:
        handled_keys.add('mode')
        metadata.append(f"- Mode: {mode}")

    center_label = data.get('center_label')
    if center_label:
        handled_keys.add('center_label')
        metadata.append(f"- Center label: {center_label}")

    scale_label = data.get('scale_label')
    if scale_label:
        handled_keys.add('scale_label')
        metadata.append(f"- Scale label: {scale_label}")

    if metadata:
        lines.append('## Metadata')
        lines.extend(metadata)
        lines.append('')

    centers = data.get('centers')
    scales = data.get('scales')

    if isinstance(centers, dict) or isinstance(scales, dict):
        handled_keys.update({'centers', 'scales'})
        centers = centers or {}
        scales = scales or {}
        features = sorted(set(centers) | set(scales))
        if features:
            center_header = center_label or 'center'
            scale_header = scale_label or 'scale'
            lines.append(f"| feature | {center_header} | {scale_header} |")
            lines.append('| --- | --- | --- |')
            for feature in features:
                lines.append('| ' + ' | '.join([
                    feature,
                    fmt(centers.get(feature)),
                    fmt(scales.get(feature))
                ]) + ' |')
        else:
            lines.append('No center/scale entries found in normalization parameters.')
    else:
        lines.append('No recognized normalization entries found (expected centers/scales).')

    diagnostics = data.get('diagnostics')
    if isinstance(diagnostics, dict):
        handled_keys.add('diagnostics')
        lines.append('')
        lines.append('## Diagnostics')
        for scope in sorted(diagnostics):
            scope_stats = diagnostics.get(scope)
            if isinstance(scope_stats, dict):
                keys = ', '.join(sorted(k for k in scope_stats if isinstance(scope_stats[k], dict)))
                descriptor = keys or 'no statistics available'
                lines.append(f"- {scope}: {descriptor}")
            else:
                lines.append(f"- {scope}: {json.dumps(scope_stats, ensure_ascii=False)}")

    extra_keys = [key for key in data.keys() if key not in handled_keys]
    if extra_keys:
        lines.append('')
        lines.append('## Additional keys')
        for key in sorted(extra_keys):
            lines.append(f"- {key}: {json.dumps(data[key], ensure_ascii=False)}")

    report_path.write_text('\n'.join(lines).strip() + '\n', encoding='utf-8')
PYCODE
fi

echo "Done. Normalization parameters saved in '$NORM_DIR'"
