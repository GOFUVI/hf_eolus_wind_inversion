#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Author: Juan Luis Herrera Cortijo <juan.luis.herrera.cortijo@gmail.com>
# Created: 2025-10-16
# Disclaimer: This script is provided as-is for research workflows and must be validated before any operational deployment.
# -----------------------------------------------------------------------------

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: get_bin_metrics.sh --train-job-name JOB_NAME --aws-profile PROFILE [--output-directory PATH]
Fetch per-wind-bin metrics CSV files from the SageMaker job's output archive.

  --train-job-name JOB_NAME    Name (or unique prefix) of the SageMaker training job
  --aws-profile PROFILE        AWS CLI profile to use
  --output-directory PATH      Local directory to save metrics files (default: current directory)
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
BIN_METRICS_DIR="${OUTPUT_DIR%/}/bin_metrics/${TRAIN_JOB}"
mkdir -p "$BIN_METRICS_DIR"

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

echo "Searching for per-bin metrics files..."
FILES=$(find "$TMPDIR" -type f \( \
  -name 'metrics_train_fold*_by_wind_bin.csv' -o \
  -name 'metrics_fold*_by_wind_bin.csv' -o \
  -name 'metrics_train_fold*_range_classification.csv' -o \
  -name 'metrics_fold*_range_classification.csv' -o \
  -name 'metrics_train_fold*_range_classification_by_wind_bin.csv' -o \
  -name 'metrics_fold*_range_classification_by_wind_bin.csv' \
\) || true)

if [[ -z "$FILES" ]]; then
  echo "Error: No per-bin metrics files found in the archive" >&2
  exit 4
fi

MOVED_FILES=()
for FILE in $FILES; do
  BASENAME=$(basename "$FILE")
  echo "Moving $FILE -> $BIN_METRICS_DIR/$BASENAME"
  mv "$FILE" "$BIN_METRICS_DIR/$BASENAME"
  MOVED_FILES+=("$BIN_METRICS_DIR/$BASENAME")
done

if [[ ${#MOVED_FILES[@]} -gt 0 ]]; then
  BIN_METRICS_DIR="$BIN_METRICS_DIR" TRAIN_JOB="$TRAIN_JOB" python3 <<'PYCODE'
import csv
import os
import statistics
from pathlib import Path

output_dir = Path(os.environ['BIN_METRICS_DIR'])
train_job = os.environ.get('TRAIN_JOB', 'unknown-job')
report_path = output_dir / 'bin_metrics_report.md'

METRIC_CANDIDATES = (
    'rmse',
    'mae_speed',
    'rmse_dir',
    'eaam_dir',
    'eam_dir',
    'bias_speed',
    'corr_speed',
    'r2_speed',
    'accuracy',
    'precision',
    'recall',
    'f1',
    'support',
    'predicted',
    'true_positives',
    'macro_precision',
    'macro_recall',
    'macro_f1',
)

def aggregate(files):
    summary = {}
    counts = {}
    per_fold = {}
    per_fold_counts = {}
    present = set()
    for fpath in files:
        if not fpath.is_file():
            continue
        name = fpath.name
        split = 'train' if 'metrics_train_' in name else 'validation'
        fold = 'unknown'
        name_squeezed = name.replace('-', '_')
        for token in name_squeezed.split('_'):
            if token.startswith('fold'):
                suffix = token[4:]
                if suffix.isdigit():
                    fold = suffix
                    break
        with fpath.open('r', encoding='utf-8') as handle:
            reader = csv.DictReader(row for row in handle if row.strip())
            for row in reader:
                wind_bin = row.get('wind_bin') or row.get('bin') or row.get('range')
                class_label = (row.get('class') or '').strip()
                if not wind_bin:
                    if class_label:
                        wind_bin = f"class={class_label}"
                    else:
                        continue
                elif class_label:
                    wind_bin = f"{wind_bin}|class={class_label}"
                if wind_bin is None:
                    continue
                key = (split, wind_bin)
                fold_key = (split, fold, wind_bin)
                counts[key] = counts.get(key, 0) + 1
                per_fold_counts[fold_key] = per_fold_counts.get(fold_key, 0) + 1
                bucket = summary.setdefault(key, {})
                bucket_fold = per_fold.setdefault(fold_key, {})
                for metric in METRIC_CANDIDATES:
                    value = row.get(metric)
                    if value in (None, ''):
                        continue
                    try:
                        numeric = float(value)
                    except ValueError:
                        continue
                    bucket.setdefault(metric, []).append(numeric)
                    bucket_fold.setdefault(metric, []).append(numeric)
                    present.add(metric)
    return summary, counts, per_fold, per_fold_counts, sorted(present)

def render_section(lines, title, files, summary, counts, metrics):
    lines.append(f"## {title}")
    if files:
        for fpath in files:
            lines.append(f"- {fpath.name}")
    else:
        lines.append('- None found')
    lines.append('')
    if not summary:
        lines.append('No aggregated data available.')
        lines.append('')
        return
    header = ['split', 'wind_bin', 'rows'] + [f"{metric} (mean±std)" for metric in metrics]
    lines.append('| ' + ' | '.join(header) + ' |')
    lines.append('| ' + ' | '.join(['---'] * len(header)) + ' |')
    for key in sorted(summary, key=lambda item: (item[0], item[1])):
        split, wind_bin = key
        row_values = [split, wind_bin, str(counts.get(key, 0))]
        metrics_data = summary[key]
        for metric in metrics:
            values = metrics_data.get(metric, [])
            if values:
                mean = statistics.mean(values)
                std = statistics.stdev(values) if len(values) > 1 else 0.0
                row_values.append(f"{mean:.4f} ± {std:.4f}")
            else:
                row_values.append('')
        lines.append('| ' + ' | '.join(row_values) + ' |')
    lines.append('')
def render_fold_section(lines, title, summary, counts, metrics):
    lines.append(f"## {title}")
    if not summary:
        lines.append('No per-fold aggregated data available.')
        lines.append('')
        return
    header = ['split', 'fold', 'wind_bin', 'rows'] + [f"{metric} (mean±std)" for metric in metrics]
    lines.append('| ' + ' | '.join(header) + ' |')
    lines.append('| ' + ' | '.join(['---'] * len(header)) + ' |')
    for key in sorted(summary, key=lambda item: (item[0], item[1], item[2])):
        split, fold, wind_bin = key
        row_values = [split, fold, wind_bin, str(counts.get(key, 0))]
        metrics_data = summary[key]
        for metric in metrics:
            values = metrics_data.get(metric, [])
            if values:
                mean = statistics.mean(values)
                std = statistics.stdev(values) if len(values) > 1 else 0.0
                row_values.append(f"{mean:.4f} ± {std:.4f}")
            else:
                row_values.append('')
        lines.append('| ' + ' | '.join(row_values) + ' |')
    lines.append('')


per_bin_files = sorted(output_dir.glob('metrics*_by_wind_bin.csv'))
node_bin_files = sorted(output_dir.glob('metrics*_by_location_id_by_wind_bin.csv'))
range_overall_files = sorted(output_dir.glob('metrics*_range_classification.csv'))
range_bin_files = sorted(output_dir.glob('metrics*_range_classification_by_wind_bin.csv'))

summary_main, counts_main, summary_main_fold, counts_main_fold, metrics_main = aggregate(per_bin_files)
summary_node, counts_node, summary_node_fold, counts_node_fold, metrics_node = aggregate(node_bin_files)
summary_range_overall, counts_range_overall, summary_range_overall_fold, counts_range_overall_fold, metrics_range_overall = aggregate(range_overall_files)
summary_range_bin, counts_range_bin, summary_range_bin_fold, counts_range_bin_fold, metrics_range_bin = aggregate(range_bin_files)

lines = [f"# Bin Metrics Summary for training job {train_job}", '']
render_section(lines, 'Source files (per wind bin)', per_bin_files, summary_main, counts_main, metrics_main)
render_section(lines, 'Source files (per node & wind bin)', node_bin_files, summary_node, counts_node, metrics_node)
render_fold_section(lines, 'Per-fold per wind bin metrics', summary_main_fold, counts_main_fold, metrics_main)
render_fold_section(lines, 'Per-fold per node & wind bin metrics', summary_node_fold, counts_node_fold, metrics_node)

render_section(lines, 'Range classification (overall)', range_overall_files, summary_range_overall, counts_range_overall, metrics_range_overall)
render_fold_section(lines, 'Range classification per fold (overall)', summary_range_overall_fold, counts_range_overall_fold, metrics_range_overall)

render_section(lines, 'Range classification by wind bin', range_bin_files, summary_range_bin, counts_range_bin, metrics_range_bin)
render_fold_section(lines, 'Range classification per fold and wind bin', summary_range_bin_fold, counts_range_bin_fold, metrics_range_bin)

report_path.write_text('\n'.join(lines).strip() + '\n', encoding='utf-8')
PYCODE
fi

echo "Done. Per-bin metrics files saved in '$BIN_METRICS_DIR'"
