#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Author: Juan Luis Herrera Cortijo <juan.luis.herrera.cortijo@gmail.com>
# Created: 2025-10-16
# Disclaimer: This script is provided as-is for research workflows and must be validated before any operational deployment.
# -----------------------------------------------------------------------------

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: get_train_metrics.sh --train-job-name JOB_NAME --aws-profile PROFILE [--output-directory PATH]
Fetch metrics_train_fold*.csv files from the SageMaker job's output tarball and build a markdown summary.

  --train-job-name JOB_NAME    Name of the SageMaker training job
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
TRAIN_METRICS_DIR="${OUTPUT_DIR%/}/training_metrics/${TRAIN_JOB}"
mkdir -p "$TRAIN_METRICS_DIR"

echo "Describing SageMaker training job '$TRAIN_JOB' (profile '$PROFILE')..."
S3_OUTPUT=$(aws sagemaker describe-training-job \
  --training-job-name "$TRAIN_JOB" \
  --profile "$PROFILE" \
  --output text \
  --query 'OutputDataConfig.S3OutputPath')

if [[ -z "$S3_OUTPUT" ]]; then
  echo "Error: Unable to retrieve S3 output path for training job '$TRAIN_JOB'" >&2
  exit 2
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

echo "Searching for metrics_train_fold*.csv files..."
FILES=$(find "$TMPDIR" -type f -name 'metrics_train_fold*.csv' || true)
if [[ -z "$FILES" ]]; then
  echo "Error: No metrics_train_fold*.csv found in the tarball" >&2
  exit 4
fi

for FILE in $FILES; do
  BASENAME=$(basename "$FILE")
  echo "Moving $FILE -> $TRAIN_METRICS_DIR/$BASENAME"
  mv "$FILE" "$TRAIN_METRICS_DIR/$BASENAME"
done

echo "Done. Metrics files saved in '$TRAIN_METRICS_DIR'"

echo "Generating training metrics report in '$TRAIN_METRICS_DIR'"

cd "$TRAIN_METRICS_DIR"
TRAIN_JOB_NAME="$TRAIN_JOB" python3 <<'PYCODE'
import csv
import glob
import math
import os
import statistics
import sys

job = os.environ.get('TRAIN_JOB_NAME', 'unknown-job')

def try_float(value):
    try:
        if value is None or value == '':
            return None
        return float(value)
    except (TypeError, ValueError):
        return None

metrics_pattern = 'metrics_train_fold*.csv'
all_files = sorted(glob.glob(metrics_pattern))
if not all_files:
    print('No training metrics files found for report generation.', file=sys.stderr)
    sys.exit(0)

classification_files = [f for f in all_files if 'range_classification' in os.path.basename(f)]
summary_files = [f for f in all_files if f not in classification_files and '_by_' not in os.path.basename(f)]

records = []
metric_names = set()
for fname in summary_files:
    try:
        with open(fname, newline='') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)
            data_rows = [row for row in reader if row]
    except Exception as exc:
        print(f'Error processing {fname}: {exc}', file=sys.stderr)
        continue

    if not data_rows:
        print(f'No data rows in {fname}', file=sys.stderr)
        continue

    row = data_rows[0]
    entry = {}
    for key, value in zip(header, row):
        numeric_value = try_float(value)
        if numeric_value is None:
            continue
        entry[key] = numeric_value
        metric_names.add(key)

    if not entry:
        print(f'No numeric metrics found in {fname}', file=sys.stderr)
        continue

    basename = os.path.splitext(os.path.basename(fname))[0]
    try:
        fold = int(basename.replace('metrics_train_fold', ''))
    except ValueError:
        fold = None

    entry['fold'] = fold
    records.append(entry)

classification_overall = []
classification_by_class = {}
for fname in classification_files:
    basename = os.path.splitext(os.path.basename(fname))[0]
    try:
        fold = int(basename.replace('metrics_train_fold', '').split('_')[0])
    except ValueError:
        fold = None

    try:
        with open(fname, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(row for row in csvfile if row.strip())
            rows = list(reader)
    except Exception as exc:
        print(f'Error processing {fname}: {exc}', file=sys.stderr)
        continue

    for row in rows:
        label = (row.get('class') or '').strip()
        metrics = {}
        for key, value in row.items():
            if key == 'class':
                continue
            numeric_value = try_float(value)
            if numeric_value is not None:
                metrics[key] = numeric_value
        metrics['fold'] = fold
        if label.lower() == 'overall':
            classification_overall.append((fold, metrics))
        elif label:
            classification_by_class.setdefault(label, []).append((fold, metrics))

report_path = os.path.join(os.getcwd(), 'training_metrics_report.md')
with open(report_path, 'w', encoding='utf-8') as handle:
    handle.write(f"# Training Metrics Report for job {job}\n\n")

    if records:
        ordered_metrics = sorted(metric_names)
        columns = ['fold'] + ordered_metrics
        handle.write('## Metrics per Fold\n\n')
        handle.write('| ' + ' | '.join(columns) + ' |\n')
        handle.write('| ' + ' | '.join(['---'] * len(columns)) + ' |\n')
        for record in sorted(records, key=lambda x: (9999 if x.get('fold') is None else x.get('fold'))):
            row = []
            for col in columns:
                value = record.get(col, '')
                if isinstance(value, float):
                    row.append(f'{value:.6f}')
                else:
                    row.append(str(value))
            handle.write('| ' + ' | '.join(row) + ' |\n')
        handle.write('\n')

        handle.write('## Mean and Standard Deviation of Metrics across Folds\n\n')
        handle.write('| metric | mean | std |\n')
        handle.write('| --- | --- | --- |\n')
        for key in ordered_metrics:
            values = [record[key] for record in records if key in record]
            if not values:
                continue
            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values) if len(values) > 1 else 0.0
            handle.write(f'| {key} | {mean_val:.6f} | {std_val:.6f} |\n')
        handle.write('\n')
    else:
        handle.write('No regression metrics were found.\n\n')

    if classification_overall:
        metrics_keys = ['accuracy', 'precision', 'recall', 'f1', 'support', 'predicted', 'true_positives']
        handle.write('## Range Classification Metrics (per fold)\n\n')
        handle.write('| fold | accuracy | precision | recall | f1 | support | predicted | true_positives |\n')
        handle.write('| --- | --- | --- | --- | --- | --- | --- | --- |\n')
        for fold, metrics in sorted(classification_overall, key=lambda item: (9999 if item[0] is None else item[0])):
            row = ['' if fold is None else str(fold)]
            for key in metrics_keys:
                value = metrics.get(key)
                if value is None:
                    row.append('')
                elif key in {'support', 'predicted', 'true_positives'}:
                    row.append(str(int(round(value))))
                else:
                    row.append(f'{value:.6f}')
            handle.write('| ' + ' | '.join(row) + ' |\n')
        handle.write('\n')

        summary_keys = ['accuracy', 'precision', 'recall', 'f1']
        handle.write('### Mean and Standard Deviation (range classification)\n\n')
        handle.write('| metric | mean | std |\n')
        handle.write('| --- | --- | --- |\n')
        for key in summary_keys:
            values = [metrics.get(key) for _, metrics in classification_overall if metrics.get(key) is not None]
            if not values:
                continue
            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values) if len(values) > 1 else 0.0
            handle.write(f'| {key} | {mean_val:.6f} | {std_val:.6f} |\n')
        handle.write('\n')

        for label in sorted(classification_by_class):
            entries = classification_by_class[label]
            handle.write(f"### Per-fold metrics for class '{label}'\n\n")
            handle.write('| fold | precision | recall | f1 | support | predicted | true_positives |\n')
            handle.write('| --- | --- | --- | --- | --- | --- | --- |\n')
            for fold, metrics in sorted(entries, key=lambda item: (9999 if item[0] is None else item[0])):
                row = ['' if fold is None else str(fold)]
                for key in ['precision', 'recall', 'f1', 'support', 'predicted', 'true_positives']:
                    value = metrics.get(key)
                    if value is None:
                        row.append('')
                    elif key in {'support', 'predicted', 'true_positives'}:
                        row.append(str(int(round(value))))
                    else:
                        row.append(f'{value:.6f}')
                handle.write('| ' + ' | '.join(row) + ' |\n')
            handle.write('\n')
    else:
        handle.write('No range classification metrics were found.\n\n')

print(f'Report generated: {report_path}')
PYCODE
