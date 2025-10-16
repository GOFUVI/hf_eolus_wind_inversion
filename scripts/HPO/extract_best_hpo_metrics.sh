#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Author: Juan Luis Herrera Cortijo <juan.luis.herrera.cortijo@gmail.com>
# Created: 2025-10-16
# Disclaimer: This script is provided as-is for research workflows and must be validated before any operational deployment.
# -----------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

usage() {
  cat <<'USAGE' >&2
Usage: extract_best_hpo_metrics.sh --report FILE --aws-profile PROFILE [options]

Identify the best training job from an aggregated HPO report and download the
associated metrics and normalization artifacts.

Required arguments:
  --report FILE           Aggregated HPO markdown report produced by integrate_hpo_reports.sh
  --aws-profile PROFILE   AWS CLI profile used to access SageMaker artifacts

Optional arguments:
  --output-directory DIR  Directory where helper scripts store downloaded artifacts (default: current directory)
  --help                  Show this help message and exit
USAGE
  exit 1
}

REPORT_PATH=""
AWS_PROFILE=""
OUTPUT_DIR=""

POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --report)
      REPORT_PATH="$2"; shift 2 ;;
    --report=*)
      REPORT_PATH="${1#*=}"; shift ;;
    --aws-profile)
      AWS_PROFILE="$2"; shift 2 ;;
    --aws-profile=*)
      AWS_PROFILE="${1#*=}"; shift ;;
    --output-directory)
      OUTPUT_DIR="$2"; shift 2 ;;
    --output-directory=*)
      OUTPUT_DIR="${1#*=}"; shift ;;
    --help)
      usage ;;
    --)
      shift; break ;;
    -* )
      echo "Error: unknown option '$1'" >&2
      usage ;;
    * )
      POSITIONAL+=("$1"); shift ;;
  esac
done

if [[ ${#POSITIONAL[@]} -gt 0 ]]; then
  echo "Error: unexpected positional arguments: ${POSITIONAL[*]}" >&2
  usage
fi

if [[ -z "$REPORT_PATH" || -z "$AWS_PROFILE" ]]; then
  echo "Error: --report and --aws-profile are required." >&2
  usage
fi

if [[ ! -f "$REPORT_PATH" ]]; then
  echo "Error: aggregated report '$REPORT_PATH' not found." >&2
  exit 1
fi

parse_row() {
  python3 - "$REPORT_PATH" <<'PY'
import sys

path = sys.argv[1]
with open(path, encoding='utf-8') as handle:
    lines = [line.rstrip('\n') for line in handle]

header_line = None
data_start = None
for idx, line in enumerate(lines):
    if line.startswith('|') and 'HPOJob' in line:
        header_line = line
        data_start = idx + 2
        break

if header_line is None:
    sys.exit(0)

headers = [col.strip() for col in header_line.strip('|').split('|')]
lower_headers = [h.lower() for h in headers]

try:
    hpo_idx = lower_headers.index('hpojob')
    train_idx = lower_headers.index('trainingjobname')
except ValueError:
    sys.exit(0)

combined_idx = None
for idx, name in enumerate(lower_headers):
    if 'combined' in name and 'std' not in name:
        combined_idx = idx
        break

if combined_idx is None:
    sys.exit(0)

for line in lines[data_start:]:
    if not line.startswith('|') or '---' in line:
        continue
    values = [col.strip() for col in line.strip('|').split('|')]
    if len(values) <= max(hpo_idx, train_idx, combined_idx):
        continue
    hpo = values[hpo_idx]
    train = values[train_idx]
    metric = values[combined_idx]
    if hpo and train:
        print(f"{hpo}|{train}|{metric}")
        break
PY
}

BEST_ENTRY="$(parse_row)"
if [[ -z "$BEST_ENTRY" ]]; then
  echo "Error: unable to locate a data row in '$REPORT_PATH'." >&2
  exit 2
fi

IFS='|' read -r BEST_HPO_JOB BEST_TRAIN_JOB BEST_LOSS <<< "$BEST_ENTRY"

TRAIN_METRICS_SCRIPT="${PROJECT_ROOT}/scripts/training/get_train_metrics.sh"
NORM_PARAMS_SCRIPT="${PROJECT_ROOT}/scripts/training/get_norm_params.sh"
BIN_METRICS_SCRIPT="${PROJECT_ROOT}/scripts/training/get_bin_metrics.sh"

for helper in "$TRAIN_METRICS_SCRIPT" "$NORM_PARAMS_SCRIPT" "$BIN_METRICS_SCRIPT"; do
  if [[ ! -x "$helper" ]]; then
    echo "Error: helper script '$helper' is missing or not executable." >&2
    exit 3
  fi
done

echo "Best HPO job: $BEST_HPO_JOB"
echo "Best training job: $BEST_TRAIN_JOB"
if [[ -n "$BEST_LOSS" ]]; then
  echo "Best combined loss: $BEST_LOSS"
fi

echo "Fetching training metrics..."
if [[ -n "$OUTPUT_DIR" ]]; then
  "$TRAIN_METRICS_SCRIPT" --aws-profile "$AWS_PROFILE" --train-job-name "$BEST_TRAIN_JOB" --output-directory "$OUTPUT_DIR"
else
  "$TRAIN_METRICS_SCRIPT" --aws-profile "$AWS_PROFILE" --train-job-name "$BEST_TRAIN_JOB"
fi

echo "Fetching normalization parameters..."
if [[ -n "$OUTPUT_DIR" ]]; then
  "$NORM_PARAMS_SCRIPT" --aws-profile "$AWS_PROFILE" --train-job-name "$BEST_TRAIN_JOB" --output-directory "$OUTPUT_DIR"
else
  "$NORM_PARAMS_SCRIPT" --aws-profile "$AWS_PROFILE" --train-job-name "$BEST_TRAIN_JOB"
fi

echo "Fetching per-bin metrics..."
if [[ -n "$OUTPUT_DIR" ]]; then
  "$BIN_METRICS_SCRIPT" --aws-profile "$AWS_PROFILE" --train-job-name "$BEST_TRAIN_JOB" --output-directory "$OUTPUT_DIR"
else
  "$BIN_METRICS_SCRIPT" --aws-profile "$AWS_PROFILE" --train-job-name "$BEST_TRAIN_JOB"
fi

echo "Done. Artifacts downloaded for training job '$BEST_TRAIN_JOB'."
