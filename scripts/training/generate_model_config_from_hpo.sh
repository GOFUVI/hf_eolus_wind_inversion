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
Usage: generate_model_config_from_hpo.sh --train-job NAME [options]

Create a model configuration JSON for scripts/training/train_model.sh using the
hyperparameters of an existing SageMaker training job.

Required arguments:
  --train-job NAME          SageMaker training job name to inspect (completed job)

Optional arguments:
  --profile PROFILE         AWS CLI profile (default: $AWS_PROFILE or 'default')
  --region REGION           AWS region (default: $AWS_REGION or 'us-east-1')
  --base-config PATH        Path to a base model JSON (default: value stored in the
                            training job hyperparameters)
  --output PATH             Output path for the generated config (default: alongside
                            the base config, named <train-job>.json)
  --force                   Allow overwriting an existing output file
  --help                    Show this help message and exit
USAGE
  exit 1
}

_check_dependency() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Error: required command '$cmd' not found in PATH." >&2
    exit 1
  fi
}

PROFILE="${AWS_PROFILE:-default}"
REGION="${AWS_REGION:-us-east-1}"
TRAIN_JOB=""
OUTPUT_PATH=""
BASE_CONFIG_OVERRIDE=""
FORCE_OVERWRITE="false"

POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --train-job)
      TRAIN_JOB="$2"; shift 2 ;;
    --train-job=*)
      TRAIN_JOB="${1#*=}"; shift ;;
    --profile)
      PROFILE="$2"; shift 2 ;;
    --profile=*)
      PROFILE="${1#*=}"; shift ;;
    --region)
      REGION="$2"; shift 2 ;;
    --region=*)
      REGION="${1#*=}"; shift ;;
    --base-config)
      BASE_CONFIG_OVERRIDE="$2"; shift 2 ;;
    --base-config=*)
      BASE_CONFIG_OVERRIDE="${1#*=}"; shift ;;
    --output)
      OUTPUT_PATH="$2"; shift 2 ;;
    --output=*)
      OUTPUT_PATH="${1#*=}"; shift ;;
    --force)
      FORCE_OVERWRITE="true"; shift ;;
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

if [[ -z "$TRAIN_JOB" ]]; then
  echo "Error: --train-job is required." >&2
  usage
fi

_check_dependency aws
_check_dependency jq
_check_dependency python3

TMP_TRAIN=""
TMP_HP=""
cleanup() {
  [[ -n "$TMP_TRAIN" && -f "$TMP_TRAIN" ]] && rm -f "$TMP_TRAIN"
  [[ -n "$TMP_HP" && -f "$TMP_HP" ]] && rm -f "$TMP_HP"
}
trap cleanup EXIT

TMP_TRAIN="$(mktemp generate_model_config_train.XXXXXX)"
if ! aws --profile "$PROFILE" --region "$REGION" \
  sagemaker describe-training-job \
  --training-job-name "$TRAIN_JOB" > "$TMP_TRAIN" 2>"$TMP_TRAIN.err"; then
  cat "$TMP_TRAIN.err" >&2 || true
  echo "Error: failed to describe training job '$TRAIN_JOB'." >&2
  exit 1
fi
rm -f "$TMP_TRAIN.err"

MODEL_CONFIG_PATH=""
if [[ -n "$BASE_CONFIG_OVERRIDE" ]]; then
  MODEL_CONFIG_PATH="$BASE_CONFIG_OVERRIDE"
else
  MODEL_CONFIG_FROM_JOB="$(jq -r '.HyperParameters["model-config"] // empty' "$TMP_TRAIN")"
  if [[ -z "$MODEL_CONFIG_FROM_JOB" ]]; then
    echo "Error: training job '$TRAIN_JOB' does not expose a 'model-config' hyperparameter. Use --base-config." >&2
    exit 2
  fi
  if [[ "$MODEL_CONFIG_FROM_JOB" == /* ]]; then
    MODEL_CONFIG_PATH="$MODEL_CONFIG_FROM_JOB"
  else
    MODEL_CONFIG_PATH="${PROJECT_ROOT}/${MODEL_CONFIG_FROM_JOB}"
  fi
fi

if [[ ! -f "$MODEL_CONFIG_PATH" ]]; then
  echo "Error: base config '$MODEL_CONFIG_PATH' not found. Provide a valid path with --base-config." >&2
  exit 3
fi

if [[ -z "$OUTPUT_PATH" ]]; then
  BASE_DIR="$(dirname "$MODEL_CONFIG_PATH")"
  SAFE_JOB="${TRAIN_JOB//\//_}"
  OUTPUT_PATH="${BASE_DIR}/${SAFE_JOB}.json"
fi

if [[ -e "$OUTPUT_PATH" && "$FORCE_OVERWRITE" != "true" ]]; then
  echo "Error: output file '$OUTPUT_PATH' already exists. Use --force to overwrite." >&2
  exit 4
fi

mkdir -p "$(dirname "$OUTPUT_PATH")"

TMP_HP="$(mktemp generate_model_config_hp.XXXXXX)"
jq '.HyperParameters' "$TMP_TRAIN" > "$TMP_HP"

python3 - "$MODEL_CONFIG_PATH" "$OUTPUT_PATH" "$TMP_HP" "$TRAIN_JOB" <<'PYCODE'
import json
import sys
from pathlib import Path

base_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
hparams_path = Path(sys.argv[3])
training_job = sys.argv[4]

with base_path.open('r', encoding='utf-8') as handle:
    config = json.load(handle)
with hparams_path.open('r', encoding='utf-8') as handle:
    raw_hparams = json.load(handle)

model_section = config.setdefault('model', {})

int_keys = {
    'hidden_layers': 'hidden_layers',
    'hidden_units': 'hidden_units',
    'epochs': 'epochs',
    'batch_size': 'batch_size',
    'patience': 'patience',
}
float_keys = {
    'dropout': 'dropout',
    'lr': 'lr',
    'weight_decay': 'weight_decay',
    'range_margin': 'range_margin',
    'range_loss_weight': 'range_loss_weight',
    'range_flag_threshold': 'range_flag_threshold',
}
bool_keys = {
    'use_mad': 'use_mad',
    'use_velocity_median': 'use_velocity_median',
    'early_stopping': 'early_stopping',
    'save_error_data': 'save_error_data',
}
str_keys = {
    'agg_stat': 'agg_stat',
}
model_str_keys = {
    'target-speed-col': 'target_speed_col',
    'target-dir-col': 'target_dir_col',
    'id-col': 'id_col',
    'normalization-mode': 'normalization_mode',
}

def parse_bool(value):
    text = str(value).strip().lower()
    if text in {'1', 'true', 'yes', 'y', 'on'}:
        return True
    if text in {'0', 'false', 'no', 'n', 'off'}:
        return False
    raise ValueError(f"Cannot parse boolean value from '{value}'")

def parse_float(value):
    return float(value)

def parse_int(value):
    return int(float(value))

def parse_norm_override(payload):
    if not payload:
        return None
    overrides = {}
    for chunk in str(payload).split(';'):
        part = chunk.strip()
        if not part or '=' not in part:
            continue
        left, value = part.split('=', 1)
        if '.' not in left:
            continue
        feature, param = left.split('.', 1)
        try:
            num_val = float(value)
        except ValueError:
            continue
        feature_entry = overrides.setdefault(feature, {})
        feature_entry[param] = num_val
    return overrides

for key, field in int_keys.items():
    if key in raw_hparams:
        model_section[field] = parse_int(raw_hparams[key])

for key, field in float_keys.items():
    if key in raw_hparams:
        model_section[field] = parse_float(raw_hparams[key])

for key, field in bool_keys.items():
    if key in raw_hparams:
        try:
            model_section[field] = parse_bool(raw_hparams[key])
        except ValueError:
            pass

for key, field in str_keys.items():
    if key in raw_hparams:
        model_section[field] = str(raw_hparams[key])

for key, field in model_str_keys.items():
    if key in raw_hparams:
        model_section[field] = str(raw_hparams[key])

if 'stations' in raw_hparams:
    stations_list = [item.strip() for item in str(raw_hparams['stations']).split(';') if item.strip()]
    if stations_list:
        config['stations'] = stations_list

norm_override = parse_norm_override(raw_hparams.get('norm-override'))
if norm_override is not None:
    config['norm_override'] = norm_override

metadata = config.setdefault('metadata', {})
metadata['source_training_job'] = training_job

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open('w', encoding='utf-8') as handle:
    json.dump(config, handle, indent=2, sort_keys=False)
    handle.write('\n')
PYCODE

cleanup
trap - EXIT

echo "Generated model config for training job '$TRAIN_JOB'."
echo "Base config: $MODEL_CONFIG_PATH"
echo "Output config: $OUTPUT_PATH"
