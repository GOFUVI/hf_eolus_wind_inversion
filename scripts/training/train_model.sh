#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Author: Juan Luis Herrera Cortijo <juan.luis.herrera.cortijo@gmail.com>
# Created: 2025-10-16
# Disclaimer: This script is provided as-is for research workflows and must be validated before any operational deployment.
# -----------------------------------------------------------------------------

#
# SageMaker Training Orchestrator
# --------------------------------
# Submits one SageMaker training job per requested fold, optionally fine-tuning
# from a previous checkpoint, collects their metrics, and renders a Markdown
# summary. The script is designed to be reproducible: everything it needs is
# derived from the provided model JSON and command-line arguments, and every
# side effect (logs, metrics, reports) lands in deterministic locations.
#
# The implementation assumes the training code lives under scripts/training/ and
# mirrors the conventions used by the rest of the HF wind inversion pipelines.
# See docs/train_model.md for the broader methodological overview.
#
# Exit immediately on error, undefined variable, or pipeline failure
set -euo pipefail
# Record original working directory before changing to script directory for correct -L handling
ORIG_PWD="$(pwd)"
# Change to script directory to ensure relative paths (Dockerfile) are correct
cd "$(dirname "$0")"
PROJECT_ROOT="$(cd ../.. && pwd)"

# ----------------------------------------------------------------------------
# ensure_jq_available
# ----------------------------------------------------------------------------
# Purpose : Bail out early when jq is unavailable so the user gets a clear
#           actionable error instead of a cryptic parse failure downstream.
# Usage   : ensure_jq_available
# Returns : nothing; exits with code 1 (after printing a message) if jq is missing
# ----------------------------------------------------------------------------
ensure_jq_available() {
  if ! command -v jq >/dev/null 2>&1; then
    echo "Error: 'jq' must be installed to parse JSON configuration files." >&2
    exit 1
  fi
}

# ----------------------------------------------------------------------------
# parse_model_config
# ----------------------------------------------------------------------------
# Purpose : Extract the subset of configuration values from the model JSON that
#           must be surfaced as SageMaker hyper-parameters (stations list, range
#           bounds, toggles, etc.). Returning them as key=value lines makes it
#           easy to consume from Bash while maintaining readability.
# ----------------------------------------------------------------------------
parse_model_config() {
  local config_path="$1"
  jq -r '
    def range_array: (.model.target_speed_range // .target_speed_range // []);
    (
      [
        "stations=" + ((.stations // []) | map(tostring) | join(";")),
        "target_speed_col=" + (.model.target_speed_col // .target_speed_col // "wind_speed"),
        "target_dir_col=" + (.model.target_dir_col // .target_dir_col // "wind_direction"),
        "agg_stat=" + (.model.agg_stat // "mean"),
        "id_col=" + (.model.id_col // .id_col // "location_id"),
        "use_mad=" + (if (.model.use_mad // false) then "1" else "0" end),
        "use_velocity_median=" + (if (.model.use_velocity_median // false) then "1" else "0" end),
        "early_stopping=" + (if (.model.early_stopping // true) then "1" else "0" end),
        "save_error_data=" + (if (.model.save_error_data // false) then "1" else "0" end),
        "normalization_mode=" + (.model.normalization_mode // .normalization_mode // "standard"),
        "range_margin=" + (((.model.range_margin // .range_margin // 0.0) | tostring)),
        "range_loss_weight=" + (((.model.range_loss_weight // .range_loss_weight // 1.0) | tostring)),
        "range_flag_threshold=" + (((.model.range_flag_threshold // .range_flag_threshold // 0.5) | tostring))
      ] + (
        if (range_array | length) == 2 then
          [
            "range_min=" + ((range_array[0]) | tostring),
            "range_max=" + ((range_array[1]) | tostring)
          ]
        else [] end
      )
    ) | .[]
  ' "$config_path"
}

# Default local output directory for logs, reports, and metrics_results
OUTPUT_DIR="."

# -----------------------------------------------------------------------------
# train_model.sh - Orchestrate SageMaker training across cross-validation folds
#
# This script submits a SageMaker training job for each fold with specified
# hyperparameters, collects logs, and aggregates results.
# It supports --stations to specify a semicolon-separated list of station names
# and --ecr-repo to override the ECR repository name.
#
# Usage:
#   ./train_model.sh --train-data-uri <s3_uri> \
#     --s3-prefix <s3_prefix> --model-config <path> [options]
# -----------------------------------------------------------------------------
# Default checkpoint config args (prevents unbound variable when no artifact URI is provided)
CHECKPOINT_CONFIG_ARGS=()

# ------------------------------------------------------------------------------
# Logging setup: initialize log file and define logging function for timestamped messages
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Function: run_aws
# Description: Execute AWS CLI commands with specified profile and region, logging command and output.
# ------------------------------------------------------------------------------
run_aws() {
  echo "Running: aws $*" >> "$LOG_FILE"
  aws --profile "$PROFILE" --region "$REGION" "$@" 2>&1 | tee -a "$LOG_FILE"
}

# ------------------------------------------------------------------------------
# Function: fetch_logs
# Description: Fetch consolidated CloudWatch logs for a SageMaker training job.
# ------------------------------------------------------------------------------
fetch_logs() {
  local job_name="$1"
  local log_group="/aws/sagemaker/TrainingJobs"
  log "Listing log streams for job $job_name"
  mkdir -p "${OUTPUT_DIR}/logs/${job_name}"
  log "Fetching consolidated CloudWatch logs for job $job_name"
  aws --profile "$PROFILE" --region "$REGION" logs filter-log-events \
    --log-group-name "$log_group" \
    --log-stream-name-prefix "$job_name" \
    --query 'events[].message' \
    --output text > "${OUTPUT_DIR}/logs/${job_name}/events.log"
}

# ------------------------------------------------------------------------------
# Function: wait_for_training_job
# Description: Wait for SageMaker training job completion; always fetch CloudWatch logs afterward.
# ------------------------------------------------------------------------------
wait_for_training_job() {
  local job_name="$1"
  log "Waiting for training job $job_name to complete..."
  set +e
  aws --profile "$PROFILE" --region "$REGION" sagemaker wait training-job-completed-or-stopped --training-job-name "$job_name"
  rc=$?
  set -e
  status=$(aws --profile "$PROFILE" --region "$REGION" sagemaker describe-training-job \
    --training-job-name "$job_name" --query 'TrainingJobStatus' --output text)
  if [ "$rc" -ne 0 ]; then
    log "Training job $job_name ended with status '$status' (waiter exit code $rc)"
  else
    log "Training job $job_name completed with status '$status'"
  fi
  log "Fetching CloudWatch logs for job $job_name"
  fetch_logs "$job_name"
}

## -----------------------------------------------------------------------------
# Function: usage
# Description: Display script usage information and exit.
## -----------------------------------------------------------------------------
usage() {
  cat <<EOF >&2
Usage: $0 \
  --train-data-uri <s3_data_geoparquet> \
  --s3-prefix <s3_prefix> \
  --model-config <path> \
  [options]

Required:
  --train-data-uri S3URI     S3 URI pointing to the GeoParquet dataset (file or prefix already in S3)
  --s3-prefix S3URI          S3 prefix for job outputs (e.g., s3://bucket/prefix)
  --model-config PATH        Path to the JSON model definition (stations, schema, hyperparameters)

Options:
  --image-uri URI            Pre-built ECR image URI for training (builds locally if omitted)
  --role-arn ARN             IAM role ARN for SageMaker execution
  --job-base-prefix NAME     Job base name prefix (default: ${JOB_BASE_PREFIX})
  --profile PROFILE          AWS CLI profile (default: ${AWS_PROFILE:-default})
  --region REGION            AWS region (default: ${AWS_REGION:-us-east-1})
  --instance-type TYPE       SageMaker instance type (default: ml.m5.large)
  --volume-size GB           EBS volume size in GB (default: 10)
  --max-runtime SECONDS      Maximum runtime in seconds (default: 3600)
  --folds COUNT              Number of CV folds (default: 5)
  --folds-list LIST          Comma-separated list of folds to run (e.g., 0,2,3)
  --wait-for-jobs true|false Wait for training jobs to complete (default: true)
  --artifact-uri URI         Existing checkpoint tarball for fine-tuning
  --clean-logs true|false    Remove existing local logs for the job prefix (default: true)
  --normalization-mode MODE  Feature normalization strategy (standard|robust)
  --output-dir DIR           Local directory for logs and reports (default: .)
  --no-cv[=BOOL]             Train without cross-validation (single fold 0; default true when flag present)
  --ecr-repo NAME            ECR repository name for Docker image (default: buoy_train)
  --seed INT                 Random seed to initialize data loaders and model weights
  --help                     Show this help message and exit
EOF
  exit 1
}

## -----------------------------------------------------------------------------
# Default configuration values
## -----------------------------------------------------------------------------
PROFILE="${AWS_PROFILE:-default}"
REGION="${AWS_REGION:-us-east-1}"
# Base name for jobs and ECR repository
JOB_BASE_PREFIX="train-fold"
JOB_BASE_NAME="${JOB_BASE_PREFIX}"
S3_PREFIX=""
ROLE_ARN=""
IMAGE_URI=""
INSTANCE_TYPE="ml.m5.large"
VOLUME_SIZE="10"
MAX_RUNTIME="7200"
# Whether to clean existing local logs for job base name (true/false)
CLEAN_LOGS="true"
SEED=""
TRAIN_DATA_URI=""
FOLDS="5"
FOLDS_LIST=""
WAIT_FOR_JOBS="true"
CLI_NORMALIZATION_MODE=""
# Default ECR repository name (unless overridden by --ecr-repo)
ECR_REPO="${ECR_REPO:-buoy_train}"

# S3 URI for existing model artifact to fine-tune from (for SageMaker checkpoint)
ARTIFACT_URI=""

# Default checkpoint config args (empty if no fine-tuning)
CHECKPOINT_CONFIG_ARGS=()

# Whether to disable cross-validation and train on full dataset only
NO_CV="false"

MODEL_CONFIG=""

# ------------------------------------------------------------------------------
# Parse command-line options (flags and parameters) using long-form flags only
# ------------------------------------------------------------------------------
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --train-data-uri)
      TRAIN_DATA_URI="$2"; shift 2 ;;
    --train-data-uri=*)
      TRAIN_DATA_URI="${1#*=}"; shift ;;
    --s3-prefix)
      S3_PREFIX="$2"; shift 2 ;;
    --s3-prefix=*)
      S3_PREFIX="${1#*=}"; shift ;;
    --role-arn)
      ROLE_ARN="$2"; shift 2 ;;
    --role-arn=*)
      ROLE_ARN="${1#*=}"; shift ;;
    --image-uri)
      IMAGE_URI="$2"; shift 2 ;;
    --image-uri=*)
      IMAGE_URI="${1#*=}"; shift ;;
    --job-base-prefix)
      JOB_BASE_PREFIX="$2"; shift 2 ;;
    --job-base-prefix=*)
      JOB_BASE_PREFIX="${1#*=}"; shift ;;
    --profile)
      PROFILE="$2"; shift 2 ;;
    --profile=*)
      PROFILE="${1#*=}"; shift ;;
    --region)
      REGION="$2"; shift 2 ;;
    --region=*)
      REGION="${1#*=}"; shift ;;
    --instance-type)
      INSTANCE_TYPE="$2"; shift 2 ;;
    --instance-type=*)
      INSTANCE_TYPE="${1#*=}"; shift ;;
    --volume-size)
      VOLUME_SIZE="$2"; shift 2 ;;
    --volume-size=*)
      VOLUME_SIZE="${1#*=}"; shift ;;
    --max-runtime)
      MAX_RUNTIME="$2"; shift 2 ;;
    --max-runtime=*)
      MAX_RUNTIME="${1#*=}"; shift ;;
    --folds)
      FOLDS="$2"; shift 2 ;;
    --folds=*)
      FOLDS="${1#*=}"; shift ;;
    --folds-list)
      FOLDS_LIST="$2"; shift 2 ;;
    --folds-list=*)
      FOLDS_LIST="${1#*=}"; shift ;;
    --wait-for-jobs)
      WAIT_FOR_JOBS="$2"; shift 2 ;;
    --wait-for-jobs=*)
      WAIT_FOR_JOBS="${1#*=}"; shift ;;
    --artifact-uri)
      ARTIFACT_URI="$2"; shift 2 ;;
    --artifact-uri=*)
      ARTIFACT_URI="${1#*=}"; shift ;;
    --clean-logs)
      CLEAN_LOGS="$2"; shift 2 ;;
    --clean-logs=*)
      CLEAN_LOGS="${1#*=}"; shift ;;
    --normalization-mode)
      CLI_NORMALIZATION_MODE="$2"; shift 2 ;;
    --normalization-mode=*)
      CLI_NORMALIZATION_MODE="${1#*=}"; shift ;;
    --output-dir)
      OUTPUT_DIR="$2"; shift 2 ;;
    --output-dir=*)
      OUTPUT_DIR="${1#*=}"; shift ;;
    --no-cv)
      NO_CV="true"; shift ;;
    --no-cv=*)
      NO_CV="${1#*=}"; shift ;;
    --seed)
      SEED="$2"; shift 2 ;;
    --seed=*)
      SEED="${1#*=}"; shift ;;
    --ecr-repo)
      ECR_REPO="$2"; shift 2 ;;
    --ecr-repo=*)
      ECR_REPO="${1#*=}"; shift ;;
    --model-config)
      MODEL_CONFIG="$2"; shift 2 ;;
    --model-config=*)
      MODEL_CONFIG="${1#*=}"; shift ;;
    --rehearsal-data-uri)
      REHEARSAL_DATA_URI="$2"; shift 2 ;;
    --rehearsal-data-uri=*)
      REHEARSAL_DATA_URI="${1#*=}"; shift ;;
    --rehearsal-target-speed-col)
      REHEARSAL_TARGET_SPEED_COL="$2"; shift 2 ;;
    --rehearsal-target-speed-col=*)
      REHEARSAL_TARGET_SPEED_COL="${1#*=}"; shift ;;
    --rehearsal-target-dir-col)
      REHEARSAL_TARGET_DIR_COL="$2"; shift 2 ;;
    --rehearsal-target-dir-col=*)
      REHEARSAL_TARGET_DIR_COL="${1#*=}"; shift ;;
    --rehearsal-fraction)
      REHEARSAL_FRACTION="$2"; shift 2 ;;
    --rehearsal-fraction=*)
      REHEARSAL_FRACTION="${1#*=}"; shift ;;
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

if [[ $# -gt 0 ]]; then
  POSITIONAL+=("$@")
fi

if [[ ${#POSITIONAL[@]} -gt 0 ]]; then
  echo "Error: unexpected positional arguments: ${POSITIONAL[*]}" >&2
  usage
fi

TRAIN_SCRIPT_ENTRYPOINT="train.py"

# Convert relative output directory to absolute path relative to original working directory
if [[ "${OUTPUT_DIR}" == "." ]]; then
  OUTPUT_DIR="${ORIG_PWD}"
elif [[ "${OUTPUT_DIR:0:1}" != "/" ]]; then
  OUTPUT_DIR="${ORIG_PWD}/${OUTPUT_DIR}"
fi

# Ensure output directory exists for logs, reports, and metrics_results
mkdir -p "${OUTPUT_DIR}"

## Initialize logging (after parsing -L)
SCRIPT_NAME=$(basename "$0" .sh)
LOG_FILE="${OUTPUT_DIR}/${SCRIPT_NAME}.log"
rm -f "$LOG_FILE"
log() { echo "$(date '+%Y-%m-%d %H:%M:%S') - $*" | tee -a "$LOG_FILE"; }
# Recompute job base name if prefix was overridden
JOB_BASE_NAME="${JOB_BASE_PREFIX}"

# Default ECR repository name to 'buoy_train' if not overridden
ECR_REPO="${ECR_REPO:-buoy_train}"

# Require model configuration for station and schema definitions
if [[ -z "$MODEL_CONFIG" ]]; then
  echo "Error: --model-config is required for training." >&2
  usage
fi

## -----------------------------------------------------------------------------
# Validate required options presence
## -----------------------------------------------------------------------------
# Ensure required options are provided
# Validate required options presence
if [ -z "$TRAIN_DATA_URI" ] || [ -z "$S3_PREFIX" ]; then
  usage
fi

# ------------------------------------------------------------------------------
# Optional cleanup of existing logs
# ------------------------------------------------------------------------------
# Clean existing local logs for this job base if requested
if [ "$CLEAN_LOGS" = "true" ]; then
  log "Cleaning existing local logs for job base prefix '$JOB_BASE_PREFIX'"
  rm -rf "${OUTPUT_DIR}/logs/${JOB_BASE_PREFIX}"* || true
fi

MODEL_CONFIG_PATH="$MODEL_CONFIG"
if [[ "$MODEL_CONFIG_PATH" != /* ]]; then
  MODEL_CONFIG_PATH="${ORIG_PWD}/${MODEL_CONFIG_PATH}"
fi
if [[ ! -f "$MODEL_CONFIG_PATH" ]]; then
  echo "Error: model config file not found: $MODEL_CONFIG_PATH" >&2
  exit 1
fi

ensure_jq_available
PARSED_VARS=$(parse_model_config "$MODEL_CONFIG_PATH")
CONFIG_STATIONS=""
CONFIG_TARGET_SPEED_COL="wind_speed"
CONFIG_TARGET_DIR_COL="wind_direction"
CONFIG_AGG_STAT="mean"
CONFIG_USE_MAD="0"
CONFIG_USE_VELO_MEDIAN="0"
CONFIG_EARLY_STOPPING="0"
CONFIG_SAVE_ERROR_DATA="0"
CONFIG_NORMALIZATION_MODE="standard"
ID_COL="location_id"
RANGE_MIN=""
RANGE_MAX=""
RANGE_MARGIN=""
RANGE_LOSS_WEIGHT=""
RANGE_FLAG_THRESHOLD=""
while IFS='=' read -r key value; do
  case "$key" in
    stations) CONFIG_STATIONS="$value" ;;
    target_speed_col) CONFIG_TARGET_SPEED_COL="$value" ;;
    target_dir_col) CONFIG_TARGET_DIR_COL="$value" ;;
    agg_stat) CONFIG_AGG_STAT="$value" ;;
    id_col) ID_COL="$value" ;;
    use_mad) CONFIG_USE_MAD="$value" ;;
    use_velocity_median) CONFIG_USE_VELO_MEDIAN="$value" ;;
    early_stopping) CONFIG_EARLY_STOPPING="$value" ;;
    save_error_data) CONFIG_SAVE_ERROR_DATA="$value" ;;
    normalization_mode) CONFIG_NORMALIZATION_MODE="$value" ;;
    range_min) RANGE_MIN="$value" ;;
    range_max) RANGE_MAX="$value" ;;
    range_margin) RANGE_MARGIN="$value" ;;
    range_loss_weight) RANGE_LOSS_WEIGHT="$value" ;;
    range_flag_threshold) RANGE_FLAG_THRESHOLD="$value" ;;
  esac
done <<< "$PARSED_VARS"

if [[ -n "$CLI_NORMALIZATION_MODE" ]]; then
  CONFIG_NORMALIZATION_MODE="$CLI_NORMALIZATION_MODE"
fi

if [[ -n "$CONFIG_NORMALIZATION_MODE" ]]; then
  NORMALIZATION_LOWER=$(printf '%s' "$CONFIG_NORMALIZATION_MODE" | tr '[:upper:]' '[:lower:]')
  case "$NORMALIZATION_LOWER" in
    standard|robust)
      CONFIG_NORMALIZATION_MODE="$NORMALIZATION_LOWER" ;;
    *)
      echo "Error: unsupported normalization mode '$CONFIG_NORMALIZATION_MODE'. Use 'standard' or 'robust'." >&2
      exit 1 ;;
  esac
else
  CONFIG_NORMALIZATION_MODE="standard"
fi

IFS=';' read -r -a STATION_ARRAY <<< "$CONFIG_STATIONS"
if [ "${#STATION_ARRAY[@]}" -lt 2 ]; then
  echo "Error: model config must define at least two stations" >&2
  exit 1
fi



log "Configuration:"
log "  Training script     : $TRAIN_SCRIPT_ENTRYPOINT"
log "  Data GeoParquet (S3 URI) : $TRAIN_DATA_URI"
log "  S3 prefix           : $S3_PREFIX"
log "  IAM Role ARN        : $ROLE_ARN"
log "  ECR Image URI       : $IMAGE_URI"
log "  Job base name       : $JOB_BASE_NAME"
log "  AWS Profile         : $PROFILE"
log "  AWS Region          : $REGION"
log "  Instance type       : $INSTANCE_TYPE"
log "  Volume size (GB)    : $VOLUME_SIZE"
log "  Max runtime (sec)   : $MAX_RUNTIME"
log "  Number of folds     : $FOLDS"
log "  No cross-validation : $NO_CV"
log "  Wait for jobs       : $WAIT_FOR_JOBS"
log "  Stations (config)   : ${CONFIG_STATIONS}"
log "  Target speed column : ${CONFIG_TARGET_SPEED_COL}"
log "  Target dir column   : ${CONFIG_TARGET_DIR_COL}"
log "  Aggregation stat    : ${CONFIG_AGG_STAT}"
log "  Use MAD features    : ${CONFIG_USE_MAD}"
log "  Use velocity median : ${CONFIG_USE_VELO_MEDIAN}"
log "  Normalization mode  : ${CONFIG_NORMALIZATION_MODE}"
log "  Metrics ID column   : ${ID_COL}"
log "  Range min/max       : ${RANGE_MIN}/${RANGE_MAX}"
log "  Range margin        : ${RANGE_MARGIN}"
log "  Range loss weight   : ${RANGE_LOSS_WEIGHT}"
log "  Range flag threshold: ${RANGE_FLAG_THRESHOLD}"


# Upload training script and data to S3
## Create or use SageMaker execution IAM role
log "Creating SageMaker execution IAM role"
ROLE_NAME="${ROLE_NAME:-MySageMakerRole}"
TMP_POLICY_FILE="$(mktemp)"
cat > "$TMP_POLICY_FILE" <<EOF
{
  "Version": "2012-10-17",
  "Statement": {
    "Effect": "Allow",
    "Principal": { "Service": "sagemaker.amazonaws.com" },
    "Action": "sts:AssumeRole"
  }
}
EOF
if run_aws iam get-role --role-name "$ROLE_NAME" > /dev/null; then
  log "Role $ROLE_NAME already exists"
else
  run_aws iam create-role --role-name "$ROLE_NAME" --assume-role-policy-document file://"$TMP_POLICY_FILE"
fi
rm "$TMP_POLICY_FILE"
run_aws iam attach-role-policy --role-name "$ROLE_NAME" --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess || log "Policy AmazonSageMakerFullAccess may already be attached"
run_aws iam attach-role-policy --role-name "$ROLE_NAME" --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess || log "Policy AmazonS3FullAccess may already be attached"
run_aws iam attach-role-policy --role-name "$ROLE_NAME" --policy-arn arn:aws:iam::aws:policy/AmazonAthenaFullAccess || log "Policy AmazonAthenaFullAccess may already be attached"
run_aws iam attach-role-policy --role-name "$ROLE_NAME" --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly || log "Policy AmazonEC2ContainerRegistryReadOnly may already be attached"
# Grant permissions for CloudWatch Logs so training containers can stream logs
run_aws iam attach-role-policy --role-name "$ROLE_NAME" --policy-arn arn:aws:iam::aws:policy/CloudWatchLogsFullAccess || log "Policy CloudWatchLogsFullAccess may already be attached"
ROLE_ARN="$(run_aws iam get-role --role-name "$ROLE_NAME" --query 'Role.Arn' --output text)"
log "Using SageMaker role ARN: $ROLE_ARN"

# Build and push Docker image for training if IMAGE_URI not provided
if [ -z "$IMAGE_URI" ]; then
  log "No image URI provided. Building and pushing Docker image to ECR repository $ECR_REPO"
  # Get AWS account ID
  ACCOUNT_ID=$(aws sts get-caller-identity --profile "$PROFILE" --region "$REGION" --query 'Account' --output text)
  # Create ECR repository if it does not exist
  if run_aws ecr describe-repositories --repository-names "$ECR_REPO" > /dev/null 2>&1; then
    log "ECR repository $ECR_REPO already exists"
  else
    run_aws ecr create-repository --repository-name "$ECR_REPO"
  fi
  # Build Docker image
  log "Building Docker image $ECR_REPO"
  docker build -t "$ECR_REPO" -f "$PWD/Dockerfile" "$PROJECT_ROOT"
  # Authenticate to ECR
  log "Logging in to ECR"
  run_aws ecr get-login-password | docker login --username AWS --password-stdin "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"
  # Tag and push image
  IMAGE_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO}:latest"
  docker tag "${ECR_REPO}:latest" "$IMAGE_URI"
  log "Pushing Docker image to $IMAGE_URI"
  docker push "$IMAGE_URI"
  log "Built and pushed Docker image to $IMAGE_URI"
fi

# Prepare image base for fold-specific tagging
IMAGE_BASE="${IMAGE_URI%:*}"

## Launch training jobs for each fold
## Prepare input data configuration as JSON for AWS CLI
TMP_INPUT_CONFIG_FILE=$(mktemp)
cat <<EOF >"$TMP_INPUT_CONFIG_FILE"
[{"ChannelName":"training","DataSource":{"S3DataSource":{"S3DataType":"S3Prefix","S3Uri":"${TRAIN_DATA_URI}","S3DataDistributionType":"FullyReplicated"}},"ContentType":"application/x-parquet","InputMode":"File"}]
EOF
## Determine list of folds to run
# Determine list of folds to run; use sentinel -1 for full-dataset mode when no cross-validation
# Construct the list of folds to run. `-1` is a sentinel meaning
# "disable cross-validation and train on the full dataset".
if [ "${NO_CV}" = "true" ]; then
  FOLD_ARRAY=(-1)
elif [ -n "${FOLDS_LIST}" ]; then
  IFS=',' read -r -a FOLD_ARRAY <<< "${FOLDS_LIST}"
else
  FOLD_ARRAY=($(seq 1 "${FOLDS}"))
fi
# Prepare human-readable labels for folds (use 'full-dataset' for no-cross-validation sentinel)
FOLD_LABELS=()
for f in "${FOLD_ARRAY[@]}"; do
  if [ "$f" -lt 0 ]; then
    FOLD_LABELS+=("full-dataset")
  else
    FOLD_LABELS+=("$f")
  fi
done
log "Launching training jobs for folds ${FOLD_LABELS[*]}"
declare -a JOB_NAMES=()
declare -a FOLD_LIST=()
for fold in "${FOLD_ARRAY[@]}"; do
  # Derive human-readable fold label for logging
  if [ "$fold" -lt 0 ]; then
    fold_label="full-dataset"
  else
    fold_label="$fold"
  fi
  # Define a timestamp-based job name for this fold
  TIMESTAMP=$(date +%Y%m%d%H%M%S)
  JOB_NAME="${JOB_BASE_PREFIX}-${TIMESTAMP}-${fold}"
  # Record job name and fold for metrics retrieval
  JOB_NAMES+=("$JOB_NAME")
  FOLD_LIST+=("$fold")
  log "Launching training job $JOB_NAME (fold $fold_label)"
  # Use the same container image for all folds
  log "Using container image $IMAGE_URI for fold $fold_label"
  # Prepare checkpoint config and environment variables for fine-tuning
  ENV_VARS="SAGEMAKER_PROGRAM=${TRAIN_SCRIPT_ENTRYPOINT},SAGEMAKER_SUBMIT_DIRECTORY=\"$S3_PREFIX/code.tar.gz\",OUTPUT_S3_URI=\"$S3_PREFIX/$JOB_NAME\""
  if [ -n "$ARTIFACT_URI" ]; then
    # Download and extract the checkpoint tar.gz, then upload model.pth as the fine-tuning checkpoint
    log "Downloading fine-tuning artifact tarball from $ARTIFACT_URI"
    TMP_ARTIFACT_TAR=$(mktemp)
    TMP_ARTIFACT_DIR=$(mktemp -d)
    run_aws s3 cp "$ARTIFACT_URI" "$TMP_ARTIFACT_TAR"
    tar -xzf "$TMP_ARTIFACT_TAR" -C "$TMP_ARTIFACT_DIR"
    MODEL_FILE=$(find "$TMP_ARTIFACT_DIR" -type f -name 'model.pth' | head -n 1)
    if [ -z "$MODEL_FILE" ]; then
      log "Error: model.pth not found in the artifact tarball"
      exit 1
    fi
    # Upload checkpoint into a job-specific fine-tuning folder to avoid collisions
    FINE_TUNE_S3_URI="${S3_PREFIX}/fine-tuning/${JOB_NAME}/checkpoint.pth"
    log "Uploading extracted checkpoint to S3 for fine-tuning: $FINE_TUNE_S3_URI"
    run_aws s3 cp "$MODEL_FILE" "$FINE_TUNE_S3_URI"
    ARTIFACT_URI="$FINE_TUNE_S3_URI"
    log "Using fine-tuning checkpoint S3 URI: $ARTIFACT_URI"
    rm -rf "$TMP_ARTIFACT_TAR" "$TMP_ARTIFACT_DIR"

    # Configure checkpoint SDK to use the folder containing the uploaded file, not the file itself
    CHECKPOINT_S3_PREFIX="${ARTIFACT_URI%/*}"
    CHECKPOINT_CONFIG_ARGS=(--checkpoint-config "S3Uri=$CHECKPOINT_S3_PREFIX,LocalPath=/opt/ml/checkpoints")
    ENV_VARS="$ENV_VARS,ARTIFACT_URI=$ARTIFACT_URI"
  else
    CHECKPOINT_CONFIG_ARGS=()
  fi
  # Assemble the SageMaker hyper-parameter string. Using a single CSV-style
  # value keeps the AWS CLI invocation readable while still allowing us to
  # thread through additional knobs from the CLI (seed, rehearsal, ...).
  MODEL_HP="fold_actual=\"$fold\""
  MODEL_HP+=",model-config=\"$MODEL_CONFIG\""
  MODEL_HP+=",normalization-mode=\"$CONFIG_NORMALIZATION_MODE\""
  if [ -n "$SEED" ]; then
    MODEL_HP+=",seed=\"$SEED\""
  fi
  if [ -n "${REHEARSAL_DATA_URI:-}" ]; then
    MODEL_HP+=",rehearsal-data-path=\"$REHEARSAL_DATA_URI\""
  fi
  if [ -n "${REHEARSAL_TARGET_SPEED_COL:-}" ]; then
    MODEL_HP+=",rehearsal-target-speed-col=\"$REHEARSAL_TARGET_SPEED_COL\""
  fi
  if [ -n "${REHEARSAL_TARGET_DIR_COL:-}" ]; then
    MODEL_HP+=",rehearsal-target-dir-col=\"$REHEARSAL_TARGET_DIR_COL\""
  fi
  if [ -n "${REHEARSAL_FRACTION:-}" ]; then
    MODEL_HP+=",rehearsal-fraction=\"$REHEARSAL_FRACTION\""
  fi
  run_aws sagemaker create-training-job \
    --training-job-name "$JOB_NAME" \
    --algorithm-specification TrainingImage="$IMAGE_URI",TrainingInputMode=File \
    --role-arn "$ROLE_ARN" \
    --input-data-config file://"$TMP_INPUT_CONFIG_FILE" \
    --output-data-config S3OutputPath="$S3_PREFIX" \
    --resource-config InstanceType="$INSTANCE_TYPE",InstanceCount=1,VolumeSizeInGB="$VOLUME_SIZE" \
    --stopping-condition MaxRuntimeInSeconds="$MAX_RUNTIME" \
    --hyper-parameters "$MODEL_HP" \
    "${CHECKPOINT_CONFIG_ARGS[@]:-}" \
    --environment "$ENV_VARS"
done

# Optionally wait for jobs and collect metrics
if [ "$WAIT_FOR_JOBS" = "true" ]; then
  log "Waiting for all training jobs to complete"
  # Wait for each job sequentially so that the logs for a failure are harvested
  # even if a later job succeeds.
  for JOB_NAME in "${JOB_NAMES[@]}"; do
    wait_for_training_job "$JOB_NAME"
  done

  log "All training jobs completed. Collecting metrics CSVs..."
  mkdir -p "${OUTPUT_DIR}/metrics_results"
  total_rmse_speed=0
  total_mae_speed=0
  total_corr_speed=0
  total_r2_speed=0
  total_bias_speed=0
  total_si_speed=0
  total_si_speed_max=0
  total_eam_dir=0
  total_eaam_dir=0
  total_rmse_dir=0
  total_compcorr_dir=0
  total_combined_loss=0
  count=0

  # Collect metrics CSVs from each fold's output path
    # Aggregate validation metrics from each fold so that the Markdown report can
    # quote both per-fold CSVs and cross-fold averages.
    for idx in "${!JOB_NAMES[@]}"; do
    JOB_NAME="${JOB_NAMES[$idx]}"
    fold="${FOLD_LIST[$idx]}"
    METRICS_FILE="metrics_fold${fold}.csv"
    # Download and extract output.tar.gz to retrieve metrics CSV for this job
    TAR_S3_PATH="$S3_PREFIX/$JOB_NAME/output/output.tar.gz"
    LOCAL_TAR="${OUTPUT_DIR}/${JOB_NAME}_output.tar.gz"
    if run_aws s3 cp "$TAR_S3_PATH" "$LOCAL_TAR"; then
      log "Extracting metrics CSVs and metadata from $LOCAL_TAR for job $JOB_NAME"
      # Ensure metrics_results folder for aggregating CSVs
      mkdir -p "${OUTPUT_DIR}/metrics_results"
      # Extract all per-fold CSVs into metrics_results
      tar -xzf "$LOCAL_TAR" -C "${OUTPUT_DIR}/metrics_results" 2>/dev/null || true
      # Extract normalization and script_args JSON into output root
      tar -xzf "$LOCAL_TAR" --strip-components=0 -C "${OUTPUT_DIR}" normalization_params.json script_args.json 2>/dev/null || true

      if [ -f "${OUTPUT_DIR}/metrics_results/$METRICS_FILE" ]; then
        # Extract metrics values from CSV (skip header)
        # Read metrics values from CSV, skipping the new 'fold' first column
        IFS=',' read -r _ rmse_speed_val mae_speed_val corr_speed_val r2_speed_val bias_speed_val si_speed_val eam_dir_val eaam_dir_val rmse_dir_val compcorr_dir_val si_speed_max_val combined_loss_val < <(awk -F, 'NR>1{print; exit}' "${OUTPUT_DIR}/metrics_results/$METRICS_FILE")
        total_rmse_speed=$(echo "${total_rmse_speed:-0} + ${rmse_speed_val:-0}" | bc)
        total_mae_speed=$(echo "${total_mae_speed:-0} + ${mae_speed_val:-0}" | bc)
        total_corr_speed=$(echo "${total_corr_speed:-0} + ${corr_speed_val:-0}" | bc)
        total_r2_speed=$(echo "${total_r2_speed:-0} + ${r2_speed_val:-0}" | bc)
        total_bias_speed=$(echo "${total_bias_speed:-0} + ${bias_speed_val:-0}" | bc)
        total_si_speed=$(echo "${total_si_speed:-0} + ${si_speed_val:-0}" | bc)
        total_si_speed_max=$(echo "${total_si_speed_max:-0} + ${si_speed_max_val:-0}" | bc)
        total_eaam_dir=$(echo "${total_eaam_dir:-0} + ${eaam_dir_val:-0}" | bc)
        total_eam_dir=$(echo "${total_eam_dir:-0} + ${eam_dir_val:-0}" | bc)
        total_rmse_dir=$(echo "${total_rmse_dir:-0} + ${rmse_dir_val:-0}" | bc)
        total_compcorr_dir=$(echo "${total_compcorr_dir:-0} + ${compcorr_dir_val:-0}" | bc)
        total_combined_loss=$(echo "${total_combined_loss:-0} + ${combined_loss_val:-0}" | bc)
        count=$((count+1))
      else
        log "Metrics file not found in extracted tarball for job $JOB_NAME"
      fi
    else
      log "output.tar.gz not found for job $JOB_NAME"
    fi
  done

  if [ $count -gt 0 ]; then
    avg_rmse_speed=$(echo "scale=4; $total_rmse_speed/$count" | bc)
    avg_mae_speed=$(echo "scale=4; $total_mae_speed/$count" | bc)
    avg_corr_speed=$(echo "scale=4; $total_corr_speed/$count" | bc)
    avg_r2_speed=$(echo "scale=4; $total_r2_speed/$count" | bc)
    avg_bias_speed=$(echo "scale=4; $total_bias_speed/$count" | bc)
    avg_si_speed=$(echo "scale=4; $total_si_speed/$count" | bc)
    avg_si_speed_max=$(echo "scale=4; $total_si_speed_max/$count" | bc)
    avg_eaam_dir=$(echo "scale=4; $total_eaam_dir/$count" | bc)
    avg_eam_dir=$(echo "scale=4; $total_eam_dir/$count" | bc)
    avg_rmse_dir=$(echo "scale=4; $total_rmse_dir/$count" | bc)
    avg_compcorr_dir=$(echo "scale=4; $total_compcorr_dir/$count" | bc)
    avg_combined_loss=$(echo "scale=4; $total_combined_loss/$count" | bc)
    log "Average RMSE Speed across $count folds: $avg_rmse_speed"
    log "Average MAE Speed across $count folds: $avg_mae_speed"
    log "Average Corr Speed across $count folds: $avg_corr_speed"
    log "Average R2 Speed across $count folds: $avg_r2_speed"
    log "Average Bias Speed across $count folds: $avg_bias_speed"
    log "Average SI Speed across $count folds: $avg_si_speed"
    log "Average SI Speed Max across $count folds: $avg_si_speed_max"
    log "Average EAM Dir across $count folds: $avg_eam_dir"
    log "Average EAAM Dir across $count folds: $avg_eaam_dir"
    log "Average RMSE Dir across $count folds: $avg_rmse_dir"
    log "Average CompCorr Dir across $count folds: $avg_compcorr_dir"
    log "Average Combined Loss across $count folds: $avg_combined_loss"
  else
    log "No metrics files found for any fold"
  fi
fi

# Generate Markdown report for folds metrics and global metrics
        REPORT_FILE="${OUTPUT_DIR}/${JOB_BASE_NAME}_report.md"
        log "Generating Markdown report $REPORT_FILE"
        {
          echo "# Report for ${JOB_BASE_NAME}"
          echo ""
          echo "## Fold metrics"
          echo ""
          echo "| Fold | RMSE Speed | MAE Speed | Corr Speed | R2 Speed | Bias Speed | SI Speed | EAM Dir | EAAM Dir | RMSE Dir | CompCorr Dir | SI Speed Max | Combined Loss |"
          echo "| ---- | ---------- | --------- | ---------- | --------- | ---------- | -------- | ------- | ------- | -------- | ------------ | ------------- | ------------- |"
          for idx in "${!JOB_NAMES[@]}"; do
            fold="${FOLD_LIST[$idx]}"
            metrics_file="${OUTPUT_DIR}/metrics_results/metrics_fold${fold}.csv"
      if [ -f "$metrics_file" ]; then
              # Extract metrics values for report (including SI_speed_max)
              # Read per-fold metrics row, skipping the 'fold' first column
              IFS=',' read -r _ rmse_speed_val mae_speed_val corr_speed_val r2_speed_val bias_speed_val si_speed_val eam_dir_val eaam_dir_val rmse_dir_val compcorr_dir_val si_speed_max_val combined_loss_val < <(awk -F, 'NR>1{print;exit}' "$metrics_file")
              echo "| $fold | $rmse_speed_val | $mae_speed_val | $corr_speed_val | $r2_speed_val | $bias_speed_val | $si_speed_val | $eam_dir_val | $eaam_dir_val | $rmse_dir_val | $compcorr_dir_val | $si_speed_max_val | $combined_loss_val |"
            fi
          done
          echo ""
          echo "## Global metrics"
          echo ""
          echo "| Metric | Value |"
          echo "| ------ | ----- |"
          echo "| Average RMSE Speed | ${avg_rmse_speed:-N/A} |"
          echo "| Average MAE Speed | ${avg_mae_speed:-N/A} |"
          echo "| Average Corr Speed | ${avg_corr_speed:-N/A} |"
          echo "| Average R2 Speed | ${avg_r2_speed:-N/A} |"
          echo "| Average Bias Speed | ${avg_bias_speed:-N/A} |"
          echo "| Average SI Speed | ${avg_si_speed:-N/A} |"
          echo "| Average SI Speed Max | ${avg_si_speed_max:-N/A} |"
          echo "| Average EAM Dir | ${avg_eam_dir:-N/A} |"
          echo "| Average EAAM Dir | ${avg_eaam_dir:-N/A} |"
          echo "| Average RMSE Dir | ${avg_rmse_dir:-N/A} |"
          echo "| Average CompCorr Dir | ${avg_compcorr_dir:-N/A} |"
          echo "| Average Combined Loss | ${avg_combined_loss:-N/A} |"

          echo ""
          echo "## Error metrics by wind bin"
          echo ""
BIN_CSV=$(ls "${OUTPUT_DIR}/metrics_results/metrics_fold"*"_by_wind_bin.csv" 2>/dev/null \
         | grep -v "_by_${ID_COL}_by_wind_bin.csv" \
         | head -n1 || true)
          if [ -f "$BIN_CSV" ]; then
            head -n1 "$BIN_CSV" | sed 's/,/ | /g; s/^/| /; s/$/ |/'
            head -n1 "$BIN_CSV" | sed 's/[^,]*/----/g; s/,/|/g; s/^/| /; s/$/ |/'
            tail -n +2 "$BIN_CSV" | sed 's/,/ | /g; s/^/| /; s/$/ |/'
          else
            echo "_No wind bin metrics found._"
          fi

          if [ -n "$ID_COL" ]; then
  # Metrics by location ID for validation
  echo ""
  echo "## Error metrics by ${ID_COL}"
  echo ""
  ID_CSV=$(ls "${OUTPUT_DIR}/metrics_results/metrics_fold"*"_by_${ID_COL}.csv" 2>/dev/null | head -n1 || true)
  if [ -f "$ID_CSV" ]; then
    head -n1 "$ID_CSV" | sed 's/,/ | /g; s/^/| /; s/$/ |/'
    head -n1 "$ID_CSV" | sed 's/[^,]*/----/g; s/,/|/g; s/^/| /; s/$/ |/'
    tail -n +2 "$ID_CSV" | sed 's/,/ | /g; s/^/| /; s/$/ |/'
  else
    echo "_No ${ID_COL} metrics found._"
  fi

  echo ""
  echo "## Error metrics by ${ID_COL} and wind bin"
  echo ""
  ID_BIN_CSV=$(ls "${OUTPUT_DIR}/metrics_results/metrics_fold"*"_by_${ID_COL}_by_wind_bin.csv" 2>/dev/null | head -n1 || true)
  if [ -f "$ID_BIN_CSV" ]; then
    head -n1 "$ID_BIN_CSV" | sed 's/,/ | /g; s/^/| /; s/$/ |/'
    head -n1 "$ID_BIN_CSV" | sed 's/[^,]*/----/g; s/,/|/g; s/^/| /; s/$/ |/'
    tail -n +2 "$ID_BIN_CSV" | sed 's/,/ | /g; s/^/| /; s/$/ |/'
  else
    echo "_No ${ID_COL} and wind bin metrics found._"
  fi

          fi

          echo ""
          echo "## Training error metrics"
echo ""
TRAIN_CSV=$(ls "${OUTPUT_DIR}/metrics_results/"*metrics_train*.*csv 2>/dev/null \
             | grep -v "by_wind_bin" \
             | grep -v "_by_${ID_COL}.csv" \
             | head -n1 || true)
if [ -f "$TRAIN_CSV" ]; then
  head -n1 "$TRAIN_CSV" | sed 's/,/ | /g; s/^/| /; s/$/ |/'
  head -n1 "$TRAIN_CSV" | sed 's/[^,]*/----/g; s/,/|/g; s/^/| /; s/$/ |/'
  tail -n +2 "$TRAIN_CSV" | sed 's/,/ | /g; s/^/| /; s/$/ |/'
else
  echo "_No training metrics found._"
fi

echo ""
echo "## Training metrics by wind bin"
echo ""
TRAIN_BIN_CSV=$(ls "${OUTPUT_DIR}/metrics_results/metrics_train_fold"*"_by_wind_bin.csv" 2>/dev/null \
               | grep -v "_by_${ID_COL}_by_wind_bin.csv" \
               | head -n1 || true)
if [ -f "$TRAIN_BIN_CSV" ]; then
  head -n1 "$TRAIN_BIN_CSV" | sed 's/,/ | /g; s/^/| /; s/$/ |/'
  head -n1 "$TRAIN_BIN_CSV" | sed 's/[^,]*/----/g; s/,/|/g; s/^/| /; s/$/ |/'
  tail -n +2 "$TRAIN_BIN_CSV" | sed 's/,/ | /g; s/^/| /; s/$/ |/'
          else
            echo "_No training bin metrics found._"
          fi

          if [ -n "$ID_COL" ]; then
          echo ""
          echo "## Training metrics by ${ID_COL}"
echo ""
TRAIN_ID_CSV=$(ls "${OUTPUT_DIR}/metrics_results/metrics_train_fold"*"_by_${ID_COL}.csv" 2>/dev/null | head -n1 || true)
if [ -f "$TRAIN_ID_CSV" ]; then
  head -n1 "$TRAIN_ID_CSV" | sed 's/,/ | /g; s/^/| /; s/$/ |/'
  head -n1 "$TRAIN_ID_CSV" | sed 's/[^,]*/----/g; s/,/|/g; s/^/| /; s/$/ |/'
  tail -n +2 "$TRAIN_ID_CSV" | sed 's/,/ | /g; s/^/| /; s/$/ |/'
else
  echo "_No training ${ID_COL} metrics found._"
fi

echo ""
echo "## Training metrics by ${ID_COL} and wind bin"
echo ""
TRAIN_ID_BIN_CSV=$(ls "${OUTPUT_DIR}/metrics_results/metrics_train_fold"*"_by_${ID_COL}_by_wind_bin.csv" 2>/dev/null | head -n1 || true)
if [ -f "$TRAIN_ID_BIN_CSV" ]; then
  head -n1 "$TRAIN_ID_BIN_CSV" | sed 's/,/ | /g; s/^/| /; s/$/ |/'
  head -n1 "$TRAIN_ID_BIN_CSV" | sed 's/[^,]*/----/g; s/,/|/g; s/^/| /; s/$/ |/'
  tail -n +2 "$TRAIN_ID_BIN_CSV" | sed 's/,/ | /g; s/^/| /; s/$/ |/'
else
  echo "_No training ${ID_COL} and wind bin metrics found._"

          fi

          fi

          echo ""
          echo "## Normalization parameters"
          echo ""
          if [ -f "${OUTPUT_DIR}/normalization_params.json" ]; then
            echo '```json'
            cat "${OUTPUT_DIR}/normalization_params.json"
            echo '```'
          else
            echo "_normalization_params.json not found._"
          fi

          echo ""
          echo "## Script arguments"
          echo ""
          if [ -f "${OUTPUT_DIR}/script_args.json" ]; then
            echo '```json'
            cat "${OUTPUT_DIR}/script_args.json"
            echo '```'
          else
            echo "_script_args.json not found._"
          fi
        } > "$REPORT_FILE"
        log "Markdown report generated: $REPORT_FILE"

        log "run_all_folds.sh completed successfully"
