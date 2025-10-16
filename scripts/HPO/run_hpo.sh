#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Author: Juan Luis Herrera Cortijo <juan.luis.herrera.cortijo@gmail.com>
# Created: 2025-10-16
# Disclaimer: This script is provided as-is for research workflows and must be validated before any operational deployment.
# -----------------------------------------------------------------------------

# Launch an AWS SageMaker Hyperparameter Tuning Job with automated image, IAM,
# and reporting orchestration tailored to the HF radar wind inversion project.
set -euo pipefail
# Directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Ensure jq is available before attempting to parse JSON config files.
ensure_jq_available() {
  if ! command -v jq >/dev/null 2>&1; then
    echo "Error: 'jq' must be installed to parse JSON configuration files." >&2
    exit 1
  fi
}

# Extract static hyperparameters from the model configuration JSON.
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

# Convert the tuning search space JSON into environment-style key/value pairs.
parse_hpo_config() {
  local config_path="$1"
  jq -r '
    [
      "HPO_STRATEGY=" + ((.strategy // "Bayesian") | tojson),
      "HPO_MAX_JOBS=" + ((.max_training_jobs // 50) | tostring),
      "HPO_MAX_PARALLEL=" + ((.max_parallel_jobs // 3) | tostring),
      "HPO_OBJECTIVE=" + ((.objective_metric // "combined_loss") | tojson),
      "HPO_PARAM_RANGES=" + ({
        "ContinuousParameterRanges": ((.ranges.continuous // {})
          | to_entries
          | map({
              "Name": .key,
              "MinValue": (.value.min | tostring),
              "MaxValue": (.value.max | tostring)
            } + ((.value.scaling | select(. != null) | {"ScalingType": .}) // {}))),
        "IntegerParameterRanges": ((.ranges.integer // {})
          | to_entries
          | map({
              "Name": .key,
              "MinValue": (.value.min | tostring),
              "MaxValue": (.value.max | tostring)
            } + ((.value.scaling | select(. != null) | {"ScalingType": .}) // {}))),
        "CategoricalParameterRanges": ((.ranges.categorical // {})
          | to_entries
          | map({
              "Name": .key,
              "Values": ((.value.values // .value // []) | map(tostring))
            }))
      } | tojson)
    ] | .[]
  ' "$config_path"
}

# Check whether a parameter name is already defined inside the HPO ranges blob.
hpo_ranges_include_param() {
  local ranges_json="$1"
  local param_name="$2"
  jq -e --arg name "$param_name" '
    ((.ContinuousParameterRanges // []) +
     (.IntegerParameterRanges // []) +
     (.CategoricalParameterRanges // []))
    | map(select(.Name == $name))
    | length > 0
  ' <<<"$ranges_json" >/dev/null 2>&1
}

# Script to launch a SageMaker Hyperparameter Tuning job (HPO)

# Render help text detailing the CLI interface.
usage() {
  cat <<EOF >&2
Usage: $0 \
  --job-name <job_name> \
  --train-data-uri <s3_uri> \
  --output-s3-uri <s3_uri> \
  --model-config <path> \
  [options]

Required:
  --job-name NAME             Name for the HPO job
  --train-data-uri S3URI      S3 URI (file or prefix) for the GeoParquet training dataset (e.g., s3://bucket/data/)
  --output-s3-uri S3URI       S3 URI prefix for tuning output (e.g., s3://bucket/output/)
  --model-config PATH         Path to the JSON model definition (stations and hyperparameters)

Options:
  --profile PROFILE           AWS CLI profile (default: ${AWS_PROFILE:-default})
  --region REGION             AWS region (default: ${AWS_REGION:-us-east-1})
  --parent-jobs LIST          Comma-separated list of previous HPO job names for warm start (max 5)
  --log-dir DIR               Local directory to write logs (default: .)
  --update-config true|false  Regenerate the HPO configuration file (default: true)
  --config-file NAME          Name of the generated HPO configuration JSON file (default: <script>-config.json)
  --hpo-config PATH           Path to the HPO search-space definition (required when regenerating the config)
  --ecr-repo NAME             ECR repository name to use for the HPO image (default: hpo_repo)
  --help                      Show this help message and exit
EOF
  exit 1
}

## Default AWS CLI settings
PROFILE="${AWS_PROFILE:-default}"
REGION="${AWS_REGION:-us-east-1}"

# Default local output directory for logs
OUTPUT_DIR="."

# Static defaults populated from model config when provided
STATIONS=""
MODEL_CONFIG=""
HPO_CONFIG=""
USE_VELOCITY_MEDIAN="0"
EARLY_STOPPING="1"
SAVE_ERROR_DATA="0"
# Default ECR repository name (unless overridden)
ECR_REPO="${ECR_REPO:-hpo_repo}"
# Whether to update the HPO configuration file (regenerate if true, or if file not present)
UPDATE_CONFIG="true"

# Parse options using long-form flags only
JOB_NAME=""
PREVIOUS_JOBS_STRING=""
TRAIN_DATA_URI=""
OUTPUT_DATA_URI=""
IMAGE_URI=""
ROLE_ARN=""
# Parse long-form CLI arguments, storing unknown tokens for validation.
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --job-name)
      JOB_NAME="$2"; shift 2 ;;
    --job-name=*)
      JOB_NAME="${1#*=}"; shift ;;
    --train-data-uri)
      TRAIN_DATA_URI="$2"; shift 2 ;;
    --train-data-uri=*)
      TRAIN_DATA_URI="${1#*=}"; shift ;;
    --output-s3-uri)
      OUTPUT_DATA_URI="$2"; shift 2 ;;
    --output-s3-uri=*)
      OUTPUT_DATA_URI="${1#*=}"; shift ;;
    --profile)
      PROFILE="$2"; shift 2 ;;
    --profile=*)
      PROFILE="${1#*=}"; shift ;;
    --region)
      REGION="$2"; shift 2 ;;
    --region=*)
      REGION="${1#*=}"; shift ;;
    --parent-jobs)
      PREVIOUS_JOBS_STRING="$2"; shift 2 ;;
    --parent-jobs=*)
      PREVIOUS_JOBS_STRING="${1#*=}"; shift ;;
    --log-dir)
      OUTPUT_DIR="$2"; shift 2 ;;
    --log-dir=*)
      OUTPUT_DIR="${1#*=}"; shift ;;
    --model-config)
      MODEL_CONFIG="$2"; shift 2 ;;
    --model-config=*)
      MODEL_CONFIG="${1#*=}"; shift ;;
    --hpo-config)
      HPO_CONFIG="$2"; shift 2 ;;
    --hpo-config=*)
      HPO_CONFIG="${1#*=}"; shift ;;
    --ecr-repo)
      ECR_REPO="$2"; shift 2 ;;
    --ecr-repo=*)
      ECR_REPO="${1#*=}"; shift ;;
    --config-file)
      CONFIG_FILE_NAME="$2"; shift 2 ;;
    --config-file=*)
      CONFIG_FILE_NAME="${1#*=}"; shift ;;
    --update-config)
      UPDATE_CONFIG="$2"; shift 2 ;;
    --update-config=*)
      UPDATE_CONFIG="${1#*=}"; shift ;;
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

# The launcher requires a model configuration to seed static hyperparameters.
if [[ -z "$MODEL_CONFIG" ]]; then
  echo "Error: --model-config is required for HPO runs." >&2
  usage
fi

MODEL_CONFIG_ORIG="$MODEL_CONFIG"
if [[ "$MODEL_CONFIG" != /* ]]; then
  MODEL_CONFIG_PATH="${PROJECT_ROOT}/${MODEL_CONFIG}"
else
  MODEL_CONFIG_PATH="$MODEL_CONFIG"
fi
if [[ ! -f "$MODEL_CONFIG_PATH" ]]; then
  echo "Model config file not found: $MODEL_CONFIG_PATH" >&2
  exit 1
fi
ensure_jq_available
PARSED_VARS=$(parse_model_config "$MODEL_CONFIG_PATH")
STATIONS=""
TARGET_SPEED_COL="wind_speed"
TARGET_DIR_COL="wind_direction"
AGG_STAT="mean"
ID_COL="location_id"
USE_VELOCITY_MEDIAN="0"
EARLY_STOPPING="1"
SAVE_ERROR_DATA="0"
while IFS='=' read -r key value; do
  case "$key" in
    stations) STATIONS="$value" ;;
    target_speed_col) TARGET_SPEED_COL="$value" ;;
    target_dir_col) TARGET_DIR_COL="$value" ;;
    agg_stat) AGG_STAT="$value" ;;
    id_col) ID_COL="$value" ;;
    use_velocity_median) USE_VELOCITY_MEDIAN="$value" ;;
    early_stopping) EARLY_STOPPING="$value" ;;
    save_error_data) SAVE_ERROR_DATA="$value" ;;
  esac
done <<< "$PARSED_VARS"
MODEL_CONFIG="$MODEL_CONFIG_ORIG"

# Logging setup: initialize log file in OUTPUT_DIR for reproducible traces.
SCRIPT_NAME=$(basename "$0" .sh)
mkdir -p "$OUTPUT_DIR"
LOG_FILE="${OUTPUT_DIR}/${SCRIPT_NAME}.log"
CONFIG_FILE_NAME="${CONFIG_FILE_NAME:-${SCRIPT_NAME}-config.json}"
rm -f "$LOG_FILE"
log() { echo "$(date '+%Y-%m-%d %H:%M:%S') - $*" | tee -a "$LOG_FILE"; }

# Function to run AWS CLI commands with logging
run_aws() {
  # Mirror the executed AWS CLI command inside the log for auditability.
  echo "Running: aws $*" >> "$LOG_FILE"
  aws --profile "$PROFILE" --region "$REGION" "$@" >> "$LOG_FILE" 2>&1
}

# Poll IAM until the newly created/updated role surfaces with the required policies.
wait_for_role_propagation() {
  local role_name="$1"
  shift
  local expected_policies=("$@")
  local max_attempts="${IAM_PROPAGATION_MAX_ATTEMPTS:-20}"
  local sleep_seconds="${IAM_PROPAGATION_SLEEP_SECONDS:-5}"
  local attempt=1

  while (( attempt <= max_attempts )); do
    set +e
    aws --profile "$PROFILE" --region "$REGION" iam get-role --role-name "$role_name" >/dev/null 2>&1
    local get_role_status=$?
    local attached=""
    local list_status=1
    if [[ $get_role_status -eq 0 ]]; then
      attached=$(aws --profile "$PROFILE" --region "$REGION" iam list-attached-role-policies \
        --role-name "$role_name" \
        --query 'AttachedPolicies[].PolicyArn' \
        --output text 2>/dev/null)
      list_status=$?
    fi
    set -e

    local ready=true
    if [[ $get_role_status -ne 0 || $list_status -ne 0 ]]; then
      ready=false
    else
      for policy_arn in "${expected_policies[@]}"; do
        if [[ -z "$policy_arn" ]]; then
          continue
        fi
        if ! grep -Fq "$policy_arn" <<<"$attached"; then
          ready=false
          break
        fi
      done
    fi

    if [[ "$ready" == true ]]; then
      log "Confirmed IAM role $role_name and required policies are available (attempt $attempt)"
      return 0
    fi

    if (( attempt == max_attempts )); then
      log "Warning: IAM role $role_name not confirmed after $max_attempts attempts"
      return 1
    fi

    log "IAM role $role_name not ready yet (attempt $attempt/$max_attempts); retrying in $sleep_seconds seconds"
    sleep "$sleep_seconds"
    ((attempt++))
  done
}

# Simulate IAM policy decisions to verify S3 access before submitting the job.
wait_for_s3_access() {
  local role_arn="$1"
  local s3_uri="$2"
  local access_mode="$3"
  local max_attempts="${IAM_S3_PROPAGATION_MAX_ATTEMPTS:-20}"
  local sleep_seconds="${IAM_S3_PROPAGATION_SLEEP_SECONDS:-6}"
  local attempt=1

  if [[ -z "$s3_uri" ]]; then
    return 0
  fi

  local without_scheme="${s3_uri#s3://}"
  local bucket="${without_scheme%%/*}"
  if [[ -z "$bucket" || "$bucket" == "$s3_uri" ]]; then
    log "Warning: unable to parse bucket from S3 URI '$s3_uri'; skipping IAM access verification"
    return 0
  fi

  local bucket_resource="arn:aws:s3:::$bucket"
  local object_resource="arn:aws:s3:::$bucket/*"

  while (( attempt <= max_attempts )); do
    local all_allowed=true
    local actions=("s3:ListBucket")
    local resources=("$bucket_resource")

    case "$access_mode" in
      read)
        actions+=("s3:GetObject")
        resources+=("$object_resource")
        ;;
      write)
        actions+=("s3:PutObject")
        resources+=("$object_resource")
        ;;
    esac

    for idx in "${!actions[@]}"; do
      local action="${actions[$idx]}"
      local resource="${resources[$idx]}"
      set +e
      local decision
      decision=$(aws --profile "$PROFILE" --region "$REGION" iam simulate-principal-policy \
        --policy-source-arn "$role_arn" \
        --action-names "$action" \
        --resource-arns "$resource" \
        --query 'EvaluationResults[0].EvalDecision' \
        --output text 2>/dev/null)
      local status=$?
      set -e

      if [[ $status -ne 0 ]]; then
        log "Warning: failed to verify IAM access for action $action on $resource (attempt $attempt)"
        all_allowed=false
        break
      fi

      if [[ "$decision" != "allowed" ]]; then
        all_allowed=false
        break
      fi
    done

    if [[ "$all_allowed" == true ]]; then
      log "Confirmed IAM role permissions for $access_mode access to $s3_uri (attempt $attempt)"
      return 0
    fi

    if (( attempt == max_attempts )); then
      log "Warning: IAM role permissions for $access_mode access to $s3_uri not confirmed after $max_attempts attempts"
      return 1
    fi

    log "Waiting for IAM permissions for $access_mode access to $s3_uri (attempt $attempt/$max_attempts); retrying in $sleep_seconds seconds"
    sleep "$sleep_seconds"
    ((attempt++))
  done
}

# Check required parameters (role ARN now auto-provisioned)
if [[ -z "$JOB_NAME" || -z "$TRAIN_DATA_URI" || -z "$OUTPUT_DATA_URI" || -z "$STATIONS" ]]; then
  usage
fi
# Sanitize JOB_NAME for AWS HPO constraints (alphanumeric and hyphens only, max 32 chars)
ORIG_JOB_NAME="$JOB_NAME"
JOB_NAME=$(echo "$JOB_NAME" | sed 's/[^A-Za-z0-9-]/-/g' | sed -E 's/^-+|-+$//g')
JOB_NAME=${JOB_NAME:0:32}
# Sanitize JOB_NAME
if [[ "$JOB_NAME" != "$ORIG_JOB_NAME" ]]; then
  log "Sanitized job name from '$ORIG_JOB_NAME' to '$JOB_NAME' for AWS compatibility"
fi

# Prevent overwriting existing HPO jobs
if aws --profile "$PROFILE" --region "$REGION" \
    sagemaker describe-hyper-parameter-tuning-job \
    --hyper-parameter-tuning-job-name "$JOB_NAME" > /dev/null 2>&1; then
  echo "Error: HPO job '$JOB_NAME' already exists. Aborting to avoid overwrite." >&2
  exit 1
fi

# Build and push Docker image if IMAGE_URI not provided
if [[ -z "$IMAGE_URI" ]]; then
  echo "Building and pushing Docker image for HPO..."
  # Get AWS account ID (use direct AWS CLI to capture output)
  ACCOUNT_ID=$(aws --profile "$PROFILE" --region "$REGION" sts get-caller-identity --query 'Account' --output text)
  # The account identifier is needed to tag the training image and reference IAM roles.
  log "AWS Account ID: $ACCOUNT_ID"
  if [[ -z "$ACCOUNT_ID" ]]; then
    log "Error: AWS Account ID is empty. Exiting."
    exit 1
  fi
  # Create ECR repository if it doesn't exist
  if run_aws ecr describe-repositories --repository-names "$ECR_REPO" > /dev/null; then
    log "ECR repository $ECR_REPO already exists"
  else
    run_aws ecr create-repository --repository-name "$ECR_REPO"
  fi
  # Build Docker image (context is project root so paths in Dockerfile.hpo are valid)
  PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
  # The build context bundles training code, configs, and cv_train.py expected by SageMaker.
  docker build -f "$SCRIPT_DIR/Dockerfile.hpo" -t "$ECR_REPO" "$PROJECT_ROOT"
  # Log in to ECR registry
  echo "Logging in to ECR registry: $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"
  # Authenticate Docker against the private registry so the push succeeds.
  aws --profile "$PROFILE" --region "$REGION" ecr get-login-password \
      | docker login --username AWS --password-stdin "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"
  # Tag the local image and push to ECR
  IMAGE_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO}:latest"
  echo "Tagging image: ${ECR_REPO}:latest as $IMAGE_URI"
  docker tag "${ECR_REPO}:latest" "$IMAGE_URI"
  echo "Pushing Docker image to ECR: $IMAGE_URI"
  docker push "$IMAGE_URI"
  echo "Built and pushed Docker image: $IMAGE_URI"
  # Provision SageMaker execution role
  echo "Creating SageMaker execution role..."
  ROLE_NAME="${JOB_NAME}-execution-role"
  # Create trust policy for SageMaker
  TMP_POLICY=$(mktemp)
  cat > "$TMP_POLICY" <<EOF
{
  "Version": "2012-10-17",
  "Statement": [{
     "Effect": "Allow",
     "Principal": { "Service": "sagemaker.amazonaws.com" },
     "Action": "sts:AssumeRole"
  }]
}
EOF
  # Create role if not exists, otherwise refresh trust policy to ensure SageMaker can assume it
  if ! run_aws iam get-role --role-name "$ROLE_NAME" > /dev/null; then
    run_aws iam create-role --role-name "$ROLE_NAME" --assume-role-policy-document file://"$TMP_POLICY"
  else
    run_aws iam update-assume-role-policy --role-name "$ROLE_NAME" --policy-document file://"$TMP_POLICY"
  fi
  rm "$TMP_POLICY"
  # Attach necessary policies
  S3_FULL_ACCESS_ARN="arn:aws:iam::aws:policy/AmazonS3FullAccess"
  ECR_READ_ONLY_ARN="arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
  # Grant full ECR access to ensure SageMaker can pull and interact with the repository
  ECR_FULL_ACCESS_ARN="arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryFullAccess"
  CLOUDWATCH_LOGS_FULL_ACCESS_ARN="arn:aws:iam::aws:policy/CloudWatchLogsFullAccess"

  run_aws iam attach-role-policy --role-name "$ROLE_NAME" --policy-arn "$S3_FULL_ACCESS_ARN"
  run_aws iam attach-role-policy --role-name "$ROLE_NAME" --policy-arn "$ECR_READ_ONLY_ARN"
  run_aws iam attach-role-policy --role-name "$ROLE_NAME" --policy-arn "$ECR_FULL_ACCESS_ARN"
  run_aws iam attach-role-policy --role-name "$ROLE_NAME" --policy-arn "$CLOUDWATCH_LOGS_FULL_ACCESS_ARN"

  if ! wait_for_role_propagation "$ROLE_NAME" \
      "$S3_FULL_ACCESS_ARN" \
      "$ECR_READ_ONLY_ARN" \
      "$ECR_FULL_ACCESS_ARN" \
      "$CLOUDWATCH_LOGS_FULL_ACCESS_ARN"; then
    log "Proceeding even though IAM role $ROLE_NAME propagation was not fully confirmed"
  fi
  # Retrieve role ARN (use direct AWS CLI to capture output)
  ROLE_ARN=$(aws --profile "$PROFILE" --region "$REGION" iam get-role --role-name "$ROLE_NAME" --query 'Role.Arn' --output text)
  log "Using SageMaker execution role: $ROLE_ARN"
  # Allow SageMaker to pull from ECR (skip if policy already set or role ARN invalid)
  log "Skipping ECR repository policy update; ensure role $ROLE_ARN has pull access to $ECR_REPO"
  if [[ -z "$ROLE_ARN" ]]; then
    log "Error: SageMaker execution role ARN is empty. Exiting."
    exit 1
  fi
fi

if [[ -n "${ROLE_ARN:-}" ]]; then
  if ! wait_for_s3_access "$ROLE_ARN" "$TRAIN_DATA_URI" read; then
    log "Proceeding even though read permissions on $TRAIN_DATA_URI could not be fully verified"
  fi
  if ! wait_for_s3_access "$ROLE_ARN" "$OUTPUT_DATA_URI" write; then
    log "Proceeding even though write permissions on $OUTPUT_DATA_URI could not be fully verified"
  fi
fi

## Create tuning job configuration JSON with optional Warm Start
# Parse and validate previous jobs if provided
if [[ -n "${PREVIOUS_JOBS_STRING:-}" ]]; then
  IFS=',' read -r -a PARENT_JOBS <<< "$PREVIOUS_JOBS_STRING"
  if (( ${#PARENT_JOBS[@]} > 5 )); then
    echo "Error: Too many parent jobs (${#PARENT_JOBS[@]}) provided; AWS HPO Warm Start supports up to 5. Aborting." >&2
    exit 1
  fi
fi

IFS=';' read -r -a STATION_ARRAY <<< "$STATIONS"
# Require at least two stations so cross-validation splits remain meaningful.
if [ "${#STATION_ARRAY[@]}" -lt 2 ]; then
  echo "Error: model config must define at least two stations" >&2
  exit 1
fi

# Determine config file path based on OUTPUT_DIR and script name
if [ "$UPDATE_CONFIG" = "true" ] || [ ! -f "${OUTPUT_DIR}/${CONFIG_FILE_NAME}" ]; then
  CONFIG_FILE="${OUTPUT_DIR}/${CONFIG_FILE_NAME}"

  INCLUDE_STATIC_USE_VELOCITY_MEDIAN=true

  if [[ -n "$HPO_CONFIG" ]]; then
    if [[ "$HPO_CONFIG" != /* ]]; then
      HPO_CONFIG="$PROJECT_ROOT/$HPO_CONFIG"
    fi
  if [[ ! -f "$HPO_CONFIG" ]]; then
    echo "HPO config file not found: $HPO_CONFIG" >&2
    exit 1
  fi
    ensure_jq_available
    PARSED_HPO=$(parse_hpo_config "$HPO_CONFIG")
    HPO_STRATEGY='"Bayesian"'
    HPO_MAX_JOBS=50
    HPO_MAX_PARALLEL=3
    HPO_OBJECTIVE='"combined_loss"'
    HPO_PARAM_RANGES='{"ContinuousParameterRanges":[],"IntegerParameterRanges":[],"CategoricalParameterRanges":[]}'
    while IFS='=' read -r key value; do
      case "$key" in
        HPO_STRATEGY) HPO_STRATEGY="$value" ;;
        HPO_MAX_JOBS) HPO_MAX_JOBS="$value" ;;
        HPO_MAX_PARALLEL) HPO_MAX_PARALLEL="$value" ;;
        HPO_OBJECTIVE) HPO_OBJECTIVE="$value" ;;
        HPO_PARAM_RANGES) HPO_PARAM_RANGES="$value" ;;
      esac
    done <<< "$PARSED_HPO"

    if hpo_ranges_include_param "$HPO_PARAM_RANGES" "use_velocity_median"; then
      INCLUDE_STATIC_USE_VELOCITY_MEDIAN=false
    fi

  else
    echo "Error: --hpo-config is required when regenerating the tuning configuration." >&2
    echo "Provide the HPO search space JSON or disable regeneration with --update-config false." >&2
    exit 1
  fi

  # Generate HPO configuration JSON blending static parameters with sampled ranges.
  {
    echo "{"
    echo "  \"HyperParameterTuningJobName\": \"$JOB_NAME\","
    echo "  \"HyperParameterTuningJobConfig\": {"
    echo "    \"Strategy\": $HPO_STRATEGY,"
    echo "    \"ResourceLimits\": { \"MaxNumberOfTrainingJobs\": $HPO_MAX_JOBS, \"MaxParallelTrainingJobs\": $HPO_MAX_PARALLEL },"
    echo "    \"HyperParameterTuningJobObjective\": { \"Type\": \"Minimize\", \"MetricName\": $HPO_OBJECTIVE },"
    echo "    \"ParameterRanges\": $HPO_PARAM_RANGES"
    echo "  },"
    if [[ -n "${PREVIOUS_JOBS_STRING:-}" ]]; then
      echo "  \"WarmStartConfig\": {"
      echo "    \"WarmStartType\": \"IdenticalDataAndAlgorithm\","
      echo "    \"ParentHyperParameterTuningJobs\": ["
      for idx in "${!PARENT_JOBS[@]}"; do
        job="${PARENT_JOBS[$idx]}"
        if (( idx == ${#PARENT_JOBS[@]} - 1 )); then
          echo "      { \"HyperParameterTuningJobName\": \"$job\" }"
        else
          echo "      { \"HyperParameterTuningJobName\": \"$job\" },"
        fi
      done
      echo "    ]"
      echo "  },"
    fi
    echo "  \"TrainingJobDefinition\": {"
    echo "    \"AlgorithmSpecification\": { \"TrainingImage\": \"$IMAGE_URI\", \"TrainingInputMode\": \"File\", \"MetricDefinitions\": [{ \"Name\": \"combined_loss\", \"Regex\": \"CombinedLoss: ([0-9.]+)\" }] },"
    echo "    \"RoleArn\": \"$ROLE_ARN\","
    echo "    \"InputDataConfig\": [{ \"ChannelName\": \"training\", \"DataSource\": { \"S3DataSource\": { \"S3DataType\": \"S3Prefix\", \"S3Uri\": \"$TRAIN_DATA_URI\", \"S3DataDistributionType\": \"FullyReplicated\" }} , \"ContentType\": \"application/x-parquet\", \"InputMode\": \"File\" }],"
    echo "    \"OutputDataConfig\": { \"S3OutputPath\": \"$OUTPUT_DATA_URI\" },"
    echo "    \"ResourceConfig\": { \"InstanceType\": \"ml.m5.xlarge\", \"InstanceCount\": 1, \"VolumeSizeInGB\": 30 },"
    echo -n "    \"StaticHyperParameters\": { \"early_stopping\": \"${EARLY_STOPPING}\""
    if [[ -n "${MODEL_CONFIG:-}" ]]; then
      echo -n ", \"model-config\": \"${MODEL_CONFIG}\""
    fi
    if [[ "$INCLUDE_STATIC_USE_VELOCITY_MEDIAN" == "true" ]]; then
      echo -n ", \"use_velocity_median\": \"${USE_VELOCITY_MEDIAN}\""
    fi
    if [[ -n "${SAVE_ERROR_DATA:-}" ]]; then
      echo -n ", \"save_error_data\": \"${SAVE_ERROR_DATA}\""
    fi
    echo " },"
    echo "    \"StoppingCondition\": { \"MaxRuntimeInSeconds\": 3600 }"
    echo "  }"
    echo "}"
  } > "$CONFIG_FILE"
else
  CONFIG_FILE="${OUTPUT_DIR}/${CONFIG_FILE_NAME}"
  log "Configuration file exists and update disabled: $CONFIG_FILE"
fi
echo "Launching SageMaker Hyperparameter Tuning Job: $JOB_NAME"
echo "Running: aws --profile $PROFILE --region $REGION sagemaker create-hyper-parameter-tuning-job --cli-input-json file://$CONFIG_FILE"
# Submit the tuning job; the CLI invocation is mirrored above for transparency.
aws --profile "$PROFILE" --region "$REGION" sagemaker create-hyper-parameter-tuning-job --cli-input-json file://"$CONFIG_FILE"
echo "HPO job submitted. Configuration file: $CONFIG_FILE"
echo
echo "To track the job status, run:"
echo "  aws --profile $PROFILE --region $REGION sagemaker describe-hyper-parameter-tuning-job --hyper-parameter-tuning-job-name $JOB_NAME"

# Automatic status polling and logging
log "Waiting for HPO job '$JOB_NAME' to complete. Polling status every 60s..."
while true; do
  status=$(aws --profile "$PROFILE" --region "$REGION" sagemaker describe-hyper-parameter-tuning-job --hyper-parameter-tuning-job-name "$JOB_NAME" --query 'HyperParameterTuningJobStatus' --output text)
  log "Current HPO job status: $status"
  if [[ "$status" != "InProgress" && "$status" != "Stopping" ]]; then
    break
  fi
  sleep 60
done
log "HPO job \"$JOB_NAME\" completed with status: $status"

REPORT_PATH="${OUTPUT_DIR}/${JOB_NAME}_hpo_report.md"
log "Generating HPO metrics report for job \"$JOB_NAME\""
# The report consolidates per-trial hyperparameters and fold statistics for auditing.
"$SCRIPT_DIR/hpo_metrics_report.sh" \
  -n "$JOB_NAME" \
  -p "$PROFILE" \
  -r "$REGION" \
  -o "$REPORT_PATH"
log "HPO metrics report saved to $REPORT_PATH"

exit 0
