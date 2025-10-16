#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Author: Juan Luis Herrera Cortijo <juan.luis.herrera.cortijo@gmail.com>
# Created: 2025-10-16
# Disclaimer: This script is provided as-is for research workflows and must be validated before any operational deployment.
# -----------------------------------------------------------------------------

set -euo pipefail

# Launch a SageMaker Processing job that executes ``scripts/inference/inference.py``.
# The workflow stages mirror those detailed in ``docs/inference.md``: build (or reuse)
# the inference container, push it to ECR when needed, and trigger a Processing job that
# writes predictions plus metadata back to S3.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

usage() {
  cat <<'USAGE' >&2
Usage: run_inference.sh --model-artifact S3_URI --input-data S3_URI --output-s3-uri S3_URI [options]

Submit a SageMaker Processing job that runs the inference container built from this
repository. The processing script preserves all original columns from the input
GeoParquet dataset and appends prediction columns.

Required arguments:
  --model-artifact S3_URI   S3 URI pointing to model.tar.gz exported by training
  --input-data S3_URI       S3 URI to the GeoParquet dataset to score
  --output-s3-uri S3_URI    S3 prefix where inference outputs will be written

Optional arguments:
  --profile PROFILE         AWS CLI profile (default: $AWS_PROFILE or 'default')
  --region REGION           AWS region (default: $AWS_REGION or 'us-east-1')
  --role-arn ROLE_ARN       SageMaker execution role ARN (auto-created if omitted)
  --instance-type TYPE      Processing instance type (default: ml.m5.xlarge)
  --instance-count N        Processing instance count (default: 1)
  --job-name NAME           Processing job name (default auto-generated)
  --dockerfile PATH         Dockerfile path (default: scripts/inference/Dockerfile)
  --ecr-repo NAME           ECR repository name (default: hf_wind_inference)
  --image-uri URI           Pre-built inference image (skip build/push)
  --output-format FORMAT    Output format: parquet (default) or csv
  --help                    Show this help message and exit
USAGE
  exit 1
}

_check_dependency() {
  local cmd="$1"
  # Surface missing tools early so the workflow does not fail halfway through AWS interactions.
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Error: required command '$cmd' not found." >&2
    exit 1
  fi
}

PROFILE="${AWS_PROFILE:-default}"
REGION="${AWS_REGION:-us-east-1}"
MODEL_ARTIFACT=""
INPUT_DATA=""
OUTPUT_S3_URI=""
ROLE_ARN=""
INSTANCE_TYPE="ml.m5.xlarge"
INSTANCE_COUNT=1
JOB_NAME=""
DOCKERFILE_PATH="${PROJECT_ROOT}/scripts/inference/Dockerfile"
ECR_REPO="hf_wind_inference"
IMAGE_URI=""
OUTPUT_FORMAT="parquet"

POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-artifact)
      MODEL_ARTIFACT="$2"; shift 2 ;;
    --model-artifact=*)
      MODEL_ARTIFACT="${1#*=}"; shift ;;
    --input-data)
      INPUT_DATA="$2"; shift 2 ;;
    --input-data=*)
      INPUT_DATA="${1#*=}"; shift ;;
    --output-s3-uri)
      OUTPUT_S3_URI="$2"; shift 2 ;;
    --output-s3-uri=*)
      OUTPUT_S3_URI="${1#*=}"; shift ;;
    --profile)
      PROFILE="$2"; shift 2 ;;
    --profile=*)
      PROFILE="${1#*=}"; shift ;;
    --region)
      REGION="$2"; shift 2 ;;
    --region=*)
      REGION="${1#*=}"; shift ;;
    --role-arn)
      ROLE_ARN="$2"; shift 2 ;;
    --role-arn=*)
      ROLE_ARN="${1#*=}"; shift ;;
    --instance-type)
      INSTANCE_TYPE="$2"; shift 2 ;;
    --instance-type=*)
      INSTANCE_TYPE="${1#*=}"; shift ;;
    --instance-count)
      INSTANCE_COUNT="$2"; shift 2 ;;
    --instance-count=*)
      INSTANCE_COUNT="${1#*=}"; shift ;;
    --job-name)
      JOB_NAME="$2"; shift 2 ;;
    --job-name=*)
      JOB_NAME="${1#*=}"; shift ;;
    --dockerfile)
      DOCKERFILE_PATH="$2"; shift 2 ;;
    --dockerfile=*)
      DOCKERFILE_PATH="${1#*=}"; shift ;;
    --ecr-repo)
      ECR_REPO="$2"; shift 2 ;;
    --ecr-repo=*)
      ECR_REPO="${1#*=}"; shift ;;
    --image-uri)
      IMAGE_URI="$2"; shift 2 ;;
    --image-uri=*)
      IMAGE_URI="${1#*=}"; shift ;;
    --output-format)
      OUTPUT_FORMAT="$2"; shift 2 ;;
    --output-format=*)
      OUTPUT_FORMAT="${1#*=}"; shift ;;
    --help)
      usage ;;
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

if [[ -z "$MODEL_ARTIFACT" || -z "$INPUT_DATA" || -z "$OUTPUT_S3_URI" ]]; then
  usage
fi

if [[ "$OUTPUT_FORMAT" != "parquet" && "$OUTPUT_FORMAT" != "csv" ]]; then
  echo "Error: --output-format must be 'parquet' or 'csv'." >&2
  exit 1
fi

_check_dependency aws
_check_dependency docker
_check_dependency jq

if [[ ! -f "$DOCKERFILE_PATH" ]]; then
  echo "Error: Dockerfile not found at $DOCKERFILE_PATH" >&2
  exit 1
fi

ACCOUNT_ID=$(aws --profile "$PROFILE" --region "$REGION" sts get-caller-identity --query Account --output text)
REGISTRY="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

if [[ -z "$ROLE_ARN" ]]; then
  ROLE_NAME="hf-wind-inference-role"
  if aws --profile "$PROFILE" --region "$REGION" iam get-role --role-name "$ROLE_NAME" >/dev/null 2>&1; then
    ROLE_ARN=$(aws --profile "$PROFILE" --region "$REGION" iam get-role --role-name "$ROLE_NAME" --query 'Role.Arn' --output text)
  else
    TRUST_DOC=$(mktemp)
    cat <<'JSON' >"$TRUST_DOC"
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": { "Service": "sagemaker.amazonaws.com" },
      "Action": "sts:AssumeRole"
    }
  ]
}
JSON
    aws --profile "$PROFILE" --region "$REGION" iam create-role \
      --role-name "$ROLE_NAME" \
      --assume-role-policy-document file://"$TRUST_DOC"
    rm -f "$TRUST_DOC"
    for policy in AmazonS3FullAccess AmazonEC2ContainerRegistryReadOnly AmazonSageMakerFullAccess; do
      aws --profile "$PROFILE" --region "$REGION" iam attach-role-policy \
        --role-name "$ROLE_NAME" \
        --policy-arn "arn:aws:iam::aws:policy/${policy}"
    done
    ROLE_ARN=$(aws --profile "$PROFILE" --region "$REGION" iam get-role --role-name "$ROLE_NAME" --query 'Role.Arn' --output text)
  fi
fi

echo "Using execution role: $ROLE_ARN"

if [[ -z "$IMAGE_URI" ]]; then
  aws --profile "$PROFILE" --region "$REGION" ecr create-repository --repository-name "$ECR_REPO" >/dev/null 2>&1 || true
  aws --profile "$PROFILE" --region "$REGION" ecr get-login-password | docker login --username AWS --password-stdin "$REGISTRY"
  # Stage 1 — build and push the inference image so SageMaker can consume it.
  docker build -t "$ECR_REPO" -f "$DOCKERFILE_PATH" "$PROJECT_ROOT"
  IMAGE_URI="${REGISTRY}/${ECR_REPO}:latest"
  docker tag "$ECR_REPO:latest" "$IMAGE_URI"
  docker push "$IMAGE_URI"
fi

echo "Using inference image: $IMAGE_URI"
TIMESTAMP=$(date +%Y%m%d%H%M%S)
PROCESSING_JOB_NAME="${JOB_NAME:-hf-wind-inference-${TIMESTAMP}}"

APP_ARGS=$(jq -nc \
  --arg model "$MODEL_ARTIFACT" \
  --arg data "$INPUT_DATA" \
  --arg fmt "$OUTPUT_FORMAT" \
  '["--model-s3-uri", $model, "--input-data", $data, "--output-format", $fmt]')

PROCESSING_INPUTS_JSON=$(mktemp)
cat > "$PROCESSING_INPUTS_JSON" <<JSON
[
  {
    "InputName": "input",
    "S3Input": {
      "S3Uri": "$INPUT_DATA",
      "LocalPath": "/opt/ml/processing/input",
      "S3DataType": "S3Prefix",
      "S3InputMode": "File"
    }
  }
]
JSON

PROCESSING_OUTPUTS_JSON=$(mktemp)
cat > "$PROCESSING_OUTPUTS_JSON" <<JSON
{
  "Outputs": [
    {
      "OutputName": "predictions",
      "S3Output": {
        "S3Uri": "$OUTPUT_S3_URI",
        "LocalPath": "/opt/ml/processing/output",
        "S3UploadMode": "EndOfJob"
      }
    }
  ]
}
JSON

aws --profile "$PROFILE" --region "$REGION" sagemaker create-processing-job \
  --processing-job-name "$PROCESSING_JOB_NAME" \
  --role-arn "$ROLE_ARN" \
  --app-specification "ImageUri=${IMAGE_URI},ContainerArguments=${APP_ARGS}" \
  --processing-resources "ClusterConfig={InstanceCount=${INSTANCE_COUNT},InstanceType=${INSTANCE_TYPE},VolumeSizeInGB=50}" \
  --processing-inputs file://"$PROCESSING_INPUTS_JSON" \
  --processing-output-config file://"$PROCESSING_OUTPUTS_JSON"

echo "Waiting for processing job ${PROCESSING_JOB_NAME}"
aws --profile "$PROFILE" --region "$REGION" sagemaker wait processing-job-completed-or-stopped \
  --processing-job-name "$PROCESSING_JOB_NAME"

echo "Inference job ${PROCESSING_JOB_NAME} finished. Outputs stored at ${OUTPUT_S3_URI}"

rm -f "$PROCESSING_INPUTS_JSON" "$PROCESSING_OUTPUTS_JSON"
