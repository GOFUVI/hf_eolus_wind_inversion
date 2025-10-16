#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Author: Juan Luis Herrera Cortijo <juan.luis.herrera.cortijo@gmail.com>
# Created: 2025-10-16
# Disclaimer: This script is provided as-is for research workflows and must be validated before any operational deployment.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# finalize_geoparquet.sh
# -----------------------------------------------------------------------------
# Description:
#   Consolidate Parquet files produced by aggregate_core.sh and add GeoParquet
#   metadata. Optionally registers the dataset in Glue and repairs partitions
#   in Athena so the table reflects the updated files.
#
# Pipeline overview:
#   1. Resolve AWS profile/region and stage the target dataset locally.
#   2. Use Docker to merge partition files, enrich them with GeoParquet
#      metadata, and materialise a Glue table definition.
#   3. Sync the enriched dataset back to S3.
#   4. Optionally register the dataset in Glue and run MSCK REPAIR to refresh
#      partition awareness.
#
# Usage:
#   ./finalize_geoparquet.sh \
#     --db-name DB_NAME \
#     --bucket-name BUCKET \
#     --output-prefix PREFIX \
#     --output-table TABLE \
#     [--partition-cols COL1,COL2] \
#     [--register-table] \
#     [--geometry-column NAME] \
#     [--profile PROFILE] \
#     [--region REGION] \
#     [--log-dir DIR] \
#     [--help]
#
# Requirements:
#   - bash
#   - AWS CLI
#   - jq
#   - docker
# -----------------------------------------------------------------------------

set -euo pipefail

usage() {
  sed -n '5,27p' "$0"
}

run_aws() {
  # Wrapper that mirrors commands to the log before delegating to the AWS CLI,
  # capturing stdout/stderr so troubleshooting remains straightforward.
  echo "Running: aws $*" >> "$LOG_FILE"
  output=$(aws "$@" 2>&1)
  rc=$?
  echo "$output" >> "$LOG_FILE"
  if [ $rc -ne 0 ]; then
    return $rc
  fi
  echo "$output"
  return 0
}

wait_for_query() {
  local qid="$1"
  log "Waiting for Athena query $qid to complete..."
  while true; do
    output=$(run_aws athena get-query-execution --query-execution-id "$qid" --region $REGION --profile "$PROFILE")
    status=$(echo "$output" | jq -r '.QueryExecution.Status.State')
    if [ "$status" = "SUCCEEDED" ]; then
      log "Athena query $qid succeeded."
      break
    elif [ "$status" = "FAILED" ] || [ "$status" = "CANCELLED" ]; then
      log "Athena query $qid failed with status $status."
      exit 1
    else
      log "Athena query $qid status: $status. Waiting..."
      sleep 5
    fi
  done
}

# Default values
PROFILE="default"
REGION=""
DB_NAME=""
BUCKET_NAME=""
OUTPUT_PREFIX=""
OUTPUT_TABLE=""
PARTITION_COLS=""
REGISTER_TABLE=false
GEOMETRY_COLUMN="geometry"

SHORTOPTS=""
LONGOPTS="db-name:,bucket-name:,output-prefix:,output-table:,partition-cols:,register-table,geometry-column:,profile:,region:,log-dir:,help"
PARSED=$(getopt --options="$SHORTOPTS" --longoptions="$LONGOPTS" --name "$0" -- "$@") || { usage; exit 2; }
eval set -- "$PARSED"
while true; do
  case "$1" in
    --db-name) DB_NAME="$2"; shift 2;;
    --bucket-name) BUCKET_NAME="$2"; shift 2;;
    --output-prefix) OUTPUT_PREFIX="$2"; shift 2;;
    --output-table) OUTPUT_TABLE="$2"; shift 2;;
    --partition-cols) PARTITION_COLS="$2"; shift 2;;
    --geometry-column) GEOMETRY_COLUMN="$2"; shift 2;;
    --register-table) REGISTER_TABLE=true; shift;;
    --profile) PROFILE="$2"; shift 2;;
    --region) REGION="$2"; shift 2;;
    --log-dir) LOG_DIR="$2"; shift 2;;
    --help) usage; exit 0;;
    --) shift; break;;
  esac
done

if [ -z "$DB_NAME" ] || [ -z "$BUCKET_NAME" ] || [ -z "$OUTPUT_PREFIX" ] || [ -z "$OUTPUT_TABLE" ]; then
  echo "Missing required arguments" >&2
  usage
  exit 1
fi

if [ -z "$REGION" ]; then
  # Prefer the profile-specific configuration over the global fallback.
  REGION=$(aws configure get region --profile "$PROFILE" 2>/dev/null || true)
fi
if [ -z "$REGION" ]; then
  REGION=$(aws configure get region 2>/dev/null || true)
fi
if [ -z "$REGION" ]; then
  # Last resort: default to us-east-1 to preserve legacy behaviour.
  REGION="us-east-1"
fi

ORIG_PWD="$(pwd)"
if [ -n "${LOG_DIR:-}" ]; then
  mkdir -p "$LOG_DIR"
else
  LOG_DIR="$ORIG_PWD"
fi
LOG_DIR="$(realpath "$LOG_DIR")"
SCRIPT_NAME=$(basename "$0")
SCRIPT_BASE="${SCRIPT_NAME%.*}"
LOG_FILE="${LOG_DIR}/${SCRIPT_BASE}_${OUTPUT_TABLE}.log"
rm -f "$LOG_FILE"
log() { echo "$(date '+%Y-%m-%d %H:%M:%S') - $*" | tee -a "$LOG_FILE"; }

S3_PATH="s3://${BUCKET_NAME}/${OUTPUT_PREFIX%/}"
S3_DATASET_LOCATION="${S3_PATH%/}/"
DATA_DIR=$(mktemp -d "${LOG_DIR}/geo_meta_XXXXXX")
GLUE_TABLE_INPUT_JSON="${DATA_DIR}/__glue_table_input.json"
cleanup() {
  rm -rf "$DATA_DIR"
}
trap cleanup EXIT

log "Using AWS region ${REGION}"
log "Syncing dataset from $S3_PATH to $DATA_DIR"
# Stage a full copy locally so merge/add-metadata steps operate on the same
# filesystem, avoiding repeated S3 round-trips.
aws s3 sync "$S3_PATH" "$DATA_DIR" --exclude "query_results/*" --profile "$PROFILE" --region "$REGION" >> "$LOG_FILE" 2>&1

DATA_FILES_COUNT=$(find "$DATA_DIR" -type f \
  ! -name '.*' ! -name '_*' ! -name 'SUCCESS' | wc -l | awk '{print $1}')
log "Downloaded files count: ${DATA_FILES_COUNT}"
if [ "$DATA_FILES_COUNT" -eq 0 ]; then
  log "No data files found under $S3_PATH. Aborting to avoid data loss."
  exit 1
fi

log "Merging Parquet files to a single file per partition"
# Containerised Python environment guarantees consistent pyarrow/shapely
# versions across hosts, and executes merge/metadata/glue helper scripts in one
# shot to keep the dataset coherent.
docker run --rm \
  -v "${PWD}":/work \
  -v "${DATA_DIR}":/data:rw \
  -w /work \
  python:3.11-slim bash -lc "pip install --no-cache-dir 'pyarrow==16.1.0' 'shapely==2.0.4' >/tmp/pip.log && python scripts/geo_utils/merge_parquet.py --root /data && python scripts/geo_utils/add_geoparquet_metadata.py --local-path /data --geometry-column '${GEOMETRY_COLUMN}' && python scripts/geo_utils/build_glue_table_input.py --dataset-root /data --output-json /data/__glue_table_input.json --table-name ${OUTPUT_TABLE} --s3-location \"${S3_DATASET_LOCATION}\" --partition-cols \"${PARTITION_COLS}\"" >> "$LOG_FILE" 2>&1

PARQUET_COUNT=$(find "$DATA_DIR" -type f -name '*.parquet' | wc -l | awk '{print $1}')
if [ "$PARQUET_COUNT" -eq 0 ]; then
  log "ERROR: No Parquet files found after processing dataset."
  exit 1
fi

log "Syncing dataset back to $S3_PATH"
aws s3 sync "$DATA_DIR" "$S3_PATH" --exclude "query_results/*" --delete --profile "$PROFILE" --region "$REGION" >> "$LOG_FILE" 2>&1

if [ "$REGISTER_TABLE" = true ]; then
  # Optionally register the dataset with Glue so Athena can discover the
  # enriched layout without manual intervention.
  if [ ! -f "$GLUE_TABLE_INPUT_JSON" ]; then
    log "ERROR: Expected schema description at $GLUE_TABLE_INPUT_JSON but it was not generated."
    exit 1
  fi

  log "Ensuring Glue database $DB_NAME exists"
  if ! run_aws --profile "$PROFILE" --region "$REGION" glue get-database --name "$DB_NAME" >/dev/null 2>&1; then
    run_aws --profile "$PROFILE" --region "$REGION" glue create-database --database-input "{\"Name\":\"$DB_NAME\"}"
  fi

  TABLE_INPUT_REALPATH="$(realpath "$GLUE_TABLE_INPUT_JSON")"
  if run_aws --profile "$PROFILE" --region "$REGION" glue get-table --database-name "$DB_NAME" --name "$OUTPUT_TABLE" --query 'Table.Name' --output text >/dev/null 2>&1; then
    log "Glue table ${DB_NAME}.${OUTPUT_TABLE} already exists; skipping creation."
  else
    log "Creating Glue table ${DB_NAME}.${OUTPUT_TABLE} referencing ${S3_DATASET_LOCATION}"
    run_aws --profile "$PROFILE" --region "$REGION" glue create-table --database-name "$DB_NAME" --table-input "file://${TABLE_INPUT_REALPATH}"
  fi
fi

if [ -n "$PARTITION_COLS" ]; then
  # Synchronise Glue/Athena partition metadata with the consolidated files.
  log "Repairing partitions in Athena for ${DB_NAME}.${OUTPUT_TABLE}"
  QID=$(run_aws athena start-query-execution \
    --query-string "MSCK REPAIR TABLE ${DB_NAME}.${OUTPUT_TABLE}" \
    --query-execution-context "Database=${DB_NAME}" \
    --result-configuration "OutputLocation=s3://${BUCKET_NAME}/${OUTPUT_PREFIX%/}_athena_results" \
    --region $REGION --profile "$PROFILE" --output text --query 'QueryExecutionId')
  wait_for_query "$QID"
fi

log "Finalization completed: GeoParquet dataset at ${S3_PATH}"
