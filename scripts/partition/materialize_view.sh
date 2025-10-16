#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Author: Juan Luis Herrera Cortijo <juan.luis.herrera.cortijo@gmail.com>
# Created: 2025-10-16
# Disclaimer: This script is provided as-is for research workflows and must be validated before any operational deployment.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# materialize_view.sh
# -----------------------------------------------------------------------------
# Description:
#   Materialise an Athena view (or table) into a standalone Parquet dataset
#   stored at a specific S3 prefix via a CREATE TABLE AS SELECT (CTAS) query.
#   Existing Glue/Athena table metadata and S3 objects are overwritten.
#
# Requirements:
#   - AWS CLI v2 configured with permissions to run Athena statements, drop Glue
#     tables, and delete/write objects under the target S3 prefix.
#   - `jq` available on PATH for extracting identifiers from AWS CLI responses.
#   - A spill bucket/prefix for Athena execution results, provided via
#     --results-s3.
#
# Usage:
#   ./materialize_view.sh \
#     --source ann_training.MY_VIEW \
#     --target-table ann_training.MY_TABLE \
#     --s3-output s3://bucket/path/prefix/ \
#     --results-s3 s3://bucket/path/athena-results/ \
#     [--profile profile-name] \
#     [--region aws-region] \
#     [--log-dir path]
# -----------------------------------------------------------------------------

set -euo pipefail

SOURCE_FQN=""
TARGET_FQN=""
S3_OUTPUT=""
RESULTS_S3=""
PROFILE="default"
REGION=""
LOG_DIR="."

# Print the usage banner embedded at the top of the script.
usage() {
  sed -n '6,24p' "$0"
}

# Validate that a fully-qualified name is supplied and split it in database and
# table identifiers to keep the rest of the script agnostic to string parsing.
parse_fqn() {
  local value="$1"
  local label="$2"
  if [[ "$value" != *.* ]]; then
    echo "Error: ${label} must be provided as database.table" >&2
    exit 1
  fi
  local db="${value%%.*}"
  local tbl="${value#*.}"
  if [[ -z "$db" || -z "$tbl" ]]; then
    echo "Error: ${label} contains empty database or table name" >&2
    exit 1
  fi
  printf '%s %s' "$db" "$tbl"
}

# Resolve the AWS region from CLI arguments or local configuration, failing fast
# if neither path produces a region.
ensure_region() {
  if [[ -n "$REGION" ]]; then
    return
  fi
  REGION=$(aws configure get region --profile "$PROFILE" 2>/dev/null || true)
  if [[ -z "$REGION" ]]; then
    REGION=$(aws configure get region 2>/dev/null || true)
  fi
  if [[ -z "$REGION" ]]; then
    echo "Error: AWS region not provided and no default configured" >&2
    exit 1
  fi
}

# Append a timestamped message to the log file as well as stdout, keeping the
# execution trace centralised regardless of whether the script completes.
log() {
  local message="$1"
  echo "$(date '+%Y-%m-%d %H:%M:%S') - ${message}" | tee -a "$LOG_FILE"
}

# Parse command-line arguments, validating unexpected flags early so that Athena
# jobs are never triggered with ambiguous settings.
while [[ $# -gt 0 ]]; do
  case "$1" in
    --source)
      SOURCE_FQN="$2"; shift 2 ;;
    --target|--target-table)
      TARGET_FQN="$2"; shift 2 ;;
    --s3-output)
      S3_OUTPUT="$2"; shift 2 ;;
    --results-s3)
      RESULTS_S3="$2"; shift 2 ;;
    --profile)
      PROFILE="$2"; shift 2 ;;
    --region)
      REGION="$2"; shift 2 ;;
    --log-dir)
      LOG_DIR="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Error: Unknown argument '$1'" >&2
      usage
      exit 1 ;;
  esac
done

# Ensure the core identifiers and output location are present before attempting
# to resolve regions or start issuing AWS calls.
if [[ -z "$SOURCE_FQN" || -z "$TARGET_FQN" || -z "$S3_OUTPUT" ]]; then
  echo "Error: --source, --target-table, and --s3-output are required" >&2
  usage
  exit 1
fi

# Athena refuses to run without a spill bucket for query metadata; surface the
# omission explicitly to keep diagnostics obvious for the caller.
if [[ -z "$RESULTS_S3" ]]; then
  echo "Error: --results-s3 is required" >&2
  exit 1
fi
if [[ "$S3_OUTPUT" != s3://* || "$RESULTS_S3" != s3://* ]]; then
  echo "Error: --s3-output and --results-s3 must be s3:// URIs" >&2
  exit 1
fi

ensure_region

# Normalise S3 prefixes so that downstream string interpolation never produces
# duplicated path delimiters.
S3_OUTPUT="${S3_OUTPUT%/}/"
RESULTS_S3="${RESULTS_S3%/}/"

read -r SOURCE_DB SOURCE_TBL <<< "$(parse_fqn "$SOURCE_FQN" "--source")"
read -r TARGET_DB TARGET_TBL <<< "$(parse_fqn "$TARGET_FQN" "--target-table")"

# Expand the log directory path and reset the per-table log so that each run
# yields a clean chronology without remnants from previous attempts.
mkdir -p "$LOG_DIR"
LOG_DIR="$(cd "$LOG_DIR" && pwd)"
LOG_FILE="${LOG_DIR}/materialize_view_${TARGET_TBL}.log"
rm -f "$LOG_FILE"

log "Materialising ${SOURCE_DB}.${SOURCE_TBL} into ${TARGET_DB}.${TARGET_TBL}"
log "Output dataset location: ${S3_OUTPUT}"

# Persist the CTAS SQL to a temporary file to avoid escaping issues when
# forwarding the statement to the AWS CLI via --query-string.
SQL_FILE="$(mktemp "${LOG_DIR}/materialize_view_${TARGET_TBL}.XXXXXX.sql")"
trap 'rm -f "$SQL_FILE"' EXIT

# Drop any pre-existing Glue metadata so that the CTAS result creates a fresh
# schema aligned with the new fileset.
log "Dropping existing Athena table ${TARGET_DB}.${TARGET_TBL} if present"
drop_output=$(aws --profile "$PROFILE" --region "$REGION" athena start-query-execution \
  --query-execution-context "Database=${TARGET_DB}" \
  --result-configuration "OutputLocation=${RESULTS_S3}" \
  --query-string "DROP TABLE IF EXISTS ${TARGET_DB}.${TARGET_TBL}" 2>&1)
printf '%s
' "$drop_output" >> "$LOG_FILE"
drop_qid=$(echo "$drop_output" | jq -r '.QueryExecutionId // empty')
if [[ -n "$drop_qid" ]]; then
  log "Waiting for DROP TABLE query ${drop_qid}"
  while true; do
    status_out=$(aws --profile "$PROFILE" --region "$REGION" athena get-query-execution --query-execution-id "$drop_qid" 2>&1)
    printf '%s
' "$status_out" >> "$LOG_FILE"
    state=$(echo "$status_out" | jq -r '.QueryExecution.Status.State')
    if [[ "$state" == "SUCCEEDED" ]]; then
      log "DROP TABLE query ${drop_qid} succeeded"
      break
    elif [[ "$state" == "FAILED" || "$state" == "CANCELLED" ]]; then
      reason=$(echo "$status_out" | jq -r '.QueryExecution.Status.StateChangeReason // "(no reason provided)"')
      log "DROP TABLE query ${drop_qid} ended with status ${state}: ${reason}"
      exit 1
    else
      sleep 3
    fi
  done
fi
# Remove stale data before the CTAS run to guarantee the prefix contains only
# the artefacts generated in this invocation and to avoid downstream joins with
# outdated partitions.
log "Removing existing objects under ${S3_OUTPUT}"
aws --profile "$PROFILE" --region "$REGION" s3 rm --recursive "$S3_OUTPUT" >> "$LOG_FILE" 2>&1 || true

cat > "$SQL_FILE" <<SQL
CREATE TABLE ${TARGET_DB}.${TARGET_TBL}
WITH (
  format = 'PARQUET',
  external_location = '${S3_OUTPUT}',
  parquet_compression = 'SNAPPY'
) AS
SELECT *
FROM ${SOURCE_DB}.${SOURCE_TBL}
;
SQL

log "Submitting CTAS query"
create_output=$(aws --profile "$PROFILE" --region "$REGION" athena start-query-execution \
  --query-execution-context "Database=${TARGET_DB}" \
  --result-configuration "OutputLocation=${RESULTS_S3}" \
  --query-string "file://${SQL_FILE}" 2>&1)
printf '%s
' "$create_output" >> "$LOG_FILE"
create_qid=$(echo "$create_output" | jq -r '.QueryExecutionId // empty')
if [[ -z "$create_qid" ]]; then
  log "Failed to submit CTAS query"
  exit 1
fi

log "Waiting for CTAS query ${create_qid}"
while true; do
  status_out=$(aws --profile "$PROFILE" --region "$REGION" athena get-query-execution --query-execution-id "$create_qid" 2>&1)
  printf '%s
' "$status_out" >> "$LOG_FILE"
  state=$(echo "$status_out" | jq -r '.QueryExecution.Status.State')
  if [[ "$state" == "SUCCEEDED" ]]; then
    log "CTAS query ${create_qid} succeeded"
    break
  elif [[ "$state" == "FAILED" || "$state" == "CANCELLED" ]]; then
    reason=$(echo "$status_out" | jq -r '.QueryExecution.Status.StateChangeReason // "(no reason provided)"')
    log "CTAS query ${create_qid} ended with status ${state}: ${reason}"
    exit 1
  else
    sleep 5
  fi
done

log "Materialisation completed successfully"
