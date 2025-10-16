#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Author: Juan Luis Herrera Cortijo <juan.luis.herrera.cortijo@gmail.com>
# Created: 2025-10-16
# Disclaimer: This script is provided as-is for research workflows and must be validated before any operational deployment.
# -----------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Script: partition.sh
# Purpose: Materialise deterministic train/test splits for an Athena-hosted HF
#          radar dataset while preserving physical stratification constraints
#          across wind direction and optional wind speed strata. The script is a
#          building block of the data preparation pipeline and is also reused by
#          downstream modelling pipelines that rely on reproducible folds,
#          detailed stratification diagnostics, and markdown reporting.
# Workflow summary:
#   1. Parse CLI flags that specify source/target Athena databases, S3 output
#      prefixes, stratification parameters, and report customisation options.
#   2. Compute wind-direction bins (and optionally speed strata) to construct a
#      combined `wind_bin` variable used for stratified sampling.
#   3. Issue CTAS statements that build Parquet-backed train/test tables in the
#      target database, enforcing hold-out quotas per bin and per identifier, as
#      well as hard exclusions provided by the caller.
#   4. Produce a comprehensive markdown report covering record counts, bin
#      balance, ID coverage, and optional fold-level descriptive statistics.
# Outputs: Parquet tables `${TABLE_NAME}_train` and `${TABLE_NAME}_test`, a log
#          file colocated with the script, and a markdown report that can be
#          shipped as part of the pipeline artefacts.
# Dependencies: AWS CLI, jq, bc, awk, python3. Requires permissions to run
#               Athena queries, manage Glue tables, and write to the specified
#               S3 prefixes.
# ------------------------------------------------------------------------------
# Fail on error, undefined vars, and on pipe failures to surface issues early.
set -euo pipefail

## Default local output directory for logs and reports
OUTPUT_DIR="."
## Default wind direction column name (overrides literal 'wind_direction' in queries)
DIR_COL="wind_direction"
## Default wind speed column name (overrides literal 'wind_speed' in queries)
SPEED_COL="wind_speed"
## Default ID column for stratification; if unset, creates a constant column named 'location_id' with value 1
ID_COL=""
# Ensure output directory exists upfront
mkdir -p "${OUTPUT_DIR}"

## CSV export removed: we now only generate Parquet-backed Athena tables


# ------------------------------------------------------------------------------
# partition.sh - Partition data table with stratified sampling by an ID column
# and wind direction bins, ensuring at least one test case per ID (if provided) and
# per directional stratum. Optionally exclude specified ID values from training
# and force them into the test set.
#
# Requirements: AWS CLI, jq, bc, awk, and appropriate AWS permissions.
# Usage: See usage() function for detailed instructions.
# Parameters:
#   --input-db <name>           Source Athena database containing the SAR table (required).
#   --table <name>              Source Athena table name with SAR observations (required).
#   --output-db <name>          Athena database where partitioned results will be created (required).
#   --s3-prefix <s3_path>       S3 prefix for output partitions (required).
#   --profile <name>            AWS CLI profile (default: ${AWS_PROFILE:-default}).
#   --region <name>             AWS CLI region (default: from AWS CLI config for the specified profile).
#   --results-s3 <s3_uri>       S3 URI for Athena query results (optional).
#   --folds <num_folds>         Number of folds for cross-validation (default: ${FOLDS:-5}).
#   --seed <seed>               Reproducibility seed for hashing (default: ${REPRO_SEED:-20250510}).
#   --direction-bin-count <n>   Number of wind direction bins (default: ${NUM_BINS:-8}).
#   --train-fraction <frac>     Training set fraction (default: ${TRAIN_FRAC:-0.85}).
#   --local-dir <path>          Local directory to write logs and reports (default: .)
#   --speed-strata <limits>     Comma-separated wind speed strata limits in m/s (outer bounds at 0 and +Inf);
#                               e.g., "5,10,15" creates bins [0,5),[5,10),[10,15),[15,+Inf].
#                               Combined with direction bins into \`wind_bin\` as \`speedBin.directionBin\`.
#   --direction-column <name>   Wind direction column name (default: wind_direction).
#   --speed-column <name>       Wind speed column name (default: wind_speed).
#   --id-column <name>          Column to stratify on (optional; default constant 'location_id').
#   --exclude-ids <list>        Comma-separated list of ID values to exclude from training.
#   --report-file <path>        Optional path for the markdown report (default: ${OUTPUT_DIR}/${OUTPUT_DB}_partition_report.md).
#   --fold-stat-columns <list>  Comma-separated list of numeric columns to profile per fold in the report.
#   --fold-stat-category <col>        Optional categorical column (repeatable) to further split numeric statistics.
#   --fold-stat-category2 <col>       Backward-compatible alias for --fold-stat-category.
#   -h|--help                   Show this help message and exit.
# ------------------------------------------------------------------------------

# Logging setup will be initialized after option parsing (to pick up --local-dir)

# run_aws "$@"
# Wrapper around AWS CLI invocations that mirrors the executed command to the
# persistent log and to STDERR/STDOUT through `tee`. The function centralises
# logging so that every Glue, S3, or Athena call leaves an auditable trace.
# Arguments:
#   $@ - Raw AWS CLI arguments (profile/region injected via environment).
# Returns: Exit status of the delegated AWS CLI command. Propagates failures so
#          that `set -e` can abort the script when an AWS call fails.
run_aws() {
  echo "Running: aws $*" >> "$LOG_FILE"
  aws --profile "$PROFILE" --region "$REGION" "$@" 2>&1 | tee -a "$LOG_FILE"
}

# wait_for_query "<query_execution_id>"
# Polls the Athena query identified by the provided execution id until the
# service reports SUCCEEDED, FAILED, or CANCELLED. Intermediate states are
# logged every five seconds to give long-running queries visibility.
# Arguments:
#   $1 - Athena QueryExecutionId returned by start-query-execution.
# Exits: Terminates the script if the query fails or is cancelled, including the
#        failure reason emitted by Athena to help the caller diagnose issues.
wait_for_query() {
  local qid="$1"
  log "Waiting for Athena query $qid to complete..."
  while true; do
    local out
    out=$(aws --profile "$PROFILE" --region "$REGION" athena get-query-execution --query-execution-id "$qid")
    local state
    state=$(echo "$out" | jq -r '.QueryExecution.Status.State')
    if [[ "$state" == "SUCCEEDED" ]]; then
      log "Athena query $qid succeeded."
      break
    elif [[ "$state" == "FAILED" || "$state" == "CANCELLED" ]]; then
      # Fetch reason for failure or cancellation
      local reason
      reason=$(echo "$out" | jq -r '.QueryExecution.Status.StateChangeReason // "(no reason provided)"')
      log "Athena query $qid ended with status $state. Reason: $reason"
      exit 1
    else
      log "Athena query $qid still in state '$state' - checking again in 5s..."
      sleep 5
    fi
  done

}

COLUMN_TYPE_CACHE_KEYS=()
COLUMN_TYPE_CACHE_VALUES=()

# get_column_type "<column_name>"
# Retrieves the Glue data type of a column from the materialised train table and
# caches the result to avoid repetitive metadata queries. The lookup is
# necessary to decide whether fold statistics should treat a column as numeric
# or categorical in later report sections.
# Arguments:
#   $1 - Column name to inspect (case-insensitive).
# Output: Prints the lowercase data type to STDOUT, or `unknown` if unresolved.
get_column_type() {
  local column="$1"
  local cached=""
  local idx
  for idx in "${!COLUMN_TYPE_CACHE_KEYS[@]}"; do
    if [[ "${COLUMN_TYPE_CACHE_KEYS[$idx]}" == "$column" ]]; then
      cached="${COLUMN_TYPE_CACHE_VALUES[$idx]}"
      break
    fi
  done
  if [[ -n "$cached" ]]; then
    echo "$cached"
    return
  fi

  local results_uri
  results_uri="${RESULTS_URI:-${S3_PREFIX%/}/athena-results/}"
  local tmp_sql
  tmp_sql=$(mktemp -t "partition.coltype.XXXXXX")
  tmp_sql="${tmp_sql}.sql"
  cat >"$tmp_sql" <<SQL
SELECT data_type
FROM information_schema.columns
WHERE lower(table_schema) = lower('${OUTPUT_DB}')
  AND lower(table_name) = lower('${TABLE_NAME}_train')
  AND lower(column_name) = lower('${column}')
LIMIT 1;
SQL

  if ! type_qid=$(aws --profile "$PROFILE" --region "$REGION" athena start-query-execution \
    --query-execution-context Database="$OUTPUT_DB" \
    --result-configuration "OutputLocation=${results_uri}" \
    --query-string "file://${tmp_sql}" \
    --output text --query 'QueryExecutionId'); then
    log "Failed to submit column type lookup for '${column}'"
    rm -f "$tmp_sql"
    COLUMN_TYPE_CACHE_KEYS+=("$column")
    COLUMN_TYPE_CACHE_VALUES+=("unknown")
    echo "unknown"
    return
  fi
  wait_for_query "$type_qid"
  if ! type_json=$(aws --profile "$PROFILE" --region "$REGION" athena get-query-results \
    --query-execution-id "$type_qid" --output json); then
    log "Failed to retrieve column type for '${column}'"
    rm -f "$tmp_sql"
    COLUMN_TYPE_CACHE_KEYS+=("$column")
    COLUMN_TYPE_CACHE_VALUES+=("unknown")
    echo "unknown"
    return
  fi
  rm -f "$tmp_sql"
  local dtype
  dtype=$(echo "$type_json" | jq -r '.ResultSet.Rows[1].Data[0].VarCharValue // ""' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' | tr 'A-Z' 'a-z')
  if [ -z "$dtype" ]; then
    log "Column '${column}' not found in ${OUTPUT_DB}.${TABLE_NAME}_train"
    COLUMN_TYPE_CACHE_KEYS+=("$column")
    COLUMN_TYPE_CACHE_VALUES+=("unknown")
    echo "unknown"
    return
  fi
  COLUMN_TYPE_CACHE_KEYS+=("$column")
  COLUMN_TYPE_CACHE_VALUES+=("$dtype")
  echo "$dtype"
}

# is_numeric_type "<glue_type>"
# Helper that normalises the Glue type string and classifies it as numeric.
# Returns success (0) for numeric-compatible types and failure (1) otherwise.
# Arguments:
#   $1 - Raw Glue type string (e.g., "double", "decimal(10,2)").
is_numeric_type() {
  local input="$1"
  local dtype
  dtype=$(printf '%s' "$input" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' | tr '[:upper:]' '[:lower:]')
  case "$dtype" in
    bigint|int|integer|smallint|tinyint|double*|float*|real*|decimal*|numeric*)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

# run_category_breakdown "<set_name>" "<numeric_column>" "<category_columns_csv>"
# Executes an aggregated Athena query that profiles a numeric column by fold and
# one or multiple categorical columns. The resulting table is appended to the
# markdown report with counts and descriptive statistics per category.
# Arguments:
#   $1 - Dataset tag (`train` or `test`) used to target the appropriate table.
#   $2 - Numeric column to describe.
#   $3 - Optional comma-separated list of categorical columns. When empty, the
#        function short-circuits without querying Athena.
run_category_breakdown() {
  local set_name="$1"
  local numeric_column="$2"
  local columns_csv="${3-}"

  if [ -z "$columns_csv" ]; then
    return
  fi

  IFS=',' read -r -a category_cols <<< "$columns_csv"
  local cleaned_cols=()
  local raw_col trimmed
  for raw_col in "${category_cols[@]}"; do
    trimmed=$(echo "$raw_col" | xargs)
    if [ -n "$trimmed" ]; then
      cleaned_cols+=("$trimmed")
    fi
  done
  category_cols=("${cleaned_cols[@]}")
  if [ ${#category_cols[@]} -eq 0 ]; then
    return
  fi

  local heading_label header_label category_expr
  if [ ${#category_cols[@]} -eq 1 ]; then
    heading_label="\`${category_cols[0]}\`"
    header_label="${category_cols[0]}"
    category_expr="COALESCE(CAST(${category_cols[0]} AS VARCHAR), '(null)')"
  else
    local heading_join
    heading_join=$(IFS=', '; echo "${category_cols[*]}")
    heading_label="\`$heading_join\`"
    header_label=$(IFS=' + '; echo "${category_cols[*]}")
    category_expr="concat_ws(' | '"
    local col
    for col in "${category_cols[@]}"; do
      category_expr+=", COALESCE(CAST($col AS VARCHAR), '(null)')"
    done
    category_expr+=")"
  fi

  log "Computing category breakdown for column '${numeric_column}' by categories ${category_cols[*]} in set '${set_name}'"
  echo "##### Breakdown by ${heading_label}" >> "$REPORT_FILE"
  echo "" >> "$REPORT_FILE"

  local tmp_cat_stats_sql
  tmp_cat_stats_sql=$(mktemp -t "${SCRIPT_NAME}.${set_name}.foldstats.by_category.XXXXXX")
  tmp_cat_stats_sql="${tmp_cat_stats_sql}.sql"

  CATEGORY_EXPR="$category_expr" NUMERIC_COL="$numeric_column" \
  cat >"$tmp_cat_stats_sql" <<SQL
WITH base AS (
  SELECT
    ${CATEGORY_EXPR} AS category_value,
    fold,
    CAST(${NUMERIC_COL} AS DOUBLE) AS value
  FROM ${OUTPUT_DB}.${TABLE_NAME}_${set_name}
)
SELECT
  category_value,
  fold,
  COUNT(value) AS count_non_null,
  COALESCE(format('%.4f', AVG(value)), 'null') AS mean_value,
  COALESCE(format('%.4f', stddev_samp(value)), 'null') AS stddev_value,
  COALESCE(format('%.4f', MIN(value)), 'null') AS min_value,
  COALESCE(format('%.4f', MAX(value)), 'null') AS max_value
FROM base
GROUP BY category_value, fold
ORDER BY category_value, fold;
SQL

  if ! cat_stats_qid=$(aws --profile "$PROFILE" --region "$REGION" athena start-query-execution \
    --query-execution-context Database="$OUTPUT_DB" \
    --result-configuration "OutputLocation=${RESULTS_URI}" \
    --query-string "file://${tmp_cat_stats_sql}" \
    --output text --query 'QueryExecutionId'); then
    log "Failed to submit category breakdown for column '${numeric_column}'"
    echo "> Warning: Athena query failed for category breakdown of column \`${numeric_column}\` in set ${set_name}." >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    rm -f "$tmp_cat_stats_sql"
    return
  fi

  wait_for_query "$cat_stats_qid"
  if ! cat_stats_json=$(aws --profile "$PROFILE" --region "$REGION" athena get-query-results \
    --query-execution-id "$cat_stats_qid" --output json); then
    log "Failed to retrieve category breakdown for column '${numeric_column}'"
    echo "> Warning: Unable to retrieve category breakdown for column \`${numeric_column}\`." >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    rm -f "$tmp_cat_stats_sql"
    return
  fi
  rm -f "$tmp_cat_stats_sql"

  local cat_stats_rows
  cat_stats_rows=$(echo "$cat_stats_json" | jq -r '.ResultSet.Rows[1:][] | [.Data[0].VarCharValue, .Data[1].VarCharValue, .Data[2].VarCharValue, .Data[3].VarCharValue, .Data[4].VarCharValue, .Data[5].VarCharValue, .Data[6].VarCharValue] | @tsv')
  if [ -z "$cat_stats_rows" ]; then
    echo "> No data available for category breakdown of column \`${numeric_column}\`." >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    return
  fi

  echo "| ${header_label} | fold | count | mean | stddev | min | max |" >> "$REPORT_FILE"
  echo "| --- | ---: | ---: | ---: | ---: | ---: | ---: |" >> "$REPORT_FILE"
  while IFS=$'	' read -r category fold count mean stddev min max; do
    [ -z "$fold" ] && continue
    [ "$mean" = "null" ] && mean="-"
    [ "$stddev" = "null" ] && stddev="-"
    [ "$min" = "null" ] && min="-"
    [ "$max" = "null" ] && max="-"
    echo "| ${category:-'(null)'} | $fold | ${count:-0} | $mean | $stddev | $min | $max |" >> "$REPORT_FILE"
  done <<< "$cat_stats_rows"
  echo "" >> "$REPORT_FILE"
}
# capitalize "<string>"
# Utility used for report headings that capitalises the first letter while
# leaving the remainder untouched, preserving any domain-specific casing.
capitalize() {
  echo "$1" | awk '{print toupper(substr($0,1,1)) substr($0,2)}'
}

# usage
# Emits CLI usage documentation to STDERR and exits with status 1. The function
# is invoked both when the user explicitly requests help and when required
# options are missing or malformed.
usage() {
  cat <<EOF >&2
Usage: $0 --input-db <db> --table <table> --output-db <db> --s3-prefix <s3://bucket/prefix> [options]

Partition data table with stratified sampling by an ID column and wind direction bins,
ensuring at least one test case per ID (if provided) and per directional stratum, while optionally
excluding specified ID values from training and enforcing them into the test set.

Required arguments:
  --input-db <db>             Source Athena database name
  --table <table>             Source Athena table name
  --output-db <db>            Output Athena database name
  --s3-prefix <s3://...>      S3 prefix for output tables

Optional arguments:
  --profile <name>            AWS CLI profile (default: default)
  --region <name>             AWS CLI region (default: from AWS CLI config for the specified profile)
  --results-s3 <s3://...>     Athena query results S3 URI (optional)
  --folds <num>               Number of folds for cross-validation (default: 5)
  --seed <seed>               Reproducibility seed for hashing (default: 20250510)
  --direction-bin-count <n>   Number of wind direction bins (default: 8)
  --train-fraction <frac>     Training set fraction (default: 0.85)
  --local-dir <path>          Local directory to write logs and reports (default: .)
  --speed-strata <limits>     Comma-separated wind speed strata limits (outer bounds at 0 and +Inf);
                              Combined with direction bins into \`wind_bin\` as \`speedBin.directionBin\`
  --direction-column <name>   Name of the wind direction column (default: wind_direction)
  --speed-column <name>       Name of the wind speed column (default: wind_speed)
  --id-column <name>          Name of the column to stratify on (optional; default constant 'location_id')
  --exclude-ids <list>        Comma-separated list of ID values to exclude from training
  --report-file <path>        Override the default markdown report path
  --fold-stat-columns <list>  Comma-separated list of numeric columns to summarise per fold in the report
  --fold-stat-category <col>        Optional categorical column (repeatable) to further split numeric statistics
  --fold-stat-category2 <col>       Backward-compatible alias for --fold-stat-category
  -h|--help                   Show this help message and exit

Example:
  $0 --input-db sourcedb --table mytable --output-db targetdb --s3-prefix s3://bucket/path \\
     --results-s3 s3://bucket/path/athena-results \\
     --folds 5 --seed 20250510 --exclude-ids nodeA,nodeB
EOF
  exit 1
}

# ensure_value "<flag>" "<candidate_value>"
# Guard invoked during CLI parsing to assert that every flag receives an
# accompanying value. Without this check, bash would silently treat the next
# flag as the value, producing confusing error states downstream.
ensure_value() {
  local opt="$1"
  local value="$2"
  if [[ -z "${value}" || "${value}" == --* ]]; then
    echo "Error: ${opt} requires a value" >&2
    usage
  fi
}

# Default AWS CLI profile and region
PROFILE="${AWS_PROFILE:-default}"
REGION="${AWS_REGION:-}"
# Seed for reproducibility hashing (override with REPRO_SEED env var or -e <seed> option)
SEED="${REPRO_SEED:-20250510}"

# Athena query results S3 URI (optional)
RESULT_LOCATION=""
# Number of folds for k-fold cross-validation
FOLDS="${FOLDS:-5}"
# Output Athena database for new tables (required)
OUTPUT_DB=""
# CSV export removed
# Comma-separated list of ID values to exclude from training and force into test set
EXCLUDE_IDS=""
# Optional report file override
REPORT_FILE=""
# Columns for which to compute per-fold descriptive statistics in the report
FOLD_STATS_COLUMNS=""
FOLD_STATS_CATEGORY_COLUMNS_RAW=""
FOLD_STATS_CATEGORY_COLUMNS=()

## Parse command-line options
while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-db)
      ensure_value "$1" "${2-}"
      DB_NAME="$2"
      shift 2
      ;;
    --table)
      ensure_value "$1" "${2-}"
      TABLE_NAME="$2"
      shift 2
      ;;
    --output-db)
      ensure_value "$1" "${2-}"
      OUTPUT_DB="$2"
      shift 2
      ;;
    --profile)
      ensure_value "$1" "${2-}"
      PROFILE="$2"
      shift 2
      ;;
    --region)
      ensure_value "$1" "${2-}"
      REGION="$2"
      shift 2
      ;;
    --s3-prefix)
      ensure_value "$1" "${2-}"
      S3_PREFIX="$2"
      shift 2
      ;;
    --results-s3)
      ensure_value "$1" "${2-}"
      RESULT_LOCATION="$2"
      shift 2
      ;;
    --folds)
      ensure_value "$1" "${2-}"
      FOLDS="$2"
      shift 2
      ;;
    --seed)
      ensure_value "$1" "${2-}"
      SEED="$2"
      shift 2
      ;;
    --direction-bin-count)
      ensure_value "$1" "${2-}"
      NUM_BINS="$2"
      shift 2
      ;;
    --train-fraction)
      ensure_value "$1" "${2-}"
      TRAIN_FRAC="$2"
      shift 2
      ;;
    --local-dir)
      ensure_value "$1" "${2-}"
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --speed-strata)
      ensure_value "$1" "${2-}"
      SPEED_LIMITS="$2"
      shift 2
      ;;
    --direction-column)
      ensure_value "$1" "${2-}"
      DIR_COL="$2"
      shift 2
      ;;
    --speed-column)
      ensure_value "$1" "${2-}"
      SPEED_COL="$2"
      shift 2
      ;;
    --id-column)
      ensure_value "$1" "${2-}"
      ID_COL="$2"
      shift 2
      ;;
    --exclude-ids)
      ensure_value "$1" "${2-}"
      EXCLUDE_IDS="$2"
      shift 2
      ;;
    --report-file)
      ensure_value "$1" "${2-}"
      REPORT_FILE="$2"
      shift 2
      ;;
    --fold-stat-columns)
      ensure_value "$1" "${2-}"
      FOLD_STATS_COLUMNS="$2"
      shift 2
      ;;
    --fold-stat-category)
      ensure_value "$1" "${2-}"
      if [ -z "$FOLD_STATS_CATEGORY_COLUMNS_RAW" ]; then
        FOLD_STATS_CATEGORY_COLUMNS_RAW="$2"
      else
        FOLD_STATS_CATEGORY_COLUMNS_RAW="${FOLD_STATS_CATEGORY_COLUMNS_RAW},$2"
      fi
      shift 2
      ;;
    --fold-stat-category2)
      ensure_value "$1" "${2-}"
      if [ -z "$FOLD_STATS_CATEGORY_COLUMNS_RAW" ]; then
        FOLD_STATS_CATEGORY_COLUMNS_RAW="$2"
      else
        FOLD_STATS_CATEGORY_COLUMNS_RAW="${FOLD_STATS_CATEGORY_COLUMNS_RAW},$2"
      fi
      shift 2
      ;;
    -h|--help)
      usage
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Error: Unknown option $1" >&2
      usage
      ;;
  esac
done
# Fail fast on unexpected positional arguments
if [[ $# -gt 0 ]]; then
  echo "Error: Unexpected positional arguments: $*" >&2
  usage
fi
# Ensure output directory exists
mkdir -p "${OUTPUT_DIR}"

## Initialize logging (after parsing --local-dir)
SCRIPT_NAME=$(basename "$0" .sh)
LOG_FILE="${OUTPUT_DIR}/${SCRIPT_NAME}.log"
rm -f "$LOG_FILE"
# log "<message>"
# Consistent logger that timestamps and duplicates messages to both STDOUT/ERR
# and the persistent log file. Enables chronological reconstruction of the
# partitioning run.
log() {
  local msg
  msg="$(date '+%Y-%m-%d %H:%M:%S') - $*"
  echo "$msg" >> "$LOG_FILE"
  >&2 echo "$msg"
}

# Ensure required options are provided
if [ -z "${DB_NAME:-}" ] || [ -z "${OUTPUT_DB:-}" ] || [ -z "${TABLE_NAME:-}" ] || [ -z "${S3_PREFIX:-}" ]; then
  usage
fi

# Determine report file path (override or default within --local-dir)
if [ -z "${REPORT_FILE}" ]; then
  REPORT_FILE="${OUTPUT_DIR%/}/${OUTPUT_DB}_partition_report.md"
fi
REPORT_DIR=$(dirname "${REPORT_FILE}")
mkdir -p "${REPORT_DIR}"

# Determine id_value_name and expression: if ID_COL set, use existing column; else create constant
# Determine id_value_name, alias expression, and concat expression for hashing
if [ -n "${ID_COL}" ]; then
  # Use existing ID column name (no duplicate alias in SELECT *)
  id_col_expr=""
  id_value_name="${ID_COL}"
  id_concat_expr="CAST(${ID_COL} AS VARCHAR)"
else
  id_col_expr="1 AS location_id,"
  id_value_name="location_id"
  id_concat_expr="'1'"
fi

# Determine default results URI if not provided
if [ -n "${RESULT_LOCATION}" ]; then
  RESULTS_URI="${RESULT_LOCATION}"
else
  RESULTS_URI="${S3_PREFIX%/}/athena-results/"
fi

# If no region specified via env var or CLI, extract AWS region from the specified profile
if [ -z "${REGION}" ]; then
  REGION=$(aws configure get region --profile "${PROFILE}")
  if [ -z "${REGION}" ]; then
    echo "Error: AWS region not specified and could not be determined from profile '${PROFILE}'." >&2
    exit 1
  fi
fi

# Stratification parameters: number of wind direction bins and training fraction.
# Values can be overridden via '-n' and '-f' flags (defaults shown).
NUM_BINS="${NUM_BINS:-8}"
TRAIN_FRAC="${TRAIN_FRAC:-0.85}"

# Compute derived fractions and bin width
VALID_THRESHOLD=${TRAIN_FRAC}
TEST_FRAC=$(echo "1 - ${TRAIN_FRAC}" | bc -l)
BIN_WIDTH=$(echo "360 / ${NUM_BINS}" | bc -l)

# Format to 6 decimal places
printf -v VALID_THRESHOLD "%.6f" "${VALID_THRESHOLD}"
printf -v TEST_FRAC      "%.6f" "${TEST_FRAC}"
printf -v BIN_WIDTH      "%.6f" "${BIN_WIDTH}"

## Build wind_bin expression combining speed strata (if provided) and direction bins
# Translate optional speed strata into a CASE expression; each clause is the
# zero-based index of the bucket so that we can compose a stable `speed.direction`
# identifier that remains comparable across datasets.
if [ -n "${SPEED_LIMITS:-}" ]; then
  IFS=',' read -ra SPEED_BOUNDS <<< "$SPEED_LIMITS"
  speed_case="CASE"
  for i in "${!SPEED_BOUNDS[@]}"; do
    speed_case+=" WHEN ${SPEED_COL} < ${SPEED_BOUNDS[$i]} THEN ${i}"
  done
  speed_case+=" ELSE ${#SPEED_BOUNDS[@]} END"
  dir_case="CAST(FLOOR(${DIR_COL}/${BIN_WIDTH}) AS INTEGER)"
  # Combine speed and direction bins into a string "speed_bin.direction_bin"
  wind_case="concat(CAST(${speed_case} AS VARCHAR), '.', CAST(${dir_case} AS VARCHAR))"
else
  # Without explicit speed strata the binning falls back to pure direction bins.
  wind_case="CAST(FLOOR(${DIR_COL}/${BIN_WIDTH}) AS INTEGER)"
fi

if [ -n "${FOLD_STATS_COLUMNS:-}" ]; then
  # Parse numeric columns list upfront so we can iterate deterministically later.
  IFS=',' read -r -a FOLD_STATS_ARRAY <<< "$FOLD_STATS_COLUMNS"
fi

if [ -n "${FOLD_STATS_CATEGORY_COLUMNS_RAW:-}" ]; then
  IFS=',' read -r -a FOLD_STATS_CATEGORY_COLUMNS <<< "$FOLD_STATS_CATEGORY_COLUMNS_RAW"
  trimmed_categories=()
  for idx in "${!FOLD_STATS_CATEGORY_COLUMNS[@]}"; do
    trimmed=$(echo "${FOLD_STATS_CATEGORY_COLUMNS[$idx]}" | xargs)
    if [ -n "$trimmed" ]; then
      trimmed_categories+=("$trimmed")
    fi
  done
  # Preserve the caller-provided order after trimming whitespace.
  FOLD_STATS_CATEGORY_COLUMNS=("${trimmed_categories[@]}")
else
  FOLD_STATS_CATEGORY_COLUMNS=()
fi

# Log configuration
log "Configuration:"
log "  Source Database : ${DB_NAME}"
log "  Source Table    : ${TABLE_NAME}"
log "  Output Database : ${OUTPUT_DB}"
log "  AWS Profile     : ${PROFILE}"
log "  AWS Region      : ${REGION}"
log "  S3 Prefix       : ${S3_PREFIX}"
if [ -n "${RESULT_LOCATION}" ]; then
  log "  Results Location: ${RESULT_LOCATION}"
fi
log "  Bins/Width      : ${NUM_BINS}/${BIN_WIDTH}"
log "  Train/Test      : ${TRAIN_FRAC}/${TEST_FRAC}"
if [ -n "${SPEED_LIMITS:-}" ]; then
  log "  Wind speed strata limits: ${SPEED_LIMITS}"
fi
log "  Wind direction column: ${DIR_COL}"
log "  Wind speed column    : ${SPEED_COL}"
log "  Reproducibility seed : ${SEED}"
log "  Number of folds : ${FOLDS}"
log "  Report file path : ${REPORT_FILE}"
if [ -n "${FOLD_STATS_COLUMNS}" ]; then
  log "  Fold statistics columns : ${FOLD_STATS_COLUMNS}"
else
  log "  Fold statistics columns : (none)"
fi
if [ ${#FOLD_STATS_CATEGORY_COLUMNS[@]} -gt 0 ]; then
  log "  Fold statistics category columns : ${FOLD_STATS_CATEGORY_COLUMNS[*]}"
else
  log "  Fold statistics category columns : (none)"
fi
# Log excluded IDs if provided
if [ -n "${EXCLUDE_IDS}" ]; then
  log "  Excluded IDs      : ${EXCLUDE_IDS}"
fi
# Log chosen stratification ID column
log "  Stratification ID column: ${id_value_name}"
log ""
# Prepare excluded IDs SQL list if provided
if [ -n "${EXCLUDE_IDS}" ]; then
  IFS=',' read -r -a EXCLUDE_IDS_ARRAY <<< "${EXCLUDE_IDS}"
  EXCLUDE_IDS_SQL=$(printf "'%s', " "${EXCLUDE_IDS_ARRAY[@]}")
  EXCLUDE_IDS_SQL=${EXCLUDE_IDS_SQL%, }
  # Inject specialised branches in the SQL CASE expressions so that excluded IDs
  # always land in the test set and are assigned deterministic fold identifiers.
  EXCLUDE_WHEN_FOLD="    WHEN ${id_value_name} IN (${EXCLUDE_IDS_SQL}) THEN 0"
  EXCLUDE_WHEN_TRAIN_BOOL="    WHEN ${id_value_name} IN (${EXCLUDE_IDS_SQL}) THEN FALSE"
  EXCLUDE_WHEN_TRAIN_NUM="    WHEN ${id_value_name} IN (${EXCLUDE_IDS_SQL}) THEN 0"
  EXCLUDE_CONDITION="${id_value_name} IN (${EXCLUDE_IDS_SQL})"
else
  EXCLUDE_IDS_SQL=""
  EXCLUDE_WHEN_FOLD=""
  EXCLUDE_WHEN_TRAIN_BOOL=""
  EXCLUDE_WHEN_TRAIN_NUM=""
  EXCLUDE_CONDITION="FALSE"
fi

# run_ctas "<set_type>"
# Issues a CREATE TABLE AS SELECT statement that materialises either the train
# or test split, replacing any previous Glue metadata and Parquet objects at the
# target S3 prefix. The SQL statement encapsulates the full stratification logic
# and is written to a temporary file to keep the bash script readable.
# Arguments:
#   $1 - Set identifier (`train` or `test`) used to build the destination table
#        name and S3 location.
run_ctas() {
  local set_type="$1"
  local new_table="${TABLE_NAME}_${set_type}"
  local output_location="${S3_PREFIX%/}/${set_type}"
  local results_uri="${RESULT_LOCATION:-${S3_PREFIX%/}/athena-results/}"

  # Ensure Athena output database exists
  log "Ensuring Athena database '$OUTPUT_DB' exists..."
  if ! aws --profile "$PROFILE" --region "$REGION" glue get-database --name "$OUTPUT_DB" >/dev/null 2>&1; then
    run_aws glue create-database --database-input "{\"Name\":\"$OUTPUT_DB\"}"
  fi

  # Drop existing table metadata if present
  log "Dropping existing table if exists: ${OUTPUT_DB}.${new_table}"
  run_aws glue delete-table --database-name "$OUTPUT_DB" --name "$new_table" || true
  # Remove existing data files from S3 to allow overwrite
  log "Removing existing S3 data at ${output_location}/"
  run_aws s3 rm --recursive "${output_location}/" || true

  # Build CTAS query into a temporary file
  local tmp_sql
  # Create a temporary SQL file (portable mktemp invocation)
  tmp_sql=$(mktemp -t "${SCRIPT_NAME}.${set_type}.XXXXXX")
  tmp_sql="${tmp_sql}.sql"
  cat >"$tmp_sql" <<EOF
CREATE TABLE ${OUTPUT_DB}.${new_table}
WITH (
  format = 'PARQUET',
  external_location = '${output_location}/',
  parquet_compression = 'SNAPPY'
) AS
-- First compute wind_bin and hash_val, then derive row numbers in a nested CTE
WITH base AS (
  -- Generate reproducible hash keys per event and append wind/bin metadata.
  SELECT
    *,
    ${wind_case} AS wind_bin,
    ${id_col_expr}
    crc32(
      CAST(concat(
        CAST(timestamp AS VARCHAR),
        ${id_concat_expr},
        '${SEED}'
      ) AS varbinary)
    ) AS hash_val
  FROM ${DB_NAME}.${TABLE_NAME}
),
ranked AS (
  -- Rank rows by hash within each identifier and wind bin to track coverage.
  SELECT
    base.*,
    row_number() OVER (PARTITION BY ${id_value_name} ORDER BY hash_val) AS row_id,
    row_number() OVER (PARTITION BY wind_bin ORDER BY hash_val) AS row_wind_bin,
    COUNT(*) OVER (PARTITION BY wind_bin) AS wind_bin_total
  FROM base
),
annotated AS (
  -- Flag rows that must belong to the test set to guarantee coverage.
  SELECT
    ranked.*,
    CASE
      WHEN ${EXCLUDE_CONDITION} THEN TRUE
      WHEN row_id = 1 THEN TRUE
      WHEN row_wind_bin = 1 THEN TRUE
      ELSE FALSE
    END AS forced_test
  FROM ranked
),
quota AS (
  -- Compute per-bin quotas and cumulative counts of forced test members.
  SELECT
    annotated.*,
    SUM(CASE WHEN forced_test THEN 1 ELSE 0 END) OVER (PARTITION BY wind_bin) AS forced_test_count,
    SUM(CASE WHEN forced_test THEN 1 ELSE 0 END) OVER (PARTITION BY wind_bin ORDER BY hash_val ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS forced_test_cume,
    CAST(ROUND(wind_bin_total * ${TEST_FRAC}) AS INTEGER) AS base_test_quota
  FROM annotated
),
allocation AS (
  -- Reconcile theoretical quotas with forced assignments to derive targets.
  SELECT
    quota.*,
    LEAST(
      wind_bin_total,
      GREATEST(base_test_quota, forced_test_count, 1)
    ) AS target_test_quota,
    row_wind_bin - forced_test_cume AS non_forced_rank
  FROM quota
),
balanced AS (
  -- Translate the gap into a residual quota available for randomised rows.
  SELECT
    allocation.*,
    GREATEST(target_test_quota - forced_test_count, 0) AS residual_test_quota
  FROM allocation
),
labeled AS (
  -- Assign each row to train/test, preserving forced placements.
  SELECT
    balanced.*,
    CASE
${EXCLUDE_WHEN_TRAIN_BOOL}
      WHEN forced_test THEN FALSE
      WHEN non_forced_rank <= residual_test_quota THEN FALSE
      ELSE TRUE
    END AS is_train
  FROM balanced
),
augmented AS (
  -- Count training rows per bin in hash order to generate fold identifiers.
  SELECT
    labeled.*,
    SUM(
      CASE
        WHEN is_train THEN 1
        ELSE 0
      END
    ) OVER (PARTITION BY wind_bin ORDER BY hash_val ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS train_row_number
  FROM labeled
),
final AS (
  -- Project fold labels and expose the split flags consumed downstream.
  SELECT
    augmented.*,
    CASE
${EXCLUDE_WHEN_FOLD}
      WHEN NOT is_train THEN 0
      ELSE ((train_row_number - 1) % ${FOLDS}) + 1
    END AS fold,
    CASE
      WHEN is_train THEN 'train'
      ELSE 'test'
    END AS set_type
  FROM augmented
)
SELECT
  *
FROM final
WHERE set_type = '${set_type}';
EOF

  log "Submitting CTAS for '${set_type}' as table '${new_table}'..."
  local response
  response=$(run_aws athena start-query-execution \
    --query-execution-context Database="$OUTPUT_DB" \
    --result-configuration "OutputLocation=${results_uri}" \
    --query-string "file://${tmp_sql}")
  # Log SQL snippet for debugging
  log "SQL snippet for '${set_type}' (first 20 lines):"
  sed -n '1,20p' "$tmp_sql" >> "$LOG_FILE"
  log ""
  rm -f "$tmp_sql"

  local qid
  qid=$(echo "$response" | jq -r '.QueryExecutionId')
  log "QueryExecutionId: $qid"
  wait_for_query "$qid"
  log "Table ${DB_NAME}.${new_table} created at ${output_location}/"
}

# Materialise both train and test partitions sequentially so that any failure
# surfaces early and leaves the log in a consistent state.
for set in train test; do
  run_ctas "$set"
done

log "All partitions created successfully (Parquet only; CSV export disabled)."

## -----------------------------------------------------------------------------
# Generate wind direction distribution report in Markdown format.
## -----------------------------------------------------------------------------
log "Generating wind direction distribution report: ${REPORT_FILE}"
echo "# Partition Report for ${OUTPUT_DB}" > "$REPORT_FILE"
echo "Generated on $(date)" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
if [ -n "${SPEED_LIMITS:-}" ]; then
  echo "## Wind Speed Strata" >> "$REPORT_FILE"
  echo "" >> "$REPORT_FILE"
  echo "| speed_bin | speed_range (m/s) |" >> "$REPORT_FILE"
  echo "| --- | --- |" >> "$REPORT_FILE"
  prev=0
  for idx in "${!SPEED_BOUNDS[@]}"; do
    limit=${SPEED_BOUNDS[$idx]}
    echo "| $idx | ${prev}-${limit} |" >> "$REPORT_FILE"
    prev=$limit
  done
  echo "| ${#SPEED_BOUNDS[@]} | ${prev}-+Inf |" >> "$REPORT_FILE"
  echo "" >> "$REPORT_FILE"
fi

# Summary of row counts for source and each partition to verify conservation of records.
log "Computing summary row counts"
 count_sql="SELECT 'source' AS set_type, COUNT(*) AS cnt FROM ${DB_NAME}.${TABLE_NAME} \
UNION ALL SELECT 'train', COUNT(*) FROM ${OUTPUT_DB}.${TABLE_NAME}_train \
UNION ALL SELECT 'test', COUNT(*) FROM ${OUTPUT_DB}.${TABLE_NAME}_test;"
 qid=$(aws --profile "$PROFILE" --region "$REGION" athena start-query-execution \
   --query-execution-context Database="$OUTPUT_DB" \
   --result-configuration "OutputLocation=${RESULTS_URI}" \
   --query-string "$count_sql" \
   --output text --query 'QueryExecutionId')
 wait_for_query "$qid"
 counts_json=$(aws --profile "$PROFILE" --region "$REGION" athena get-query-results \
   --query-execution-id "$qid" --output json)

 # Extract counts
   source_cnt=$(echo "$counts_json" | jq -r '.ResultSet.Rows[] | select(.Data[0].VarCharValue=="source") .Data[1].VarCharValue')
   train_cnt=$(echo "$counts_json" | jq -r '.ResultSet.Rows[] | select(.Data[0].VarCharValue=="train") .Data[1].VarCharValue')
   test_cnt=$(echo "$counts_json" | jq -r '.ResultSet.Rows[] | select(.Data[0].VarCharValue=="test") .Data[1].VarCharValue')

 echo "## Summary of Row Counts" >> "$REPORT_FILE"
 echo "" >> "$REPORT_FILE"
 echo "| set | count | % of source |" >> "$REPORT_FILE"
 echo "| --- | ---: | ---: |" >> "$REPORT_FILE"
echo "| source | $source_cnt | 100.00% |" >> "$REPORT_FILE"
for set in train test; do
   cnt_var="${set}_cnt"
   cnt=${!cnt_var}
   pct=$(awk "BEGIN {printf \"%.2f\", ($cnt/$source_cnt*100)}")
   echo "| $set | $cnt | ${pct}% |" >> "$REPORT_FILE"
 done
 echo "" >> "$REPORT_FILE"

 for set in train test; do
  title=$(capitalize "$set")
  table_name="${TABLE_NAME}_${set}"
  log "Running distribution query for ${table_name}"
  qsql="SELECT wind_bin, COUNT(*) AS cnt FROM ${OUTPUT_DB}.${table_name} GROUP BY wind_bin ORDER BY wind_bin;"
  qid=$(aws --profile "$PROFILE" --region "$REGION" athena start-query-execution \
    --query-execution-context Database="$OUTPUT_DB" \
    --result-configuration "OutputLocation=${RESULTS_URI}" \
    --query-string "$qsql" \
    --output text --query 'QueryExecutionId')
  wait_for_query "$qid"
  log "Fetching generic distribution for ${table_name}"
  count_results_json=$(aws --profile "$PROFILE" --region "$REGION" athena get-query-results \
    --query-execution-id "$qid" --output json)

  echo "## ${title} Set" >> "$REPORT_FILE"
  echo "" >> "$REPORT_FILE"
  if [ "$set" = "train" ]; then
    echo "### Wind Stratified Distribution by Fold and Bin" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    echo "| fold | wind_bin | speed_range (m/s) | dir_range (°) | count | % of fold |" >> "$REPORT_FILE"
    echo "| --- | --- | --- | --- | ---: | ---: |" >> "$REPORT_FILE"
    fold_dist_sql="WITH dist AS (SELECT fold, wind_bin, COUNT(*) AS cnt FROM ${OUTPUT_DB}.${TABLE_NAME}_${set} GROUP BY fold, wind_bin) SELECT fold, wind_bin, cnt, SUM(cnt) OVER (PARTITION BY fold) AS total_cnt FROM dist ORDER BY fold, wind_bin;"
    qid=$(aws --profile "$PROFILE" --region "$REGION" athena start-query-execution --query-execution-context Database="$OUTPUT_DB" --result-configuration "OutputLocation=${RESULTS_URI}" --query-string "$fold_dist_sql" --output text --query 'QueryExecutionId')
    wait_for_query "$qid"
    fold_results_json=$(aws --profile "$PROFILE" --region "$REGION" athena get-query-results --query-execution-id "$qid" --output json)
    echo "$fold_results_json" | jq -r '.ResultSet.Rows[1:][] | [.Data[0].VarCharValue, .Data[1].VarCharValue, .Data[2].VarCharValue, .Data[3].VarCharValue] | "\(.[0]) \(.[1]) \(.[2]) \(.[3])"' | \
    while read fold bin cnt total_cnt; do
      # Decode speed_bin.direction_bin
      if [[ "$bin" == *.* ]]; then
        speed_bin=${bin%%.*}
        dir_bin=${bin##*.}
        if [ $speed_bin -eq 0 ]; then speed_start=0; else speed_start=${SPEED_BOUNDS[$((speed_bin-1))]}; fi
        if [ $speed_bin -lt ${#SPEED_BOUNDS[@]} ]; then speed_end=${SPEED_BOUNDS[$speed_bin]}; else speed_end="+Inf"; fi
        speed_range="${speed_start}-${speed_end}"
        range_start=$(awk "BEGIN {printf \"%.0f\", $dir_bin * $BIN_WIDTH}")
        range_end=$(awk "BEGIN {printf \"%.0f\", ($dir_bin + 1) * $BIN_WIDTH}")
        dir_range="${range_start}-${range_end}"
      else
        speed_range=""
        dir_bin=$bin
        range_start=$(awk "BEGIN {printf \"%.0f\", $bin * $BIN_WIDTH}")
        range_end=$(awk "BEGIN {printf \"%.0f\", ($bin + 1) * $BIN_WIDTH}")
        dir_range="${range_start}-${range_end}"
      fi
      pct=$(awk "BEGIN {printf \"%.2f\", ($cnt/$total_cnt*100)}")
      echo "| $fold | $bin | $speed_range | $dir_range | $cnt | ${pct}% |" >> "$REPORT_FILE"
    done
    echo "" >> "$REPORT_FILE"

    # Evaluate fold balance by wind_bin and capture potential imbalances
    balance_summary_tmp=$(mktemp -t "${SCRIPT_NAME}.balance_summary.XXXXXX")
    balance_warnings_tmp=$(mktemp -t "${SCRIPT_NAME}.balance_warnings.XXXXXX")
    fold_results_tmp=$(mktemp -t "${SCRIPT_NAME}.fold_results.XXXXXX")
    printf '%s\n' "$fold_results_json" >"$fold_results_tmp"
    # Use an inline Python helper to inspect the Athena payload and summarise
    # fold-level balance, surfacing warnings when ratios exceed acceptable
    # thresholds or when empty folds are detected.
    python3 - "$fold_results_tmp" <<'PY' >"$balance_summary_tmp" 2>"$balance_warnings_tmp"
import json
import sys
from collections import defaultdict

# sys.argv[1] contains the temporary file path with the Athena JSON payload
with open(sys.argv[1], 'r', encoding='utf-8') as fh:
    data = json.load(fh)
rows = data.get("ResultSet", {}).get("Rows", [])
if len(rows) <= 1:
    print("- No training rows to evaluate fold balance.")
    sys.exit(0)

records = []
for row in rows[1:]:
    cells = row.get("Data", [])
    if len(cells) < 3:
        continue
    fold_val = cells[0].get("VarCharValue")
    bin_val = cells[1].get("VarCharValue")
    cnt_val = cells[2].get("VarCharValue")
    if fold_val is None or bin_val is None or cnt_val is None:
        continue
    try:
        cnt = int(cnt_val)
    except ValueError:
        continue
    records.append((bin_val, fold_val, cnt))

by_bin = defaultdict(list)
for bin_val, fold_val, cnt in records:
    by_bin[bin_val].append((fold_val, cnt))

if not by_bin:
    print("- No training rows to evaluate fold balance.")
    sys.exit(0)

lines = []
warnings = []
for bin_val in sorted(by_bin.keys()):
    entries = by_bin[bin_val]
    counts = [cnt for _, cnt in entries]
    total = sum(counts)
    folds = len(entries)
    avg = total / folds if folds else 0.0
    min_cnt = min(counts)
    max_cnt = max(counts)
    zero_folds = [fold for fold, cnt in entries if cnt == 0]
    ratio = (max_cnt / min_cnt) if min_cnt else float("inf")
    if zero_folds:
        status = f"[WARN] fold(s) {', '.join(zero_folds)} empty"
        warnings.append(f"wind_bin {bin_val} has empty folds: {', '.join(zero_folds)}")
    elif ratio > 1.5:
        status = f"[WARN] max/min ratio {ratio:.2f}"
        warnings.append(f"wind_bin {bin_val} is unbalanced (ratio {ratio:.2f})")
    else:
        status = f"[OK] max/min ratio {ratio:.2f}"
    lines.append(f"- wind_bin `{bin_val}`: total {total}, expected avg {avg:.1f}, min {min_cnt}, max {max_cnt} -> {status}")

if not lines:
    lines.append("- No training rows to evaluate fold balance.")

print("\n".join(lines))
for warning in warnings:
    print(warning, file=sys.stderr)
PY

    if [ -s "$balance_summary_tmp" ]; then
      echo "#### Fold Balance Checks" >> "$REPORT_FILE"
      echo "" >> "$REPORT_FILE"
      cat "$balance_summary_tmp" >> "$REPORT_FILE"
      echo "" >> "$REPORT_FILE"
    fi
    if [ -s "$balance_warnings_tmp" ]; then
      while IFS= read -r warning_line; do
        [ -z "$warning_line" ] && continue
        log "Fold balance warning: ${warning_line}"
        echo "> Warning: ${warning_line}" >> "$REPORT_FILE"
      done <"$balance_warnings_tmp"
      echo "" >> "$REPORT_FILE"
    else
      log "Fold balance check: all wind_bin distributions appear within threshold."
    fi
    rm -f "$balance_summary_tmp" "$balance_warnings_tmp" "$fold_results_tmp"
  fi
  # Generic distribution per bin with percentage of set
  echo "| wind_bin | speed_range (m/s) | dir_range (°) | count | % of set |" >> "$REPORT_FILE"
  echo "| --- | --- | --- | ---: | ---: |" >> "$REPORT_FILE"
  echo "$count_results_json" | jq -r '.ResultSet.Rows[1:][] | [.Data[0].VarCharValue, (.Data[1].VarCharValue // "0")] | "\(.[0]) \(.[1])"' | \
  while read bin cnt; do
    # Decode speed_bin.direction_bin format
    if [[ "$bin" == *.* ]]; then
      speed_bin=${bin%%.*}
      dir_bin=${bin##*.}
      if [ $speed_bin -eq 0 ]; then speed_start=0; else speed_start=${SPEED_BOUNDS[$((speed_bin-1))]}; fi
      if [ $speed_bin -lt ${#SPEED_BOUNDS[@]} ]; then speed_end=${SPEED_BOUNDS[$speed_bin]}; else speed_end="+Inf"; fi
      speed_range="${speed_start}-${speed_end}"
      range_start=$(awk "BEGIN {printf \"%.0f\", $dir_bin * $BIN_WIDTH}")
      range_end=$(awk "BEGIN {printf \"%.0f\", ($dir_bin + 1) * $BIN_WIDTH}")
      dir_range="${range_start}-${range_end}"
    else
      speed_range=""
      dir_bin=$bin
      range_start=$(awk "BEGIN {printf \"%.0f\", $bin * $BIN_WIDTH}")
      range_end=$(awk "BEGIN {printf \"%.0f\", ($bin + 1) * $BIN_WIDTH}")
      dir_range="${range_start}-${range_end}"
    fi
    total_var="${set}_cnt"
    total=${!total_var}
    if [ "$total" -eq 0 ]; then
      pct="0.00"
    else
      pct=$(awk "BEGIN {printf \"%.2f\", ($cnt / $total * 100)}")
    fi
    echo "| $bin | $speed_range | $dir_range | $cnt | ${pct}% |" >> "$REPORT_FILE"
    done
    echo "" >> "$REPORT_FILE"

  # Distribution by ${id_value_name}
  echo "### Distribution by ${id_value_name}" >> "$REPORT_FILE"
  echo "" >> "$REPORT_FILE"
  echo "| ${id_value_name} | count | % of set |" >> "$REPORT_FILE"
  echo "| --- | ---: | ---: |" >> "$REPORT_FILE"
  id_sql="SELECT ${id_value_name}, COUNT(*) AS cnt FROM ${OUTPUT_DB}.${TABLE_NAME}_${set} GROUP BY ${id_value_name} ORDER BY ${id_value_name};"
  qid=$(aws --profile "$PROFILE" --region "$REGION" athena start-query-execution \
    --query-execution-context Database="$OUTPUT_DB" \
    --result-configuration "OutputLocation=${RESULTS_URI}" \
    --query-string "$id_sql" \
    --output text --query 'QueryExecutionId')
  wait_for_query "$qid"
  id_json=$(aws --profile "$PROFILE" --region "$REGION" athena get-query-results \
    --query-execution-id "$qid" --output json)
  total_var="${set}_cnt"
  total=${!total_var}
  echo "$id_json" | jq -r '.ResultSet.Rows[1:][] | [.Data[0].VarCharValue, .Data[1].VarCharValue] | "\(.[0]) \(.[1])"' \
    | while read id cnt; do
      pct=$(awk "BEGIN {printf \"%.2f\", ($cnt/$total*100)}")
      echo "| $id | $cnt | ${pct}% |" >> "$REPORT_FILE"
    done
  echo "" >> "$REPORT_FILE"

  # Distribution by ${id_value_name} and wind_bin
  echo "### Distribution by ${id_value_name} and wind_bin" >> "$REPORT_FILE"
  echo "" >> "$REPORT_FILE"
  echo "| ${id_value_name} | wind_bin | speed_range (m/s) | dir_range (°) | count | % of id |" >> "$REPORT_FILE"
  echo "| --- | --- | --- | --- | ---: | ---: |" >> "$REPORT_FILE"
  id_bin_sql="WITH dist AS (SELECT ${id_value_name}, wind_bin, COUNT(*) AS cnt FROM ${OUTPUT_DB}.${TABLE_NAME}_${set} GROUP BY ${id_value_name}, wind_bin) SELECT ${id_value_name}, wind_bin, cnt, SUM(cnt) OVER (PARTITION BY ${id_value_name}) AS total FROM dist ORDER BY ${id_value_name}, wind_bin;"
  qid=$(aws --profile "$PROFILE" --region "$REGION" athena start-query-execution \
    --query-execution-context Database="$OUTPUT_DB" \
    --result-configuration "OutputLocation=${RESULTS_URI}" \
    --query-string "$id_bin_sql" \
    --output text --query 'QueryExecutionId')
  wait_for_query "$qid"
  id_bin_json=$(aws --profile "$PROFILE" --region "$REGION" athena get-query-results \
    --query-execution-id "$qid" --output json)
  echo "$id_bin_json" | jq -r '.ResultSet.Rows[1:][] | [.Data[0].VarCharValue, .Data[1].VarCharValue, .Data[2].VarCharValue, .Data[3].VarCharValue] | "\(.[0]) \(.[1]) \(.[2]) \(.[3])"' | \
    while read id bin cnt total; do
      # Decode speed_bin.direction_bin
      if [[ "$bin" == *.* ]]; then
        speed_bin=${bin%%.*}
        dir_bin=${bin##*.}
        if [ $speed_bin -eq 0 ]; then speed_start=0; else speed_start=${SPEED_BOUNDS[$((speed_bin-1))]}; fi
        if [ $speed_bin -lt ${#SPEED_BOUNDS[@]} ]; then speed_end=${SPEED_BOUNDS[$speed_bin]}; else speed_end="+Inf"; fi
        speed_range="${speed_start}-${speed_end}"
        range_start=$(awk "BEGIN {printf \"%.0f\", $dir_bin * $BIN_WIDTH}")
        range_end=$(awk "BEGIN {printf \"%.0f\", ($dir_bin + 1) * $BIN_WIDTH}")
        dir_range="${range_start}-${range_end}"
      else
        speed_range=""
        dir_bin=$bin
        range_start=$(awk "BEGIN {printf \"%.0f\", $bin * $BIN_WIDTH}")
        range_end=$(awk "BEGIN {printf \"%.0f\", ($bin + 1) * $BIN_WIDTH}")
        dir_range="${range_start}-${range_end}"
      fi
      pct=$(awk "BEGIN {printf \"%.2f\", ($cnt/$total*100)}")
      echo "| $id | $bin | $speed_range | $dir_range | $cnt | ${pct}% |" >> "$REPORT_FILE"
    done
  echo "" >> "$REPORT_FILE"

  # Optional fold-level analytics when the caller requested additional columns.
  if [ -n "${FOLD_STATS_COLUMNS:-}" ]; then
    set_heading=$(capitalize "$set")
    fold_numeric_written=false
    fold_categorical_written=false
    fold_any_written=false
    for raw_column in "${FOLD_STATS_ARRAY[@]}"; do
      column=$(echo "$raw_column" | xargs)
      if [ -z "$column" ]; then
        continue
      fi
      # Inspect Glue metadata to decide whether to treat the column as numeric.
      column_type=$(get_column_type "$column")
      log "Column '${column}' type resolved to '${column_type}'"
      if is_numeric_type "$column_type"; then
        if [ "$fold_numeric_written" = false ]; then
          echo "### ${set_heading} fold-level descriptive statistics" >> "$REPORT_FILE"
          echo "" >> "$REPORT_FILE"
          fold_numeric_written=true
        fi
        fold_any_written=true
        log "Treating column '${column}' as numeric (${column_type}) in set '${set}'"
        echo "#### Column \`${column}\`" >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
        tmp_stats_sql=$(mktemp -t "${SCRIPT_NAME}.${set}.foldstats.XXXXXX")
        tmp_stats_sql="${tmp_stats_sql}.sql"
        cat >"$tmp_stats_sql" <<SQL
SELECT
  fold,
  COUNT(${column}) AS count_non_null,
  COALESCE(format('%.4f', AVG(CAST(${column} AS DOUBLE))), 'null') AS mean_value,
  COALESCE(format('%.4f', stddev_samp(CAST(${column} AS DOUBLE))), 'null') AS stddev_value,
  COALESCE(format('%.4f', MIN(CAST(${column} AS DOUBLE))), 'null') AS min_value,
  COALESCE(format('%.4f', MAX(CAST(${column} AS DOUBLE))), 'null') AS max_value
FROM ${OUTPUT_DB}.${TABLE_NAME}_${set}
GROUP BY fold
ORDER BY fold;
SQL
        if ! stats_qid=$(aws --profile "$PROFILE" --region "$REGION" athena start-query-execution \
          --query-execution-context Database="$OUTPUT_DB" \
          --result-configuration "OutputLocation=${RESULTS_URI}" \
          --query-string "file://${tmp_stats_sql}" \
          --output text --query 'QueryExecutionId'); then
          log "Failed to submit fold statistics query for column '${column}'"
          echo "> Warning: Athena query failed for column \`${column}\` in set ${set}." >> "$REPORT_FILE"
          echo "" >> "$REPORT_FILE"
          rm -f "$tmp_stats_sql"
          continue
        fi
        wait_for_query "$stats_qid"
        if ! stats_json=$(aws --profile "$PROFILE" --region "$REGION" athena get-query-results \
          --query-execution-id "$stats_qid" --output json); then
          log "Failed to retrieve fold statistics results for column '${column}'"
          echo "> Warning: Unable to retrieve fold statistics for column \`${column}\`." >> "$REPORT_FILE"
          echo "" >> "$REPORT_FILE"
          rm -f "$tmp_stats_sql"
          continue
        fi
        rm -f "$tmp_stats_sql"
        stats_rows=$(echo "$stats_json" | jq -r '.ResultSet.Rows[1:][] | [.Data[0].VarCharValue, .Data[1].VarCharValue, .Data[2].VarCharValue, .Data[3].VarCharValue, .Data[4].VarCharValue, .Data[5].VarCharValue] | @tsv')
        if [ -z "$stats_rows" ]; then
          echo "> No data available to compute fold statistics for column \`${column}\`." >> "$REPORT_FILE"
          echo "" >> "$REPORT_FILE"
          continue
        fi
        echo "| fold | count | mean | stddev | min | max |" >> "$REPORT_FILE"
        echo "| ---: | ---: | ---: | ---: | ---: | ---: |" >> "$REPORT_FILE"
        while IFS=$'\t' read -r fold count mean stddev min max; do
          [ -z "$fold" ] && continue
          [ "$mean" = "null" ] && mean="-"
          [ "$stddev" = "null" ] && stddev="-"
          [ "$min" = "null" ] && min="-"
          [ "$max" = "null" ] && max="-"
          echo "| $fold | ${count:-0} | $mean | $stddev | $min | $max |" >> "$REPORT_FILE"
        done <<< "$stats_rows"
        echo "" >> "$REPORT_FILE"

        if [ ${#FOLD_STATS_CATEGORY_COLUMNS[@]} -gt 0 ]; then
          for category_col in "${FOLD_STATS_CATEGORY_COLUMNS[@]}"; do
            run_category_breakdown "${set}" "${column}" "${category_col}"
          done
          if [ ${#FOLD_STATS_CATEGORY_COLUMNS[@]} -gt 1 ]; then
            combined_cols="${FOLD_STATS_CATEGORY_COLUMNS[0]}"
            for extra_col in "${FOLD_STATS_CATEGORY_COLUMNS[@]:1}"; do
              combined_cols+="${combined_cols:+,}${extra_col}"
            done
            run_category_breakdown "${set}" "${column}" "${combined_cols}"
          fi
        fi
      elif [ "$column_type" = "unknown" ]; then
        log "Skipping fold statistics for unknown column '${column}'"
        echo "> Warning: Column \`${column}\` could not be found in the partition outputs; skipping." >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
        continue
      else
        if [ "$fold_categorical_written" = false ]; then
          echo "### ${set_heading} fold-level categorical distributions" >> "$REPORT_FILE"
          echo "" >> "$REPORT_FILE"
          fold_categorical_written=true
        fi
        fold_any_written=true
        log "Treating column '${column}' as categorical (${column_type}) in set '${set}'"
        echo "#### Column \`${column}\`" >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
        tmp_cat_sql=$(mktemp -t "${SCRIPT_NAME}.${set}.foldcat.XXXXXX")
        tmp_cat_sql="${tmp_cat_sql}.sql"
        cat >"$tmp_cat_sql" <<SQL
WITH value_counts AS (
  SELECT
    fold,
    COALESCE(CAST(${column} AS VARCHAR), '(null)') AS value,
    COUNT(*) AS cnt
  FROM ${OUTPUT_DB}.${TABLE_NAME}_${set}
  GROUP BY fold, COALESCE(CAST(${column} AS VARCHAR), '(null)')
),
ranked AS (
  SELECT
    fold,
    value,
    cnt,
    format('%.2f', cnt * 100.0 / SUM(cnt) OVER (PARTITION BY fold)) AS pct,
    ROW_NUMBER() OVER (PARTITION BY fold ORDER BY cnt DESC, value) AS value_rank,
    COUNT(*) OVER (PARTITION BY fold) AS category_count
  FROM value_counts
)
SELECT fold, value, cnt, pct, value_rank, category_count
FROM ranked
WHERE value_rank <= 10
ORDER BY fold, value_rank;
SQL
        if ! cat_qid=$(aws --profile "$PROFILE" --region "$REGION" athena start-query-execution \
          --query-execution-context Database="$OUTPUT_DB" \
          --result-configuration "OutputLocation=${RESULTS_URI}" \
          --query-string "file://${tmp_cat_sql}" \
          --output text --query 'QueryExecutionId'); then
          log "Failed to submit categorical fold distribution query for column '${column}'"
          echo "> Warning: Athena query failed for categorical distribution of column \`${column}\` in set ${set}." >> "$REPORT_FILE"
          echo "" >> "$REPORT_FILE"
          rm -f "$tmp_cat_sql"
          continue
        fi
        wait_for_query "$cat_qid"
        if ! cat_json=$(aws --profile "$PROFILE" --region "$REGION" athena get-query-results \
          --query-execution-id "$cat_qid" --output json); then
          log "Failed to retrieve categorical distribution results for column '${column}'"
          echo "> Warning: Unable to retrieve categorical distribution for column \`${column}\`." >> "$REPORT_FILE"
          echo "" >> "$REPORT_FILE"
          rm -f "$tmp_cat_sql"
          continue
        fi
        rm -f "$tmp_cat_sql"
        cat_rows=$(echo "$cat_json" | jq -r '.ResultSet.Rows[1:][] | [.Data[0].VarCharValue, .Data[1].VarCharValue, .Data[2].VarCharValue, .Data[3].VarCharValue, .Data[4].VarCharValue, .Data[5].VarCharValue] | @tsv')
        if [ -z "$cat_rows" ]; then
          echo "> No data available to compute categorical distribution for column \`${column}\`." >> "$REPORT_FILE"
          echo "" >> "$REPORT_FILE"
          continue
        fi
        echo "| fold | value | count | % of fold |" >> "$REPORT_FILE"
        echo "| ---: | --- | ---: | ---: |" >> "$REPORT_FILE"
        fold_summaries=""
        while IFS=$'\t' read -r fold value count pct rank category_total; do
          [ -z "$fold" ] && continue
          display_value="$value"
          [ -z "$display_value" ] && display_value="(null)"
          echo "| $fold | \`$display_value\` | $count | ${pct}% |" >> "$REPORT_FILE"
          if [[ "$rank" == "1" ]]; then
            printf -v fold_summaries '%s> Fold %s: top category \`%s\` covers %s%% (%s rows) across %s distinct values.\n' "$fold_summaries" "$fold" "$display_value" "$pct" "$count" "$category_total"
          fi
        done <<< "$cat_rows"
        echo "" >> "$REPORT_FILE"
        echo "> Showing top 10 categories per fold for column \`${column}\`. Additional categories are truncated." >> "$REPORT_FILE"
        if [ -n "$fold_summaries" ]; then
          printf "%s" "$fold_summaries" >> "$REPORT_FILE"
        fi
        echo "" >> "$REPORT_FILE"
      fi
    done
    if [ "$fold_any_written" = false ]; then
      echo "> Fold statistics requested but no valid column names were provided." >> "$REPORT_FILE"
      echo "" >> "$REPORT_FILE"
    fi
  fi

done

log "Partition report written to ${REPORT_FILE}"
