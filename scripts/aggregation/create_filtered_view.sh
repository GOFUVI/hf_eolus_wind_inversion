#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Author: Juan Luis Herrera Cortijo <juan.luis.herrera.cortijo@gmail.com>
# Created: 2025-10-16
# Disclaimer: This script is provided as-is for research workflows and must be validated before any operational deployment.
# -----------------------------------------------------------------------------

# Exit on error, unset var, or pipefail
set -euo pipefail

# ----------------------------------------------------------------------------
# create_filtered_view.sh
#
# Description:
#   Create (or replace) an Athena view that filters the output of
#   the pivot/join workflow (e.g., scripts/aggregation/join_pivoted_tables.sh) using user-provided SQL. The script
#   reads a SQL snippet or template from disk, builds the final SELECT query,
#   and submits a CREATE OR REPLACE VIEW statement via the AWS CLI.
#
# Requirements:
#   - AWS CLI v2
#
# Usage:
#   scripts/aggregation/create_filtered_view.sh \
#     --source analytics_source_db.source_table \
#     --view analytics_target_db.filtered_view \
#     --sql-file filters/apply_quality_mask.sql \
#     --results-s3 s3://example-bucket/athena-results/ \
#     --profile data_profile --region us-east-1
#
# SQL file conventions:
#   - If the file contains the token {{SOURCE_TABLE}}, it will be replaced by
#     the fully-qualified source table name and used as-is as the SELECT body.
#   - Otherwise, the file is assumed to hold either a WHERE clause (starting
#     with WHERE/AND/OR) or a raw boolean expression that will be appended to
#     "SELECT * FROM <source>". Leading and trailing semicolons are stripped.
#   - Provide an empty SQL file to copy all rows from the source table.
#
# Flags:
#   --preview    Print the generated CREATE VIEW statement and exit without
#                executing it.
# ----------------------------------------------------------------------------

SCRIPT_NAME=$(basename "$0")
PROFILE="${AWS_PROFILE:-default}"
REGION="${AWS_REGION:-}"
RESULTS_S3=""
SOURCE_SPEC=""
VIEW_SPEC=""
SQL_FILE=""
PREVIEW=false

# usage()
#   Print CLI usage instructions and exit with status 0. Called when the user
#   requests help or provides incomplete arguments.
usage() {
  cat << EOF
Usage: $SCRIPT_NAME --source <db.table> --view <db.view> --sql-file <path> --results-s3 <s3://bucket/prefix/> [options]

Required:
  --source <db.table>       Fully-qualified source table produced by join_pivoted_tables.sh
  --view <db.view>          Target view (database.table) to create or replace
  --sql-file <path>         Path to SQL snippet or template with filter logic
  --results-s3 <s3://...>   S3 location for Athena query results

Optional:
  --profile <name>          AWS CLI profile (default: ${PROFILE})
  --region <name>           AWS region (default: from profile)
  --preview                 Print the CREATE VIEW statement without executing it
  -h|--help                 Show this help message

SQL file conventions:
  * Use {{SOURCE_TABLE}} to reference the source table inside a custom SELECT.
  * Without the placeholder, the file may start with WHERE/AND/OR or hold a raw
    boolean expression that will be wrapped in a WHERE clause.
EOF
}

# log()
#   Emit a timestamped message to stdout, providing consistent diagnostics.
log() { echo "$(date '+%Y-%m-%d %H:%M:%S') - $*"; }

# run_aws()
#   Invoke the AWS CLI with the configured profile and optional region.
#   All AWS calls in this script go through this helper to ensure consistent
#   credential scoping.
run_aws() {
  if [ -n "$REGION" ]; then
    aws --profile "$PROFILE" --region "$REGION" "$@"
  else
    aws --profile "$PROFILE" "$@"
  fi
}

# wait_for_query(qid)
#   Poll Athena until the submitted query finishes. The function exits with an
#   error if the query fails or is cancelled, surfacing the reason provided by
#   Athena for easier debugging.
wait_for_query() {
  local qid="$1"
  while true; do
    local state reason
    state=$(run_aws athena get-query-execution \
      --query-execution-id "$qid" \
      --query 'QueryExecution.Status.State' \
      --output text)
    case "$state" in
      SUCCEEDED)
        return 0
        ;;
      FAILED|CANCELLED)
        reason=$(run_aws athena get-query-execution \
          --query-execution-id "$qid" \
          --query 'QueryExecution.Status.StateChangeReason' \
          --output text)
        log "Athena query $qid ended with $state: ${reason:-"(no reason provided)"}"
        exit 1
        ;;
      *)
        sleep 3
        ;;
    esac
  done
}

# ensure_region()
#   Resolve the AWS region either from the environment flag or from the named
#   profile. Athena requires an explicit region, so the script aborts if none is
#   found.
ensure_region() {
  if [ -z "$REGION" ]; then
    REGION=$(aws configure get region --profile "$PROFILE")
  fi
  if [ -z "$REGION" ]; then
    echo "Error: AWS region not specified and not set in profile $PROFILE" >&2
    exit 1
  fi
}

# ensure_db(db)
#   Guarantee that the Glue database referenced by the target view exists. The
#   call is idempotent: it queries for the database and creates it only when
#   missing.
ensure_db() {
  local db="$1"
  if ! run_aws glue get-database --name "$db" >/dev/null 2>&1; then
    log "Creating Glue database: $db"
    run_aws glue create-database --database-input "{\"Name\":\"$db\"}"
  fi
}

# parse_db_tbl(spec)
#   Split a fully qualified identifier into database and table components.
#   Exits if the specification does not match the expected database.table form.
parse_db_tbl() {
  local spec="$1"
  if [[ "$spec" != *.* ]]; then
    echo "Error: Expected database.table format, got '$spec'" >&2
    exit 1
  fi
  local db="${spec%%.*}"
  local tbl="${spec#*.}"
  if [ -z "$db" ] || [ -z "$tbl" ]; then
    echo "Error: Invalid database.table format '$spec'" >&2
    exit 1
  fi
  echo "$db" "$tbl"
}

# read_sql_file(path)
#   Load the SQL template from disk, remove Windows carriage returns, and strip
#   trailing semicolons. Preserves the original formatting to maintain
#   indentation in complex statements or CTEs.
read_sql_file() {
  local path="$1"
  if [ ! -f "$path" ]; then
    echo "Error: SQL file '$path' not found" >&2
    exit 1
  fi
  # Preserve formatting while removing trailing semicolons and CR characters.
  local content
  content=$(cat "$path")
  content=${content//$'\r'/}
  printf '%s' "$content" | sed 's/[[:space:]]*;[[:space:]]*$//'
}

# build_select_query(raw_sql, source_fqn)
#   Assemble the SELECT body for the view. The logic supports template
#   substitution, WHERE/AND/OR snippets, raw boolean expressions, or full SELECT
#   statements. Empty input defaults to a passthrough SELECT * projection.
build_select_query() {
  local raw_sql="$1"
  local source_fqn="$2"
  local trimmed leading_word

  if [[ "$raw_sql" == *"{{SOURCE_TABLE}}"* ]]; then
    local replaced
    replaced="${raw_sql//\{\{SOURCE_TABLE\}\}/$source_fqn}"
    printf '%s' "$replaced"
    return
  fi

  trimmed=$(printf '%s' "$raw_sql" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
  if [ -z "$trimmed" ]; then
    printf 'SELECT * FROM %s' "$source_fqn"
    return
  fi

  leading_word=$(printf '%s' "$trimmed" | awk '{print toupper($1); exit}')
  case "$leading_word" in
    SELECT|WITH)
      printf '%s' "$raw_sql"
      ;;
    WHERE|AND|OR)
      printf 'SELECT * FROM %s
%s' "$source_fqn" "$trimmed"
      ;;
    *)
      printf 'SELECT * FROM %s
WHERE %s' "$source_fqn" "$trimmed"
      ;;
  esac
}

# ---------------------------------------------------------------------------
# Main execution flow: parse arguments, prepare supporting context, build the
# CREATE OR REPLACE VIEW statement, optionally preview it, otherwise run it and
# wait for completion.
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --source)
      SOURCE_SPEC="$2"; shift 2 ;;
    --view)
      VIEW_SPEC="$2"; shift 2 ;;
    --sql-file)
      SQL_FILE="$2"; shift 2 ;;
    --results-s3)
      RESULTS_S3="$2"; shift 2 ;;
    --profile)
      PROFILE="$2"; shift 2 ;;
    --region)
      REGION="$2"; shift 2 ;;
    --preview)
      PREVIEW=true; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Error: Unknown argument '$1'" >&2
      usage
      exit 1
      ;;
  esac
done

if [ -z "$SOURCE_SPEC" ] || [ -z "$VIEW_SPEC" ] || [ -z "$SQL_FILE" ]; then
  echo "Error: --source, --view and --sql-file are required" >&2
  usage
  exit 1
fi

if [ ! -f "$SQL_FILE" ]; then
  echo "Error: SQL file '$SQL_FILE' not found" >&2
  exit 1
fi

read -r SOURCE_DB SOURCE_TBL < <(parse_db_tbl "$SOURCE_SPEC")
read -r VIEW_DB VIEW_TBL < <(parse_db_tbl "$VIEW_SPEC")

SOURCE_FQN="$SOURCE_DB.$SOURCE_TBL"
VIEW_FQN="$VIEW_DB.$VIEW_TBL"

RAW_SQL=$(read_sql_file "$SQL_FILE")
SELECT_QUERY=$(build_select_query "$RAW_SQL" "$SOURCE_FQN")

ATHENA_QUERY=$(cat <<SQL
CREATE OR REPLACE VIEW $VIEW_FQN AS
$SELECT_QUERY
;
SQL
)

if $PREVIEW; then
  printf '%s
' "$ATHENA_QUERY"
  exit 0
fi

if [ -z "$RESULTS_S3" ]; then
  echo "Error: --results-s3 is required when executing the query" >&2
  exit 1
fi

ensure_region
# Ensure the Glue catalog has the database so the CREATE VIEW statement does not
# fail because of a missing schema.
ensure_db "$VIEW_DB"

log "Submitting CREATE OR REPLACE VIEW for $VIEW_FQN"
QID=$(run_aws athena start-query-execution \
  --query-string "$ATHENA_QUERY" \
  --query-execution-context "Database=$VIEW_DB" \
  --result-configuration "OutputLocation=$RESULTS_S3" \
  --query 'QueryExecutionId' \
  --output text)

wait_for_query "$QID"
log "View $VIEW_FQN created successfully"
