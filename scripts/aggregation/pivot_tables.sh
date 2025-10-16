#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Author: Juan Luis Herrera Cortijo <juan.luis.herrera.cortijo@gmail.com>
# Created: 2025-10-16
# Disclaimer: This script is provided as-is for research workflows and must be validated before any operational deployment.
# -----------------------------------------------------------------------------

# Exit on error, unset var, or pipefail
set -euo pipefail

# ----------------------------------------------------------------------------
# pivot_tables.sh
#
# Description:
#   Drives the pivot of one or more HF-radar aggregate tables catalogued in
#   Athena so that Bragg sidebands become explicit columns while temporal,
#   node, and geometry identifiers remain untouched. The script creates or
#   reuses Glue databases, validates that each node is tied to a single geometry
#   (when requested), and materialises the pivot through CREATE TABLE AS (CTAS)
#   statements that publish Parquet-backed external tables to S3.
#   This standalone pivot stage generalises the behaviour previously embedded in
#   pivot_and_join.sh, enabling pipelines to reuse a single entry point.
#
# Usage highlights:
#   1. Accepts one or more --pivot specifications (input table, output table,
#      and destination S3 prefix).
#   2. Negotiates the AWS region and Glue database lifecycle before issuing
#      Athena queries.
#   3. Drops any pre-existing S3 payload and Glue metadata so that the CTAS run
#      is reproducible.
#
# Requirements:
#   - AWS CLI v2
#   - jq
# ----------------------------------------------------------------------------

SCRIPT_NAME=$(basename "$0")
PROFILE="${AWS_PROFILE:-default}"
REGION="${AWS_REGION:-}"
RESULTS_S3=""

TS_COL="timestamp"
NODE_COL="node_id"
GEOM_COL="geometry"
POSBRAGG_COL="pos_bragg"
PREFIX_MODE="table"    # table | none
REQUIRE_GEOM_UNIQUE=false

declare -a PIVOT_SPECS

# Print CLI help including the repeatable --pivot argument shape.
usage() {
  cat <<EOF
Usage: $SCRIPT_NAME --pivot <in_db.in_tbl=out_db.out_tbl@s3://path/> [options]

Repeat --pivot to process multiple tables. All pivot outputs are materialized as
Athena external tables backed by Parquet files.

Options:
  --pivot <in_db>.<in_tbl>=<out_db>.<out_tbl>@s3://path/   Pivot specification (repeatable)
  --results-s3 <s3://...>        Athena query results location (defaults to first pivot path)
  --profile <name>               AWS CLI profile (default: ${PROFILE})
  --region <name>                AWS region (default: from profile configuration)
  --ts-col <name>                Timestamp column name (default: ${TS_COL})
  --node-col <name>              Node identifier column (default: ${NODE_COL})
  --geom-col <name>              Geometry column name (default: ${GEOM_COL})
  --posbragg-col <name>          pos_bragg column used for the pivot (default: ${POSBRAGG_COL})
  --prefix-mode <table|none>     Prefix pivoted columns with sanitized table name (default: table)
  --require-geometry-unique      Fail if a node maps to more than one geometry
  -h|--help                      Show this help message
EOF
}

# Emit timestamped log lines to ease cross-referencing with Athena runs.
log() { echo "$(date '+%Y-%m-%d %H:%M:%S') - $*"; }

# Wrapper around aws CLI that applies the configured profile and, when present, region.
run_aws() {
  if [ -n "$REGION" ]; then
    aws --profile "$PROFILE" --region "$REGION" "$@"
  else
    aws --profile "$PROFILE" "$@"
  fi
}

# Block until the target Athena query finishes, failing fast on errors.
wait_for_query() {
  local qid="$1"
  while true; do
    local out state
    out=$(run_aws athena get-query-execution --query-execution-id "$qid")
    state=$(echo "$out" | jq -r '.QueryExecution.Status.State')
    case "$state" in
      SUCCEEDED) return 0 ;;
      FAILED|CANCELLED)
        local reason
        reason=$(echo "$out" | jq -r '.QueryExecution.Status.StateChangeReason // "(no reason provided)"')
        log "Athena query $qid ended with $state: $reason"
        exit 1
        ;;
      *) sleep 3 ;;
    esac
  done
}

# Infer region if not provided, mirroring the user's AWS CLI configuration.
ensure_region() {
  if [ -z "$REGION" ]; then
    REGION=$(aws configure get region --profile "$PROFILE" 2>/dev/null || true)
  fi
  if [ -z "$REGION" ]; then
    echo "Error: AWS region not specified and not found in profile $PROFILE" >&2
    exit 1
  fi
}

# Create the destination Glue database when it does not already exist.
ensure_db() {
  local db="$1"
  if ! run_aws glue get-database --name "$db" >/dev/null 2>&1; then
    log "Creating Glue database: $db"
    run_aws glue create-database --database-input "{"Name":"$db"}"
  fi
}

# Split a db.table specification and return both components.
parse_db_tbl() {
  local spec="$1"
  local db="${spec%%.*}"
  local tbl="${spec#*.}"
  echo "$db $tbl"
}

# Generate a sanitised prefix used to namespace pivoted columns.
sanitize_prefix() {
  echo "$1" | tr '[:upper:]' '[:lower:]' | tr -cd 'a-z0-9_'
}

# Optionally prepend the prefix to a column depending on the selected mode.
prefix_col() {
  local pfx="$1" col="$2"
  if [ "$PREFIX_MODE" = "table" ]; then
    echo "${pfx}__${col}"
  else
    echo "$col"
  fi
}

# Recover column metadata from Glue using information_schema, preserving order.
get_columns_json() {
  local db="$1" tbl="$2"
  local db_lc tbl_lc
  db_lc=$(echo "$db" | tr '[:upper:]' '[:lower:]')
  tbl_lc=$(echo "$tbl" | tr '[:upper:]' '[:lower:]')
  local q="SELECT column_name, data_type FROM information_schema.columns WHERE table_schema='${db_lc}' AND table_name='${tbl_lc}' ORDER BY ordinal_position;"
  local qid
  qid=$(run_aws athena start-query-execution     --query-execution-context Database="$db"     --result-configuration OutputLocation="${RESULTS_S3}"     --query-string "$q" --output text --query 'QueryExecutionId')
  wait_for_query "$qid"
  run_aws athena get-query-results --query-execution-id "$qid" --output json     | jq '{cols: [.ResultSet.Rows[1:][] | {name: .Data[0].VarCharValue, type: .Data[1].VarCharValue}]}'
}

# Detect nodes associated with more than one geometry, which would break the pivot assumption.
geom_uniqueness_check() {
  local db="$1" tbl="$2"
  local q="SELECT ${NODE_COL} AS node_id, COUNT(DISTINCT ${GEOM_COL}) AS distinct_geoms FROM ${db}.${tbl} GROUP BY ${NODE_COL} HAVING COUNT(DISTINCT ${GEOM_COL}) > 1 LIMIT 20;"
  local qid
  qid=$(run_aws athena start-query-execution     --query-execution-context Database="$db"     --result-configuration OutputLocation="${RESULTS_S3}"     --query-string "$q" --output text --query 'QueryExecutionId')
  wait_for_query "$qid"
  local res
  res=$(run_aws athena get-query-results --query-execution-id "$qid" --output json)
  local n
  n=$(echo "$res" | jq '.ResultSet.Rows | length')
  if [ "$n" -gt 1 ]; then
    log "WARNING: Non-unique geometry per node detected in ${db}.${tbl}. Showing offenders:"
    echo "$res" | jq -r '.ResultSet.Rows[1:][] | [.Data[0].VarCharValue, .Data[1].VarCharValue] | @tsv' | sed 's/^/  - /'
    if [ "$REQUIRE_GEOM_UNIQUE" = true ]; then
      echo "Error: Geometry uniqueness check failed and --require-geometry-unique is set" >&2
      exit 1
    fi
  else
    log "Geometry uniqueness OK for ${db}.${tbl}"
  fi
}

# Drop both Glue table metadata and its backing S3 folder so the pivot is rebuilt from scratch.
drop_table_and_s3() {
  local odb="$1" otbl="$2" s3="$3"
  log "Dropping Glue table if exists: ${odb}.${otbl}"
  run_aws glue delete-table --database-name "$odb" --name "$otbl" >/dev/null 2>&1 || true
  log "Removing existing S3 data at: ${s3}"
  run_aws s3 rm --recursive "$s3" >/dev/null 2>&1 || true
}

# Execute a CREATE TABLE AS statement within Athena, blocking until completion.
ctas() {
  local db="$1" sql="$2"
  local qid
  qid=$(run_aws athena start-query-execution     --query-execution-context Database="$db"     --result-configuration OutputLocation="${RESULTS_S3}"     --query-string "$sql" --output text --query 'QueryExecutionId')
  wait_for_query "$qid"
}

# Compose the CTAS payload that widens the pos_bragg dimension into explicit columns.
build_pivot_sql() {
  local idb="$1" itbl="$2" odb="$3" otbl="$4" s3="$5" pfx="$6"
  local cols_json pivot_cols keep_cols select_list
  cols_json=$(get_columns_json "$idb" "$itbl")
  # Ensure we keep the temporal, node, and geometry identifiers untouched.
  keep_cols=$(echo "$cols_json" | jq -r --arg TS "$TS_COL" --arg NODE "$NODE_COL" --arg GEOM "$GEOM_COL" --arg PB "$POSBRAGG_COL" '.cols | map(select(.name==$TS or .name==$NODE or .name==$GEOM)) | map(.name) | @tsv')
  if [ -z "$keep_cols" ]; then
    echo "Error: Could not find key columns (${TS_COL}, ${NODE_COL}, ${GEOM_COL}) in ${idb}.${itbl}" >&2
    exit 1
  fi
  # All remaining metrics are candidates for the pivot; each is duplicated for both Bragg sides.
  pivot_cols=$(echo "$cols_json" | jq -r --arg TS "$TS_COL" --arg NODE "$NODE_COL" --arg GEOM "$GEOM_COL" --arg PB "$POSBRAGG_COL" '.cols | map(select((.name!=$TS) and (.name!=$NODE) and (.name!=$GEOM) and (.name!=$PB))) | map(.name) | @tsv')
  local ts node geom
  read -r ts node geom <<< "$keep_cols"
  select_list="${ts}, ${node}, ${geom}"
  if [ -n "$pivot_cols" ]; then
    for col in $pivot_cols; do
      local c0 c1
      c0=$(prefix_col "$pfx" "${col}_0")
      c1=$(prefix_col "$pfx" "${col}_1")
      # Compute sideband-specific aggregates, yielding at most one value per dimension tuple.
      select_list+=$'
  , MAX(CASE WHEN '
      select_list+="${POSBRAGG_COL}=0 THEN ${col} END) AS ${c0}"
      select_list+=$'
  , MAX(CASE WHEN '
      select_list+="${POSBRAGG_COL}=1 THEN ${col} END) AS ${c1}"
    done
  fi
  # Require both sidebands to exist so that the wide record is physically meaningful.
  cat <<EOSQL
CREATE TABLE ${odb}.${otbl}
WITH (
  format = 'PARQUET',
  external_location = '${s3%/}/'
)
AS
SELECT
  ${select_list}
FROM ${idb}.${itbl}
GROUP BY ${ts}, ${node}, ${geom}
HAVING
  SUM(CASE WHEN ${POSBRAGG_COL}=0 THEN 1 ELSE 0 END) > 0
  AND SUM(CASE WHEN ${POSBRAGG_COL}=1 THEN 1 ELSE 0 END) > 0
EOSQL
}

if [ $# -eq 0 ]; then
  usage
  exit 1
fi

while [ $# -gt 0 ]; do
  # Parse CLI arguments, capturing each pivot specification verbatim.
  case "$1" in
    --pivot)
      [ $# -lt 2 ] && { echo "Error: --pivot requires an argument" >&2; exit 1; }
      PIVOT_SPECS+=("$2"); shift 2 ;;
    --results-s3)
      [ $# -lt 2 ] && { echo "Error: --results-s3 requires an argument" >&2; exit 1; }
      RESULTS_S3="$2"; shift 2 ;;
    --profile)
      [ $# -lt 2 ] && { echo "Error: --profile requires an argument" >&2; exit 1; }
      PROFILE="$2"; shift 2 ;;
    --region)
      [ $# -lt 2 ] && { echo "Error: --region requires an argument" >&2; exit 1; }
      REGION="$2"; shift 2 ;;
    --ts-col)
      TS_COL="$2"; shift 2 ;;
    --node-col)
      NODE_COL="$2"; shift 2 ;;
    --geom-col)
      GEOM_COL="$2"; shift 2 ;;
    --posbragg-col)
      POSBRAGG_COL="$2"; shift 2 ;;
    --prefix-mode)
      PREFIX_MODE="$2"; shift 2 ;;
    --require-geometry-unique)
      REQUIRE_GEOM_UNIQUE=true; shift 1 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1 ;;
  esac
done

if [ ${#PIVOT_SPECS[@]} -eq 0 ]; then
  echo "Error: at least one --pivot must be provided" >&2
  exit 1
fi

# Region resolution must happen before any Athena or Glue interaction.
ensure_region

if [ -z "$RESULTS_S3" ]; then
  # Default Athena spill bucket to the first pivot destination so metadata lives together.
  first_s3="${PIVOT_SPECS[0]#*@}"
  RESULTS_S3="${first_s3%/}/athena-results/"
fi

log "AWS Profile: ${PROFILE} | Region: ${REGION}"
log "Athena results: ${RESULTS_S3}"

# Iterate over each pivot job, rebuilding the target table and its data payload.
for spec in "${PIVOT_SPECS[@]}"; do
  in_part="${spec%%=*}"
  right="${spec#*=}"
  out_part="${right%@*}"
  s3_out="${right#*@}"
  read -r in_db in_tbl   <<< "$(parse_db_tbl "$in_part")"
  read -r out_db out_tbl <<< "$(parse_db_tbl "$out_part")"
  ensure_db "$out_db"
  drop_table_and_s3 "$out_db" "$out_tbl" "$s3_out"
  geom_uniqueness_check "$in_db" "$in_tbl"
  pfx=$(sanitize_prefix "$in_tbl")
  log "Pivoting ${in_db}.${in_tbl} -> ${out_db}.${out_tbl} (prefix=${pfx})"
  # Materialise the pivot immediately after composing the CTAS statement.
  pivot_sql=$(build_pivot_sql "$in_db" "$in_tbl" "$out_db" "$out_tbl" "$s3_out" "$pfx")
  ctas "$in_db" "$pivot_sql"
  log "Pivot finished for ${out_db}.${out_tbl}"
done

log "All pivots completed."
