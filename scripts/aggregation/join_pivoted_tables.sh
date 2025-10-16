#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Author: Juan Luis Herrera Cortijo <juan.luis.herrera.cortijo@gmail.com>
# Created: 2025-10-16
# Disclaimer: This script is provided as-is for research workflows and must be validated before any operational deployment.
# -----------------------------------------------------------------------------

# Exit on error, unset var, or pipefail
set -euo pipefail

# ----------------------------------------------------------------------------
# join_pivoted_tables.sh
#
# Description:
#   Performs a chained inner join across multiple pivoted HF-radar tables using
#   (timestamp, node_id, geometry) as composite keys and optionally enriches the
#   resulting feature set with SAR and buoy measurements. Each stage is
#   materialised through Athena CTAS queries into Parquet-backed Glue tables so
#   downstream pipelines can rely on consistent schemas and partition layouts.
#
# Typical flow:
#   1. Drop and recreate the destination Glue tables and clear the target S3
#      prefixes to guarantee idempotent reruns.
#   2. Build a CTAS query that anchors the schema on the first pivoted source
#      and inner joins all other sources on the shared key trio.
#   3. Optionally join SAR and buoy datasets, prefixing their columns when
#      requested to avoid name collisions.
#
# Requirements:
#   - AWS CLI v2
#   - jq
#   - An Athena workgroup/profile with permissions to create CTAS tables and
#     manage Glue metadata
# ----------------------------------------------------------------------------

SCRIPT_NAME=$(basename "$0")
PROFILE="${AWS_PROFILE:-default}"
REGION="${AWS_REGION:-}"
RESULTS_S3=""

TS_COL="timestamp"
NODE_COL="node_id"
GEOM_COL="geometry"
PREFIX_MODE="table"  # table | none

UNION_OUT_SPEC=""

declare -a SOURCE_SPECS

SAR_TABLE=""
SAR_OUT_SPEC=""
BUOY_TABLE=""
BUOY_OUT_SPEC=""

# usage:
#   Print CLI help covering mandatory and optional flags.
usage() {
  cat <<EOF
Usage: $SCRIPT_NAME --source <db.table>... --union-out <db.table@s3://path/> [options]

Options:
  --source <db.table>            Pivoted table/view to include in the join chain (repeatable; at least one)
  --union-out <db.table@s3://>   Output location for the joined sources (required)
  --sar <db.table>               Optional SAR table to join after the union
  --sar-out <db.table@s3://>     Output for the SAR join result (requires --sar)
  --buoy <db.table>              Optional buoy table to join after the union
  --buoy-out <db.table@s3://>    Output for the buoy join result (requires --buoy)
  --results-s3 <s3://...>        Athena query results location (defaults to <union_out>/athena-results/)
  --profile <name>               AWS CLI profile (default: ${PROFILE})
  --region <name>                AWS region (default: from profile configuration)
  --ts-col <name>                Timestamp column (default: ${TS_COL})
  --node-col <name>              Node identifier column (default: ${NODE_COL})
  --geom-col <name>              Geometry column (default: ${GEOM_COL})
  --prefix-mode <table|none>     Prefix columns when joining auxiliary tables (default: table)
  -h|--help                      Show this help message
EOF
}

# log:
#   Emit timestamped messages to aid troubleshooting and auditing.
log() { echo "$(date '+%Y-%m-%d %H:%M:%S') - $*"; }

# run_aws:
#   Execute an AWS CLI command honouring the configured profile and region.
run_aws() {
  if [ -n "$REGION" ]; then
    aws --profile "$PROFILE" --region "$REGION" "$@"
  else
    aws --profile "$PROFILE" "$@"
  fi
}

# wait_for_query:
#   Poll athena until the query finishes, relaying failures to the caller.
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

# ensure_region:
#   Resolve the AWS region from CLI flags or the profile configuration.
ensure_region() {
  if [ -z "$REGION" ]; then
    REGION=$(aws configure get region --profile "$PROFILE" 2>/dev/null || true)
  fi
  if [ -z "$REGION" ]; then
    echo "Error: AWS region not specified and not found in profile $PROFILE" >&2
    exit 1
  fi
}

# ensure_db:
#   Create the Glue database if it is missing, enabling CTAS outputs.
ensure_db() {
  local db="$1"
  if ! run_aws glue get-database --name "$db" >/dev/null 2>&1; then
    log "Creating Glue database: $db"
    run_aws glue create-database --database-input "{"Name":"$db"}"
  fi
}

# parse_db_tbl:
#   Decompose a db.table specification into database and table tokens.
parse_db_tbl() {
  local spec="$1"
  local db="${spec%%.*}"
  local tbl="${spec#*.}"
  echo "$db $tbl"
}

# parse_out_spec:
#   Expand an output directive formatted as db.table@s3://path/.
parse_out_spec() {
  local spec="$1"
  local left="${spec%@*}"
  local s3="${spec#*@}"
  local db="${left%%.*}"
  local tbl="${left#*.}"
  echo "$db $tbl $s3"
}

# sanitize_prefix:
#   Prepare a safe lowercase prefix for auxiliary column names.
sanitize_prefix() {
  echo "$1" | tr '[:upper:]' '[:lower:]' | tr -cd 'a-z0-9_'
}

# prefix_col:
#   Apply the desired prefixing policy when adding auxiliary columns.
prefix_col() {
  local pfx="$1" col="$2"
  if [ "$PREFIX_MODE" = "table" ]; then
    echo "${pfx}__${col}"
  else
    echo "$col"
  fi
}

# get_columns_json:
#   Query information_schema to gather ordered column names for projection.
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

# drop_table_and_s3:
#   Remove existing Glue metadata and clear the destination S3 prefix to make
#   subsequent CTAS runs deterministic.
drop_table_and_s3() {
  local odb="$1" otbl="$2" s3="$3"
  log "Dropping Glue table if exists: ${odb}.${otbl}"
  run_aws glue delete-table --database-name "$odb" --name "$otbl" >/dev/null 2>&1 || true
  log "Removing existing S3 data at: ${s3}"
  run_aws s3 rm --recursive "$s3" >/dev/null 2>&1 || true
}

# ctas:
#   Execute a CTAS statement and wait for Athena to finish before continuing.
ctas() {
  local db="$1" sql="$2"
  local qid
  qid=$(run_aws athena start-query-execution     --query-execution-context Database="$db"     --result-configuration OutputLocation="${RESULTS_S3}"     --query-string "$sql" --output text --query 'QueryExecutionId')
  wait_for_query "$qid"
}

# build_union_sql:
#   Construct the CTAS body that inner joins all pivoted sources on the shared
#   key columns, projecting non-key fields from each contributor.
build_union_sql() {
  local odb="$1" otbl="$2" s3="$3"; shift 3
  local n=$#
  if [ $((n % 2)) -ne 0 ]; then
    echo "Internal error: build_union_sql expects db/tbl pairs" >&2
    exit 1
  fi
  local first_db="$1" first_tbl="$2"; shift 2
  local first_alias="t1"
  local from_clause="FROM ${first_db}.${first_tbl} ${first_alias}"
  local selects="${first_alias}.${TS_COL}, ${first_alias}.${NODE_COL}, ${first_alias}.${GEOM_COL}"
  # Pull every non-key column from the leading source to define the schema order.
  local cols_json cols
  cols_json=$(get_columns_json "$first_db" "$first_tbl")
  cols=$(echo "$cols_json" | jq -r --arg TS "$TS_COL" --arg NODE "$NODE_COL" --arg GEOM "$GEOM_COL" '.cols | map(select((.name!=$TS) and (.name!=$NODE) and (.name!=$GEOM))) | .[].name')
  while IFS= read -r c; do
    selects+=$'
  , '
    selects+="${first_alias}.${c}"
  done <<< "$cols"

  local join_chain=""
  local idx=2
  # Join the remaining sources using the same key trio, appending their payload columns.
  while [ $# -gt 0 ]; do
    local db="$1" tbl="$2"; shift 2
    local alias="t${idx}"
    join_chain+=$'
'
    join_chain+="INNER JOIN ${db}.${tbl} ${alias} ON ${alias}.${TS_COL}=${first_alias}.${TS_COL} AND ${alias}.${NODE_COL}=${first_alias}.${NODE_COL} AND ${alias}.${GEOM_COL}=${first_alias}.${GEOM_COL}"
    cols_json=$(get_columns_json "$db" "$tbl")
    cols=$(echo "$cols_json" | jq -r --arg TS "$TS_COL" --arg NODE "$NODE_COL" --arg GEOM "$GEOM_COL" '.cols | map(select((.name!=$TS) and (.name!=$NODE) and (.name!=$GEOM))) | .[].name')
    while IFS= read -r c; do
      selects+=$'
  , '
      selects+="${alias}.${c}"
    done <<< "$cols"
    idx=$((idx+1))
  done

  cat <<EOSQL
CREATE TABLE ${odb}.${otbl}
WITH (
  format='PARQUET',
  external_location='${s3%/}/'
)
AS
SELECT
  ${selects}
${from_clause}
${join_chain}
EOSQL
}

# build_join2_sql:
#   Build the CTAS statement that joins the union output with an auxiliary SAR
#   or buoy table, prefixing columns when needed and checking geometry support.
build_join2_sql() {
  local ldb="$1" ltbl="$2" rdb="$3" rtbl="$4" odb="$5" otbl="$6" s3="$7" pfx="$8"
  local cols_json cols select_list right_has_geom
  # Inspect the right-hand table to understand which non-key columns to project.
  cols_json=$(get_columns_json "$rdb" "$rtbl")
  right_has_geom=$(echo "$cols_json" | jq -r --arg GEOM "$GEOM_COL" '.cols | any(.name==$GEOM)')
  cols=$(echo "$cols_json" | jq -r --arg TS "$TS_COL" --arg NODE "$NODE_COL" --arg GEOM "$GEOM_COL" '.cols | map(select((.name!=$TS) and (.name!=$NODE) and (.name!=$GEOM))) | .[].name')
  select_list="L.*"
  while IFS= read -r c; do
    local outc
    outc=$(prefix_col "$pfx" "$c")
    select_list+=$'
  , '
    select_list+="R.${c} AS ${outc}"
  done <<< "$cols"
  local join_cond
  join_cond="L.${TS_COL} = R.${TS_COL} AND L.${NODE_COL} = R.${NODE_COL}"
  if [ "$right_has_geom" = "true" ]; then
    join_cond+=" AND L.${GEOM_COL} = R.${GEOM_COL}"
  else
    log "WARNING: ${rdb}.${rtbl} has no '${GEOM_COL}' column; joining only on ${TS_COL}, ${NODE_COL}"
  fi
  cat <<EOSQL
CREATE TABLE ${odb}.${otbl}
WITH (
  format='PARQUET',
  external_location='${s3%/}/'
)
AS
SELECT
  ${select_list}
FROM ${ldb}.${ltbl} L
INNER JOIN ${rdb}.${rtbl} R
  ON ${join_cond}
EOSQL
}

if [ $# -eq 0 ]; then
  usage
  exit 1
fi

# Parse CLI arguments sequentially to support repeated flags such as --source.
while [ $# -gt 0 ]; do
  case "$1" in
    --source)
      [ $# -lt 2 ] && { echo "Error: --source requires an argument" >&2; exit 1; }
      SOURCE_SPECS+=("$2"); shift 2 ;;
    --union-out)
      [ $# -lt 2 ] && { echo "Error: --union-out requires an argument" >&2; exit 1; }
      UNION_OUT_SPEC="$2"; shift 2 ;;
    --sar)
      [ $# -lt 2 ] && { echo "Error: --sar requires an argument" >&2; exit 1; }
      SAR_TABLE="$2"; shift 2 ;;
    --sar-out)
      [ $# -lt 2 ] && { echo "Error: --sar-out requires an argument" >&2; exit 1; }
      SAR_OUT_SPEC="$2"; shift 2 ;;
    --buoy)
      [ $# -lt 2 ] && { echo "Error: --buoy requires an argument" >&2; exit 1; }
      BUOY_TABLE="$2"; shift 2 ;;
    --buoy-out)
      [ $# -lt 2 ] && { echo "Error: --buoy-out requires an argument" >&2; exit 1; }
      BUOY_OUT_SPEC="$2"; shift 2 ;;
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
    --prefix-mode)
      PREFIX_MODE="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1 ;;
  esac
done

# Validate mandatory parameters and paired options before triggering AWS jobs.
if [ ${#SOURCE_SPECS[@]} -eq 0 ]; then
  echo "Error: at least one --source must be provided" >&2
  exit 1
fi
if [ -z "$UNION_OUT_SPEC" ]; then
  echo "Error: --union-out is required" >&2
  exit 1
fi

if { [ -n "$SAR_TABLE" ] && [ -z "$SAR_OUT_SPEC" ]; } || { [ -z "$SAR_TABLE" ] && [ -n "$SAR_OUT_SPEC" ]; }; then
  echo "Error: --sar and --sar-out must be provided together" >&2
  exit 1
fi
if { [ -n "$BUOY_TABLE" ] && [ -z "$BUOY_OUT_SPEC" ]; } || { [ -z "$BUOY_TABLE" ] && [ -n "$BUOY_OUT_SPEC" ]; }; then
  echo "Error: --buoy and --buoy-out must be provided together" >&2
  exit 1
fi

# Ensure the AWS client has an explicit region before issuing requests.
ensure_region

read -r UN_OUT_DB UN_OUT_TBL UN_OUT_S3 <<< "$(parse_out_spec "$UNION_OUT_SPEC")"
if [ -z "$RESULTS_S3" ]; then
  RESULTS_S3="${UN_OUT_S3%/}/athena-results/"
fi

log "AWS Profile: ${PROFILE} | Region: ${REGION}"
log "Athena results: ${RESULTS_S3}"

ensure_db "$UN_OUT_DB"
drop_table_and_s3 "$UN_OUT_DB" "$UN_OUT_TBL" "$UN_OUT_S3"

# Break down every source specification into tokens for SQL generation.
declare -a SOURCE_DB_TBLS
for spec in "${SOURCE_SPECS[@]}"; do
  read -r src_db src_tbl <<< "$(parse_db_tbl "$spec")"
  SOURCE_DB_TBLS+=("$src_db" "$src_tbl")
done

log "Building union CTAS for ${UN_OUT_DB}.${UN_OUT_TBL}"
union_sql=$(build_union_sql "$UN_OUT_DB" "$UN_OUT_TBL" "$UN_OUT_S3" "${SOURCE_DB_TBLS[@]}")
ctas "$UN_OUT_DB" "$union_sql"

if [ -n "$SAR_TABLE" ]; then
  # Materialise the SAR-enriched dataset when SAR inputs are provided.
  read -r SAR_DB SAR_TBL <<< "$(parse_db_tbl "$SAR_TABLE")"
  read -r SAR_OUT_DB SAR_OUT_TBL SAR_OUT_S3 <<< "$(parse_out_spec "$SAR_OUT_SPEC")"
  ensure_db "$SAR_OUT_DB"
  drop_table_and_s3 "$SAR_OUT_DB" "$SAR_OUT_TBL" "$SAR_OUT_S3"
  log "Building SAR join CTAS for ${SAR_OUT_DB}.${SAR_OUT_TBL}"
  sar_sql=$(build_join2_sql "$UN_OUT_DB" "$UN_OUT_TBL" "$SAR_DB" "$SAR_TBL" "$SAR_OUT_DB" "$SAR_OUT_TBL" "$SAR_OUT_S3" "sar")
  ctas "$SAR_OUT_DB" "$sar_sql"
fi

if [ -n "$BUOY_TABLE" ]; then
  # Materialise the buoy-enriched dataset when buoy inputs are provided.
  read -r B_DB B_TBL <<< "$(parse_db_tbl "$BUOY_TABLE")"
  read -r B_OUT_DB B_OUT_TBL B_OUT_S3 <<< "$(parse_out_spec "$BUOY_OUT_SPEC")"
  ensure_db "$B_OUT_DB"
  drop_table_and_s3 "$B_OUT_DB" "$B_OUT_TBL" "$B_OUT_S3"
  log "Building buoy join CTAS for ${B_OUT_DB}.${B_OUT_TBL}"
  buoy_sql=$(build_join2_sql "$UN_OUT_DB" "$UN_OUT_TBL" "$B_DB" "$B_TBL" "$B_OUT_DB" "$B_OUT_TBL" "$B_OUT_S3" "buoy")
  ctas "$B_OUT_DB" "$buoy_sql"
fi

# All requested datasets have been materialised successfully.
log "Union/join phase completed."
