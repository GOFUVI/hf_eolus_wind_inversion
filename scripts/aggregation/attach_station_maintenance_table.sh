#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Author: Juan Luis Herrera Cortijo <juan.luis.herrera.cortijo@gmail.com>
# Created: 2025-10-16
# Disclaimer: This script is provided as-is for research workflows and must be validated before any operational deployment.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# attach_station_maintenance_table.sh
#
# Enriches a pivoted HF-radar station table with maintenance metadata sourced
# from a local CSV. The script materialises an Athena CTAS that embeds the
# maintenance catalogue as an inline VALUES clause, ranks interventions by
# recency, and appends interval identifiers, operation type, start timestamp,
# and elapsed hours since the last calibration. It deletes any previous Glue
# table/S3 prefix before writing the refreshed dataset.
# -----------------------------------------------------------------------------
set -euo pipefail

SCRIPT_NAME=$(basename "$0")
PROFILE="${AWS_PROFILE:-default}"
REGION="${AWS_REGION:-}"
RESULTS_S3=""
SOURCE_SPEC=""
OUT_SPEC=""
MAINTENANCE_CSV=""
STATION_ID=""
PREFIX=""
TIMESTAMP_COL="timestamp"

# usage: render CLI help and exit. Kept as a function to reuse in error paths.
usage() {
  cat <<USAGE
Usage: $SCRIPT_NAME --source <db.table> --out <db.table@s3://path/> --maintenance-csv <path> --station-id <id> --prefix <prefix> [options]

Required:
  --source <db.table>            Pivoted station table/view to enrich
  --out <db.table@s3://path/>    Destination table and S3 location for CTAS output
  --maintenance-csv <path>       CSV file with station_id,event_type/effective_start
  --station-id <id>              Station identifier (case-insensitive)
  --prefix <prefix>              Column prefix for maintenance metrics (e.g., vila)

Optional:
  --timestamp-col <column>       Timestamp column in the source (default: ${TIMESTAMP_COL})
  --results-s3 <s3://...>        Athena result bucket (required if not set)
  --profile <name>               AWS CLI profile (default: ${PROFILE})
  --region <name>                AWS region (default: from profile)
  -h|--help                      Show this help message
USAGE
}

# log: print timestamped messages to standard output, used across the workflow.
log() { echo "$(date '+%Y-%m-%d %H:%M:%S') - $*"; }

# run_aws: thin wrapper around aws CLI to inject profile/region consistently.
run_aws() {
  if [ -n "$REGION" ]; then
    aws --profile "$PROFILE" --region "$REGION" "$@"
  else
    aws --profile "$PROFILE" "$@"
  fi
}

# ensure_region: resolve AWS region lazily if not provided on the command line.
ensure_region() {
  if [ -z "$REGION" ]; then
    REGION=$(aws configure get region --profile "$PROFILE" 2>/dev/null || true)
  fi
  if [ -z "$REGION" ]; then
    echo "Error: AWS region not specified and not found in profile $PROFILE" >&2
    exit 1
  fi
}

# parse_db_tbl: validate and split a database.table specification.
parse_db_tbl() {
  local spec="$1"
  if [[ "$spec" != *.* ]]; then
    echo "Error: expected database.table format, got '$spec'" >&2
    exit 1
  fi
  local db="${spec%%.*}"
  local tbl="${spec#*.}"
  if [ -z "$db" ] || [ -z "$tbl" ]; then
    echo "Error: invalid database.table format '$spec'" >&2
    exit 1
  fi
  echo "$db" "$tbl"
}

# parse_out_spec: validate database.table@s3://path and extract its components.
parse_out_spec() {
  local spec="$1"
  if [[ "$spec" != *@* ]]; then
    echo "Error: expected database.table@s3://path format, got '$spec'" >&2
    exit 1
  fi
  local left="${spec%@*}"
  local s3="${spec#*@}"
  if [[ "$s3" != s3://* ]]; then
    echo "Error: output location must start with s3://, got '$s3'" >&2
    exit 1
  fi
  read -r db tbl < <(parse_db_tbl "$left")
  echo "$db" "$tbl" "$s3"
}

# wait_for_query: poll Athena until the query settles, aborting on failures.
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

# Parse CLI arguments; the script fails fast when required flags are missing.
while [[ $# -gt 0 ]]; do
  case "$1" in
    --source) SOURCE_SPEC="$2"; shift 2 ;;
    --out) OUT_SPEC="$2"; shift 2 ;;
    --maintenance-csv) MAINTENANCE_CSV="$2"; shift 2 ;;
    --station-id) STATION_ID="$2"; shift 2 ;;
    --prefix) PREFIX="$2"; shift 2 ;;
    --timestamp-col) TIMESTAMP_COL="$2"; shift 2 ;;
    --results-s3) RESULTS_S3="$2"; shift 2 ;;
    --profile) PROFILE="$2"; shift 2 ;;
    --region) REGION="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Error: unknown argument '$1'" >&2; usage; exit 1 ;;
  esac
done

if [ -z "$SOURCE_SPEC" ] || [ -z "$OUT_SPEC" ] || [ -z "$MAINTENANCE_CSV" ] || [ -z "$STATION_ID" ] || [ -z "$PREFIX" ]; then
  echo "Error: missing required arguments" >&2
  usage
  exit 1
fi

if [ ! -f "$MAINTENANCE_CSV" ]; then
  echo "Error: maintenance CSV '$MAINTENANCE_CSV' not found" >&2
  exit 1
fi

read -r SRC_DB SRC_TBL < <(parse_db_tbl "$SOURCE_SPEC")
read -r OUT_DB OUT_TBL OUT_S3 < <(parse_out_spec "$OUT_SPEC")

if [ -z "$RESULTS_S3" ]; then
  echo "Error: --results-s3 must be provided" >&2
  exit 1
fi

ensure_region

log "Creating maintenance-enriched table ${OUT_DB}.${OUT_TBL} from ${SRC_DB}.${SRC_TBL}"

# Replace artefacts created by previous runs so the Glue table and S3 prefix
# always mirror the latest maintenance catalogue.
log "Dropping existing Glue table ${OUT_DB}.${OUT_TBL} if present"
run_aws glue delete-table --database-name "$OUT_DB" --name "$OUT_TBL" >/dev/null 2>&1 || true

log "Cleaning S3 prefix ${OUT_S3}"
run_aws s3 rm --recursive "$OUT_S3" >/dev/null 2>&1 || true

TMP_SQL=$(mktemp)
trap 'rm -f "$TMP_SQL"' EXIT

# Delegate SQL generation to an embedded Python helper so the CSV is parsed
# safely and converted into an Athena VALUES clause without relying on jq/sed.
SOURCE_DB="$SRC_DB" SOURCE_TBL="$SRC_TBL" OUT_DB="$OUT_DB" OUT_TBL="$OUT_TBL" OUT_S3="$OUT_S3" \
  MAINTENANCE_CSV="$MAINTENANCE_CSV" STATION_ID="$STATION_ID" PREFIX="$PREFIX" \
  TIMESTAMP_COL="$TIMESTAMP_COL" python3 - <<'PY' > "$TMP_SQL"
import csv
import os
from pathlib import Path

source_db = os.environ["SOURCE_DB"]
source_tbl = os.environ["SOURCE_TBL"]
out_db = os.environ["OUT_DB"]
out_tbl = os.environ["OUT_TBL"]
out_s3 = os.environ["OUT_S3"]
station_id = os.environ["STATION_ID"].strip().lower()
prefix = os.environ["PREFIX"].strip().lower()
timestamp_col = os.environ["TIMESTAMP_COL"].strip()
csv_path = Path(os.environ["MAINTENANCE_CSV"])

rows = []
with csv_path.open(newline='', encoding='utf-8') as fh:
    reader = csv.DictReader(fh)
    for line in reader:
        station = (line.get('station_id') or '').strip().lower()
        if station != station_id:
            continue
        mtype = (line.get('maintenance_type') or line.get('event_type') or '').strip()
        start = (line.get('maintenance_start') or line.get('effective_start') or '').strip()
        if not start:
            continue
        rows.append((station, mtype, start))

if rows:
    lines = []
    for station, mtype, start in rows:
        station_sql = station.replace("'", "''")
        if mtype:
            sanitized = mtype.replace("'", "''")
            mtype_sql = "'" + sanitized + "'"
        else:
            mtype_sql = "NULL"
        start_sql = start.replace("'", "''")
        lines.append(
            "    ('{}', {}, from_iso8601_timestamp('{}'))".format(station_sql, mtype_sql, start_sql)
        )
    values_block = ",\n".join(lines)
else:
    values_block = "    (NULL, NULL, CAST(NULL AS timestamp))"

sql = f"""CREATE TABLE {out_db}.{out_tbl}
WITH (
  format = 'PARQUET',
  external_location = '{out_s3}',
  write_compression = 'SNAPPY'
) AS
WITH maintenance_prepared AS (
  SELECT station_id, maintenance_type, maintenance_start_ts
  FROM (VALUES
{values_block}
  ) AS v(station_id, maintenance_type, maintenance_start_ts)
  WHERE station_id IS NOT NULL
    AND maintenance_start_ts IS NOT NULL
),
maintenance_ranked AS (
  SELECT
    src.{timestamp_col} AS obs_timestamp,
    src.node_id,
    src.geometry,
    mp.maintenance_type,
    mp.maintenance_start_ts,
    ROW_NUMBER() OVER (
      PARTITION BY src.{timestamp_col}, src.node_id, src.geometry
      ORDER BY mp.maintenance_start_ts DESC
    ) AS maintenance_rank
  FROM {source_db}.{source_tbl} src
  LEFT JOIN maintenance_prepared mp
    ON mp.station_id = '{station_id}'
   AND mp.maintenance_start_ts <= src.{timestamp_col}
)
SELECT
  src.*,
  CASE
    WHEN mr.maintenance_start_ts IS NOT NULL THEN CONCAT('{prefix}_', date_format(mr.maintenance_start_ts, '%Y%m%dT%H%i%SZ'))
    ELSE NULL
  END AS {prefix}_maintenance_interval_id,
  mr.maintenance_type AS {prefix}_maintenance_type,
  CASE
    WHEN mr.maintenance_start_ts IS NOT NULL THEN date_format(mr.maintenance_start_ts, '%Y-%m-%dT%H:%i:%sZ')
    ELSE NULL
  END AS {prefix}_maintenance_start,
  CASE
    WHEN mr.maintenance_start_ts IS NOT NULL THEN CAST(date_diff('second', mr.maintenance_start_ts, src.{timestamp_col}) / 3600.0 AS double)
    ELSE NULL
  END AS {prefix}_hours_since_last_calibration
FROM {source_db}.{source_tbl} src
LEFT JOIN maintenance_ranked mr
  ON mr.obs_timestamp = src.{timestamp_col}
 AND mr.node_id = src.node_id
 AND mr.geometry = src.geometry
 AND mr.maintenance_rank = 1;"""

print(sql)
PY

SQL=$(cat "$TMP_SQL")

log "Submitting CTAS query"
QUERY_ID=$(run_aws athena start-query-execution \
  --query-execution-context Database="$OUT_DB" \
  --result-configuration OutputLocation="$RESULTS_S3" \
  --query-string "$SQL" \
  --output text --query 'QueryExecutionId')

wait_for_query "$QUERY_ID"

log "Created maintenance table ${OUT_DB}.${OUT_TBL}"
