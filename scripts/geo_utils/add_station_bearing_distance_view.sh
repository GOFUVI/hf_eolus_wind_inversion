#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Author: Juan Luis Herrera Cortijo <juan.luis.herrera.cortijo@gmail.com>
# Created: 2025-10-16
# Disclaimer: This script is provided as-is for research workflows and must be validated before any operational deployment.
# -----------------------------------------------------------------------------

# Exit on error, unset var, or pipefail
set -euo pipefail

# ----------------------------------------------------------------------------
# add_station_bearing_distance_view.sh
#
# Description:
#   Create (or replace) an Athena view that augments a source table containing
#   WKB geometries with per-station bearing and distance columns relative to one
#   or more reference stations. For each station, the view appends:
#     * <station>_bearing   - Bearing in degrees from geographic north from the
#                             station to the geometry centroid (identical to
#                             the original coordinate for point geometries).
#     * <station>_dist_km   - Great-circle distance in kilometres between the
#                             station and the geometry.
#   The script assumes the geometry column stores longitude/latitude coordinates
#   in WGS84 (CRS84) and that AWS Athena geospatial functions are available.
#
# Usage example:
#   scripts/geo_utils/add_station_bearing_distance_view.sh \
#     --source analytics_source_db.observations \
#     --view analytics_source_db.observations_with_features \
#     --station station_a:43.1589:-9.2108 \
#     --station station_b:43.5680:-8.3140 \
#     --results-s3 s3://example-bucket/athena-results/ \
#     --profile data_profile --region us-east-1
# ----------------------------------------------------------------------------

SCRIPT_NAME=$(basename "$0")
PROFILE="${AWS_PROFILE:-default}"
REGION="${AWS_REGION:-}"
RESULTS_S3=""
SOURCE_SPEC=""
VIEW_SPEC=""
GEOMETRY_COLUMN="geometry"
PREVIEW=false

# Station metadata
# Arrays keep the original station descriptors plus the alias used to craft
# deterministic Athena column names.
declare -a STATION_NAMES=()
declare -a STATION_LATS=()
declare -a STATION_LONS=()
declare -a STATION_ALIASES=()

# Print CLI usage guidance and exit.
usage() {
  cat <<EOF
Usage: $SCRIPT_NAME --source <db.table> --view <db.view> --station <name:lat:lon> [--station ...] --results-s3 <s3://bucket/prefix/> [options]

Required:
  --source <db.table>        Fully-qualified source table/view (database.table)
  --view <db.view>           Target view (database.view) to create or replace
  --station <name:lat:lon>   Station identifier and coordinates in decimal degrees
                             (repeatable; e.g. --station vila:29.15:-80.93)
  --results-s3 <s3://...>    S3 location for Athena query results (ignored with --preview)

Optional:
  --geometry-column <name>   Geometry column name in the source (default: geometry)
  --profile <name>           AWS CLI profile (default: ${PROFILE})
  --region <name>            AWS region (default: from profile configuration)
  --preview                  Print the CREATE VIEW statement without executing it
  -h|--help                  Show this help message
EOF
}

# Emit a timestamped log message to stdout.
log() {
  echo "$(date '+%Y-%m-%d %H:%M:%S') - $*"
}

# Wrapper around aws cli honouring the configured region.
run_aws() {
  if [ -n "$REGION" ]; then
    aws --profile "$PROFILE" --region "$REGION" "$@"
  else
    aws --profile "$PROFILE" "$@"
  fi
}

# Poll Athena until the submitted query finishes or fails.
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

# Resolve the AWS region from flags or profile configuration.
ensure_region() {
  if [ -z "$REGION" ]; then
    REGION=$(aws configure get region --profile "$PROFILE")
  fi
  if [ -z "$REGION" ]; then
    echo "Error: AWS region not specified and not set in profile $PROFILE" >&2
    exit 1
  fi
}

# Lazily create the Glue database backing the target view.
ensure_db() {
  local db="$1"
  if ! run_aws glue get-database --name "$db" >/dev/null 2>&1; then
    log "Creating Glue database: $db"
    run_aws glue create-database --database-input "{\"Name\":\"$db\"}"
  fi
}

# Split a database.table spec into its components with validation.
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

# Assert that an option is followed by a non-flag value.
ensure_value() {
  local opt="$1"
  local value="$2"
  if [ -z "$value" ] || [[ "$value" == --* ]]; then
    echo "Error: $opt requires a value" >&2
    usage
    exit 1
  fi
}

# Normalise station names into safe Athena column suffixes.
sanitize_column_name() {
  local raw="$1"
  local lowered sanitized
  lowered=$(printf '%s' "$raw" | tr '[:upper:]' '[:lower:]')
  sanitized=$(printf '%s' "$lowered" | sed 's/[^a-z0-9_]/_/g')
  sanitized=$(printf '%s' "$sanitized" | sed 's/_\{2,\}/_/g')
  sanitized=$(printf '%s' "$sanitized" | sed 's/^_\+//; s/_\+$//')
  if [[ -z "$sanitized" ]]; then
    sanitized="station"
  fi
  if [[ $sanitized =~ ^[0-9] ]]; then
    sanitized="_$sanitized"
  fi
  echo "$sanitized"
}

# Parse --station descriptors and stash validated metadata.
add_station() {
  local spec="$1"
  local name lat lon
  IFS=':' read -r name lat lon <<< "$spec"
  if [ -z "$name" ] || [ -z "$lat" ] || [ -z "$lon" ]; then
    echo "Error: --station expects name:lat:lon, got '$spec'" >&2
    exit 1
  fi
  if ! [[ "$lat" =~ ^-?[0-9]+([.][0-9]+)?$ ]]; then
    echo "Error: Invalid latitude '$lat' for station '$name'" >&2
    exit 1
  fi
  if ! [[ "$lon" =~ ^-?[0-9]+([.][0-9]+)?$ ]]; then
    echo "Error: Invalid longitude '$lon' for station '$name'" >&2
    exit 1
  fi
  if ! awk -v v="$lat" 'BEGIN{exit !(v >= -90 && v <= 90)}'; then
    echo "Error: Latitude '$lat' for station '$name' is outside [-90, 90]" >&2
    exit 1
  fi
  if ! awk -v v="$lon" 'BEGIN{exit !(v >= -180 && v <= 180)}'; then
    echo "Error: Longitude '$lon' for station '$name' is outside [-180, 180]" >&2
    exit 1
  fi
  local alias
  alias=$(sanitize_column_name "$name")
  if [ ${#STATION_ALIASES[@]} -gt 0 ]; then
    for existing in "${STATION_ALIASES[@]}"; do
      if [ "$alias" = "$existing" ]; then
        echo "Error: Duplicate station alias '$alias' derived from station '$name'" >&2
        exit 1
      fi
    done
  fi
  STATION_NAMES+=("$name")
  STATION_LATS+=("$lat")
  STATION_LONS+=("$lon")
  STATION_ALIASES+=("$alias")
}

# Parse CLI arguments. We avoid eval tricks so that the script remains easy to
# audit and copy/paste from the pipelines.
while [[ $# -gt 0 ]]; do
  case "$1" in
    --source)
      ensure_value "$1" "${2-}"
      SOURCE_SPEC="$2"
      shift 2
      ;;
    --view)
      ensure_value "$1" "${2-}"
      VIEW_SPEC="$2"
      shift 2
      ;;
    --station)
      ensure_value "$1" "${2-}"
      add_station "$2"
      shift 2
      ;;
    --station-lat|--station-lon)
      echo "Error: use --station name:lat:lon instead of $1" >&2
      exit 1
      ;;
    --geometry-column)
      ensure_value "$1" "${2-}"
      GEOMETRY_COLUMN="$2"
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
    --results-s3)
      ensure_value "$1" "${2-}"
      RESULTS_S3="$2"
      shift 2
      ;;
    --preview)
      PREVIEW=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [ -z "$SOURCE_SPEC" ] || [ -z "$VIEW_SPEC" ]; then
  echo "Error: --source and --view are required" >&2
  usage
  exit 1
fi

if [ ${#STATION_NAMES[@]} -eq 0 ]; then
  echo "Error: At least one --station is required" >&2
  usage
  exit 1
fi

if ! $PREVIEW && [ -z "$RESULTS_S3" ]; then
  echo "Error: --results-s3 is required when executing the query" >&2
  exit 1
fi

read -r SOURCE_DB SOURCE_TBL < <(parse_db_tbl "$SOURCE_SPEC")
read -r VIEW_DB VIEW_TBL < <(parse_db_tbl "$VIEW_SPEC")

SOURCE_FQN="$SOURCE_DB.$SOURCE_TBL"
VIEW_FQN="$VIEW_DB.$VIEW_TBL"

# Derive a representative coordinate from the WKB geometry. The inner TRY
# prevents malformed blobs from crashing `ST_GeomFromBinary`, while the outer
# TRY keeps concave polygons that fail centroid computation from aborting the
# query altogether.
POINT_ON_SURFACE="TRY(ST_Centroid(TRY(ST_GeomFromBinary(src.${GEOMETRY_COLUMN}))))"
POINT_LON="TRY(ST_X(${POINT_ON_SURFACE}))"
POINT_LAT="TRY(ST_Y(${POINT_ON_SURFACE}))"

log "Generating view columns for ${#STATION_NAMES[@]} station(s)"
for idx in "${!STATION_NAMES[@]}"; do
  log "  Station ${STATION_NAMES[$idx]} (lat=${STATION_LATS[$idx]}, lon=${STATION_LONS[$idx]}) -> columns ${STATION_ALIASES[$idx]}_bearing, ${STATION_ALIASES[$idx]}_dist_km"
done

# Build the bearing/distance expressions once per station. We expand the
# formulas explicitly instead of delegating to ST_Distance to retain full
# control over null propagation and to keep the results in kilometres.
SELECT_LINES=()
for idx in "${!STATION_NAMES[@]}"; do
  lat="${STATION_LATS[$idx]}"
  lon="${STATION_LONS[$idx]}"
  alias="${STATION_ALIASES[$idx]}"

  # Precompute all angular quantities in radians; embedding literals in the
  # SQL prevents Athena from recomputing trigonometric conversions per row.
  lat_rad="radians(${lat})"
  lon_rad="radians(${lon})"
  obs_lat_rad="radians(${POINT_LAT})"
  obs_lon_rad="radians(${POINT_LON})"
  delta_lat="(${obs_lat_rad} - ${lat_rad})"
  delta_lon="(${obs_lon_rad} - ${lon_rad})"
  sin_half_lat="sin(${delta_lat} / 2)"
  sin_half_lon="sin(${delta_lon} / 2)"
  a_expr="(pow(${sin_half_lat}, 2) + cos(${lat_rad}) * cos(${obs_lat_rad}) * pow(${sin_half_lon}, 2))"
  c_expr="2 * atan2(sqrt(least(1.0, ${a_expr})), sqrt(greatest(0.0, 1 - ${a_expr})))"
  distance_expr="TRY(6371.0088 * (${c_expr}))"
  bearing_inner="TRY(degrees(atan2(sin(${delta_lon}) * cos(${obs_lat_rad}), cos(${lat_rad}) * sin(${obs_lat_rad}) - sin(${lat_rad}) * cos(${obs_lat_rad}) * cos(${delta_lon}))))"
  bearing_expr="mod(${bearing_inner} + 360.0, 360.0)"

# Wrap the metrics in CASE expressions so rows with missing geometries carry
# nulls instead of throwing during the trigonometric evaluation.
  printf -v bearing_sql '  , CASE
      WHEN src.%s IS NULL THEN NULL
      ELSE %s
    END AS %s_bearing' "$GEOMETRY_COLUMN" "$bearing_expr" "$alias"
  SELECT_LINES+=("$bearing_sql")

  printf -v distance_sql '  , CASE
      WHEN src.%s IS NULL THEN NULL
      ELSE %s
    END AS %s_dist_km' "$GEOMETRY_COLUMN" "$distance_expr" "$alias"
  SELECT_LINES+=("$distance_sql")
done

# Assemble the final CREATE VIEW statement. Keeping it in a single variable
# simplifies both the preview mode and the actual submission to Athena.
ATHENA_QUERY="$({
  printf 'CREATE OR REPLACE VIEW %s AS\n' "$VIEW_FQN"
  printf 'SELECT\n  src.*\n'
  for line in "${SELECT_LINES[@]}"; do
    printf '%s\n' "$line"
  done
  printf 'FROM %s AS src\n;\n' "$SOURCE_FQN"
})"

if $PREVIEW; then
  printf '%s\n' "$ATHENA_QUERY"
  exit 0
fi

ensure_region
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
