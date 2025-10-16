#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Author: Juan Luis Herrera Cortijo <juan.luis.herrera.cortijo@gmail.com>
# Created: 2025-10-16
# Disclaimer: This script is provided as-is for research workflows and must be validated before any operational deployment.
# -----------------------------------------------------------------------------

set -euo pipefail

LOG_FILE=""
OUTPUT_FILE=""
USAGE="Usage: record_final_job.sh --log PATH --output PATH"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --log)
      LOG_FILE="$2"; shift 2 ;;
    --log=*)
      LOG_FILE="${1#*=}"; shift ;;
    --output)
      OUTPUT_FILE="$2"; shift 2 ;;
    --output=*)
      OUTPUT_FILE="${1#*=}"; shift ;;
    --help|-h)
      echo "$USAGE"; exit 0 ;;
    --)
      shift; break ;;
    -* )
      echo "Error: unknown option '$1'" >&2
      echo "$USAGE" >&2
      exit 1 ;;
    * )
      echo "Error: unexpected positional argument '$1'" >&2
      echo "$USAGE" >&2
      exit 1 ;;
  esac
done

if [[ -z "$LOG_FILE" || -z "$OUTPUT_FILE" ]]; then
  echo "$USAGE" >&2
  exit 1
fi

if [[ ! -f "$LOG_FILE" ]]; then
  echo "Error: log file '$LOG_FILE' not found." >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT_FILE")"

final_job=$(grep -F 'Launching training job' "$LOG_FILE" | tail -1 | sed -E 's/.*Launching training job ([^ ]+).*/\1/')
if [[ -z "$final_job" ]]; then
  echo "Error: unable to determine final training job name from '$LOG_FILE'." >&2
  exit 1
fi

printf '%s\n' "$final_job" > "$OUTPUT_FILE"
echo "$final_job"
