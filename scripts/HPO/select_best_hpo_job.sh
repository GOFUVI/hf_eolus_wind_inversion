#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Author: Juan Luis Herrera Cortijo <juan.luis.herrera.cortijo@gmail.com>
# Created: 2025-10-16
# Disclaimer: This script is provided as-is for research workflows and must be validated before any operational deployment.
# -----------------------------------------------------------------------------

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: select_best_hpo_job.sh --report PATH --output PATH

Parse an aggregated HPO report (Markdown) to determine the best job and write
its name into an output file. If the output file already exists, the value is
read from the first non-empty, non-comment line and reused without touching the
report.
USAGE
  exit 1
}

REPORT=""
OUTPUT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --report)
      REPORT="$2"; shift 2 ;;
    --report=*)
      REPORT="${1#*=}"; shift ;;
    --output)
      OUTPUT="$2"; shift 2 ;;
    --output=*)
      OUTPUT="${1#*=}"; shift ;;
    --help|-h)
      usage ;;
    --)
      shift; break ;;
    -* )
      echo "Error: unknown option '$1'" >&2
      usage ;;
    * )
      echo "Error: unexpected positional argument '$1'" >&2
      usage ;;
  esac
done

[[ -n "$REPORT" && -n "$OUTPUT" ]] || usage

mkdir -p "$(dirname "$OUTPUT")"

if [[ -f "$OUTPUT" ]]; then
  cached=""
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%%#*}"
    trimmed="$(printf '%s' "$line" | xargs 2>/dev/null || true)"
    if [[ -n "$trimmed" ]]; then
      cached="$trimmed"
      break
    fi
  done < "$OUTPUT"
  if [[ -n "$cached" ]]; then
    has_metric=$(awk -F'|' -v target="$cached" '
      function trim(s) {
        sub(/^[ \t\r\n]+/, "", s)
        sub(/[ \t\r\n]+$/, "", s)
        return s
      }
      BEGIN {
        header_seen = 0
        metric_col = 0
      }
      /^\|/ {
        if (!header_seen) {
          if (trim($2) != "HPOJob") next
          for (i = 1; i <= NF; i++) {
            col = trim($i)
            if (col ~ /^combined[ _]?loss$/I) {
              metric_col = i
              break
            }
          }
          header_seen = 1
          next
        }
        if (trim($2) == "---") next
        job = trim($3)
        if (job != target) next
        metric = (metric_col > 0 && metric_col <= NF) ? trim($metric_col) : ""
        if (metric != "" && metric ~ /^-?[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?$/) {
          print "yes"
          exit
        }
      }
    ' "$REPORT" 2>/dev/null || true)
    if [[ "$has_metric" == "yes" ]]; then
      printf '%s\n' "$cached"
      exit 0
    fi
  fi
fi

if [[ ! -f "$REPORT" ]]; then
  echo "Error: aggregated HPO report '$REPORT' not found." >&2
  exit 1
fi

best_job=$(awk -F'|' '
  function trim(s) {
    sub(/^[ \t\r\n]+/, "", s)
    sub(/[ \t\r\n]+$/, "", s)
    return s
  }
  BEGIN {
    header_seen = 0
    metric_col = 0
    first_data_job = ""
  }
  /^\|/ {
    # Identify the header row to locate the combined loss column index
    if (!header_seen) {
      if (trim($2) != "HPOJob") {
        next
      }
      for (i = 1; i <= NF; i++) {
        col = trim($i)
        if (col ~ /^combined[ _]?loss$/I) {
          metric_col = i
          break
        }
      }
      header_seen = 1
      next
    }
    # Skip separator row made of --- markers
    if (trim($2) == "---") {
      next
    }
    job = trim($3)
    if (job == "" || job == "TrainingJobName") {
      next
    }
    if (first_data_job == "") {
      first_data_job = job
    }
    metric = ""
    if (metric_col > 0 && metric_col <= NF) {
      metric = trim($metric_col)
    }
    if (metric != "" && metric ~ /^-?[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?$/) {
      print job
      exit
    }
  }
  END {
    if (first_data_job != "") {
      print first_data_job
    }
  }
' "$REPORT")

if [[ -z "$best_job" ]]; then
  echo "Error: could not extract best job from '$REPORT'." >&2
  exit 1
fi

printf '%s\n' "$best_job" > "$OUTPUT"
printf '%s\n' "$best_job"
