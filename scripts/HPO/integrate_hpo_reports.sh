#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Author: Juan Luis Herrera Cortijo <juan.luis.herrera.cortijo@gmail.com>
# Created: 2025-10-16
# Disclaimer: This script is provided as-is for research workflows and must be validated before any operational deployment.
# -----------------------------------------------------------------------------

# Merge multiple single-job HPO Markdown reports into a combined leaderboard.
set -euo pipefail

# Print CLI usage instructions for the aggregator.
usage() {
    cat <<EOF
Usage: $0 -i REPORT_FILES -o GLOBAL_REPORT

Aggregate multiple SageMaker HPO markdown reports into one global report.

Options:
  -i REPORT_FILES   Comma-separated list of HPO report markdown files (required)
  -o GLOBAL_REPORT  Output file path for the aggregated report (required)
  -h                Show this help message and exit
EOF
    exit 1
}

# Parse arguments
REPORTS_ARG=""
GLOBAL_OUT=""
while getopts "i:o:h" opt; do
    case "$opt" in
        i) REPORTS_ARG="$OPTARG" ;; 
        o) GLOBAL_OUT="$OPTARG" ;; 
        h|*) usage ;; 
    esac
done

if [ -z "$REPORTS_ARG" ] || [ -z "$GLOBAL_OUT" ]; then
    echo "Error: both -i and -o are required."
    usage
fi

# Prepare list of report files
# Expand the comma-separated input list into an array of report paths.
IFS=',' read -r -a REPORT_FILES <<< "$REPORTS_ARG"
# Validate that every referenced report exists before continuing.
for f in "${REPORT_FILES[@]}"; do
    if [ ! -f "$f" ]; then
        echo "Error: report file '$f' not found."
        exit 1
    fi
done

# Extract table header and separator from first report
HEADER_LINE=""
SEP_LINE=""
found=false
# Identify the table header from the first report so we can reuse it globally.
for file in "${REPORT_FILES[@]}"; do
    while IFS= read -r line; do
        if [[ $line == "## Training Job Summaries"* ]]; then
            # After this, next non-empty line is header, then next is separator
            while IFS= read -r line && [[ -z $line ]]; do :; done
            HEADER_LINE="$line"
            IFS= read -r SEP_LINE
            found=true
            break
        fi
    done < "$file"
    $found && break
done

if [ -z "$HEADER_LINE" ] || [ -z "$SEP_LINE" ]; then
    echo "Error: could not find table header in reports."
    exit 1
fi

## Write global report header
# Bootstrap the aggregated report with metadata and the shared header row.
{
    echo "# Aggregated HPO Job Reports"
    echo ""
    echo "Generated on $(date)"
    echo ""
    echo "## Training Job Summaries (Global)"
    echo ""
    # Prepend HPOJob column to header and separator
    echo "| HPOJob | ${HEADER_LINE#| }"
    echo "| --- | ${SEP_LINE#| }"
} > "$GLOBAL_OUT"

# Initialize array to collect all rows for sorting
# Accumulate rows tagged with their originating job, ready for sorting.
ROWS=()

# Collect rows from each report
for file in "${REPORT_FILES[@]}"; do
    # Derive HPO job name from filename (strip _hpo_report.md or .md)
    base=$(basename "$file" .md)
    if [[ $base == *_hpo_report ]]; then
        HPOJOB="${base%_hpo_report}"
    else
        HPOJOB="$base"
    fi
    # Extract and append table rows
    in_table=false
    while IFS= read -r line; do
        if [[ $line == "## Training Job Summaries"* ]]; then
            in_table=true
            continue
        fi
        if $in_table; then
            # Stop at next section
            if [[ $line == '## '* ]]; then
                break
            fi
            # Skip empty, header, and separator lines
            if [[ -z $line || $line == "$HEADER_LINE" || $line == "$SEP_LINE" ]]; then
                continue
            fi
            # Collect row with HPOJob prefix
            ROWS+=("| $HPOJOB |${line#|}")
        fi
    done < "$file"
# End of per-file collection
done

# Determine CombinedLoss column index in global table header (case-insensitive match)
GLOBAL_HEADER="| HPOJob | ${HEADER_LINE#| }"
# Determine CombinedLoss column index in global table header (case-insensitive match);
# use only the first matching column to avoid matching StdDev-combined_loss header
COL_IDX=$(echo "$GLOBAL_HEADER" | awk -F'|' 'BEGIN{IGNORECASE=1} {for (i=1; i<=NF; i++) if ($i ~ /combined[ _]?loss/) {print i; exit}}')

# Sort collected rows by CombinedLoss (numeric) and append to output
# Sort rows by CombinedLoss when available, pushing incomplete entries to the bottom.
if [ -n "$COL_IDX" ]; then
    printf "%s\n" "${ROWS[@]}" | awk -F'|' -v idx="$COL_IDX" '
        {
            val = $idx
            gsub(/^[ \t]+|[ \t]+$/, "", val)
            missing = (val == "")
            if (missing || val !~ /^-?[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?$/) {
                missing = 1
                sortval = "9999999999"
            } else {
                sortval = val
            }
            printf "%d|%s|%s\n", missing, sortval, $0
        }
    ' | sort -t '|' -n -k1,1 -k2,2 | while IFS= read -r entry; do
        trimmed="${entry#*|}"
        trimmed="${trimmed#*|}"
        printf "%s\n" "$trimmed"
    done >> "$GLOBAL_OUT"
else
    # Fallback: append unsorted rows
    printf "%s\n" "${ROWS[@]}" >> "$GLOBAL_OUT"
fi

echo "Global aggregated report generated at: $GLOBAL_OUT"
