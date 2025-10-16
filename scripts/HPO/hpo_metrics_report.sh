#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Author: Juan Luis Herrera Cortijo <juan.luis.herrera.cortijo@gmail.com>
# Created: 2025-10-16
# Disclaimer: This script is provided as-is for research workflows and must be validated before any operational deployment.
# -----------------------------------------------------------------------------

# Generate markdown summaries for SageMaker HPO jobs, extracting hyperparameters
# and fold metrics produced by the cv-based training loop.
set -euo pipefail

# Print CLI usage instructions.
usage() {
    cat <<EOF
Usage: $0 -n JOB_NAMES [-p PROFILE] [-r REGION] [-o OUTPUT_FILE]

Generate a markdown report from a SageMaker HyperParameter Tuning (HPO) job.

Options:
  -n JOB_NAMES   Comma-separated HPO job names (required)
  -p PROFILE     AWS CLI profile (default: \$AWS_PROFILE or 'default')
  -r REGION      AWS region (default: \$AWS_REGION or 'us-east-1')
  -o OUTPUT_FILE Path to output markdown report (default: firstJOB_hpo_report.md or hpo_multi_report.md)
  -h             Show this help message and exit
EOF
    exit 1
}

# Default AWS CLI configuration
PROFILE="${AWS_PROFILE:-default}"
REGION="${AWS_REGION:-us-east-1}"
OUTPUT_FILE=""
JOB_NAMES_ARG=""

# Parse command-line arguments
while getopts "n:p:r:o:h" opt; do
    case "$opt" in
        n) JOB_NAMES_ARG="${OPTARG}" ;; 
        p) PROFILE="${OPTARG}" ;; 
        r) REGION="${OPTARG}" ;; 
        o) OUTPUT_FILE="${OPTARG}" ;; 
        h|*) usage ;; 
    esac
done

if [ -z "$JOB_NAMES_ARG" ]; then
    echo "Error: JOB_NAMES (-n) is required."
    usage
fi
IFS=',' read -r -a HPO_JOBS <<< "$JOB_NAMES_ARG"
# Allow callers to request multiple jobs in one pass; aggregate after per-job runs.
if [ "${#HPO_JOBS[@]}" -gt 1 ]; then
    # Set default output file for multi-report
    if [ -z "$OUTPUT_FILE" ]; then
        OUTPUT_FILE="hpo_multi_report.md"
    fi
    REPORT_FILES=()
    for hpo in "${HPO_JOBS[@]}"; do
        rfile="${hpo}_hpo_report.md"
        echo "Generating individual report for HPO job: $hpo"
        "$0" -n "$hpo" -p "$PROFILE" -r "$REGION" -o "$rfile"
        REPORT_FILES+=("$rfile")
    done
    # Aggregate individual reports
    REPORT_LIST="$(IFS=,; echo "${REPORT_FILES[*]}")"
    AGG_SCRIPT="$(dirname "$0")/integrate_hpo_reports.sh"
    bash "$AGG_SCRIPT" -i "$REPORT_LIST" -o "$OUTPUT_FILE"
    exit 0
fi
# For backward compatibility, use first job for single-job behavior
JOB_NAME="${HPO_JOBS[0]}"

# Set default report file if not provided
if [ -z "$OUTPUT_FILE" ]; then
    if [ "${#HPO_JOBS[@]}" -gt 1 ]; then
        OUTPUT_FILE="hpo_multi_report.md"
    else
        OUTPUT_FILE="${HPO_JOBS[0]}_hpo_report.md"
    fi
fi

# Query the tuning job definition; keep the payload locally to avoid TMPDIR restrictions.
TMP_JSON=$(mktemp hpo_metrics_report.XXXXXX)
aws --profile "$PROFILE" --region "$REGION" \
    sagemaker describe-hyper-parameter-tuning-job \
    --hyper-parameter-tuning-job-name "$JOB_NAME" > "$TMP_JSON"

# Extract ordered list of hyperparameter names as JSON array
PARAM_NAMES_JSON=$(jq -c '[.HyperParameterTuningJobConfig.ParameterRanges.ContinuousParameterRanges[].Name, .HyperParameterTuningJobConfig.ParameterRanges.IntegerParameterRanges[].Name, .HyperParameterTuningJobConfig.ParameterRanges.CategoricalParameterRanges[].Name]' "$TMP_JSON")

# Extract objective metric name
METRIC_NAME=$(jq -r '.HyperParameterTuningJobConfig.HyperParameterTuningJobObjective.MetricName' "$TMP_JSON")
# Extract S3 output prefix for training job artifacts
S3_OUT=$(jq -r '.TrainingJobDefinition.OutputDataConfig.S3OutputPath' "$TMP_JSON")

# Generate markdown report header capturing job metadata for traceability.
{
    echo "# HPO Job Report: $JOB_NAME"
    echo ""
    echo "Generated on $(date)"
    echo ""
    echo "## Training Job Summaries"
    echo ""
} > "$OUTPUT_FILE"

# List the training jobs for this HPO job
TMP_LIST=$(mktemp hpo_metrics_list.XXXXXX)
aws --profile "$PROFILE" --region "$REGION" \
    sagemaker list-training-jobs-for-hyper-parameter-tuning-job \
    --hyper-parameter-tuning-job-name "$JOB_NAME" > "$TMP_LIST"
JOB_NAMES=( $(jq -r '.TrainingJobSummaries[].TrainingJobName' "$TMP_LIST") )
rm -f "$TMP_LIST"

if [ ${#JOB_NAMES[@]} -eq 0 ]; then
    echo "Error: No training jobs found for HPO job $JOB_NAME" >&2
    rm -f "$TMP_JSON"
    exit 1
fi

# Convert PARAM_NAMES_JSON into Bash array
PARAM_NAMES=()
while IFS= read -r name; do
    PARAM_NAMES+=("$name")
done < <(echo "$PARAM_NAMES_JSON" | jq -r '.[]')

# Build and append Markdown table header
header="| TrainingJobName"
for name in "${PARAM_NAMES[@]}"; do
    header+=" | $name"
done
header+=" | $METRIC_NAME | StdDev-${METRIC_NAME} | AvgRMSE-speed | StdDev-AvgRMSE-speed | AvgRMSE-dir | StdDev-AvgRMSE-dir | AvgRangeAccuracy | StdDev-RangeAccuracy | AvgRangeMacroF1 | StdDev-RangeMacroF1 |"
printf '%s\n' "$header" >> "$OUTPUT_FILE"

sep="| ---"
for name in "${PARAM_NAMES[@]}"; do
    sep+=" | ---"
done
sep+=" | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
printf '%s\n' "$sep" >> "$OUTPUT_FILE"
# Initialize array to collect rows for sorting
ROWS=()

# For each training job, fetch hyperparameters and final metric
for job in "${JOB_NAMES[@]}"; do
    TMP_JOB=$(mktemp hpo_metrics_job.XXXXXX)
    aws --profile "$PROFILE" --region "$REGION" \
        sagemaker describe-training-job \
        --training-job-name "$job" > "$TMP_JOB"

    # Extract hyperparameter values per parameter
    row="| $job"
    for pname in "${PARAM_NAMES[@]}"; do
        val=$(jq -r --arg p "$pname" '.HyperParameters[$p] // ""' "$TMP_JOB")
        row+=" | $val"
    done
    # Extract the objective metric value (combined loss)
    loss=$(jq -r --arg m "$METRIC_NAME" \
        '.FinalMetricDataList[] | select(.MetricName == $m) | .Value' \
        "$TMP_JOB" 2>/dev/null || echo "")
    # Append combined loss
    row+=" | $loss"

    # Compute average RMSE and EAAM across folds
    TARBALL_S3="${S3_OUT}/${job}/output/output.tar.gz"
    TMP_TAR_FOLD=$(mktemp hpo_fold_tar.XXXXXX)
    avg_rmse=""
    std_rmse_speed=""
    avg_rmse_dir=""
    std_rmse_dir=""
    std_combined_loss=""
    avg_range_acc=""
    std_range_acc=""
    avg_range_f1=""
    std_range_f1=""
    # Download the tarball generated by SageMaker; it bundles per-fold CSV metrics.
    if aws --profile "$PROFILE" --region "$REGION" s3 cp "$TARBALL_S3" "$TMP_TAR_FOLD" >/dev/null 2>&1; then
        agg_output=$(TMP_TAR_PATH="$TMP_TAR_FOLD" python3 <<'PY'
import csv
import os
import statistics
import tarfile
import re

tar_path = os.environ['TMP_TAR_PATH']

def try_float(value):
    try:
        if value is None or value == '':
            return None
        return float(value)
    except (TypeError, ValueError):
        return None

rmse_speed = []
rmse_dir = []
combined_loss = []
range_accuracy = []
range_macro_f1 = []

pat_metrics = re.compile(r'^metrics_fold(\d+)\.csv$')
pat_class = re.compile(r'^metrics_fold(\d+)_range_classification\.csv$')

with tarfile.open(tar_path, 'r:gz') as tar:
    for member in tar.getmembers():
        name = os.path.basename(member.name)
        if pat_metrics.match(name):
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            # Each metrics_fold*.csv contains a single row with fold-level regression KPIs.
            text = extracted.read().decode('utf-8').splitlines()
            if len(text) < 2:
                continue
            header = text[0].split(',')
            row = text[1].split(',')
            data = {k: v for k, v in zip(header, row)}
            rmse = try_float(data.get('rmse_speed'))
            if rmse is not None:
                rmse_speed.append(rmse)
            rmse_dir_val = try_float(data.get('rmse_dir'))
            if rmse_dir_val is not None:
                rmse_dir.append(rmse_dir_val)
            combined = try_float(data.get('combined_loss'))
            if combined is not None:
                combined_loss.append(combined)
        elif pat_class.match(name):
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            # Range classification CSVs enumerate per-class statistics; only the 'overall' row is required here.
            reader = csv.DictReader(line.decode('utf-8') if isinstance(line, bytes) else line for line in extracted)
            for row in reader:
                label = (row.get('class') or '').strip().lower()
                if label == 'overall':
                    acc = try_float(row.get('accuracy'))
                    if acc is not None:
                        range_accuracy.append(acc)
                    macro_f1 = try_float(row.get('f1'))
                    if macro_f1 is not None:
                        range_macro_f1.append(macro_f1)
                    break

def mean_std(values):
    if not values:
        return '', ''
    mean_val = statistics.mean(values)
    std_val = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{mean_val:.6f}", f"{std_val:.6f}"

avg_rmse, std_rmse_speed = mean_std(rmse_speed)
avg_rmse_dir, std_rmse_dir = mean_std(rmse_dir)

if combined_loss:
    std_combined = statistics.stdev(combined_loss) if len(combined_loss) > 1 else 0.0
    std_combined_loss = f"{std_combined:.6f}"
else:
    std_combined_loss = ''

avg_acc, std_acc = mean_std(range_accuracy)
avg_f1, std_f1 = mean_std(range_macro_f1)

print('|'.join([
    avg_rmse,
    std_rmse_speed,
    avg_rmse_dir,
    std_rmse_dir,
    std_combined_loss,
    avg_acc,
    std_acc,
    avg_f1,
    std_f1,
]))
        PY)
        IFS='|' read -r avg_rmse std_rmse_speed avg_rmse_dir std_rmse_dir std_combined_loss avg_range_acc std_range_acc avg_range_f1 std_range_f1 <<< "$agg_output"
    fi
    rm -f "$TMP_TAR_FOLD"
    # Append metrics: combined loss, its stddev, average RMSE-speed with stddev, and average RMSE-dir with stddev
    row+=" | $std_combined_loss | $avg_rmse | $std_rmse_speed | $avg_rmse_dir | $std_rmse_dir | $avg_range_acc | $std_range_acc | $avg_range_f1 | $std_range_f1 |"
    # Collect row prefixed with sorting helpers so missing metrics fall to the bottom
    sort_metric="$loss"
    missing_flag=0
    if [[ -z "$loss" || ! "$loss" =~ ^-?[0-9]+([.][0-9]+)?$ ]]; then
        missing_flag=1
        sort_metric="9999999999"
    fi
    ROWS+=("$missing_flag|$sort_metric$row")
    rm -f "$TMP_JOB"
done

# Sort collected rows by metric (best to worst) and append to report
printf '%s\n' "${ROWS[@]}" | sort -t '|' -n -k1,1 -k2,2 | while IFS= read -r entry; do
    trimmed="${entry#*|}"
    trimmed="${trimmed#*|}"
    printf '|%s\n' "$trimmed" >> "$OUTPUT_FILE"
done
printf '\n' >> "$OUTPUT_FILE"
echo "## Detailed Fold Metrics per Training Job" >> "$OUTPUT_FILE"
    for job in "${JOB_NAMES[@]}"; do
        echo "" >> "$OUTPUT_FILE"
        echo "### Job: $job" >> "$OUTPUT_FILE"
        # Include hyperparameters specific to this training job
        TMP_JOB_PARAMS=$(mktemp hpo_metrics_params.XXXXXX)
        aws --profile "$PROFILE" --region "$REGION" \
            sagemaker describe-training-job \
            --training-job-name "$job" > "$TMP_JOB_PARAMS"
        echo "" >> "$OUTPUT_FILE"
        echo "#### Hyperparameters" >> "$OUTPUT_FILE"
        echo "" >> "$OUTPUT_FILE"
        echo "| Parameter | Value |" >> "$OUTPUT_FILE"
        echo "| --- | --- |" >> "$OUTPUT_FILE"
        jq -r '.HyperParameters | to_entries[] | "| \(.key) | \(.value) |"' "$TMP_JOB_PARAMS" >> "$OUTPUT_FILE"
        rm -f "$TMP_JOB_PARAMS"
        echo "" >> "$OUTPUT_FILE"
        echo "| Fold | RMSE Speed | MAE Speed | Corr Speed | R2 Speed | Bias Speed | SI Speed | EAM Dir | EAAM Dir | RMSE Dir | CompCorr Dir | SI Speed Max | Combined Loss |" >> "$OUTPUT_FILE"
        echo "| ---- | ---------- | --------- | ---------- | --------- | ---------- | -------- | ------- | -------- | -------- | ------------ | ------------- | ------------- |" >> "$OUTPUT_FILE"
        TARBALL_S3="${S3_OUT}/${job}/output/output.tar.gz"
        TMP_TAR=$(mktemp hpo_metrics_tar.XXXXXX)
        if aws --profile "$PROFILE" --region "$REGION" s3 cp "$TARBALL_S3" "$TMP_TAR" 2>/dev/null; then
            # Iterate over fold CSVs inside the tarball to populate the detailed table.
            for file in $(tar -tzf "$TMP_TAR" | grep -E '^metrics_fold[0-9]+\.csv$'); do
                # Extract the fold number and metrics from the CSV (fold now in first column)
                line=$(tar -xzOf "$TMP_TAR" "$file" | sed -n '2p')
                IFS=',' read -r fold rmse_speed mae_speed corr_speed r2_speed bias_speed si_speed eam_dir eaam_dir rmse_dir compcorr_dir si_speed_max combined_loss <<< "$line"
                echo "| $fold | $rmse_speed | $mae_speed | $corr_speed | $r2_speed | $bias_speed | $si_speed | $eam_dir | $eaam_dir | $rmse_dir | $compcorr_dir | $si_speed_max | $combined_loss |" >> "$OUTPUT_FILE"
            done
            # Summarise macro-F1/accuracy per fold in a nested Markdown table.
            CLASS_SECTION=$(TMP_TAR_PATH="$TMP_TAR" python3 <<'PY'
import csv
import io
import os
import tarfile
import re

tar_path = os.environ['TMP_TAR_PATH']
pat_class = re.compile(r'^metrics_fold(\d+)_range_classification\.csv$')
rows = []

with tarfile.open(tar_path, 'r:gz') as tar:
    for member in tar.getmembers():
        name = os.path.basename(member.name)
        match = pat_class.match(name)
        if not match:
            continue
        fold = int(match.group(1))
        extracted = tar.extractfile(member)
        if extracted is None:
            continue
        reader = csv.DictReader(io.TextIOWrapper(extracted, encoding='utf-8'))
        for row in reader:
            if (row.get('class') or '').strip().lower() == 'overall':
                rows.append((fold, row))
                break

if rows:
    rows.sort(key=lambda item: item[0])
    def fmt(value, digits=6, integer=False):
        if value is None or value == '':
            return ''
        try:
            num = float(value)
        except (TypeError, ValueError):
            return ''
        if integer:
            return str(int(round(num)))
        return f"{num:.{digits}f}"

    lines = [
        '| Fold | Accuracy | Precision | Recall | F1 | Support | Predicted | TruePositives |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |',
    ]
    for fold, metrics in rows:
        line = "| {fold} | {acc} | {prec} | {rec} | {f1} | {sup} | {pred} | {tp} |".format(
            fold=fold,
            acc=fmt(metrics.get('accuracy')),
            prec=fmt(metrics.get('precision')),
            rec=fmt(metrics.get('recall')),
            f1=fmt(metrics.get('f1')),
            sup=fmt(metrics.get('support'), integer=True),
            pred=fmt(metrics.get('predicted'), integer=True),
            tp=fmt(metrics.get('true_positives'), integer=True),
        )
        lines.append(line)
    print('\n'.join(lines))
else:
    print('', end='')
PY)
            # Append the range-classification summary if available.
            if [ -n "$CLASS_SECTION" ]; then
                echo "" >> "$OUTPUT_FILE"
                echo "#### Range Classification (validation folds)" >> "$OUTPUT_FILE"
                echo "" >> "$OUTPUT_FILE"
                printf "%s\n" "$CLASS_SECTION" >> "$OUTPUT_FILE"
            fi
        fi
        rm -f "$TMP_TAR"
    done

    # Clean up config JSON file
    rm -f "$TMP_JSON"
    echo "Markdown report generated: $OUTPUT_FILE"
