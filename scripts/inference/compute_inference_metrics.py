#!/usr/bin/env python3
"""Compute regression and range-classification metrics from inference output.

The script consolidates regression diagnostics, range-aware classification
scores, and maintenance/group breakdowns into Markdown and CSV artefacts. It is
designed to run inside the Docker workflow invoked by
``compute_inference_metrics.sh`` so that analysts can reproduce model
evaluation outside SageMaker.

Author: Juan Luis Herrera Cortijo <juan.luis.herrera.cortijo@gmail.com>
Created: 2025-10-16
Disclaimer: This script is provided as-is for research workflows and must be validated before any operational deployment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments controlling the metrics post-processing job.

    Returns:
        argparse.Namespace: Parameters defining input paths, group breakdowns,
        and truth column overrides.
    """
    parser = argparse.ArgumentParser(description="Compute metrics for inference outputs")
    parser.add_argument("--predictions", required=True, help="Local path to predictions parquet file")
    parser.add_argument("--metadata", required=False, help="Optional path to inference_metadata.json")
    parser.add_argument("--output-dir", required=True, help="Directory where the metrics report will be written")
    parser.add_argument("--truth-speed-col", default="wind_speed", help="Column name for true wind speed in predictions parquet")
    parser.add_argument("--truth-dir-col", default="wind_dir", help="Column name for true wind direction in predictions parquet")
    parser.add_argument(
        "--group-column",
        action="append",
        default=[],
        help=(
            "Column name to compute per-group metrics (may be repeated). "
            "Comma-separated lists are also accepted."
        ),
    )
    parser.add_argument(
        "--wind-bin-column",
        default="wind_bin",
        help=(
            "Column containing wind-speed bin labels. Leave empty to skip "
            "wind-bin diagnostics."
        ),
    )
    return parser.parse_args()


def load_metadata(path: Path | None) -> Dict[str, float]:
    """Return persisted metadata when available, defaulting to an empty mapping.

    Args:
        path: Optional path to ``inference_metadata.json``.

    Returns:
        dict: Metadata contents or an empty dictionary when the file is absent.
    """
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def compute_direction_metrics(df: pd.DataFrame, pred_column: str, true_column: str) -> Tuple[np.ndarray, np.ndarray]:
    """Compute angular residuals (signed and absolute) in radians.

    Args:
        df: Predictions dataframe containing the specified columns.
        pred_column: Column with predicted direction angles in degrees.
        true_column: Column with reference direction angles in degrees.

    Returns:
        tuple[np.ndarray, np.ndarray]: Signed and absolute angular errors.
    """
    pred_rad = np.deg2rad(df[pred_column].values)
    true_rad = np.deg2rad(df[true_column].values)
    diff = np.arctan2(np.sin(pred_rad - true_rad), np.cos(pred_rad - true_rad))
    return diff, np.abs(diff)


def compute_classification_metrics(
    df: pd.DataFrame,
    range_min: float,
    range_max: float,
    truth_speed_col: str,
    pred_label_col: str = "pred_range_label",
) -> Dict[str, float]:
    """Evaluate range-classification performance against the deterministic gating.

    Args:
        df: Predictions dataframe containing the probability outputs.
        range_min: Lower bound of the valid wind-speed range.
        range_max: Upper bound of the valid wind-speed range.
        truth_speed_col: Column holding reference wind speed values.
        pred_label_col: Column with categorical predictions.

    Returns:
        dict: Accuracy, precision/recall/F1 triplets, and macro-F1 summary.
    """
    if pred_label_col not in df.columns:
        return {}

    true_speeds = df[truth_speed_col].values
    true_labels = np.where(true_speeds < range_min, "below",
                           np.where(true_speeds > range_max, "above", "in"))
    pred_labels = df[pred_label_col].astype(str).values

    accuracy = np.mean(pred_labels == true_labels)

    metrics = {"accuracy": float(accuracy)}
    for label in ["below", "in", "above"]:
        mask_true = true_labels == label
        mask_pred = pred_labels == label
        tp = np.sum(mask_true & mask_pred)
        precision = tp / np.sum(mask_pred) if np.any(mask_pred) else np.nan
        recall = tp / np.sum(mask_true) if np.any(mask_true) else np.nan
        if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0:
            f1 = np.nan
        else:
            f1 = 2 * precision * recall / (precision + recall)
        metrics[f"precision_{label}"] = precision
        metrics[f"recall_{label}"] = recall
        metrics[f"f1_{label}"] = f1

    valid_f1 = [metrics[f"f1_{label}"] for label in ("below", "in", "above") if not np.isnan(metrics[f"f1_{label}"])]
    if valid_f1:
        metrics["macro_f1"] = float(np.mean(valid_f1))
    else:
        metrics["macro_f1"] = float("nan")

    return metrics


def normalize_interval_value(value: object) -> str:
    """Return a human-readable label for interval values, preserving NULL state.

    Args:
        value: Interval/group value coming from the predictions dataframe.

    Returns:
        str: ``"NULL"`` when the input is missing, otherwise ``str(value)``.
    """
    if pd.isna(value):
        return "NULL"
    return str(value)


def resolve_truth_column(
    df: pd.DataFrame,
    requested: str,
    fallbacks: List[str],
    role: str,
) -> str:
    """Resolve the column used for a ground-truth field, allowing smart fallbacks.

    Args:
        df: Predictions dataframe.
        requested: User-specified column name.
        fallbacks: Ordered list of alternative names to try.
        role: Human-readable description for logging.

    Returns:
        str: Column name found in ``df``.

    Raises:
        KeyError: If none of the requested or fallback columns exist.
    """

    candidates: List[str] = []
    if requested:
        candidates.append(requested)
    for option in fallbacks:
        if option and option not in candidates:
            candidates.append(option)

    for candidate in candidates:
        if candidate in df.columns:
            if requested and candidate != requested:
                print(
                    f"[compute_inference_metrics] Column '{requested}' not found; "
                    f"using '{candidate}' for {role} instead.",
                    flush=True,
                )
            return candidate

    primary = requested or (fallbacks[0] if fallbacks else "")
    raise KeyError(
        f"Expected column '{primary}' in predictions parquet produced by inference.py"
    )


def relative_increase(value: float, baseline: float) -> float:
    """Compute relative increase vs. baseline; NaN-safe with zero guard.

    Args:
        value: Metric value for a subset.
        baseline: Baseline metric value used for comparison.

    Returns:
        float: Relative increase ``(value - baseline) / baseline`` or ``nan``
        when the baseline is zero or any operand is non-finite.
    """
    if not np.isfinite(value) or not np.isfinite(baseline) or baseline == 0:
        return float("nan")
    return (value - baseline) / baseline


def main() -> None:
    """Coordinate metrics computation, reporting, and CSV exports.

    The routine mirrors the diagnostic stages described in ``docs/inference.md``:
    it ingests the inference parquet, reconciles truth columns, computes global
    regression/classification summaries, and then stratifies them by maintenance
    intervals, arbitrary groupings, and wind-speed bins before rendering
    Markdown and CSV artefacts.
    """
    args = parse_args()

    predictions_path = Path(args.predictions)
    metadata_path = Path(args.metadata) if args.metadata else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parquet outputs keep the schema stable regardless of upstream storage formats.
    # Stage 1 — load inference predictions (parquet ensures schema continuity across runs).
    df = pd.read_parquet(predictions_path)

    # Consolidate repeated ``--group-column`` occurrences into a flat set of names.
    # Consolidate repeated ``--group-column`` occurrences into a flat list while preserving order.
    raw_group_columns: List[str] = []
    for entry in args.group_column:
        if not entry:
            continue
        raw_group_columns.extend([part.strip() for part in entry.split(",") if part.strip()])

    group_columns: List[str] = []
    for column in raw_group_columns:
        if column not in group_columns:
            group_columns.append(column)

    wind_bin_column = (args.wind_bin_column or "").strip()

    # Guard early against typos so the report never skips a requested breakdown silently.
    missing_columns = [column for column in group_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(
            "Requested group columns not found in predictions parquet: "
            + ", ".join(sorted(missing_columns))
        )

    pred_speed_col = "pred_wind_speed"
    pred_dir_col = "pred_wind_direction"
    # Resolve the ground-truth columns, respecting CLI overrides and falling back to canonical names.
    truth_speed_col = resolve_truth_column(
        df,
        args.truth_speed_col,
        ["wind_speed", "ground_truth_wind_speed"],
        "truth wind speed",
    )
    truth_dir_col = resolve_truth_column(
        df,
        args.truth_dir_col,
        ["wind_dir", "wind_direction", "ground_truth_wind_direction"],
        "truth wind direction",
    )

    for col in (pred_speed_col, pred_dir_col, truth_speed_col, truth_dir_col):
        if col not in df.columns:
            raise KeyError(f"Expected column '{col}' in predictions parquet produced by inference.py")

    has_truth_speed = True
    has_truth_dir = True

    if not (has_truth_speed and has_truth_dir):
        note = "Ground-truth wind columns not found in inference output. Skipping metric computation."
        report_path = output_dir / "inference_metrics_report.md"
        with report_path.open("w", encoding="utf-8") as handle:
            handle.write("# Inference Metrics\n\n")
            handle.write(f"{note}\n")
        metrics_csv = output_dir / "inference_metrics.csv"
        pd.DataFrame([{"metric": "note", "value": note}]).to_csv(metrics_csv, index=False)
        return

    # Metadata (range bounds, labels) keeps the diagnostics aligned with the inference job when available.
    meta = load_metadata(metadata_path)
    range_min = float(meta.get("range_min", 5.7))
    range_max = float(meta.get("range_max", 17.8))

    def regression_metrics(df_subset: pd.DataFrame) -> Dict[str, float]:
        """Compute scalar regression diagnostics for a subset of the predictions."""
        if df_subset.empty:
            return {"samples": 0}
        # Core scalar diagnostics mirror the training reports (RMSE, MAE, bias, etc.).
        error_speed = df_subset[pred_speed_col].values - df_subset[truth_speed_col].values
        n_samples = len(df_subset)
        rmse = float(np.sqrt(np.mean(error_speed ** 2)))
        mae = float(np.mean(np.abs(error_speed)))
        bias = float(np.mean(error_speed))
        corr = float(np.corrcoef(df_subset[pred_speed_col], df_subset[truth_speed_col])[0, 1])
        r2_num = np.sum((df_subset[pred_speed_col] - df_subset[truth_speed_col].mean()) ** 2)
        r2_den = np.sum((df_subset[truth_speed_col] - df_subset[truth_speed_col].mean()) ** 2)
        r2 = float(1 - ((rmse ** 2 * n_samples) / r2_den)) if r2_den > 0 else float("nan")
        std_error = float(np.sqrt(np.mean((error_speed - bias) ** 2)))
        mean_true_speed = float(np.mean(np.abs(df_subset[truth_speed_col].values)))
        max_true_speed = float(np.max(np.abs(df_subset[truth_speed_col].values)))
        si = std_error / mean_true_speed if mean_true_speed > 0 else float("nan")
        si_max = std_error / max_true_speed if max_true_speed > 0 else float("nan")
        # Directional components are measured in radians and converted back to degrees for readability.
        diff_rad, abs_diff_rad = compute_direction_metrics(df_subset, pred_dir_col, truth_dir_col)
        eam_dir = float(np.degrees(np.mean(diff_rad)))
        eaam_dir = float(np.degrees(np.mean(abs_diff_rad)))
        rmse_dir = float(np.degrees(np.sqrt(np.mean(diff_rad ** 2))))
        compcorr_real = np.sum(np.cos(np.deg2rad(df_subset[pred_dir_col])) * np.cos(np.deg2rad(df_subset[truth_dir_col])) +
                               np.sin(np.deg2rad(df_subset[pred_dir_col])) * np.sin(np.deg2rad(df_subset[truth_dir_col])))
        compcorr_imag = np.sum(np.sin(np.deg2rad(df_subset[pred_dir_col])) * np.cos(np.deg2rad(df_subset[truth_dir_col])) -
                               np.cos(np.deg2rad(df_subset[pred_dir_col])) * np.sin(np.deg2rad(df_subset[truth_dir_col])))
        compcorr_dir = float(np.sqrt(compcorr_real ** 2 + compcorr_imag ** 2) / len(df_subset))
        return {
            "samples": int(n_samples),
            "rmse_speed": rmse,
            "mae_speed": mae,
            "bias_speed": bias,
            "corr_speed": corr,
            "r2_speed": r2,
            "si_speed": si,
            "si_speed_max": si_max,
            "eam_dir": eam_dir,
            "eaam_dir": eaam_dir,
            "rmse_dir": rmse_dir,
            "compcorr_dir": compcorr_dir,
        }

    # Stage 2 — global regression diagnostics.
    metrics_full = regression_metrics(df)

    in_range_mask = (df[truth_speed_col].values >= range_min) & (df[truth_speed_col].values <= range_max)
    metrics_in_range = regression_metrics(df[in_range_mask])

    match_mask: np.ndarray | None = None
    metrics_match = {}
    if "pred_range_label" in df.columns:
        speed_based_labels = np.where(df[truth_speed_col].values < range_min, "below",
                                      np.where(df[truth_speed_col].values > range_max, "above", "in"))
        match_mask = df["pred_range_label"].astype(str).values == speed_based_labels
        metrics_match = regression_metrics(df[in_range_mask & match_mask])

    # Stage 3 — range classification diagnostics, benchmarked against deterministic speed gating.
    classification_metrics = compute_classification_metrics(
        df,
        range_min,
        range_max,
        truth_speed_col=truth_speed_col,
    )
    classification_metrics_in_range = compute_classification_metrics(
        df[in_range_mask],
        range_min,
        range_max,
        truth_speed_col=truth_speed_col,
    ) if in_range_mask.any() else {}
    classification_metrics_matched = {}
    if match_mask is not None:
        matched_mask = in_range_mask & match_mask
        if matched_mask.any():
            classification_metrics_matched = compute_classification_metrics(
                df[matched_mask],
                range_min,
                range_max,
                truth_speed_col=truth_speed_col,
            )

    interval_records: List[Dict[str, float | int | str]] = []

    maintenance_columns = [
        col for col in df.columns
        if "maintenance_interval" in col.lower()
    ]

    # Create boolean masks that will be reused when stratifying diagnostics.
    df["__subset_all__"] = True
    df["__subset_in_range__"] = in_range_mask
    if match_mask is not None:
        df["__subset_matched__"] = in_range_mask & match_mask

    # Describe the subsets for which regression/classification baselines will be generated.
    subset_configs = [
        ("all", "__subset_all__", "All samples", metrics_full, classification_metrics),
        ("in_range", "__subset_in_range__", "Within range", metrics_in_range, classification_metrics_in_range),
    ]
    if match_mask is not None:
        subset_configs.append(
            ("matched", "__subset_matched__", "Within range & classification match", metrics_match, classification_metrics_matched)
        )

    baseline_regression = {name: baseline for name, _, _, baseline, _ in subset_configs}
    baseline_macro = {
        name: subset_class.get("macro_f1", float("nan")) if subset_class else float("nan")
        for name, _, _, _, subset_class in subset_configs
    }

    min_support_ratio = 0.05
    min_support_absolute = 24
    degradation_threshold = 0.15
    macro_drop_threshold = 0.15
    wind_bin_csv_map: Dict[str, List[Dict[str, object]]] = {}

    report_path = output_dir / "inference_metrics_report.md"
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("# Inference Metrics\n\n")

        def write_table(title: str, metrics: Dict[str, float]) -> None:
            """Render a simple two-column Markdown table with metric values."""
            handle.write(f"## {title}\n\n")
            handle.write("| metric | value |\n| --- | --- |\n")
            for key, value in metrics.items():
                handle.write(f"| {key} | {value:.6f} |\n" if isinstance(value, float) else f"| {key} | {value} |\n")
            handle.write("\n")

        write_table("Regression (all samples)", metrics_full)
        write_table("Regression (within range)", metrics_in_range)
        if metrics_match:
            write_table("Regression (within range & classification match)", metrics_match)
        write_table("Range classification", classification_metrics)
        def render_group_section(columns: List[str], section_title: str, *, is_maintenance: bool) -> None:
            """Emit maintenance- or group-level breakdowns with degradation alerts."""
            if not columns:
                return

            handle.write(f"## {section_title}\n\n")
            if is_maintenance:
                handle.write(
                    "Maintenance-aware diagnostics complement the global aggregates by highlighting "
                    "intervals where inference deviates from the fleet-wide baseline. NULL entries "
                    "correspond to predictions without a recorded maintenance interval and should be "
                    "benchmarked directly against the global metrics above.\n\n"
                )
            else:
                handle.write(
                    "Per-group diagnostics highlight how inference quality varies across categorical "
                    "partitions. Use these tables to spot subsets where regression or classification "
                    "depart from the global baseline.\n\n"
                )

            label_heading = "interval" if is_maintenance else "group"
            label_word = "Interval" if is_maintenance else "Group"
            sentinel_value = "__NULL_INTERVAL__" if is_maintenance else "__NULL_GROUP__"

            for column in columns:
                df_group = df.copy()
                df_group["__interval_group__"] = df_group[column].astype("object").where(
                    ~df_group[column].isna(), sentinel_value
                )

                column_subset_records: Dict[str, List[Dict[str, float | int | str]]] = {
                    name: [] for name, _, _, _, _ in subset_configs
                }

                for group_value, df_subset in df_group.groupby("__interval_group__"):
                    if df_subset.empty:
                        continue

                    label = "NULL" if group_value == sentinel_value else normalize_interval_value(group_value)

                    subset_metrics_map: Dict[str, Dict[str, float]] = {}
                    subset_macro_map: Dict[str, float] = {}

                    for subset_name, subset_column, _, _, _ in subset_configs:
                        if subset_column in df_subset.columns:
                            subset_df = df_subset[df_subset[subset_column]]
                        else:
                            subset_df = df_subset

                        metrics_interval = regression_metrics(subset_df)
                        subset_metrics_map[subset_name] = metrics_interval

                        if "pred_range_label" in subset_df.columns and not subset_df.empty:
                            subset_class_metrics = compute_classification_metrics(
                                subset_df,
                                range_min,
                                range_max,
                                truth_speed_col=truth_speed_col,
                            )
                            subset_macro_map[subset_name] = subset_class_metrics.get("macro_f1", float("nan"))
                        else:
                            subset_macro_map[subset_name] = float("nan")

                    interval_record: Dict[str, float | int | str] = {
                        "interval_column": column,
                        "interval_value": label,
                    }

                    for subset_name, _, _, _, _ in subset_configs:
                        metrics_interval = subset_metrics_map.get(subset_name, {})
                        baseline_metrics = baseline_regression.get(subset_name, {})
                        baseline_macro_value = baseline_macro.get(subset_name, float("nan"))

                        prefix = "all" if subset_name == "all" else subset_name

                        rmse_delta = relative_increase(
                            metrics_interval.get("rmse_speed", float("nan")),
                            baseline_metrics.get("rmse_speed", float("nan"))
                        )
                        eaam_delta = relative_increase(
                            metrics_interval.get("eaam_dir", float("nan")),
                            baseline_metrics.get("eaam_dir", float("nan"))
                        )
                        macro_interval = subset_macro_map.get(subset_name, float("nan"))
                        macro_delta = (
                            macro_interval - baseline_macro_value
                            if np.isfinite(macro_interval) and np.isfinite(baseline_macro_value)
                            else float("nan")
                        )

                        record_entry = {
                            "label": label,
                            "samples": int(metrics_interval.get("samples", 0)),
                            "rmse_speed": metrics_interval.get("rmse_speed", float("nan")),
                            "mae_speed": metrics_interval.get("mae_speed", float("nan")),
                            "rmse_dir": metrics_interval.get("rmse_dir", float("nan")),
                            "eaam_dir": metrics_interval.get("eaam_dir", float("nan")),
                            "macro_f1": macro_interval,
                            "delta_rmse_pct": rmse_delta * 100 if np.isfinite(rmse_delta) else float("nan"),
                            "delta_eaam_pct": eaam_delta * 100 if np.isfinite(eaam_delta) else float("nan"),
                            "delta_macro": macro_delta,
                        }
                        column_subset_records[subset_name].append(record_entry)

                        interval_record[f"{prefix}_samples"] = record_entry["samples"]
                        interval_record[f"{prefix}_rmse_speed"] = record_entry["rmse_speed"]
                        interval_record[f"{prefix}_mae_speed"] = record_entry["mae_speed"]
                        interval_record[f"{prefix}_eaam_dir"] = record_entry["eaam_dir"]
                        interval_record[f"{prefix}_rmse_dir"] = record_entry["rmse_dir"]
                        interval_record[f"{prefix}_macro_f1"] = record_entry["macro_f1"]
                        interval_record[f"{prefix}_rmse_delta_vs_baseline"] = rmse_delta
                        interval_record[f"{prefix}_eaam_delta_vs_baseline"] = eaam_delta
                        interval_record[f"{prefix}_macro_delta_vs_baseline"] = macro_delta

                        if subset_name == "all":
                            interval_record["samples"] = record_entry["samples"]
                            interval_record["rmse_speed"] = record_entry["rmse_speed"]
                            interval_record["mae_speed"] = record_entry["mae_speed"]
                            interval_record["eaam_dir"] = record_entry["eaam_dir"]
                            interval_record["rmse_dir"] = record_entry["rmse_dir"]
                            interval_record["macro_f1"] = record_entry["macro_f1"]
                            interval_record["rmse_delta_vs_global"] = rmse_delta
                            interval_record["eaam_delta_vs_global"] = eaam_delta
                            interval_record["macro_f1_delta_vs_global"] = macro_delta

                    interval_records.append(interval_record)

                threshold_pct = degradation_threshold * 100

                for subset_name, _, subset_title, _, _ in subset_configs:
                    subset_entries = column_subset_records.get(subset_name, [])
                    if not subset_entries:
                        continue

                    handle.write(f"### Column `{column}` — {subset_title}\n\n")
                    handle.write(
                        f"| {label_heading} | samples | rmse_speed | mae_speed | rmse_dir | eaam_dir | macro_f1 | Δrmse (%) | Δeaam (%) | Δmacro_f1 |\n"
                    )
                    handle.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
                    subset_entries.sort(key=lambda rec: (rec["samples"], rec["label"]), reverse=True)
                    for entry in subset_entries:
                        handle.write(
                            "| {label} | {samples} | {rmse:.6f} | {mae:.6f} | {rmse_dir:.6f} | {eaam:.6f} | {macro:.6f} | {rmse_pct:.2f} | {eaam_pct:.2f} | {macro_delta:.6f} |\n".format(
                                label=entry["label"],
                                samples=entry["samples"],
                                rmse=entry["rmse_speed"],
                                mae=entry["mae_speed"],
                                rmse_dir=entry["rmse_dir"],
                                eaam=entry["eaam_dir"],
                                macro=entry["macro_f1"],
                                rmse_pct=entry["delta_rmse_pct"],
                                eaam_pct=entry["delta_eaam_pct"],
                                macro_delta=entry["delta_macro"],
                            )
                        )
                    handle.write("\n")

                    if subset_name == "all":
                        degrade_lines: List[str] = []
                        # Only issue alerts when the subset carries enough support to be reliable.
                        for entry in subset_entries:
                            samples = entry["samples"]
                            if samples < max(int(min_support_ratio * metrics_full.get("samples", samples)), min_support_absolute):
                                continue

                            rmse_increase = entry["delta_rmse_pct"] / 100.0
                            eaam_increase = entry["delta_eaam_pct"] / 100.0
                            macro_delta = entry["delta_macro"]

                            if np.isfinite(rmse_increase) and rmse_increase > degradation_threshold:
                                degrade_lines.append(
                                    f"{label_word} `{entry['label']}` exhibits an RMSE {rmse_increase * 100:.1f}% above the global baseline."
                                )
                            if np.isfinite(eaam_increase) and eaam_increase > degradation_threshold:
                                degrade_lines.append(
                                    f"{label_word} `{entry['label']}` shows angular MAE {eaam_increase * 100:.1f}% above the global reference."
                                )
                            if np.isfinite(macro_delta) and macro_delta < -macro_drop_threshold:
                                degrade_lines.append(
                                    f"{label_word} `{entry['label']}` reduces macro-F1 by {abs(macro_delta):.3f} relative to the baseline, indicating classification drift."
                                )

                        if degrade_lines:
                            narrative = " ".join(degrade_lines)
                            if is_maintenance:
                                handle.write(
                                    f"**Degradation alert:** {narrative} Intervals exceeding the {threshold_pct:.0f}% degradation threshold merit calibration checks and should be cross-referenced with the maintenance chronology described in `docs/attach_maintenance_windows.md`.\n\n"
                                )
                            else:
                                handle.write(f"**Degradation alert:** {narrative}\n\n")
                        else:
                            if is_maintenance:
                                handle.write(
                                    f"No maintenance interval surpassed the {threshold_pct:.0f}% degradation threshold against the global RMSE or angular MAE baseline. Minor oscillations remain within expected variability.\n\n"
                                )
                            else:
                                handle.write(
                                    f"No group within `{column}` exceeded the {threshold_pct:.0f}% degradation threshold relative to the global baseline.\n\n"
                                )

        def render_wind_bin_section(column: str) -> Dict[str, List[Dict[str, object]]]:
            """Summarise metrics per wind bin and collect CSV payloads for export."""
            csv_records: Dict[str, List[Dict[str, object]]] = {}
            sentinel_value = "__NULL_WIND_BIN__"
            df_bins = df.copy()
            df_bins["__wind_bin_label__"] = df_bins[column].astype("object").where(
                ~df_bins[column].isna(), sentinel_value
            )

            handle.write("## Wind bin analysis\n\n")
            handle.write(
                "Per-bin diagnostics mirror the training-time reporting, allowing direct comparison "
                "of inference quality across wind-speed regimes. Bins marked as NULL correspond to "
                "predictions without an associated wind category and should be benchmarked against the "
                "global aggregates above.\n\n"
            )

            for subset_name, subset_column, subset_title, _, _ in subset_configs:
                if subset_column in df_bins.columns:
                    subset_df = df_bins[df_bins[subset_column]]
                else:
                    subset_df = df_bins

                if subset_df.empty:
                    continue

                subset_entries: List[Dict[str, float | int | str]] = []
                csv_entries: List[Dict[str, object]] = []

                for label, bin_df in subset_df.groupby("__wind_bin_label__"):
                    if bin_df.empty:
                        continue

                    label_value = "NULL" if label == sentinel_value else normalize_interval_value(bin_df.iloc[0][column])
                    metrics_bin = regression_metrics(bin_df)
                    class_metrics_bin = compute_classification_metrics(
                        bin_df,
                        range_min,
                        range_max,
                        truth_speed_col=truth_speed_col,
                    )

                    baseline_metrics = baseline_regression.get(subset_name, {})
                    baseline_macro_value = baseline_macro.get(subset_name, float("nan"))

                    rmse_delta = relative_increase(
                        metrics_bin.get("rmse_speed", float("nan")),
                        baseline_metrics.get("rmse_speed", float("nan")),
                    )
                    eaam_delta = relative_increase(
                        metrics_bin.get("eaam_dir", float("nan")),
                        baseline_metrics.get("eaam_dir", float("nan")),
                    )
                    macro_bin = class_metrics_bin.get("macro_f1", float("nan"))
                    macro_delta = (
                        macro_bin - baseline_macro_value
                        if np.isfinite(macro_bin) and np.isfinite(baseline_macro_value)
                        else float("nan")
                    )

                    subset_entries.append(
                        {
                            "wind_bin": label_value,
                            "samples": int(metrics_bin.get("samples", 0)),
                            "rmse_speed": metrics_bin.get("rmse_speed", float("nan")),
                            "mae_speed": metrics_bin.get("mae_speed", float("nan")),
                            "rmse_dir": metrics_bin.get("rmse_dir", float("nan")),
                            "eaam_dir": metrics_bin.get("eaam_dir", float("nan")),
                            "macro_f1": macro_bin,
                            "delta_rmse_pct": rmse_delta * 100 if np.isfinite(rmse_delta) else float("nan"),
                            "delta_eaam_pct": eaam_delta * 100 if np.isfinite(eaam_delta) else float("nan"),
                            "delta_macro": macro_delta,
                        }
                    )

                    csv_entries.append(
                        {
                            "wind_bin": label_value,
                            "rmse": metrics_bin.get("rmse_speed", float("nan")),
                            "mae_speed": metrics_bin.get("mae_speed", float("nan")),
                            "corr_speed": metrics_bin.get("corr_speed", float("nan")),
                            "r2_speed": metrics_bin.get("r2_speed", float("nan")),
                            "bias_speed": metrics_bin.get("bias_speed", float("nan")),
                            "si_speed": metrics_bin.get("si_speed", float("nan")),
                            "si_speed_max": metrics_bin.get("si_speed_max", float("nan")),
                            "eam_dir": metrics_bin.get("eam_dir", float("nan")),
                            "eaam_dir": metrics_bin.get("eaam_dir", float("nan")),
                            "rmse_dir": metrics_bin.get("rmse_dir", float("nan")),
                            "compcorr_dir": metrics_bin.get("compcorr_dir", float("nan")),
                        }
                    )

                if not subset_entries:
                    continue

                subset_entries.sort(key=lambda rec: rec["wind_bin"])
                csv_entries.sort(key=lambda rec: rec["wind_bin"])
                csv_records[subset_name] = csv_entries

                handle.write(f"### Wind bin distribution — {subset_title}\n\n")
                handle.write(
                    "| wind_bin | samples | rmse_speed | mae_speed | rmse_dir | eaam_dir | macro_f1 | Δrmse (%) | Δeaam (%) | Δmacro_f1 |\n"
                )
                handle.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
                for entry in subset_entries:
                    handle.write(
                        "| {wind_bin} | {samples} | {rmse:.6f} | {mae:.6f} | {rmse_dir:.6f} | {eaam:.6f} | {macro:.6f} | {rmse_pct:.2f} | {eaam_pct:.2f} | {macro_delta:.6f} |\n".format(
                            wind_bin=entry["wind_bin"],
                            samples=entry["samples"],
                            rmse=entry["rmse_speed"],
                            mae=entry["mae_speed"],
                            rmse_dir=entry["rmse_dir"],
                            eaam=entry["eaam_dir"],
                            macro=entry["macro_f1"],
                            rmse_pct=entry["delta_rmse_pct"],
                            eaam_pct=entry["delta_eaam_pct"],
                            macro_delta=entry["delta_macro"],
                        )
                    )
                handle.write("\n")

                if subset_name == "all":
                    degrade_lines: List[str] = []
                    all_baseline = baseline_regression.get("all", {})
                    baseline_samples = all_baseline.get("samples", metrics_full.get("samples", 0))
                    support_floor = max(int(min_support_ratio * max(baseline_samples, 0)), min_support_absolute)

                    for entry in subset_entries:
                        if entry["samples"] < support_floor:
                            continue

                        rmse_increase = entry["delta_rmse_pct"] / 100.0
                        eaam_increase = entry["delta_eaam_pct"] / 100.0
                        macro_delta_value = entry["delta_macro"]

                        # Compare deltas against the user-defined degradation thresholds.
                        if np.isfinite(rmse_increase) and rmse_increase > degradation_threshold:
                            degrade_lines.append(
                                f"Wind bin `{entry['wind_bin']}` records an RMSE {rmse_increase * 100:.1f}% above the global baseline."
                            )
                        if np.isfinite(eaam_increase) and eaam_increase > degradation_threshold:
                            degrade_lines.append(
                                f"Wind bin `{entry['wind_bin']}` shows an angular MAE {eaam_increase * 100:.1f}% above the overall reference."
                            )
                        if np.isfinite(macro_delta_value) and macro_delta_value < -macro_drop_threshold:
                            degrade_lines.append(
                                f"Wind bin `{entry['wind_bin']}` reduces macro-F1 by {abs(macro_delta_value):.3f} relative to the baseline, signalling classification drift."
                            )

                    if degrade_lines:
                        narrative = " ".join(degrade_lines)
                        handle.write(f"**Degradation alert:** {narrative}\n\n")
                    else:
                        handle.write(
                            f"No wind bin exceeded the {degradation_threshold * 100:.0f}% degradation threshold relative to the global RMSE or angular MAE baseline.\n\n"
                        )

            return csv_records

        # Stage 4 — wind-bin diagnostics quantify how errors and macro-F1 behave across engineered speed strata.
        if wind_bin_column:
            if wind_bin_column in df.columns:
                wind_bin_csv_map = render_wind_bin_section(wind_bin_column)
            else:
                print(
                    f"[compute_inference_metrics] Wind bin column '{wind_bin_column}' not found; skipping per-bin metrics.",
                    flush=True,
                )

        # Stage 4b — maintenance intervals and arbitrary groupings.
        if maintenance_columns:
            render_group_section(maintenance_columns, "Maintenance interval analysis", is_maintenance=True)
        if group_columns:
            render_group_section(group_columns, "Group breakdown", is_maintenance=False)

    metrics_csv = output_dir / "inference_metrics.csv"
    csv_row = {}
    for prefix, metrics in (("full", metrics_full), ("in_range", metrics_in_range), ("matched", metrics_match)):
        for key, value in metrics.items():
            label = f"{prefix}_{key}" if prefix else key
            csv_row[label] = value
    for key, value in classification_metrics.items():
        csv_row[key] = value
    # Stage 5 — export flat CSV tables for programmatic consumption.
    pd.DataFrame([csv_row]).to_csv(metrics_csv, index=False)

    if interval_records:
        intervals_csv = output_dir / "inference_metrics_by_interval.csv"
        interval_df = pd.DataFrame(interval_records)
        interval_df.to_csv(intervals_csv, index=False)

    # Persist per-subset wind-bin CSVs, mirroring the Markdown narrative.
    for subset_name, records in wind_bin_csv_map.items():
        if not records:
            continue

        csv_df = pd.DataFrame(records)
        csv_columns = [
            "wind_bin",
            "rmse",
            "mae_speed",
            "corr_speed",
            "r2_speed",
            "bias_speed",
            "si_speed",
            "si_speed_max",
            "eam_dir",
            "eaam_dir",
            "rmse_dir",
            "compcorr_dir",
        ]
        csv_df = csv_df[csv_columns]
        suffix = "" if subset_name == "all" else f"_{subset_name}"
        csv_path = output_dir / f"inference_metrics{suffix}_by_wind_bin.csv"
        csv_df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    main()
