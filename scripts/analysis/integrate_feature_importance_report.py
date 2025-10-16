#!/usr/bin/env python3
"""Integrate feature-importance analysis outputs into a Markdown report.

The utility consumes the artefacts emitted by
``scripts/analysis/feature_importance.py`` and produces a self-contained
Markdown section that blends narrative context, tabular highlights, and figure
embeds. It is designed to be idempotent: each invocation rewrites the target
file with a fresh section built from the supplied analysis directory.

Inputs
------
- ``--analysis-dir``: Path to a single analysis output directory (it must
  contain ``analysis_metadata.json`` and ``baseline_metrics.json``; CSVs and
  PNGs are optional but will be embedded when present).
- ``--report-file``: Destination Markdown file (default
  ``hf_eolus/ann_training_report.md``).
- ``--section-title``: Optional explicit heading; otherwise the script derives
  one from the model artefact and dataset names.
- ``--max-rows``: Maximum number of rows shown in each summary table (default
  10).

The report is written in English at a level appropriate for a mixed technical
and scientific audience, echoing the style requirements of the wider project
documentation.

Author: Juan Luis Herrera Cortijo <juan.luis.herrera.cortijo@gmail.com>
Created: 2025-10-16
Disclaimer: This script is provided as-is for research workflows and must be validated before any operational deployment.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Dict


def read_json(path: Path) -> dict:
    """Load and return the JSON payload stored at ``path``."""

    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    """Load a CSV into a list of dictionaries preserving header names."""

    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)
    return rows


def to_float(v: str, default: float = 0.0) -> float:
    """Best-effort conversion to float, falling back to ``default``."""

    try:
        return float(v)
    except Exception:
        return default


def _sanitize_cell(text: str) -> str:
    """Normalise cell content to keep Markdown tables well-formed."""

    if text is None:
        return ""
    s = str(text)
    # Escape pipe characters to avoid breaking Markdown tables
    s = s.replace("|", "\\|")
    # Avoid hard newlines inside cells
    s = s.replace("\n", " ")
    return s


def format_table(headers: List[str], rows: List[List[str]]) -> str:
    """Render headers/rows as a GitHub-flavoured Markdown table."""

    out = []
    headers = [_sanitize_cell(h) for h in headers]
    out.append("| " + " | ".join(headers) + " |")
    out.append("|" + "|".join([" --- " for _ in headers]) + "|")
    for r in rows:
        r = [_sanitize_cell(c) for c in r]
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out) + "\n"


def guess_labels(meta: dict) -> tuple[str, str]:
    """Derive human-readable labels for the model artefact and dataset."""

    model = Path(meta.get("model_artifact", "")).name or "model"
    data = Path(meta.get("data_path", "")).name or "dataset"
    return model, data


def build_section(analysis_dir: Path, title: str | None, max_rows: int) -> str:
    """Construct the Markdown section describing a feature-importance run.

    Args:
        analysis_dir: Directory containing the JSON/CSV/PNG artefacts produced by
            the analysis script.
        title: Optional heading to use; when ``None`` the title is generated
            from the artefact names.
        max_rows: Truncation threshold for summary tables.

    Returns:
        A Markdown string ready to be written to disk.

    Raises:
        FileNotFoundError: If mandatory JSON files are missing.
    """

    meta = read_json(analysis_dir / "analysis_metadata.json")
    baseline = read_json(analysis_dir / "baseline_metrics.json")

    if not title:
        model, data = guess_labels(meta)
        title = f"Feature Importance and Sensitivity — {model} on {data}"

    # Load optional CSVs
    groups_csv = analysis_dir / "permutation_importance_groups.csv"
    feats_csv = analysis_dir / "permutation_importance_features.csv"
    sens_csv = analysis_dir / "local_sensitivities.csv"
    weights_csv = analysis_dir / "weight_path_importance.csv"

    groups = read_csv_rows(groups_csv) if groups_csv.exists() else []
    feats = read_csv_rows(feats_csv) if feats_csv.exists() else []
    sens = read_csv_rows(sens_csv) if sens_csv.exists() else []
    weights = read_csv_rows(weights_csv) if weights_csv.exists() else []

    # Sort and select top rows
    groups_sorted = sorted(groups, key=lambda r: to_float(r.get("delta_speed_rmse", "0")), reverse=True)[:max_rows]
    sens_sorted = sorted(sens, key=lambda r: to_float(r.get("sensitivity_speed", "0")), reverse=True)[:max_rows]
    weights_sorted = sorted(weights, key=lambda r: to_float(r.get("weight_importance_speed", "0")), reverse=True)[:max_rows]

    # Build Markdown
    md: List[str] = []
    md.append(f"# {title}\n")
    md.append(
        "This report explains which inputs matter most for the model's predictions, using three complementary lenses that are simple to read: "
        "(i) permutation importance (how much the error worsens if an input is shuffled), (ii) local sensitivity (how much the output changes when an input is perturbed), and (iii) weight-path aggregation (a quick visual check based on the network's weights). "
        "All regression metrics are computed only where the observed wind speed falls within the physically supported range.\n"
    )
    # Quick reading guide
    md.append(
        "\n" \
        "Reading guide:\n\n" \
        "- Higher ΔRMSE means the input (or group) is more influential for speed predictions.\n" \
        "- Sensitivities reflect local responsiveness around the data; they are not global effects.\n" \
        "- Weight-path charts are qualitative: use them as a sanity check alongside the other two.\n"
    )

    # Baseline metrics table
    md.append("\n## Baseline Performance (In-Range)\n")
    md.append(
        "The baseline quantifies the model's error before any diagnostic perturbation. Speed RMSE measures magnitude errors (in m/s or the training units). Direction MAE is the mean absolute angular error in degrees. Only samples whose observed wind speed lies within the valid range contribute to these statistics.\n"
    )
    # Build baseline rows prioritising direction RMSE, include MAE if available
    speed_rmse = baseline.get('speed_rmse')
    dir_rmse = baseline.get('dir_rmse_deg')
    dir_mae = baseline.get('dir_mae_deg')
    rows = []
    rows.append(["Speed RMSE", f"{speed_rmse:.4f}" if isinstance(speed_rmse, (int, float)) else str(speed_rmse)])
    rows.append(["Direction RMSE (deg)", f"{dir_rmse:.2f}" if isinstance(dir_rmse, (int, float)) else str(dir_rmse)])
    if dir_mae is not None:
        rows.append(["Direction MAE (deg)", f"{dir_mae:.2f}" if isinstance(dir_mae, (int, float)) else str(dir_mae)])
    md.append(format_table(["Metric", "Value"], rows))
    # Add sample counts when available
    if 'total_count' in baseline and 'in_range_count' in baseline:
        total = int(baseline.get('total_count', 0) or 0)
        inrng = int(baseline.get('in_range_count', 0) or 0)
        frac = float(baseline.get('in_range_fraction', 0.0) or 0.0)
        md.append(format_table(["Samples", "In-Range", "In-Range %"], [[str(total), str(inrng), f"{frac*100:.1f}%"]]))
    # Range
    rmin = meta.get('range_min', None)
    rmax = meta.get('range_max', None)
    if rmin is not None and rmax is not None:
        md.append(f"Valid wind-speed range for regression: [{rmin}, {rmax}].\n")

    # Groups table if available
    if groups_sorted:
        md.append("\n## Permutation Importance — Groups\n")
        md.append(
            "Permutation importance reports how much the error increases when we randomly shuffle all the values of a group of inputs. "
            "A larger ΔRMSE means the group is more critical for accurate speed predictions. Grouping aggregates logically related inputs (e.g., both Bragg peaks, bearing sine/cosine).\n"
        )
    rows = []
    for r in groups_sorted:
        rows.append([
            r.get("group", ""),
            f"{to_float(r.get('delta_speed_rmse','0')):.4f}",
            f"{to_float(r.get('delta_dir_rmse_deg', r.get('delta_dir_mae_deg','0'))):.2f}",
            r.get("group_size", ""),
        ])
        md.append(format_table(["Group", "ΔRMSE (speed)", "ΔRMSE° (dir)", "Size"], rows))
        md.append("Interpretation: read from top to bottom; larger bars in the corresponding figure match larger ΔRMSE values here. Beware that highly correlated groups may share influence.\n")
        # Embed group figures when available
        if (analysis_dir / "perm_speed_groups.png").exists():
            md.append("\n### Figure — Grouped permutation importance (speed RMSE)\n")
            md.append("Higher bars indicate larger error increase after shuffling the entire group.\n")
            md.append("![Grouped permutation importance — speed](perm_speed_groups.png)\n")
        if (analysis_dir / "perm_dir_groups.png").exists():
            md.append("\n### Figure — Grouped permutation importance (direction RMSE)\n")
            md.append("Focuses on directional accuracy (degrees).\n")
            md.append("![Grouped permutation importance — direction](perm_dir_groups.png)\n")

    # Per-feature permutation table (optional)
    if feats:
        feats_sorted = sorted(feats, key=lambda r: to_float(r.get("delta_speed_rmse", "0")), reverse=True)[:max_rows]
        md.append("\n## Permutation Importance — Individual Features\n")
        md.append(
            "This table drills down into individual inputs. As with grouped results, higher ΔRMSE indicates greater influence. "
            "When inputs are strongly correlated, consider the grouped view as more robust and use this table for context.\n"
        )
        rows = []
        for r in feats_sorted:
            rows.append([
                r.get("feature", ""),
                f"{to_float(r.get('delta_speed_rmse','0')):.4f}",
                f"{to_float(r.get('delta_dir_rmse_deg', r.get('delta_dir_mae_deg','0'))):.2f}",
            ])
        md.append(format_table(["Feature", "ΔRMSE (speed)", "ΔRMSE° (dir)"], rows))
        # Embed feature figures when available
        if (analysis_dir / "perm_speed_features.png").exists():
            md.append("\n### Figure — Feature permutation importance (speed RMSE)\n")
            md.append("Relative influence at the single-feature level.\n")
            md.append("![Feature permutation importance — speed](perm_speed_features.png)\n")
        if (analysis_dir / "perm_dir_features.png").exists():
            md.append("\n### Figure — Feature permutation importance (direction RMSE)\n")
            md.append("Highlights inputs most critical for directional accuracy.\n")
            md.append("![Feature permutation importance — direction](perm_dir_features.png)\n")

    # Sensitivities table
    if sens_sorted:
        md.append("\n## Local Sensitivity (Gradients)\n")
        md.append(
            "Local sensitivity measures how much the output changes when an input is nudged, averaged across in-range samples. "
            "Values are reported in the normalised input space used by the model; larger values indicate greater responsiveness around the data.\n"
        )
        # Speed sensitivity table
        rows_speed = []
        for r in sens_sorted:
            rows_speed.append([r.get("feature", ""), f"{to_float(r.get('sensitivity_speed','0')):.3e}"])
        md.append(format_table(["Feature", "Median absolute gradient (speed)"], rows_speed))
        # Direction sensitivity table (use its own ranking)
        sens_sorted_dir = sorted(sens, key=lambda r: to_float(r.get("sensitivity_dir_vec", "0")), reverse=True)[:max_rows]
        rows_dir = []
        for r in sens_sorted_dir:
            rows_dir.append([r.get("feature", ""), f"{to_float(r.get('sensitivity_dir_vec','0')):.3e}"])
        md.append(format_table(["Feature", "Median gradient magnitude (direction)"], rows_dir))
        # Embed sensitivity figures when available
        if (analysis_dir / "sens_speed.png").exists():
            md.append("\n### Figure — Local sensitivity for speed\n")
            md.append("Bar height is the median absolute gradient; higher means stronger local effect.\n")
            md.append("![Local sensitivity — speed](sens_speed.png)\n")
        if (analysis_dir / "sens_dir.png").exists():
            md.append("\n### Figure — Local sensitivity for direction vector\n")
            md.append("Magnitude of the gradient combining direction cosine/sine outputs.\n")
            md.append("![Local sensitivity — direction](sens_dir.png)\n")
    # Weight-path aggregation (qualitative) — always last
    if (analysis_dir / "weights_speed.png").exists() or (analysis_dir / "weights_dir.png").exists():
        md.append("\n## Weight-Path Aggregation (Qualitative)\n")
        md.append(
            "A qualitative view that combines absolute weights along the network paths from inputs to each output head. "
            "Use these plots as a sanity check, not a ranking, and read them together with permutation and sensitivity results.\n"
        )
        if (analysis_dir / "weights_speed.png").exists():
            md.append("\n### Figure — Weight-path aggregation for speed head\n")
            md.append("![Weight-path — speed](weights_speed.png)\n")
        if (analysis_dir / "weights_dir.png").exists():
            md.append("\n### Figure — Weight-path aggregation for direction head\n")
            md.append("![Weight-path — direction](weights_dir.png)\n")

    # Limitations and caveats
    md.append(
        "\n## Limitations and Caveats\n"
    )
    md.append(
        "- Correlated inputs can split influence: grouped importance is often more reliable than per-feature in such cases.\n"
        "- Local sensitivities describe behaviour near the observed data; they may not capture global non-linearities.\n"
        "- Weight-path charts ignore activation dynamics and are intended as a qualitative cross-check.\n"
        "- All regression diagnostics apply only within the configured valid wind-speed range; outside it, the model is not calibrated for speed.\n"
    )

    # Provenance
    md.append("\n## Provenance\n")
    model, data = guess_labels(meta)
    topk = meta.get('top_k', None)
    feat_count = len(meta.get('feature_cols', []))
    md.append(
        format_table(
            ["Model artifact", "Dataset", "Features", "Top-K"],
            [[model, data, str(feat_count), str(topk) if topk is not None else "-"]],
        )
    )

    return "\n".join(md) + "\n"


def write_report(report_file: Path, content: str) -> None:
    """Write ``content`` to ``report_file``, creating parent directories."""

    report_file.parent.mkdir(parents=True, exist_ok=True)
    with report_file.open("w", encoding="utf-8") as fh:
        fh.write(content)


def main():
    """Parse CLI arguments, build the section, and write it to disk."""

    ap = argparse.ArgumentParser(description="Integrate feature-importance outputs into a Markdown report")
    ap.add_argument("--analysis-dir", required=True, help="Path to a feature_importance_* directory")
    ap.add_argument("--report-file", default="hf_eolus/ann_training_report.md")
    ap.add_argument("--section-title")
    ap.add_argument("--max-rows", type=int, default=10)
    args = ap.parse_args()

    analysis_dir = Path(args.analysis_dir)
    if not analysis_dir.exists():
        raise FileNotFoundError(f"Analysis directory not found: {analysis_dir}")
    section = build_section(analysis_dir, args.section_title, args.max_rows)
    write_report(Path(args.report_file), section)
    print(f"Wrote analysis report to {args.report_file}")


if __name__ == "__main__":
    main()
