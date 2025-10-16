#!/usr/bin/env python3
"""
Feature importance and sensitivity analysis for the HF wind inversion MLP.

This script is non-intrusive: it loads an already-trained model artifact and a
validation/test dataset, reconstructs the exact feature pipeline used at
inference (engineering + normalization), and computes:

- Permutation importance (per-feature and logical groups)
- Local sensitivities via input–output gradients (Jacobian-based)
- Weight-path importance (Olden/Garson-style) as a quick sanity-check

Outputs are stored under `artifacts_root/analysis/<timestamp>/` as CSVs and PNGs.

Notes
-----
- All regression metrics and attributions are computed “in-range” by default,
  honoring the physical gating used in training: samples whose true wind speed
  lies outside [range_min, range_max] are excluded from regression metrics.
- The feature pipeline mirrors scripts/inference/inference.py, but is copied
  locally to avoid import packaging frictions.

Author: Juan Luis Herrera Cortijo <juan.luis.herrera.cortijo@gmail.com>
Created: 2025-10-16
Disclaimer: This script is provided as-is for research workflows and must be validated before any operational deployment.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import math
import os
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch


# -----------------------------------------------------------------------------
# Minimal logging setup
# -----------------------------------------------------------------------------
logger = logging.getLogger("feature_importance")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# -----------------------------------------------------------------------------
# Local copy of feature-spec and engineering utilities (from inference.py)
# -----------------------------------------------------------------------------

from types import SimpleNamespace


def build_feature_spec(model_cfg: dict) -> SimpleNamespace:
    """Construct a compact feature specification from a model configuration.

    The helper mirrors the layout produced during training, extracting the
    station roster, schema template, aggregation strategy, and target column
    names so that the analysis step can rebuild the inference feature pipeline
    without importing the original training package.

    Args:
        model_cfg: Dictionary recovered from the model artifact describing the
            architecture, available stations, and feature schema patterns.

    Returns:
        A `SimpleNamespace` carrying the station list, per-station schema
        mapping, aggregation metadata, and target column names required to
        standardise inputs before feature engineering.

    Raises:
        KeyError: If the configuration is missing the station list or schema
            entries that are mandatory to locate raw feature columns.
    """

    stations = model_cfg.get("stations", [])
    if not stations:
        raise KeyError("Model configuration must include a 'stations' list")
    model_section = model_cfg.get("model", {})
    schema_section = model_cfg.get("schema", {})
    if "stations" in schema_section:
        schema_section = schema_section["stations"]
    return SimpleNamespace(
        station_names=stations,
        station_schema=schema_section,
        agg=model_section.get("agg_stat", model_cfg.get("agg_stat", "mean")),
        use_mad=model_section.get("use_mad", False),
        use_velocity_median=model_section.get("use_velocity_median", False),
        target_speed_col=model_section.get(
            "target_speed_col", model_cfg.get("target_speed_col", "wind_speed")
        ),
        target_dir_col=model_section.get(
            "target_dir_col", model_cfg.get("target_dir_col", "wind_direction")
        ),
        id_col=model_section.get("id_col", model_cfg.get("id_col", "location_id")),
    )


def _format_pattern(pattern: str, agg: str, peak: str) -> str:
    """Resolve a schema pattern by progressively relaxing placeholder support.

    Args:
        pattern: Template string that may reference `{agg}` and/or `{peak}`.
        agg: Aggregation statistic used when the template exposes `{agg}`.
        peak: Bragg-peak index used when the template exposes `{peak}`.

    Returns:
        The resolved column name matching the requested aggregation and peak.
    """

    try:
        return pattern.format(agg=agg, peak=peak)
    except (KeyError, IndexError):
        try:
            return pattern.format(peak=peak)
        except (KeyError, IndexError):
            return pattern


def ensure_standard_columns(df: pd.DataFrame, feature_spec: SimpleNamespace) -> pd.DataFrame:
    """Clone the input frame and expose canonical column names.

    The training pipelines expect station-specific columns to follow naming
    conventions (e.g., `<station>_pwr_0`). Real datasets carry these values
    under a variety of schema-dependent names, so this routine copies the
    source fields into a deterministic set of aliases while leaving the
    original dataset untouched.

    Args:
        df: DataFrame containing the raw aggregated HF-Radar measurements and
            optional wind truth columns.
        feature_spec: Namespace produced by :func:`build_feature_spec` holding
            the schema templates and station list.

    Returns:
        A dataframe mirroring the input but with every required feature copied
        into the canonical names expected downstream.

    Raises:
        KeyError: If any of the mandatory inputs cannot be located in the
            source dataframe.
    """

    working = df.copy()
    columns = set(working.columns)

    def copy_column(source: str, dest: str, *, required: bool = True) -> bool:
        """Copy `source` into `dest`, optionally tolerating missing origins."""

        if source not in columns:
            if required:
                raise KeyError(f"Required column '{source}' not found in dataset")
            return False
        if dest not in working.columns:
            working[dest] = working[source]
        return True

    speed_candidates = [feature_spec.target_speed_col, "wind_speed"]
    for cand in speed_candidates:
        if cand and copy_column(cand, "wind_speed", required=False):
            break

    dir_candidates = [feature_spec.target_dir_col, "wind_dir", "wind_direction"]
    for cand in dir_candidates:
        if cand and copy_column(cand, "wind_dir", required=False):
            break

    for station in feature_spec.station_names:
        info = feature_spec.station_schema.get(station)
        if info is None:
            raise KeyError(f"Schema mapping missing for station '{station}'")

        power_pattern = info.get("power_pattern")
        if not power_pattern:
            raise KeyError(f"Schema for station '{station}' lacks 'power_pattern'")
        for peak in ("0", "1"):
            raw_col = _format_pattern(power_pattern, feature_spec.agg, peak)
            copy_column(raw_col, f"{station}_pwr_{peak}")

        # Optional features follow the same pattern resolution, but are only
        # enforced when the model configuration indicates they were used during
        # training.

        if feature_spec.use_mad:
            mad_pattern = info.get("mad_pattern")
            if not mad_pattern:
                raise KeyError(
                    f"Schema for station '{station}' lacks 'mad_pattern' but model requires MAD features"
                )
            for peak in ("0", "1"):
                raw_col = _format_pattern(mad_pattern, feature_spec.agg, peak)
                copy_column(raw_col, f"{station}_pwr_mad_{peak}")

        if feature_spec.use_velocity_median:
            vel_pattern = info.get("velocity_median_pattern") if isinstance(info, dict) else None
            for peak in ("0", "1"):
                candidates = []
                if vel_pattern:
                    candidates.append(_format_pattern(vel_pattern, feature_spec.agg, peak))
                candidates.extend(
                    [
                        f"{station}__velo_median_{peak}",
                        f"{station}_velo_median_{peak}",
                        f"{station}_stats_velo_median_{peak}",
                    ]
                )
                for cand in candidates:
                    if cand in columns:
                        copy_column(cand, f"{station}_velo_median_{peak}")
                        break
                else:
                    raise KeyError(
                        f"Median radial velocity column not found for station '{station}' (peak {peak})"
                    )

        bearing_pattern = info.get("bearing")
        if not bearing_pattern:
            raise KeyError(f"Schema for station '{station}' lacks 'bearing'")
        bearing_col = _format_pattern(bearing_pattern, feature_spec.agg, "")
        copy_column(bearing_col, f"{station}_bearing_source")

        distance_pattern = info.get("distance")
        if not distance_pattern:
            raise KeyError(f"Schema for station '{station}' lacks 'distance'")
        dist_col = _format_pattern(distance_pattern, feature_spec.agg, "")
        copy_column(dist_col, f"{station}_dist_source")

    return working


def engineer_features(df: pd.DataFrame, feature_spec: SimpleNamespace) -> tuple[pd.DataFrame, list[str]]:
    """Derive engineered inputs from the standardised dataset.

    The analysis replicates the feature generation used for model inference:
    MAD proxies, bearing trigonometric components, station distances, and the
    optional velocity medians. Returning the feature list alongside the frame
    guarantees deterministic ordering when later converting to tensors.

    Args:
        df: DataFrame with canonical column names produced by
            :func:`ensure_standard_columns`.
        feature_spec: Namespace with station metadata and feature toggles.

    Returns:
        A tuple of (engineered dataframe, ordered feature column names).

    Raises:
        KeyError: If a derived feature cannot be produced because its source
            column is missing.
    """

    stations = feature_spec.station_names
    use_mad = feature_spec.use_mad
    use_velocity_median = feature_spec.use_velocity_median

    if use_mad:
        mad_cols = [f"{station}_pwr_mad_{peak}" for station in stations for peak in ("0", "1")]
        stacked = pd.concat([df[col].replace(-np.inf, np.nan) for col in mad_cols], axis=1)
        df["pwr_mad"] = stacked.max(axis=1, skipna=True)
        fallback_max = df[[f"{station}_pwr_{peak}" for station in stations for peak in ("0", "1")]].max(axis=1)
        df["pwr_mad"] = df["pwr_mad"].fillna(fallback_max)

    for station in stations:
        bearing_col = f"{station}_bearing_source"
        rad = np.deg2rad(df[bearing_col])
        df[f"cos_{station}_bearing"] = np.cos(rad)
        df[f"sin_{station}_bearing"] = np.sin(rad)

    for station in stations:
        dist_col = f"{station}_dist_source"
        df.rename(columns={dist_col: f"{station}_dist"}, inplace=True)

    # Build the deterministic feature ordering expected by the trained
    # checkpoint. The order mirrors the concatenation performed during
    # training.
    feature_cols = []
    for station in stations:
        for peak in ("0", "1"):
            feature_cols.append(f"{station}_pwr_{peak}")
    if use_velocity_median:
        for station in stations:
            for peak in ("0", "1"):
                feature_cols.append(f"{station}_velo_median_{peak}")
    if use_mad:
        feature_cols.append("pwr_mad")
    for station in stations:
        feature_cols.append(f"{station}_dist")
    for station in stations:
        feature_cols.extend([f"cos_{station}_bearing", f"sin_{station}_bearing"])

    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing engineered feature columns: {missing}")

    return df, feature_cols


def normalize_features(feature_df: pd.DataFrame, feature_cols, norm_params):
    """Apply persisted global and conditional scalers to feature columns.

    Args:
        feature_df: DataFrame containing engineered features.
        feature_cols: Ordered list of feature names to keep after scaling.
        norm_params: Dictionary with global means/standard deviations and
            optional conditional normalisation metadata as saved inside the
            checkpoint.

    Returns:
        A dataframe restricted to `feature_cols` whose numeric entries follow
        the same normalisation applied during training.

    Raises:
        KeyError: When any feature lacks associated normalisation parameters.
    """

    working_df = feature_df.copy()

    numeric_cols = [c for c in feature_cols if not (c.startswith("cos_") or c.startswith("sin_"))]
    centers = norm_params.get("centers") or norm_params.get("means", {})
    scales = norm_params.get("scales") or norm_params.get("stds", {})
    conditional_params = norm_params.get("conditional", {}) or {}

    # Deferred import (package is under scripts/training)
    import sys as _sys, os as _os
    # Inject the training package into sys.path to reuse the original model and
    # normalisation helpers without requiring an installed package.
    _sys.path.insert(0, _os.path.join(Path(__file__).resolve().parent.parent, "training"))
    from train_lib.normalization import apply_conditional_normalization_from_params as _apply_cond

    conditional_features = set()
    for station_spec in conditional_params.values():
        for feature in station_spec.get("features", {}):
            conditional_features.add(feature)

    missing = [col for col in numeric_cols if col not in centers or col not in scales]
    if missing:
        raise KeyError(
            f"Normalization parameters missing for feature(s) {missing}. Available centers: {list(centers.keys())}"
        )

    # Apply the simple global (mean, std) scaling first for features that are
    # not part of any maintenance-interval specific normalisation block.
    global_cols = [col for col in numeric_cols if col not in conditional_features]
    for col in global_cols:
        working_df[col] = (working_df[col] - centers[col]) / scales[col]

    if conditional_features and conditional_params:
        # The helper mutates the dataframe in place, matching the behaviour of
        # the training inference code where scaling parameters depend on
        # maintenance interval labels.
        _apply_cond(working_df, conditional_params, stage="analysis")
    else:
        for col in conditional_features:
            working_df[col] = (working_df[col] - centers[col]) / scales[col]

    return working_df[feature_cols]


# -----------------------------------------------------------------------------
# Model loading helpers
# -----------------------------------------------------------------------------


@dataclass
class LoadedModel:
    """Container bundling the materialised model and its preprocessing assets."""

    model: torch.nn.Module
    feature_spec: SimpleNamespace
    feature_cols: List[str]
    norm_params: dict
    script_args: dict


def _load_artifact_paths(artifact_path: Path) -> Tuple[Path, Optional[Path]]:
    """Resolve the concrete locations of model weights and auxiliary metadata.

    Args:
        artifact_path: Filesystem path pointing to a directory, raw
            `model.pth`, or a compressed SageMaker bundle.

    Returns:
        Tuple containing the path to `model.pth` and, when present, the
        companion `script_args.json` with training arguments.
    """
    if artifact_path.is_dir():
        m = artifact_path / "model.pth"
        s = artifact_path / "script_args.json"
        return m, (s if s.exists() else None)
    if artifact_path.suffixes[-2:] == [".tar", ".gz"] or artifact_path.suffix == ".tgz" or artifact_path.suffix == ".gz":
        tmpdir = Path(tempfile.mkdtemp(prefix="hf-model-"))
        with tarfile.open(artifact_path, "r:gz") as tar:
            # SageMaker bundles contain model.pth and script_args.json at the
            # archive root; expand them into a temporary directory so the rest
            # of the loader can operate transparently.
            tar.extractall(path=tmpdir)
        m = tmpdir / "model.pth"
        s = tmpdir / "script_args.json"
        return m, (s if s.exists() else None)
    return artifact_path, None


def load_model_and_spec(artifact: str) -> LoadedModel:
    """Instantiate the MLP together with its feature pipeline configuration.

    Args:
        artifact: Location of the trained model (directory, checkpoint file,
            or compressed archive).

    Returns:
        A :class:`LoadedModel` bundling the PyTorch module, the feature
        specification, the ordered feature list, and normalisation parameters.

    Raises:
        FileNotFoundError: When required assets cannot be located inside the
            artifact.
        KeyError: When the model configuration lacks mandatory entries to
            rebuild the feature schema.
    """
    from pathlib import Path as _Path
    model_path, script_args_path = _load_artifact_paths(_Path(artifact).resolve())
    if not model_path.exists():
        raise FileNotFoundError(f"model.pth not found at {model_path}")

    state = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        norm_params = state.get("normalization_params")
        saved_args = state.get("args")
        payload_cfg = state.get("model_config_payload")
        model_state = state["model_state_dict"]
    else:
        model_state = state
        norm_params = None
        saved_args = None
        payload_cfg = None

    if norm_params is None:
        raise FileNotFoundError("Normalization parameters not found in model checkpoint")

    if script_args_path and script_args_path.exists():
        with open(script_args_path, "r", encoding="utf-8") as fh:
            script_args = json.load(fh)
    else:
        if not saved_args:
            raise FileNotFoundError("Saved script arguments not found in checkpoint or sidecar file")
        script_args = saved_args

    model_cfg = None
    model_config_rel = script_args.get("model_config")
    if model_config_rel:
        # Try relative to repo root
        cfg_path = Path(model_config_rel)
        if not cfg_path.exists():
            # Try under scripts/ to mirror container layout
            alt = Path("/opt/ml/code") / model_config_rel
            if alt.exists():
                cfg_path = alt
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as fh:
                model_cfg = json.load(fh)
    if model_cfg is None:
        if payload_cfg:
            model_cfg = json.loads(payload_cfg) if isinstance(payload_cfg, str) else payload_cfg
        else:
            raise FileNotFoundError("Model configuration not found in artifact or payload")

    feature_spec = build_feature_spec(model_cfg)

    # Construct the engineered feature list deterministically from the spec,
    # without requiring source columns to exist (the real dataset will be used
    # later to compute features and validate presence).
    feature_cols: List[str] = []
    for s in feature_spec.station_names:
        for p in ("0", "1"):
            feature_cols.append(f"{s}_pwr_{p}")
    if feature_spec.use_velocity_median:
        for s in feature_spec.station_names:
            for p in ("0", "1"):
                feature_cols.append(f"{s}_velo_median_{p}")
    if feature_spec.use_mad:
        feature_cols.append("pwr_mad")
    for s in feature_spec.station_names:
        feature_cols.append(f"{s}_dist")
    for s in feature_spec.station_names:
        feature_cols.extend([f"cos_{s}_bearing", f"sin_{s}_bearing"])

    # Instantiate the model
    # Ensure 'scripts/training' is on sys.path so we can import train_lib
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.join(Path(__file__).resolve().parent.parent, "training"))
    from train_lib.model import MLP as _MLP

    hidden_layers = int(script_args.get("hidden_layers", 2))
    hidden_units = int(script_args.get("hidden_units", 128))
    dropout = float(script_args.get("dropout", 0.0))
    class_labels = script_args.get("range_class_labels", ['below', 'in', 'above'])
    if isinstance(class_labels, str):
        class_labels = [p.strip() for p in class_labels.split(',') if p.strip()]
    num_classes = len(class_labels)
    model = _MLP(len(feature_cols), hidden_layers, hidden_units, drop_rate=dropout, num_classes=num_classes)
    model.load_state_dict(model_state)
    model.eval()

    return LoadedModel(model=model, feature_spec=feature_spec, feature_cols=feature_cols, norm_params=norm_params, script_args=script_args)


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------


def load_dataset(path: str) -> pd.DataFrame:
    """Load a dataset from a Parquet directory, Parquet file, or CSV.

    Args:
        path: Filesystem path to the dataset root or file.

    Returns:
        DataFrame containing the full dataset in memory.

    Raises:
        RuntimeError: When parquet assets cannot be read.
        ValueError: When the data format is not supported.
    """

    p = Path(path)
    if p.is_dir():
        # Try to read as a Parquet dataset directory
        try:
            import pyarrow.dataset as ds
            # Loading through pyarrow.dataset preserves schema partitions when
            # present, closely matching the training data access path.
            table = ds.dataset(str(p), format="parquet").to_table()
            return table.to_pandas()
        except Exception as exc:
            raise RuntimeError(f"Failed to read Parquet dataset at {path}: {exc}")
    # Single file
    if p.suffix.lower() in {".parquet", ".pq"}:
        try:
            return pd.read_parquet(p)
        except Exception as exc:
            raise RuntimeError(f"Failed to read Parquet file {path}: {exc}")
    if p.suffix.lower() in {".csv", ".gz", ".bz2"}:
        return pd.read_csv(p)
    raise ValueError(f"Unsupported data path: {path}. Provide a Parquet dataset/Parquet file or CSV.")


# -----------------------------------------------------------------------------
# Prediction and metrics
# -----------------------------------------------------------------------------


def run_model(model: torch.nn.Module, features_df: pd.DataFrame, feature_cols: List[str]):
    """Execute the regression/classification heads and return decoded outputs.

    Returns the speed scalar, the direction angle in degrees, normalised
    cosine/sine components, and class probabilities, mirroring the metrics
    required downstream.
    
    Args:
        model: Trained PyTorch MLP.
        features_df: DataFrame containing the features to evaluate.
        feature_cols: Ordered feature list used to slice the dataframe.

    Returns:
        Tuple `(speed, angle_deg, cos_unit, sin_unit, probs)` where each entry
        is a NumPy array aligned with the input rows.
    """

    with torch.no_grad():
        X = torch.tensor(features_df[feature_cols].values, dtype=torch.float32)
        regression, logits = model(X)
        regression = regression.numpy()
        probs = torch.softmax(logits, dim=1).numpy()
    speed = regression[:, 0]
    cos_vals = regression[:, 1]
    sin_vals = regression[:, 2]
    norm = np.sqrt(cos_vals ** 2 + sin_vals ** 2)
    norm = np.where(norm < 1e-6, 1e-6, norm)
    cos_unit = cos_vals / norm
    sin_unit = sin_vals / norm
    # The network predicts cosine/sine heads; convert them back into
    # compass-like degrees for downstream circular metrics.
    angle_rad = np.arctan2(sin_unit, cos_unit)
    angle_deg = np.degrees(angle_rad) % 360.0
    return speed, angle_deg, cos_unit, sin_unit, probs


def circular_diff_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute absolute angular differences on the circle (degrees).

    Args:
        a: First set of angles in degrees.
        b: Second set of angles in degrees.

    Returns:
        Absolute angular difference confined to the range [0, 180].
    """

    d = (a - b + 180.0) % 360.0 - 180.0
    return np.abs(d)


def compute_metrics(y_true_speed, y_pred_speed, y_true_dir_deg, y_pred_dir_deg, mask):
    """Return RMSE/MAE diagnostics restricted to the in-range mask.

    Args:
        y_true_speed: Ground-truth wind speed values.
        y_pred_speed: Model-predicted wind speed values.
        y_true_dir_deg: Ground-truth wind directions in degrees.
        y_pred_dir_deg: Predicted wind directions in degrees.
        mask: Boolean array selecting the physically valid range.

    Returns:
        Dictionary with speed RMSE and direction MAE/RMSE (degrees). NaNs are
        returned when no samples satisfy the mask.
    """

    idx = mask
    if idx.sum() == 0:
        return {"speed_rmse": np.nan, "dir_mae_deg": np.nan, "dir_rmse_deg": np.nan}
    se = (y_pred_speed[idx] - y_true_speed[idx]) ** 2
    rmse = float(np.sqrt(se.mean()))
    # Circular difference automatically wraps values to [-180, 180], so the
    # absolute can be interpreted as a directional error in degrees.
    ang_err = circular_diff_deg(y_pred_dir_deg[idx], y_true_dir_deg[idx])
    mae_ang = float(np.mean(ang_err))
    rmse_ang = float(np.sqrt(np.mean(ang_err ** 2)))
    return {"speed_rmse": rmse, "dir_mae_deg": mae_ang, "dir_rmse_deg": rmse_ang}


# -----------------------------------------------------------------------------
# Permutation importance
# -----------------------------------------------------------------------------


def build_groups(feature_cols: List[str], spec: SimpleNamespace) -> Dict[str, List[str]]:
    """Assemble station-level permutation groups from individual features.

    Args:
        feature_cols: Ordered list of engineered feature names.
        spec: Feature specification exposing the station roster.

    Returns:
        Dictionary mapping group labels to the columns they contain.
    """

    groups: Dict[str, List[str]] = {}
    # Power by station
    for s in spec.station_names:
        # Treat station-level contributions together so that shuffling preserves
        # relationships between peaks, distances, and bearings.
        cols = [c for c in feature_cols if c.startswith(f"{s}_pwr_")]
        if cols:
            groups[f"{s}__pwr"] = cols
        # Median velocity if present
        vel_cols = [c for c in feature_cols if c.startswith(f"{s}_velo_median_")]
        if vel_cols:
            groups[f"{s}__velo_median"] = vel_cols
        # Distance
        d = f"{s}_dist"
        if d in feature_cols:
            groups[f"{s}__dist"] = [d]
        # Bearing cos/sin
        bcols = [f"cos_{s}_bearing", f"sin_{s}_bearing"]
        bcols = [c for c in bcols if c in feature_cols]
        if bcols:
            groups[f"{s}__bearing"] = bcols
    # MAD proxy
    if "pwr_mad" in feature_cols:
        groups["pwr_mad"] = ["pwr_mad"]
    return groups


def permutation_importance(
    model: torch.nn.Module,
    features_norm: pd.DataFrame,
    feature_cols: List[str],
    y_true_speed: np.ndarray,
    y_true_dir_deg: np.ndarray,
    in_range_mask: np.ndarray,
    *,
    rng: np.random.Generator,
    group_map: Optional[Dict[str, List[str]]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute permutation deltas for individual features and logical groups.

    Args:
        model: Trained MLP model to evaluate.
        features_norm: Normalised feature dataframe.
        feature_cols: Ordered feature names.
        y_true_speed: Ground-truth speed targets.
        y_true_dir_deg: Ground-truth direction targets in degrees.
        in_range_mask: Boolean mask restricting metrics to the valid band.
        rng: Numpy random generator to drive feature shuffles.
        group_map: Optional mapping of group labels to feature subsets.

    Returns:
        Tuple with (per-feature DataFrame, per-group DataFrame). Either entry
        may be empty when the corresponding permutation mode is disabled.
    """

    # Baseline predictions guide both the metric deltas and the report.
    base_speed, base_dir_deg, *_ = run_model(model, features_norm, feature_cols)
    base_metrics = compute_metrics(y_true_speed, base_speed, y_true_dir_deg, base_dir_deg, in_range_mask)

    def eval_with_permutation(cols: List[str]) -> Dict[str, float]:
        """Shuffle the selected columns and measure the degradation."""

        tmp = features_norm.copy()
        idx = rng.permutation(len(tmp))
        for c in cols:
            tmp[c] = tmp[c].values[idx]
        ps, pdg, *_ = run_model(model, tmp, feature_cols)
        m = compute_metrics(y_true_speed, ps, y_true_dir_deg, pdg, in_range_mask)
        return {
            "delta_speed_rmse": float(m["speed_rmse"] - base_metrics["speed_rmse"]),
            "delta_dir_rmse_deg": float(m["dir_rmse_deg"] - base_metrics["dir_rmse_deg"]),
            "delta_dir_mae_deg": float(m["dir_mae_deg"] - base_metrics["dir_mae_deg"]),
            "baseline_speed_rmse": base_metrics["speed_rmse"],
            "baseline_dir_mae_deg": base_metrics["dir_mae_deg"],
            "baseline_dir_rmse_deg": base_metrics["dir_rmse_deg"],
        }

    # Per-feature
    rows_feat = []
    for c in feature_cols:
        res = eval_with_permutation([c])
        res.update({"feature": c})
        rows_feat.append(res)
    df_feat = pd.DataFrame(rows_feat).sort_values("delta_speed_rmse", ascending=False)

    # Grouped
    rows_group = []
    if group_map:
        for name, cols in group_map.items():
            res = eval_with_permutation(cols)
            res.update({"group": name, "group_size": len(cols)})
            rows_group.append(res)
    df_group = pd.DataFrame(rows_group).sort_values("delta_speed_rmse", ascending=False) if rows_group else pd.DataFrame()

    return df_feat, df_group


# -----------------------------------------------------------------------------
# Local sensitivities (Jacobian-based)
# -----------------------------------------------------------------------------


def local_sensitivities(
    model: torch.nn.Module,
    features_norm: pd.DataFrame,
    feature_cols: List[str],
    in_range_mask: np.ndarray,
    *,
    max_samples: int = 512,
) -> pd.DataFrame:
    """Estimate median absolute Jacobian magnitudes for speed and direction.

    Args:
        model: Trained MLP loaded from the artifact.
        features_norm: Normalised feature dataframe.
        feature_cols: Ordered feature names used to index the dataframe.
        in_range_mask: Boolean mask marking valid samples.
        max_samples: Optional cap on the number of samples evaluated via
            autograd.

    Returns:
        DataFrame with one row per feature and the median absolute gradients
        for the speed and direction heads.
    """

    idxs = np.flatnonzero(in_range_mask)
    if len(idxs) == 0:
        raise ValueError("No in-range samples available for sensitivity analysis")
    # Cap the number of samples to control the cost of autograd-based
    # sensitivities on large validation sets.
    sel = idxs[: max_samples] if len(idxs) > max_samples else idxs
    X = torch.tensor(features_norm.iloc[sel][feature_cols].values, dtype=torch.float32)
    X.requires_grad_(True)

    # Speed gradient: mean absolute per-feature gradient across samples
    speed = model(X)[0][:, 0]
    grads_speed = []
    for i in range(len(sel)):
        # Compute gradients sample by sample to mirror the approach used in the
        # training diagnostics; retaining the graph allows reuse across heads.
        model.zero_grad(set_to_none=True)
        grad = torch.autograd.grad(speed[i], X, retain_graph=True, create_graph=False, allow_unused=False)[0][i]
        grads_speed.append(grad.detach().abs().cpu().numpy())
    speed_sens = np.median(np.stack(grads_speed, axis=0), axis=0)

    # Direction gradient: combine cos/sin gradients into a magnitude per feature
    dir_out = model(X)[0][:, 1:3]
    grads_dir = []
    for i in range(len(sel)):
        model.zero_grad(set_to_none=True)
        gx = torch.autograd.grad(dir_out[i, 0], X, retain_graph=True, create_graph=False)[0][i]
        gy = torch.autograd.grad(dir_out[i, 1], X, retain_graph=True, create_graph=False)[0][i]
        # Combine cosine and sine gradients into a single magnitude so that the
        # direction sensitivity remains invariant to rotation in the unit circle.
        gmag = torch.sqrt(gx.pow(2) + gy.pow(2))
        grads_dir.append(gmag.detach().cpu().numpy())
    dir_sens = np.median(np.stack(grads_dir, axis=0), axis=0)

    return pd.DataFrame({
        "feature": feature_cols,
        "sensitivity_speed": speed_sens,
        "sensitivity_dir_vec": dir_sens,
    }).sort_values("sensitivity_speed", ascending=False)


# -----------------------------------------------------------------------------
# Olden/Garson-style weight path importance
# -----------------------------------------------------------------------------


def weight_path_importance(model: torch.nn.Module, feature_cols: List[str]) -> pd.DataFrame:
    """Propagate absolute weights along backbone paths as a quick heuristic.

    Args:
        model: Trained MLP containing the backbone and two heads.
        feature_cols: Ordered feature names to label the resulting scores.

    Returns:
        DataFrame with heuristic importances for the speed and direction heads.
    """

    import torch.nn as nn

    # Extract backbone linear layers in order
    linear_layers: List[nn.Linear] = []
    if hasattr(model, "backbone"):
        for m in model.backbone:
            if isinstance(m, nn.Linear):
                linear_layers.append(m)

    def combine_path(head: nn.Linear) -> np.ndarray:
        A = head.weight.detach().abs().cpu().numpy()  # shape: (out_dim, hidden)
        for lin in reversed(linear_layers):
            W = lin.weight.detach().abs().cpu().numpy()  # (out, in)
            # Use matrix multiplication to back-propagate the absolute weight
            # contribution through each hidden layer until reaching the input.
            A = A @ W
        # Now A shape: (out_dim, input_dim)
        if A.shape[0] > 1:
            # Aggregate multi-output head by L2 over outputs
            A = np.sqrt((A ** 2).sum(axis=0, keepdims=True))
        return A.reshape(-1)

    speed_vec = combine_path(model.speed_head)
    dir_vec = combine_path(model.direction_head)

    return pd.DataFrame({
        "feature": feature_cols,
        "weight_importance_speed": speed_vec,
        "weight_importance_dir_vec": dir_vec,
    }).sort_values("weight_importance_speed", ascending=False)


# -----------------------------------------------------------------------------
# Plotting helpers (optional)
# -----------------------------------------------------------------------------


def plot_bars(df: pd.DataFrame, value_col: str, label_col: str, title: str, outfile: Path, top_k: int = 20):
    """Render a horizontal bar plot, gracefully degrading when matplotlib fails.

    Args:
        df: DataFrame with the data to plot.
        value_col: Column containing the height of each bar.
        label_col: Column providing the labels for the Y axis.
        title: Plot title for readability when inspecting artefacts.
        outfile: Destination path for the PNG output.
        top_k: Maximum number of entries to display.
    """

    try:
        import matplotlib.pyplot as plt
    except Exception:
        logger.warning("matplotlib not available; skipping plot %s", outfile.name)
        return
    # Restrict the plot to the requested top-K entries for readability.
    top = df.sort_values(value_col, ascending=False).head(top_k)
    plt.figure(figsize=(10, max(4, 0.3 * len(top))))
    plt.barh(top[label_col][::-1], top[value_col][::-1])
    plt.xlabel(value_col)
    plt.title(title)
    plt.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=150)
    plt.close()


# -----------------------------------------------------------------------------
# Main CLI
# -----------------------------------------------------------------------------


def main():
    """CLI entry point that orchestrates model loading, analysis, and reporting."""

    ap = argparse.ArgumentParser(description="Feature importance and sensitivity analysis for HF wind MLP")
    ap.add_argument("--model-artifact", required=True, help="Path to model.pth or model.tar.gz (or extracted dir)")
    ap.add_argument("--data-path", required=True, help="Path to validation/test data (Parquet dataset dir, Parquet file, or CSV)")
    ap.add_argument("--output-dir", default="artifacts_root/analysis", help="Base output directory")
    ap.add_argument("--grouping", choices=["grouped", "feature", "both"], default="grouped", help="Compute permutation importance at group-level, per-feature, or both")
    ap.add_argument("--top-k", type=int, default=20, help="Top-K bars to plot")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-samples-sensitivity", type=int, default=512, help="Max in-range samples for Jacobian sensitivities")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # Load model and spec
    loaded = load_model_and_spec(args.model_artifact)
    model, spec, feature_cols, norm_params, sargs = (
        loaded.model,
        loaded.feature_spec,
        loaded.feature_cols,
        loaded.norm_params,
        loaded.script_args,
    )

    # Load data and build features
    original = load_dataset(args.data_path)
    std = ensure_standard_columns(original, spec)
    engineered, feature_cols = engineer_features(std, spec)
    # Keep the feature ordering aligned with training and apply persisted
    # normalisation so that the model sees values in the same scale it expects.
    features_norm = normalize_features(engineered, feature_cols, norm_params)

    # Truth columns and range gating (robust resolution across domains)
    def _pick_col(df: pd.DataFrame, candidates: list[str]) -> tuple[str, pd.DataFrame]:
        """Return the first column from `candidates` available in `df`."""

        for c in candidates:
            if c and c in df.columns:
                return c, df
        return "", df

    speed_candidates = [
        sargs.get("target_speed_col", None),
        "wind_speed",
        "buoy__wind_speed",
        "sar__owiwindspeed_mean",
    ]
    dir_candidates = [
        sargs.get("target_dir_col", None),
        "wind_dir",
        "wind_direction",
        "buoy__wind_dir",
        "sar__owiwinddirection_mean",
    ]

    col_s, src_df_s = _pick_col(original, speed_candidates)
    if not col_s:
        col_s, src_df_s = _pick_col(std, speed_candidates)
    col_d, src_df_d = _pick_col(original, dir_candidates)
    if not col_d:
        col_d, src_df_d = _pick_col(std, dir_candidates)
    if not col_s or not col_d:
        raise KeyError(
            f"Could not resolve truth columns. Tried speed candidates {speed_candidates} and direction candidates {dir_candidates}."
        )
    y_true_speed = src_df_s[col_s].to_numpy()
    y_true_dir_deg = src_df_d[col_d].to_numpy()

    rmin = float(sargs.get("range_min", 5.7))
    rmax = float(sargs.get("range_max", 17.8))
    in_range_mask = (y_true_speed >= rmin) & (y_true_speed <= rmax)
    total_count = int(len(y_true_speed))
    in_range_count = int(in_range_mask.sum())

    # Output directory
    stamp = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
    outdir = Path(args.output_dir) / f"feature_importance_{stamp}"
    outdir.mkdir(parents=True, exist_ok=True)

    # Baseline metrics for context
    base_speed, base_dir, *_ = run_model(model, features_norm, feature_cols)
    base_metrics = compute_metrics(y_true_speed, base_speed, y_true_dir_deg, base_dir, in_range_mask)
    base_metrics.update({
        "total_count": total_count,
        "in_range_count": in_range_count,
        "in_range_fraction": (float(in_range_count) / float(total_count)) if total_count else 0.0,
    })
    with open(outdir / "baseline_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(base_metrics, fh, indent=2)

    # Permutation importance
    groups = build_groups(feature_cols, spec)
    df_feat, df_group = permutation_importance(
        model,
        features_norm,
        feature_cols,
        y_true_speed,
        y_true_dir_deg,
        in_range_mask,
        rng=rng,
        group_map=groups if args.grouping in {"grouped", "both"} else None,
    )
    if args.grouping in {"feature", "both"}:
        df_feat.to_csv(outdir / "permutation_importance_features.csv", index=False)
        plot_bars(df_feat, "delta_speed_rmse", "feature", "Permutation importance (speed RMSE, per-feature)", outdir / "perm_speed_features.png", top_k=args.top_k)
        plot_bars(df_feat, "delta_dir_rmse_deg", "feature", "Permutation importance (direction RMSE°, per-feature)", outdir / "perm_dir_features.png", top_k=args.top_k)
    if args.grouping in {"grouped", "both"} and not df_group.empty:
        df_group.to_csv(outdir / "permutation_importance_groups.csv", index=False)
        plot_bars(df_group, "delta_speed_rmse", "group", "Permutation importance (speed RMSE, groups)", outdir / "perm_speed_groups.png", top_k=args.top_k)
        plot_bars(df_group, "delta_dir_rmse_deg", "group", "Permutation importance (direction RMSE°, groups)", outdir / "perm_dir_groups.png", top_k=args.top_k)

    # Local sensitivities
    sens = local_sensitivities(model, features_norm, feature_cols, in_range_mask, max_samples=args.max_samples_sensitivity)
    sens.to_csv(outdir / "local_sensitivities.csv", index=False)
    plot_bars(sens, "sensitivity_speed", "feature", "Local sensitivity (|∂speed/∂x|, median)", outdir / "sens_speed.png", top_k=args.top_k)
    plot_bars(sens, "sensitivity_dir_vec", "feature", "Local sensitivity (dir vector, median)", outdir / "sens_dir.png", top_k=args.top_k)

    # Weight-path importance
    wimp = weight_path_importance(model, feature_cols)
    wimp.to_csv(outdir / "weight_path_importance.csv", index=False)
    plot_bars(wimp, "weight_importance_speed", "feature", "Weight-path importance (speed head)", outdir / "weights_speed.png", top_k=args.top_k)
    plot_bars(wimp, "weight_importance_dir_vec", "feature", "Weight-path importance (direction head)", outdir / "weights_dir.png", top_k=args.top_k)

    # Write metadata
    meta = {
        "model_artifact": str(Path(args.model_artifact).resolve()),
        "data_path": str(Path(args.data_path).resolve()),
        "output_dir": str(outdir.resolve()),
        "range_min": rmin,
        "range_max": rmax,
        "grouping": args.grouping,
        "seed": args.seed,
        "top_k": args.top_k,
        "feature_cols": feature_cols,
        "station_names": spec.station_names,
    }
    with open(outdir / "analysis_metadata.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    # Brief textual summary
    with open(outdir / "report.txt", "w", encoding="utf-8") as fh:
        fh.write("Baseline (in-range) metrics\n")
        for k, v in base_metrics.items():
            fh.write(f"- {k}: {v}\n")
        if not df_group.empty:
            fh.write("\nTop groups by ΔRMSE (speed)\n")
            for _, row in df_group.head(10).iterrows():
                fh.write(f"- {row['group']}: +{row['delta_speed_rmse']:.4f} RMSE\n")
        fh.write("\nTop features by |∂speed/∂x| (median)\n")
        for _, row in sens.head(10).iterrows():
            fh.write(f"- {row['feature']}: {row['sensitivity_speed']:.4e}\n")

    print(f"Analysis completed. Results in {outdir}")


if __name__ == "__main__":
    main()
