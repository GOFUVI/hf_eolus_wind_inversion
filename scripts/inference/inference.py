#!/usr/bin/env python3
"""Run HF-radar wind inference inside the SageMaker Processing container.

This module drives the end-to-end prediction workflow: it downloads the model
artefact from S3, rebuilds the engineered feature space expected by the
network, executes the forward pass, and emits deterministic post-processing
layers (range gating, confidence scoring, and diagnostics metadata). Every
step mirrors the training-time conventions so that inference remains auditable
and reproducible across processing jobs.

Author: Juan Luis Herrera Cortijo <juan.luis.herrera.cortijo@gmail.com>
Created: 2025-10-16
Disclaimer: This script is provided as-is for research workflows and must be validated before any operational deployment.
"""

import argparse
import json
import logging
import os
import tarfile
import tempfile
from pathlib import Path
from types import SimpleNamespace

import boto3
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import torch

from train_lib.model import MLP
from train_lib.normalization import apply_conditional_normalization_from_params


logger = logging.getLogger(__name__)


def parse_args():
    """Return CLI arguments expected inside the SageMaker Processing job.

    Returns:
        argparse.Namespace: Parsed flags controlling model downloads, optional
            input S3 references, and the output serialization format.
    """
    parser = argparse.ArgumentParser(description="Run HF wind inference on SageMaker Processing")
    parser.add_argument("--model-s3-uri", required=True, help="S3 URI to model.tar.gz artifact")
    parser.add_argument(
        "--input-data",
        required=False,
        help="Optional S3 URI for the input dataset (data is mounted under /opt/ml/processing/input)",
    )
    parser.add_argument(
        "--output-format",
        choices=["csv", "parquet"],
        default="parquet",
        help="Output format for predictions (default: parquet)",
    )
    return parser.parse_args()


def parse_s3_uri(uri: str):
    """Split an S3 URI into bucket and key components.

    Args:
        uri: Full S3 URI (e.g. ``s3://bucket/path/to/object``).

    Returns:
        tuple[str, str]: Bucket name and object key.

    Raises:
        ValueError: If the URI does not use the ``s3://`` scheme.
    """
    if not uri.lower().startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {uri}")
    bucket, key = uri[5:].split("/", 1)
    return bucket, key


def download_model_artifact(s3_uri: str, destination: Path) -> Path:
    """Download the packaged model artefact to a temporary path.

    Args:
        s3_uri: Location of ``model.tar.gz`` in S3.
        destination: Local file path where the archive will be stored.

    Returns:
        Path: Resolved local path to the downloaded archive.
    """
    bucket, key = parse_s3_uri(s3_uri)
    client = boto3.client("s3")
    destination.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(bucket, key, str(destination))
    return destination


def extract_model_artifact(artifact_path: Path, extract_dir: Path) -> Path:
    """Extract the tarball that holds the checkpoint and auxiliary files.

    Args:
        artifact_path: Local path to ``model.tar.gz``.
        extract_dir: Directory where the archive will be decompressed.

    Returns:
        Path: Directory containing the unpacked artefacts (weights, configs).
    """
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(artifact_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)
    return extract_dir


def load_json(path: Path, description: str):
    """Load a JSON file and raise an explicit error if it is missing.

    Args:
        path: File to read.
        description: Human-readable label used in error messages.

    Returns:
        Any: Parsed JSON payload.

    Raises:
        FileNotFoundError: If the file does not exist at the expected location.
    """
    if not path.exists():
        raise FileNotFoundError(f"Expected {description} at {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_feature_spec(model_cfg: dict) -> SimpleNamespace:
    """Compose the feature configuration required to rebuild engineered inputs.

    Args:
        model_cfg: Model configuration fragment persisted during training.

    Returns:
        SimpleNamespace: Resolved feature switches, schema templates, and
            target/identifier columns used downstream.

    Raises:
        KeyError: If mandatory schema attributes are absent from the
            configuration payload.
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
    """Format schema patterns while tolerating partially specified templates.

    Args:
        pattern: Pattern string that may include ``{agg}`` and/or ``{peak}``.
        agg: Aggregation suffix (e.g. ``mean``) injected when present.
        peak: Peak identifier used to disambiguate Bragg lobes.

    Returns:
        str: Interpolated pattern falling back to the raw input when
            placeholders are absent.
    """
    try:
        return pattern.format(agg=agg, peak=peak)
    except (KeyError, IndexError):
        try:
            return pattern.format(peak=peak)
        except (KeyError, IndexError):
            return pattern


def ensure_standard_columns(df: pd.DataFrame, feature_spec: SimpleNamespace) -> pd.DataFrame:
    """Guarantee that the raw inference frame exposes the expected columns.

    Args:
        df: DataFrame loaded from the pivoted GeoParquet shards.
        feature_spec: Feature descriptor produced by :func:`build_feature_spec`.

    Returns:
        pandas.DataFrame: Copy of ``df`` with canonical column names added.

    Raises:
        KeyError: When required source columns cannot be located.
    """
    working = df.copy()
    columns = set(working.columns)

    def copy_column(source: str, dest: str, *, required: bool = True) -> bool:
        """Clone a column if present, optionally failing when it is missing.

        Args:
            source: Existing column expected in the raw dataframe.
            dest: Target column name to populate.
            required: When ``True`` raise ``KeyError`` if ``source`` is missing.

        Returns:
            bool: ``True`` when the column existed (and was copied if needed).
        """
        if source not in columns:
            if required:
                raise KeyError(f"Required column '{source}' not found in inference dataset")
            return False
        if dest not in working.columns:
            working[dest] = working[source]
        return True

    # Harmonise target speed/direction fields, tolerating legacy naming.
    speed_candidates = [feature_spec.target_speed_col, "wind_speed"]
    for cand in speed_candidates:
        if cand and copy_column(cand, "wind_speed", required=False):
            break

    dir_candidates = [feature_spec.target_dir_col, "wind_dir", "wind_direction"]
    for cand in dir_candidates:
        if cand and copy_column(cand, "wind_dir", required=False):
            break

    if "wind_bin" in columns:
        copy_column("wind_bin", "wind_bin", required=False)
    if feature_spec.id_col:
        copy_column(feature_spec.id_col, feature_spec.id_col, required=False)

    # Iterate through each radar station to materialise the power, MAD,
    # velocity, bearing, and distance columns consumed by the network.
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
    """Derive engineered features to mirror the training data layout.

    Args:
        df: DataFrame after :func:`ensure_standard_columns` harmonisation.
        feature_spec: Feature descriptor detailing optional switches and
            station-level schema.

    Returns:
        tuple[pandas.DataFrame, List[str]]: The augmented frame and the ordered
        list of feature columns expected by the neural network.

    Raises:
        KeyError: If any engineered feature cannot be constructed.
    """
    stations = feature_spec.station_names
    use_mad = feature_spec.use_mad
    use_velocity_median = feature_spec.use_velocity_median

    if use_mad:
        mad_cols = [f"{station}_pwr_mad_{peak}" for station in stations for peak in ("0", "1")]
        # Replicate the maximum-MAD aggregation used during training, using raw
        # power as a deterministic fallback when MAD observations are missing.
        stacked = pd.concat([df[col].replace(-np.inf, np.nan) for col in mad_cols], axis=1)
        df["pwr_mad"] = stacked.max(axis=1, skipna=True)
        df["pwr_mad"].fillna(
            df[[f"{station}_pwr_{peak}" for station in stations for peak in ("0", "1")]].max(axis=1),
            inplace=True,
        )

    for station in stations:
        bearing_col = f"{station}_bearing_source"
        # Convert bearings to sine/cosine so directional information remains
        # continuous for the regression head.
        rad = np.deg2rad(df[bearing_col])
        df[f"cos_{station}_bearing"] = np.cos(rad)
        df[f"sin_{station}_bearing"] = np.sin(rad)

    for station in stations:
        dist_col = f"{station}_dist_source"
        # Preserve the short distance naming expected by downstream metrics.
        df.rename(columns={dist_col: f"{station}_dist"}, inplace=True)

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
    """Apply the saved scaling parameters, including conditional branches.

    Args:
        feature_df: Frame containing engineered features.
        feature_cols: Ordered list of feature names used by the model.
        norm_params: Normalisation payload embedded inside ``model.pth``.

    Returns:
        pandas.DataFrame: Normalised features restricted to ``feature_cols``.

    Raises:
        KeyError: If scaling coefficients are missing for any requested feature.
    """
    working_df = feature_df.copy()

    numeric_cols = [c for c in feature_cols if not (c.startswith("cos_") or c.startswith("sin_"))]
    centers = norm_params.get("centers") or norm_params.get("means", {})
    scales = norm_params.get("scales") or norm_params.get("stds", {})
    conditional_params = norm_params.get("conditional", {}) or {}

    conditional_features = set()
    for station_spec in conditional_params.values():
        for feature in station_spec.get("features", {}):
            conditional_features.add(feature)

    # Sanity-check that every numeric feature has global scaling coefficients.
    missing = [
        col for col in numeric_cols
        if col not in centers or col not in scales
    ]
    if missing:
        raise KeyError(
            f"Normalization parameters missing for feature(s) {missing}. Available centers: {list(centers.keys())}"
        )

    global_cols = [col for col in numeric_cols if col not in conditional_features]
    # Apply the canonical z-score transform to non-conditional features.
    for col in global_cols:
        working_df[col] = (working_df[col] - centers[col]) / scales[col]

    if conditional_features and conditional_params:
        fallback_events = apply_conditional_normalization_from_params(
            working_df,
            conditional_params,
            stage='inference',
        )
        if fallback_events:
            # The helper returns the rows that could not be matched to a station-specific mapping.
            logger.info(
                "Conditional normalization applied %d fallback adjustments during inference",
                len(fallback_events),
            )
    else:
        for col in conditional_features:
            working_df[col] = (working_df[col] - centers[col]) / scales[col]

    return working_df[feature_cols]


def run_inference(model, features: pd.DataFrame, feature_cols):
    """Execute the MLP forward pass and reconstruct directional predictions.

    Args:
        model: Torch model instantiated with training-time topology.
        features: Normalised feature frame.
        feature_cols: Ordered feature list used to build the tensor input.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        Predicted speed, cosine unit vector, sine unit vector, direction in
        degrees, and range probabilities.
    """
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(features[feature_cols].values, dtype=torch.float32)
        regression, logits = model(inputs)
        regression = regression.numpy()
        probs = torch.softmax(logits, dim=1).numpy()
    speed = regression[:, 0]
    cos_vals = regression[:, 1]
    sin_vals = regression[:, 2]
    norm = np.sqrt(cos_vals ** 2 + sin_vals ** 2)
    # Avoid division-by-zero when the regression head outputs a degenerate
    # direction vector; clamp to a tiny epsilon instead.
    norm = np.where(norm < 1e-6, 1e-6, norm)
    cos_unit = cos_vals / norm
    sin_unit = sin_vals / norm
    angle_rad = np.arctan2(sin_unit, cos_unit)
    angle_deg = np.degrees(angle_rad) % 360.0
    return speed, cos_unit, sin_unit, angle_deg, probs


def write_output(df: pd.DataFrame, output_dir: Path, output_format: str):
    """Persist predictions in the requested tabular format.

    Args:
        df: Fully assembled inference DataFrame.
        output_dir: Directory under ``/opt/ml/processing/output`` where
            artefacts will be written.
        output_format: Either ``csv`` or ``parquet``.

    Returns:
        Path: Destination file path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_format == "csv":
        output_path = output_dir / "predictions.csv"
        df.to_csv(output_path, index=False)
    else:
        output_path = output_dir / "predictions.parquet"
        df.to_parquet(output_path, index=False)
    return output_path


def main():
    """Controller orchestrating S3 download, feature engineering, and output writing.

    This entrypoint is invoked by SageMaker Processing. It wires together the
    ten-stage sequence documented in ``docs/inference.md``: argument parsing,
    dataset staging, model reconstruction, feature engineering, normalisation,
    neural inference, range-aware post-processing, and final materialisation of
    predictions plus metadata.
    """
    args = parse_args()

    input_dir = Path("/opt/ml/processing/input")
    if not input_dir.exists():
        raise FileNotFoundError("Processing input directory not found: /opt/ml/processing/input")

    # SageMaker mounts Parquet shards under /opt/ml/processing/input; load them in a single frame.
    dataset = ds.dataset(str(input_dir), format="parquet")
    table = dataset.to_table()
    original_df = table.to_pandas()

    # Persist artefacts in a scratch directory to avoid polluting the mounted volumes.
    tmp_dir = Path(tempfile.mkdtemp(prefix="hf-wind-model-"))
    artifact_path = download_model_artifact(args.model_s3_uri, tmp_dir / "model.tar.gz")
    model_dir = extract_model_artifact(artifact_path, tmp_dir / "extracted")

    model_path = model_dir / "model.pth"
    # Torch checkpoints embed both weights and the normalisation payload required at inference.
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    norm_params = None
    if isinstance(checkpoint, dict):
        norm_params = checkpoint.get("normalization_params")
    if norm_params is None:
        raise FileNotFoundError("Normalization parameters not found in model checkpoint")

    script_args_path = model_dir / "script_args.json"
    if script_args_path.exists():
        script_args = load_json(script_args_path, "saved script arguments")
    else:
        # Legacy checkpoints embed the arguments directly in the Torch payload; keep backwards compatibility.
        saved_args = checkpoint.get("args") if isinstance(checkpoint, dict) else None
        if not saved_args:
            raise FileNotFoundError("Saved script arguments not found in checkpoint")
        script_args = saved_args

    # Step 3 in docs/inference.md: rebuild the configuration that encodes station
    # catalogues and feature switches so downstream engineering stays in lockstep with training.
    model_config_rel = script_args.get("model_config")
    model_cfg = None
    if model_config_rel:
        model_config_path = Path("/opt/ml/code") / model_config_rel
        if model_config_path.exists():
            model_cfg = load_json(model_config_path, "model configuration")
    if model_cfg is None:
        payload = checkpoint.get("model_config_payload") if isinstance(checkpoint, dict) else None
        if payload:
            model_cfg = json.loads(payload)
        else:
            raise FileNotFoundError("Model configuration not found in container or checkpoint payload")

    if isinstance(model_cfg, str):
        model_cfg = json.loads(model_cfg)

    # Construct the feature recipe before touching any raw inference columns.
    feature_spec = build_feature_spec(model_cfg)
    # Step 4 (schema harmonisation): clone/rename raw fields to the canonical training layout.
    standardized_df = ensure_standard_columns(original_df, feature_spec)
    # Step 5 (feature engineering): recompute derived attributes (MAD, bearings, distances, etc.).
    engineered_df, feature_cols = engineer_features(standardized_df, feature_spec)
    # Step 6 (normalisation): use checkpoint-embedded statistics to obtain z-scored features.
    normalized_features = normalize_features(engineered_df, feature_cols, norm_params)

    hidden_layers = int(script_args.get("hidden_layers", 2))
    hidden_units = int(script_args.get("hidden_units", 128))
    dropout = float(script_args.get("dropout", 0.0))
    # Step 7 (forward propagation preparation): recover the label taxonomy for the range classifier.
    class_labels = script_args.get("range_class_labels", ['below', 'in', 'above'])
    if isinstance(class_labels, str):
        class_labels = [part.strip() for part in class_labels.split(',') if part.strip()]
    class_labels = [str(label) for label in class_labels]
    num_classes = len(class_labels)
    # Rebuild the neural network using the exact topology stored in script_args.
    model = MLP(len(feature_cols), hidden_layers, hidden_units, drop_rate=dropout, num_classes=num_classes)
    model_path = model_dir / "model.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found at {model_path}")
    state = torch.load(model_path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)

    range_min = float(script_args.get("range_min", 5.7))
    range_max = float(script_args.get("range_max", 17.8))
    range_margin = max(0.0, float(script_args.get("range_margin", 0.5)))
    flag_threshold = float(script_args.get("range_flag_threshold", 0.5))
    flag_threshold = min(max(flag_threshold, 0.0), 1.0)
    in_index = int(script_args.get("range_in_class_index", 1))
    if in_index < 0 or in_index >= num_classes:
        in_index = min(1, num_classes - 1) if num_classes > 1 else 0

    # Step 7 (continued): execute the MLP in evaluation mode to obtain raw predictions.
    speed, cos_unit, sin_unit, angle_deg, probs = run_inference(
        model,
        normalized_features,
        feature_cols,
    )

    truth_speed_col = script_args.get("target_speed_col", "wind_speed")
    truth_dir_col = script_args.get("target_dir_col", "wind_dir")

    # Step 8 (deterministic post-processing): preserve raw truths while adding prediction artefacts.
    output_df = original_df.copy()
    output_df.rename(columns={truth_speed_col: "wind_speed", truth_dir_col: "wind_dir"}, inplace=True)
    output_df["pred_wind_speed"] = speed
    output_df["pred_cos_wind_dir"] = cos_unit
    output_df["pred_sin_wind_dir"] = sin_unit
    output_df["pred_wind_direction"] = angle_deg

    # Range classification outputs
    pred_class_indices = probs.argmax(axis=1)
    def normalize_label(label: str, idx: int) -> str:
        if label:
            sanitized = ''.join(ch if ch.isalnum() else '_' for ch in label.lower())
            return sanitized or f"class_{idx}"
        return f"class_{idx}"

    # Preserve the human-readable labels defined at training time, falling back
    # to synthetic identifiers when necessary.
    label_map = [class_labels[idx] if idx < len(class_labels) else f"class_{idx}" for idx in range(probs.shape[1])]
    sanitized_labels = [normalize_label(label_map[i], i) for i in range(len(label_map))]

    for idx, label in enumerate(sanitized_labels):
        col_name = f"prob_range_{label}"
        if idx < probs.shape[1]:
            output_df[col_name] = probs[:, idx]

    confidence = probs[np.arange(len(probs)), pred_class_indices]
    predicted_labels = [label_map[idx] if idx < len(label_map) else f"class_{idx}" for idx in pred_class_indices]
    output_df["pred_range_label"] = predicted_labels
    output_df["pred_range_confidence"] = confidence
    output_df["prob_range_in"] = probs[:, in_index] if in_index < probs.shape[1] else np.nan

    # Flag handling with configurable threshold and margin
    def classify_speed(val: float) -> str:
        if val < range_min:
            return 'below'
        if val > range_max:
            return 'above'
        return 'in'

    # Deterministic speed-based gating acts as a guardrail for the classifier predictions.
    speed_based_label = np.vectorize(classify_speed)(speed)
    output_df["pred_speed_range_label"] = speed_based_label
    near_lower = np.abs(speed - range_min) <= range_margin
    near_upper = np.abs(speed - range_max) <= range_margin
    output_df["range_near_lower_margin"] = near_lower
    output_df["range_near_upper_margin"] = near_upper
    output_df["range_near_any_margin"] = near_lower | near_upper

    # Build range flags that balance classifier confidence with speed gating.
    range_flags = []
    flag_confident = []
    consistency = []
    for label, conf, speed_label in zip(predicted_labels, confidence, speed_based_label):
        confident = conf >= flag_threshold
        flag_confident.append(confident)
        if confident:
            range_flags.append(label)
        else:
            range_flags.append('uncertain')
        consistency.append(label == speed_label)

    output_df["range_flag"] = range_flags
    output_df["range_flag_confident"] = flag_confident
    output_df["range_prediction_consistent"] = consistency

    # Step 9 (output materialisation): persist predictions and accompanying metadata for audits.
    output_dir = Path("/opt/ml/processing/output")
    output_path = write_output(output_df, output_dir, args.output_format)

    # Persist lightweight metadata so downstream diagnostics can trace the inference job.
    metadata = {
        "model_artifact": args.model_s3_uri,
        "input_rows": len(original_df),
        "output_format": args.output_format,
        "feature_columns": feature_cols,
        "range_min": range_min,
        "range_max": range_max,
        "range_margin": range_margin,
        "range_flag_threshold": flag_threshold,
        "range_class_labels": class_labels,
    }
    with (output_dir / "inference_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Inference completed. Output saved to {output_path}")


if __name__ == "__main__":
    main()
