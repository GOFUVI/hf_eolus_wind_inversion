"""Feature normalization utilities for training and inference pipelines.

Author: Juan Luis Herrera Cortijo <juan.luis.herrera.cortijo@gmail.com>
Created: 2025-10-16
Disclaimer: This script is provided as-is for research workflows and must be validated before any operational deployment.
"""

import collections
import logging
import math
import os
from typing import Dict, Iterable, List, Tuple, Set

import torch

import pandas as pd

logger = logging.getLogger(__name__)

MAD_SCALE = 1.4826
ROBUST_EPS = 1e-9
ROBUST_TOL = 0.05
MISSING_INTERVAL_TOKEN = "__missing_interval__"
MIN_INTERVAL_SAMPLES = 24


def _compute_single_feature_stats(
    series: pd.Series,
    mode: str,
    center_override: Dict[str, float],
    scale_override: Dict[str, float],
    feature: str,
) -> Tuple[float, float, bool]:
    """Return center/scale for a single feature, honouring overrides."""

    fallback_to_std = False
    if mode == 'robust':
        center = series.median()
        mad = (series - center).abs().median()
        scale = mad * MAD_SCALE
        if pd.isna(scale) or scale <= ROBUST_EPS:
            fallback_to_std = True
            scale = series.std()
    else:
        center = series.mean()
        scale = series.std()

    if pd.isna(scale) or scale <= ROBUST_EPS:
        logger.warning(
            "Feature %s has near-zero scale (%.3e) while using %s normalization; defaulting to 1.0",
            feature,
            scale if scale is not None else float('nan'),
            mode,
        )
        scale = 1.0

    if feature in center_override:
        logger.info("Overriding center for %s to %s", feature, center_override[feature])
        center = center_override[feature]
    if feature in scale_override:
        logger.info("Overriding scale for %s to %s", feature, scale_override[feature])
        scale = scale_override[feature]

    return float(center), float(scale), fallback_to_std


def compute_norm_from_data(
    train_df,
    val_df,
    numeric_features,
    center_override,
    scale_override,
    mode,
):
    """Compute normalization parameters from training data and apply scaling to both datasets."""

    feature_centers, feature_scales = {}, {}
    fallback_to_std = []

    for col in numeric_features:
        center, scale, used_std = _compute_single_feature_stats(
            train_df[col],
            mode,
            center_override,
            scale_override,
            col,
        )
        if used_std:
            fallback_to_std.append(col)
        feature_centers[col] = center
        feature_scales[col] = scale
        train_df[col] = (train_df[col] - center) / scale
        val_df[col] = (val_df[col] - center) / scale

    if fallback_to_std:
        logger.info(
            "Robust normalization fell back to standard deviation for columns: %s",
            ', '.join(sorted(fallback_to_std)),
        )

    diagnostics = {}
    if numeric_features:
        normalized_train = train_df[numeric_features]
        train_stats = {'mean': {}, 'std': {}}
        for col in numeric_features:
            col_series = normalized_train[col]
            train_stats['mean'][col] = float(col_series.mean())
            col_std = float(col_series.std())
            train_stats['std'][col] = col_std if not math.isnan(col_std) else math.nan
        if mode == 'robust':
            train_stats['median'] = {
                col: float(normalized_train[col].median())
                for col in numeric_features
            }
            scaled_mad = {}
            out_of_tolerance = []
            for col in numeric_features:
                col_series = normalized_train[col]
                med = col_series.median()
                mad = (col_series - med).abs().median() * MAD_SCALE
                mad = float(mad)
                scaled_mad[col] = mad
                if abs(med) > ROBUST_TOL or abs(mad - 1.0) > ROBUST_TOL:
                    out_of_tolerance.append(col)
            train_stats['scaled_mad'] = scaled_mad
            if out_of_tolerance:
                logger.warning(
                    "Robust normalization diagnostics exceeded tolerance %.3f for columns: %s",
                    ROBUST_TOL,
                    ', '.join(sorted(out_of_tolerance)),
                )

        diagnostics['train'] = train_stats
        if fallback_to_std:
            diagnostics['fallback_to_std'] = sorted(fallback_to_std)

    logger.info(
        "Computed %s normalization parameters for features: %s",
        mode,
        numeric_features,
    )
    logger.debug(f"Computed feature centers: {feature_centers}")
    logger.debug(f"Computed feature scales: {feature_scales}")

    return feature_centers, feature_scales, diagnostics


def _normalize_interval_key(value) -> str:
    """Normalize maintenance interval IDs into stable string keys."""

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return MISSING_INTERVAL_TOKEN
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else MISSING_INTERVAL_TOKEN
    return str(value)


def _interval_sort_key(interval_key: str) -> Tuple[str, str]:
    """Return a lexicographic key that preserves chronological order."""

    if interval_key == MISSING_INTERVAL_TOKEN:
        return ('', interval_key)
    if '_' in interval_key:
        _, remainder = interval_key.split('_', 1)
        return (remainder, interval_key)
    return (interval_key, interval_key)


def apply_conditional_normalization_from_params(
    df: pd.DataFrame,
    conditional_params: Dict[str, dict],
    stage: str = 'train',
) -> List[dict]:
    """
    Apply conditional normalization based on persisted parameters.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame whose columns will be scaled in place.
    conditional_params : dict
        Mapping persisted in normalization_params['conditional'].
    stage : str
        Label used for logging/audit (e.g., 'train', 'validation', 'inference').

    Returns
    -------
    list of dict
        Fallback events observed while applying the transform.
    """

    if not conditional_params:
        return []

    fallback_events: List[dict] = []

    for station, station_spec in conditional_params.items():
        maintenance_col = station_spec.get('maintenance_column')
        if not maintenance_col or maintenance_col not in df.columns:
            continue

        maintenance_keys = df[maintenance_col].map(_normalize_interval_key).tolist()

        for feature, feature_spec in station_spec.get('features', {}).items():
            if feature not in df.columns:
                continue

            per_interval = feature_spec.get('per_interval', {})
            interval_order = feature_spec.get('interval_order', [])
            global_params = feature_spec.get('global', {})
            interval_lookup = {
                interval: {
                    'center': float(values['center']),
                    'scale': float(values['scale']),
                    'sort_key': values.get('sort_key', ''),
                }
                for interval, values in per_interval.items()
            }

            centers = []
            scales = []

            for raw_key in maintenance_keys:
                key = raw_key
                params = interval_lookup.get(key)
                fallback_source = None
                fallback_interval = None

                if params is None:
                    key_sort = _interval_sort_key(key)
                    candidate = None
                    for interval in interval_order:
                        interval_sort = _interval_sort_key(interval)
                        if interval_sort <= key_sort:
                            candidate = interval
                        else:
                            break
                    if candidate and candidate in interval_lookup:
                        params = interval_lookup[candidate]
                        fallback_source = 'previous_interval'
                        fallback_interval = candidate
                    else:
                        params = {
                            'center': float(global_params.get('center', 0.0)),
                            'scale': float(global_params.get('scale', 1.0)),
                        }
                        fallback_source = 'global'

                scale = params.get('scale', 1.0)
                if scale <= ROBUST_EPS:
                    scale = 1.0
                centers.append(float(params.get('center', 0.0)))
                scales.append(float(scale))

                if fallback_source is not None:
                    fallback_events.append(
                        {
                            'stage': stage,
                            'station': station,
                            'feature': feature,
                            'interval': key,
                            'fallback_to': fallback_source,
                            'source_interval': fallback_interval,
                        }
                    )

            centers_series = pd.Series(centers, index=df.index, dtype='float64')
            scales_series = pd.Series(scales, index=df.index, dtype='float64')
            df[feature] = (df[feature] - centers_series) / scales_series

    return fallback_events


def apply_global_normalization_from_params(
    df: pd.DataFrame,
    feature_cols: List[str],
    centers: Dict[str, float],
    scales: Dict[str, float],
) -> None:
    """
    Apply global (per-feature) normalization given centers and scales to the provided DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to normalize in place.
    feature_cols : list[str]
        Feature columns to normalize.
    centers : dict
        Mapping feature -> center value.
    scales : dict
        Mapping feature -> scale value.
    """

    for col in feature_cols:
        if col not in df.columns:
            continue
        c = float(centers.get(col, 0.0))
        s = float(scales.get(col, 1.0))
        if s <= ROBUST_EPS:
            s = 1.0
        df[col] = (df[col] - c) / s


def _station_prefix_candidates(station_name: str) -> List[str]:
    """Generate candidate prefixes to locate maintenance columns."""

    candidates = []
    if station_name:
        candidates.append(station_name)
        if station_name.endswith('_aggregated'):
            candidates.append(station_name[: -len('_aggregated')])
        if '_' in station_name:
            candidates.append(station_name.split('_', 1)[0])
    return list(dict.fromkeys([c for c in candidates if c]))


def _resolve_maintenance_column(df_columns: Iterable[str], station_info: dict, station_name: str) -> str:
    """Pick the maintenance interval column associated with a station."""

    preferred = station_info.get('maintenance_interval_column') if isinstance(station_info, dict) else None
    if preferred and preferred in df_columns:
        return preferred

    for prefix in _station_prefix_candidates(station_name):
        candidate = f"{prefix}_maintenance_interval_id"
        if candidate in df_columns:
            return candidate
    return ''


def _compute_conditional_parameters(
    train_df: pd.DataFrame,
    station: str,
    maintenance_col: str,
    features: List[str],
    mode: str,
    center_override: Dict[str, float],
    scale_override: Dict[str, float],
    min_samples: int,
) -> Tuple[Dict[str, dict], Dict[str, dict], List[str]]:
    """Compute per-interval normalization stats for a station."""

    station_features: Dict[str, dict] = {}
    diagnostics: Dict[str, dict] = {}
    fallback_std_features: List[str] = []

    if maintenance_col not in train_df.columns:
        logger.warning(
            "Maintenance column %s not found while computing conditional stats for station %s",
            maintenance_col,
            station,
        )
        return station_features, diagnostics, fallback_std_features

    grouped = train_df.groupby(maintenance_col, dropna=False)

    for feature in features:
        if feature not in train_df.columns:
            logger.warning(
                "Feature %s missing from training frame; skipping conditional stats for station %s",
                feature,
                station,
            )
            continue

        series = train_df[feature]
        global_center, global_scale, global_used_std = _compute_single_feature_stats(
            series,
            mode,
            center_override,
            scale_override,
            feature,
        )
        if global_used_std:
            fallback_std_features.append(feature)

        interval_records = []
        for interval_value, group in grouped:
            key = _normalize_interval_key(interval_value)
            values = group[feature].dropna()
            count = len(values)

            if count == 0:
                interval_records.append(
                    {
                        'interval': key,
                        'count': 0,
                        'valid': False,
                        'reason': 'no_samples',
                        'center': None,
                        'scale': None,
                        'used_std': False,
                        'sort_key': _interval_sort_key(key),
                    }
                )
                continue

            center_val, scale_val, used_std = _compute_single_feature_stats(
                values,
                mode,
                {},
                {},
                feature,
            )
            valid = (count >= min_samples) and (scale_val > ROBUST_EPS)
            interval_records.append(
                {
                    'interval': key,
                    'count': count,
                    'center': center_val,
                    'scale': scale_val,
                    'used_std': used_std,
                    'valid': valid,
                    'reason': 'insufficient_samples' if count < min_samples else ('degenerate_scale' if scale_val <= ROBUST_EPS else ''),
                    'sort_key': _interval_sort_key(key),
                }
            )

        interval_records.sort(key=lambda rec: rec['sort_key'])
        per_interval = {}
        interval_order = []
        training_fallbacks = []
        conditional_diag = {}

        assigned = collections.OrderedDict()
        for record in interval_records:
            interval_key = record['interval']
            interval_order.append(interval_key)

            if record['valid']:
                assigned_params = {
                    'center': float(record['center']),
                    'scale': float(record['scale']),
                    'count': int(record['count']),
                    'source': 'interval',
                    'source_interval': None,
                    'used_std': bool(record['used_std']),
                    'sort_key': record['sort_key'][0],
                }
            else:
                fallback_source = None
                fallback_interval = None
                if assigned:
                    fallback_interval = next(reversed(assigned))
                    fallback_params = assigned[fallback_interval]
                    fallback_source = 'previous_interval'
                else:
                    fallback_params = {
                        'center': float(global_center),
                        'scale': float(global_scale),
                    }
                    fallback_source = 'global'
                assigned_params = {
                    'center': float(fallback_params['center']),
                    'scale': float(fallback_params['scale']),
                    'count': int(record['count']),
                    'source': fallback_source,
                    'source_interval': fallback_interval,
                    'used_std': False,
                    'sort_key': record['sort_key'][0],
                }
                training_fallbacks.append(
                    {
                        'feature': feature,
                        'interval': interval_key,
                        'fallback_to': fallback_source,
                        'source_interval': fallback_interval,
                        'available_samples': int(record['count']),
                        'reason': record['reason'] or 'insufficient_samples',
                    }
                )

            assigned[interval_key] = assigned_params
            per_interval[interval_key] = assigned_params
            conditional_diag[interval_key] = {
                'count': int(record['count']),
                'source': assigned_params['source'],
                'source_interval': assigned_params['source_interval'],
                'used_std': assigned_params['used_std'],
            }

        station_features[feature] = {
            'global': {
                'center': float(global_center),
                'scale': float(global_scale),
            },
            'per_interval': per_interval,
            'interval_order': interval_order,
            'min_samples': int(min_samples),
        }
        if training_fallbacks:
            station_features[feature]['training_fallbacks'] = training_fallbacks

        diagnostics[feature] = {
            'global_used_std': bool(global_used_std),
            'intervals': conditional_diag,
            'training_fallbacks': training_fallbacks,
        }

        if global_used_std:
            fallback_std_features.append(feature)

    return station_features, diagnostics, fallback_std_features


def _summarize_fallback_events(events: List[dict]) -> List[dict]:
    """Aggregate fallback events per station/feature for diagnostics."""

    summary_map: Dict[Tuple[str, str], collections.Counter] = {}
    for event in events or []:
        key = (event.get('station', ''), event.get('feature', ''))
        if key not in summary_map:
            summary_map[key] = collections.Counter()
        summary_map[key][event.get('fallback_to', 'global')] += 1

    summary = []
    for (station, feature), counter in sorted(summary_map.items()):
        entry = {'station': station, 'feature': feature}
        entry.update({fallback: int(count) for fallback, count in counter.items()})
        summary.append(entry)
    return summary
def normalize_features(train_df, val_df, feature_cols, config):
    """
    Identify numeric and angular features, then compute or load normalization parameters.

    Args:
        train_df (pd.DataFrame): Training DataFrame.
        val_df (pd.DataFrame): Validation DataFrame.
        feature_cols (list[str]): All input feature columns.
        config (argparse.Namespace): Configuration with overrides.

    Returns:
        tuple[dict, dict, dict, dict]:
            feature centers, feature scales, diagnostics metadata, conditional params.
    """

    angle_features = [c for c in feature_cols if c.startswith('cos_') or c.startswith('sin_')]
    numeric_features = [c for c in feature_cols if c not in angle_features]

    mode = getattr(config, 'normalization_mode', 'standard')
    center_override = getattr(config, 'center_override', {})
    scale_override = getattr(config, 'scale_override', {})

    station_schema = getattr(config, 'station_schema', {})
    conditional_candidates = {}
    conditional_feature_set = set()
    for station in getattr(config, 'station_names', []):
        info = station_schema.get(station, {}) if isinstance(station_schema, dict) else {}
        if not info.get('maintenance_interval_column'):
            logger.info(
                "Skipping conditional normalization for station %s: 'maintenance_interval_column' not declared in schema",
                station,
            )
            continue
        features = []
        for peak in ('0', '1'):
            col_name = f"{station}_pwr_{peak}"
            if col_name in numeric_features:
                features.append(col_name)
        if getattr(config, 'use_mad', False):
            for peak in ('0', '1'):
                col_name = f"{station}_pwr_mad_{peak}"
                if col_name in numeric_features:
                    features.append(col_name)
        if not features:
            continue
        maintenance_col = info.get('maintenance_interval_column')
        if maintenance_col not in train_df.columns:
            logger.warning(
                "Declared maintenance column %s for station %s not present in training data; skipping conditional normalization",
                maintenance_col,
                station,
            )
            continue
        conditional_candidates[station] = {
            'maintenance_column': maintenance_col,
            'features': features,
        }
        conditional_feature_set.update(features)

    global_features = [col for col in numeric_features if col not in conditional_feature_set]

    checkpoint_dir = os.path.join('/opt/ml/checkpoints')
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pth')
    logger.info(
        "Looking for fine-tuning checkpoint at %s: exists=%s",
        checkpoint_path,
        os.path.exists(checkpoint_path),
    )
    if os.path.exists(checkpoint_path):
        tmp_ckpt = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(tmp_ckpt, dict) and 'normalization_params' in tmp_ckpt:
            norm = tmp_ckpt['normalization_params']
            centers = norm.get('centers') or norm.get('means') or {}
            scales = norm.get('scales') or norm.get('stds') or {}
            conditional_params = norm.get('conditional', {})
            mode = norm.get('mode', mode)
            config.normalization_mode = mode
            logger.info("Loading normalization parameters from fine-tuning checkpoint (mode=%s)", mode)
            missing = [col for col in numeric_features if col not in centers or col not in scales]
            if missing:
                raise KeyError(
                    f"Checkpoint normalization parameters missing columns: {missing}. Available centers: {list(centers.keys())}, scales: {list(scales.keys())}"
                )
            for col in global_features:
                center, scale = centers[col], scales[col]
                train_df[col] = (train_df[col] - center) / scale
                val_df[col] = (val_df[col] - center) / scale

            if conditional_params:
                apply_conditional_normalization_from_params(train_df, conditional_params, stage='train')
                apply_conditional_normalization_from_params(val_df, conditional_params, stage='validation')
            else:
                for col in conditional_feature_set:
                    center, scale = centers[col], scales[col]
                    train_df[col] = (train_df[col] - center) / scale
                    val_df[col] = (val_df[col] - center) / scale

            diagnostics = norm.get('diagnostics', {})
            if diagnostics:
                logger.debug(f"Loaded normalization diagnostics: {diagnostics}")
            centers_out = {col: float(val) for col, val in centers.items()}
            scales_out = {col: float(val) for col, val in scales.items()}
            return centers_out, scales_out, diagnostics, conditional_params or {}
        logger.info("No normalization parameters in checkpoint; computing from training data.")
    else:
        logger.info("Fine-tuning checkpoint not found; computing normalization parameters from training data.")

    feature_centers: Dict[str, float] = {}
    feature_scales: Dict[str, float] = {}
    diagnostics: Dict[str, dict] = {}
    conditional_params: Dict[str, dict] = {}
    conditional_diagnostics: Dict[str, dict] = {}
    fallback_std_features: Set[str] = set()

    if global_features:
        global_centers, global_scales, global_diag = compute_norm_from_data(
            train_df,
            val_df,
            global_features,
            center_override,
            scale_override,
            mode,
        )
        feature_centers.update({col: float(val) for col, val in global_centers.items()})
        feature_scales.update({col: float(val) for col, val in global_scales.items()})
        fallback_std_features.update(global_diag.get('fallback_to_std', []))
    else:
        global_diag = {'train': {}}

    for station, spec in conditional_candidates.items():
        station_features, station_diag, station_fallback_std = _compute_conditional_parameters(
            train_df,
            station,
            spec['maintenance_column'],
            spec['features'],
            mode,
            center_override,
            scale_override,
            MIN_INTERVAL_SAMPLES,
        )
        if not station_features:
            continue
        conditional_params[station] = {
            'maintenance_column': spec['maintenance_column'],
            'features': station_features,
        }
        conditional_diagnostics[station] = station_diag
        for feature, info in station_features.items():
            feature_centers[feature] = float(info['global']['center'])
            feature_scales[feature] = float(info['global']['scale'])
        fallback_std_features.update(station_fallback_std)

    train_fallback_events = apply_conditional_normalization_from_params(train_df, conditional_params, stage='train')
    val_fallback_events = apply_conditional_normalization_from_params(val_df, conditional_params, stage='validation')

    if train_fallback_events:
        logger.info(
            "Conditional normalization applied %d fallback adjustments on training data",
            len(train_fallback_events),
        )
    if val_fallback_events:
        logger.info(
            "Conditional normalization applied %d fallback adjustments on validation data",
            len(val_fallback_events),
        )

    if numeric_features:
        normalized_train = train_df[numeric_features]
        train_stats = {'mean': {}, 'std': {}}
        for col in numeric_features:
            col_series = normalized_train[col]
            train_stats['mean'][col] = float(col_series.mean())
            train_stats['std'][col] = float(col_series.std())
        if mode == 'robust':
            train_stats['median'] = {}
            scaled_mad = {}
            out_of_tolerance = []
            for col in numeric_features:
                col_series = normalized_train[col]
                med = float(col_series.median())
                mad = float((col_series - med).abs().median() * MAD_SCALE)
                train_stats['median'][col] = med
                scaled_mad[col] = mad
                if abs(med) > ROBUST_TOL or abs(mad - 1.0) > ROBUST_TOL:
                    out_of_tolerance.append(col)
            train_stats['scaled_mad'] = scaled_mad
            if out_of_tolerance:
                logger.warning(
                    "Robust normalization diagnostics exceeded tolerance %.3f for columns: %s",
                    ROBUST_TOL,
                    ', '.join(sorted(out_of_tolerance)),
                )
        diagnostics['train'] = train_stats

    if fallback_std_features:
        diagnostics['fallback_to_std'] = sorted(set(fallback_std_features))

    if conditional_params:
        diagnostics['conditional_normalization'] = {
            'min_samples': MIN_INTERVAL_SAMPLES,
            'stations': conditional_diagnostics,
            'fallback_summary': {
                'train': _summarize_fallback_events(train_fallback_events),
                'validation': _summarize_fallback_events(val_fallback_events),
            },
        }

    logger.info(
        "Computed %s normalization parameters for features: %s",
        mode,
        numeric_features,
    )
    logger.debug(f"Computed feature centers: {feature_centers}")
    logger.debug(f"Computed feature scales: {feature_scales}")

    return feature_centers, feature_scales, diagnostics, conditional_params
