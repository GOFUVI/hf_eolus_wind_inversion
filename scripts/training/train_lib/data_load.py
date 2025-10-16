"""Data loading and train/validation split logic for GeoParquet datasets.

Author: Juan Luis Herrera Cortijo <juan.luis.herrera.cortijo@gmail.com>
Created: 2025-10-16
Disclaimer: This script is provided as-is for research workflows and must be validated before any operational deployment.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import pyarrow.dataset as ds

logger = logging.getLogger(__name__)


def _resolve_dataset_path(config) -> Path:
    """Resolve the GeoParquet dataset path from CLI arguments or SageMaker channels."""

    search_paths: List[Path] = []

    if getattr(config, 'data_path', None):
        search_paths.append(Path(config.data_path))

    channel = os.environ.get('SM_CHANNEL_TRAINING', os.environ.get('SM_CHANNEL_TRAIN', ''))
    if channel:
        search_paths.append(Path(channel))

    # Common local fallbacks for development use
    search_paths.append(Path('data.parquet'))
    search_paths.append(Path('data'))

    for candidate in search_paths:
        if candidate.is_file() and candidate.suffix.lower() in {'.parquet', '.pq'}:
            return candidate.resolve()
        if candidate.is_dir():
            try:
                next(candidate.rglob('*.parquet'))
            except StopIteration:
                continue
            return candidate.resolve()

    raise FileNotFoundError(
        'GeoParquet dataset not found. Provide it with --data_path or via the training channel.'
    )


def _ensure_column(
    schema_names: Iterable[str],
    candidates: List[Optional[str]],
    *,
    required: bool,
    description: str,
) -> Optional[str]:
    """Return the first candidate present in schema_names, optionally enforcing presence."""

    available = set(schema_names)
    for name in candidates:
        if name and name in available:
            return name
    if required:
        raise KeyError(f"Required column for {description} not found in dataset schema")
    return None


def _collect_required_columns(schema_names: List[str], config) -> Tuple[List[str], Dict[str, str]]:
    """Infer required columns and canonical renames for feature engineering."""

    ordered_columns: List[str] = []
    rename_map: Dict[str, str] = {}

    def add(col: Optional[str]) -> None:
        if col and col not in ordered_columns:
            ordered_columns.append(col)

    # Target columns and metadata
    try:
        speed_source = _ensure_column(
            schema_names,
            [getattr(config, 'target_speed_col', None), 'wind_speed'],
            required=True,
            description='wind speed target',
        )
    except KeyError as exc:
        if getattr(config, 'target_speed_col', None):
            raise KeyError(
                f"Custom target speed column '{config.target_speed_col}' not found in dataset schema"
            ) from exc
        raise

    try:
        dir_source = _ensure_column(
            schema_names,
            [getattr(config, 'target_dir_col', None), 'wind_dir', 'wind_direction'],
            required=True,
            description='wind direction target',
        )
    except KeyError as exc:
        if getattr(config, 'target_dir_col', None):
            raise KeyError(
                f"Custom target direction column '{config.target_dir_col}' not found in dataset schema"
            ) from exc
        raise

    add(speed_source)
    add(dir_source)
    if speed_source != 'wind_speed':
        rename_map[speed_source] = 'wind_speed'
    if dir_source != 'wind_dir':
        rename_map[dir_source] = 'wind_dir'

    add(_ensure_column(schema_names, ['fold'], required=False, description='fold index'))
    add(_ensure_column(schema_names, ['wind_bin'], required=False, description='wind bin'))
    add(
        _ensure_column(
            schema_names,
            [getattr(config, 'id_col', None)],
            required=False,
            description='ID grouping column',
        )
    )

    # Station-derived columns
    schema = getattr(config, 'station_schema', {})
    include_velocity_median = getattr(config, 'use_velocity_median', False)
    for station in getattr(config, 'station_names', []):
        info = schema.get(station)
        if info is None:
            raise KeyError(f"No schema mapping found for station '{station}'")

        power_pattern = info.get('power_pattern')
        if not power_pattern:
            raise KeyError(f"Schema mapping for station '{station}' lacks 'power_pattern'")

        for peak in ['0', '1']:
            power_col = power_pattern.format(agg=config.agg, peak=peak)
            if power_col not in schema_names:
                raise KeyError(
                    f"Column '{power_col}' not found for station '{station}' (agg={config.agg}, peak={peak})"
                )
            add(power_col)
            rename_map[power_col] = f"{station}_pwr_{peak}"

        if getattr(config, 'use_mad', False):
            mad_pattern = info.get('mad_pattern')
            if not mad_pattern:
                raise KeyError(
                    f"Schema mapping for station '{station}' lacks 'mad_pattern' but --use_mad was requested"
                )
            for peak in ['0', '1']:
                mad_col = mad_pattern.format(agg=config.agg, peak=peak)
                if mad_col not in schema_names:
                    raise KeyError(
                        f"MAD column '{mad_col}' not found for station '{station}' (peak={peak})"
                    )
                add(mad_col)
                rename_map[mad_col] = f"{station}_pwr_mad_{peak}"

        if include_velocity_median:
            vel_pattern = info.get('velocity_median_pattern') if isinstance(info, dict) else None
            for peak in ['0', '1']:
                candidates = []
                if vel_pattern:
                    try:
                        candidates.append(vel_pattern.format(agg=config.agg, peak=peak))
                    except (KeyError, IndexError):
                        candidates.append(vel_pattern.format(peak=peak))
                candidates.extend([
                    f"{station}__velo_median_{peak}",
                    f"{station}_velo_median_{peak}",
                    f"{station}_stats_velo_median_{peak}",
                ])
                match = next((cand for cand in candidates if cand in schema_names), None)
                if match is None:
                    raise KeyError(
                        f"Median radial velocity column not found for station '{station}' (peak={peak})."
                    )
                add(match)
                rename_map[match] = f"{station}_velo_median_{peak}"

        bearing_col = info.get('bearing')
        if not bearing_col:
            raise KeyError(f"Schema mapping for station '{station}' lacks 'bearing'")
        bearing_col = bearing_col.format(agg=config.agg)
        if bearing_col not in schema_names:
            raise KeyError(f"Bearing column '{bearing_col}' not found for station '{station}'")
        add(bearing_col)
        rename_map[bearing_col] = f"{station}_bearing_source"

        dist_col = info.get('distance')
        if not dist_col:
            raise KeyError(f"Schema mapping for station '{station}' lacks 'distance'")
        dist_col = dist_col.format(agg=config.agg)
        if dist_col not in schema_names:
            raise KeyError(f"Distance column '{dist_col}' not found for station '{station}'")
        add(dist_col)
        rename_map[dist_col] = f"{station}_dist_source"

    # Maintenance interval identifiers (optional but required for conditional normalization)
    for column_name in schema_names:
        if column_name.endswith('_maintenance_interval_id') or column_name.endswith('_maintenance_start'):
            add(column_name)

    return ordered_columns, rename_map


def load_data(config):
    """
    Discover and read the input GeoParquet dataset, then split into training and validation DataFrames.

    Args:
        config (argparse.Namespace): Configuration with data_path, station metadata, and fold index.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: train_df, val_df
    """

    dataset_path = _resolve_dataset_path(config)
    if dataset_path.is_dir():
        dataset = ds.dataset(str(dataset_path), format='parquet', partitioning='hive')
    else:
        dataset = ds.dataset([str(dataset_path)], format='parquet')

    schema_names = list(dataset.schema.names)
    columns, rename_map = _collect_required_columns(schema_names, config)
    logger.info(
        "Loading GeoParquet dataset from %s with columns %s",
        dataset_path,
        columns,
    )

    table = dataset.to_table(columns=columns)
    df = table.to_pandas()
    if rename_map:
        df.rename(columns=rename_map, inplace=True)

    # Split into train/validation folds
    if config.fold_actual < 0:
        train_df = df.copy()
        val_df = df.copy()
        logger.info(
            "Loaded full dataset: %s rows for training and validation",
            len(train_df),
        )
    else:
        train_df = df[df['fold'] != config.fold_actual].copy()
        val_df = df[df['fold'] == config.fold_actual].copy()
        logger.info(
            "Loaded data: %s train rows, %s validation rows",
            len(train_df),
            len(val_df),
        )

    return train_df, val_df


def _clone_for_rehearsal(config, speed_col: Optional[str], dir_col: Optional[str]):
    """Create a shallow clone of config overriding target columns for rehearsal loading."""

    import argparse

    clone = argparse.Namespace(**vars(config))
    if speed_col:
        clone.target_speed_col = speed_col
    if dir_col:
        clone.target_dir_col = dir_col
    # Force full-dataset mode for rehearsal ingestion
    clone.fold_actual = -1
    return clone


def load_rehearsal_frame(config) -> Optional[Tuple[object, List[str], List[str]]]:
    """
    Optionally load a rehearsal (source-domain) DataFrame using the provided paths/columns.

    Returns
    -------
    tuple(pd.DataFrame, list[str], list[str]) or None
        The rehearsal DataFrame with engineered features and targets, and the feature/target column lists.
        Returns None if no rehearsal path was specified.
    """

    path = getattr(config, 'rehearsal_data_path', None)
    if not path:
        return None

    # Clone config to adopt the rehearsal target column names for loading/engineering
    reh_cfg = _clone_for_rehearsal(
        config,
        getattr(config, 'rehearsal_target_speed_col', None),
        getattr(config, 'rehearsal_target_dir_col', None),
    )

    path_str = str(path)
    # Support S3 prefixes directly for rehearsal data
    if path_str.startswith('s3://'):
        import pyarrow as pa
        fs = pa.fs.S3FileSystem()
        dataset = ds.dataset(path_str, format='parquet', filesystem=fs)
    else:
        dataset_path = _resolve_dataset_path(reh_cfg)
        if dataset_path.is_dir():
            dataset = ds.dataset(str(dataset_path), format='parquet', partitioning='hive')
        else:
            dataset = ds.dataset([str(dataset_path)], format='parquet')

    schema_names = list(dataset.schema.names)
    columns, rename_map = _collect_required_columns(schema_names, reh_cfg)
    table = dataset.to_table(columns=columns)
    df = table.to_pandas()
    if rename_map:
        df.rename(columns=rename_map, inplace=True)

    # Engineer features with the rehearsal-config targets
    from .features import engineer_features
    df, df_val, feature_cols, target_cols = engineer_features(df.copy(), df.copy(), reh_cfg)
    # We only need the training portion; discard df_val
    return (df, feature_cols, target_cols)
