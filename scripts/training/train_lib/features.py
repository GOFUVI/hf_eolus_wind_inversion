"""Feature engineering: power statistics, MAD features, directional and station transforms.

Author: Juan Luis Herrera Cortijo <juan.luis.herrera.cortijo@gmail.com>
Created: 2025-10-16
Disclaimer: This script is provided as-is for research workflows and must be validated before any operational deployment.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def engineer_features(train_df, val_df, config):
    """
    Build input features from raw station data, including power stats, MAD-based features,
    directional encoding, and station bearing/distance transforms.

    Args:
        train_df (pd.DataFrame): Training DataFrame.
        val_df (pd.DataFrame): Validation DataFrame.
        config (argparse.Namespace): Configuration with station_names, agg, use_mad.

    Returns:
        tuple: (train_df, val_df, feature_cols, target_cols)
    """
    station_names = config.station_names
    use_mad = config.use_mad
    use_velocity_median = getattr(config, 'use_velocity_median', False)
    range_min = float(config.range_min)
    range_max = float(config.range_max)
    range_labels = list(getattr(config, 'range_class_labels', ['below', 'in', 'above']))
    below_idx = range_labels.index('below') if 'below' in range_labels else 0
    in_idx = getattr(config, 'range_in_class_index', 1)
    above_idx = range_labels.index('above') if 'above' in range_labels else 2

    def annotate_range(df):
        speeds = df['wind_speed']
        mask = ((speeds >= range_min) & (speeds <= range_max)).astype(np.float32)
        labels = np.full(len(df), in_idx, dtype=np.int64)
        labels[speeds < range_min] = below_idx
        labels[speeds > range_max] = above_idx
        df['range_mask'] = mask
        df['range_class'] = labels

    annotate_range(train_df)
    annotate_range(val_df)

    # Validate power statistic columns per station and Bragg peak
    power_cols = []
    for source in station_names:
        for peak in ['0', '1']:
            col_name = f"{source}_pwr_{peak}"
            if col_name not in train_df.columns:
                available = train_df.columns.tolist()
                raise KeyError(
                    f"Expected column '{col_name}' for station {source} peak {peak}. Available columns: {available}"
                )
            power_cols.append(col_name)

    # MAD-based power feature
    if use_mad:
        mad_cols = []
        for source in station_names:
            for peak in ['0', '1']:
                mad_col = f"{source}_pwr_mad_{peak}"
                if mad_col not in train_df.columns:
                    raise KeyError(f"No MAD column found for {source} Bragg peak {peak}")
                mad_cols.append(mad_col)
        train_mad_matrix = train_df[mad_cols].replace(-np.inf, np.nan)
        val_mad_matrix = val_df[mad_cols].replace(-np.inf, np.nan)
        train_pwr_mad = train_mad_matrix.max(axis=1, skipna=True)
        val_pwr_mad = val_mad_matrix.max(axis=1, skipna=True)
        train_pwr_mad.fillna(train_df[power_cols].max(axis=1), inplace=True)
        val_pwr_mad.fillna(val_df[power_cols].max(axis=1), inplace=True)
        train_df['pwr_mad'] = train_pwr_mad
        val_df['pwr_mad'] = val_pwr_mad

    velocity_median_cols = []
    if use_velocity_median:
        for source in station_names:
            for peak in ['0', '1']:
                vel_col = f"{source}_velo_median_{peak}"
                if vel_col not in train_df.columns:
                    raise KeyError(
                        f"Median radial velocity column '{vel_col}' not found in training data"
                    )
                velocity_median_cols.append(vel_col)

    # Directional target encoding for wind direction (cos/sin)
    train_df['cos_wind_dir'] = np.cos(np.deg2rad(train_df['wind_dir']))
    train_df['sin_wind_dir'] = np.sin(np.deg2rad(train_df['wind_dir']))
    val_df['cos_wind_dir'] = np.cos(np.deg2rad(val_df['wind_dir']))
    val_df['sin_wind_dir'] = np.sin(np.deg2rad(val_df['wind_dir']))
    train_df.drop(columns=['wind_dir'], inplace=True)
    val_df.drop(columns=['wind_dir'], inplace=True)

    # Bearing and distance features per station
    for station in station_names:
        bear_col = f'{station}_bearing_source'
        if bear_col not in train_df.columns:
            raise KeyError(f"No bearing column found for station {station}")
        for df in (train_df, val_df):
            df[f'cos_{station}_bearing'] = np.cos(np.deg2rad(df[bear_col]))
            df[f'sin_{station}_bearing'] = np.sin(np.deg2rad(df[bear_col]))
        train_df.drop(columns=[bear_col], inplace=True)
        val_df.drop(columns=[bear_col], inplace=True)

    for station in station_names:
        dist_col = f'{station}_dist_source'
        if dist_col not in train_df.columns:
            raise KeyError(f"No distance column found for station {station}")
        train_df.rename(columns={dist_col: f'{station}_dist'}, inplace=True)
        val_df.rename(columns={dist_col: f'{station}_dist'}, inplace=True)

    # Drop fold column if present
    for df in (train_df, val_df):
        if 'fold' in df.columns:
            df.drop(columns=['fold'], inplace=True)

    # Build feature and target columns lists
    feature_cols = []
    for station in station_names:
        for peak in ['0', '1']:
            feature_cols.append(f"{station}_pwr_{peak}")
    if use_velocity_median:
        feature_cols.extend(velocity_median_cols)
    if use_mad:
        feature_cols.append('pwr_mad')
    for station in station_names:
        feature_cols.append(f'{station}_dist')
    for station in station_names:
        feature_cols += [f'cos_{station}_bearing', f'sin_{station}_bearing']

    target_cols = ['wind_speed', 'cos_wind_dir', 'sin_wind_dir', 'range_class', 'range_mask']

    # Validate presence of required columns
    for col in feature_cols:
        if col not in train_df.columns:
            raise KeyError(f"Feature column {col} not found in training data")
    for col in target_cols:
        if col not in train_df.columns:
            raise KeyError(f"Target column {col} not found in data")

    return train_df, val_df, feature_cols, target_cols
