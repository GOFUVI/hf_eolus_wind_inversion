#!/usr/bin/env python3
"""
Author: Juan Luis Herrera Cortijo <juan.luis.herrera.cortijo@gmail.com>
Created: 2025-10-16
Disclaimer: This script is provided as-is for research workflows and must be validated before any operational deployment.
"""

#
# Detailed Script Overview:
# This script implements a full training pipeline for an MLP-based wind prediction model,
# tailored for SageMaker. It handles data ingestion, feature construction, normalization,
# model definition, multi-task training with dynamic weight averaging (DWA) and optional
# early stopping, evaluation metrics computation, checkpoint management, and output
# serialization for downstream inference.
#
# Core sections of this script:
#  1) IMPORTS: Load standard library, AWS SDK, data processing, and PyTorch modules,
#     with logging configuration and device detection.
#  2) ARGUMENT PARSING: Define flags for cross-validation folds, feature aggregation,
#     model hyperparameters, data paths, early stopping, normalization overrides,
#     station selection, and custom target column names.
#  3) STATION VALIDATION: Ensure the user-specified station list contains at least two
#     stations, and determine whether to train on the full dataset or perform fold-based
#     validation.
#  4) AGGREGATION STATISTIC NORMALIZATION: Validate the --agg_stat parameter to control
#     how per-station power features are aggregated (mean/median/max).
#  5) MAD FEATURE & EARLY STOPPING CONFIG: Parse flags for including MAD-based features,
#     enabling early stopping, and saving raw error data for analysis.
#  6) NORMALIZATION OVERRIDES: Parse optional user-defined mean/std overrides for numeric
#     features before normalization.
#  7) DEVICE SETUP: Determine whether to use GPU (CUDA) or CPU for model training.
#  8) DATA LOADING: Locate and load the input GeoParquet dataset from --data_path or SageMaker channels,
#     and split the data into training and validation folds.
#  9) FEATURE ENGINEERING: Construct input features, including per-station power,
#     optional MAD-based power, directional encoding (cos/sin), and station bearing/distance.
# 10) FEATURE NORMALIZATION: Separate numeric and angular features, compute or load
#     normalization parameters (means/stds), and apply scaling to numeric features.
# 11) MODEL DEFINITION: Initialize hyperparameters, define the MLP class, instantiate
#     the model and optimizer, and optionally load checkpoints for fine-tuning.
# 12) EVALUATION FUNCTION: Define evaluate_metrics_full() to compute speed and direction
#     error metrics (RMSE, MAE, correlation, R², scatter index, angular errors).
# 13) TRAINING LOOP: Train the model over epochs, computing multi-task losses,
#     updating weights with DWA, and applying early stopping based on validation loss.
# 14) CHECKPOINTING & OUTPUT SERIALIZATION: Save best-model checkpoints, normalization
#     parameters, CLI arguments, and periodic checkpoints for recovery.
# 15) FINAL MODEL LOADING & REPORTING: Reload the best model for final evaluation and
#     print combined loss for automated capture.
#
"""
train.py - Train MLP model for wind prediction per fold in SageMaker

This script is executed by train_model.sh inside SageMaker to train the model
on the specified fold. It loads data, builds and trains an MLP, evaluates metrics,
and saves the trained model. It supports a --stations flag to specify a
semicolon-separated list of station names for feature extraction (required; at least two names).
It also supports optional flags --target-speed-col and --target-dir-col to specify
custom column names for the wind speed and wind direction targets (defaults to
"wind_speed" and "wind_direction").
"""
import os
import random
from typing import Optional

import numpy as np
import torch

# train.py

from train_lib.utils import configure_logging
from train_lib.cli import get_train_args
from train_lib.config import build_config
from train_lib.device import select_device
from train_lib.data_load import load_data, load_rehearsal_frame
from train_lib.features import engineer_features
from train_lib.normalization import normalize_features
from train_lib.model import build_model_and_optimizer
from train_lib.train_loop import run_training_loop
from train_lib.reporting import final_evaluation_and_report


def set_random_seed(seed: Optional[int]) -> None:
    """Set library-wide random seeds when a deterministic run is requested."""
    if seed is None:
        return
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    """Execute the full SageMaker training lifecycle for a single fold."""
    configure_logging()
    args = get_train_args()
    config = build_config(args)
    set_random_seed(getattr(config, 'seed', None))
    device = select_device()

    # Data ingestion and feature engineering ---------------------------------
    # We keep the transformations explicit so that every stage can consume the
    # same artefacts (dataframes, normalization params) and remain reproducible
    # across folds and downstream pipelines.
    train_df, val_df = load_data(config)
    train_df, val_df, feature_cols, target_cols = engineer_features(train_df, val_df, config)
    feature_centers, feature_scales, norm_diagnostics, conditional_params = normalize_features(
        train_df,
        val_df,
        feature_cols,
        config,
    )
    norm_centers = {col: float(val) for col, val in feature_centers.items()}
    norm_scales = {col: float(val) for col, val in feature_scales.items()}
    center_label = 'median' if config.normalization_mode == 'robust' else 'mean'
    scale_label = 'scaled_mad' if config.normalization_mode == 'robust' else 'std'
    norm_params = {
        'mode': config.normalization_mode,
        'center_label': center_label,
        'scale_label': scale_label,
        'centers': dict(norm_centers),
        'scales': dict(norm_scales),
    }
    if norm_diagnostics:
        norm_params['diagnostics'] = norm_diagnostics
    if conditional_params:
        norm_params['conditional'] = conditional_params
    (
        model,
        optimizer,
        train_loader,
        val_loader,
        start_epoch,
        teacher_model,
    ) = build_model_and_optimizer(
        train_df,
        val_df,
        feature_cols,
        target_cols,
        config,
        device,
    )
    # Optional rehearsal loader (fine-tuning only)
    rehearsal_loader = None
    try:
        if getattr(config, 'rehearsal_data_path', None) and float(getattr(config, 'rehearsal_fraction', 0.0)) > 0.0:
            import logging
            logging.getLogger(__name__).info(
                "Rehearsal requested: path=%s, fraction=%.3f",
                getattr(config, 'rehearsal_data_path', None),
                float(getattr(config, 'rehearsal_fraction', 0.0)),
            )
            reh = load_rehearsal_frame(config)
            if reh is not None:
                reh_df, reh_feature_cols, reh_target_cols = reh
                # Apply same normalization parameters as main training
                from train_lib.normalization import (
                    apply_global_normalization_from_params,
                    apply_conditional_normalization_from_params,
                )
                apply_global_normalization_from_params(reh_df, reh_feature_cols, feature_centers, feature_scales)
                if conditional_params:
                    apply_conditional_normalization_from_params(reh_df, conditional_params, stage='rehearsal')
                import torch
                X_reh = torch.tensor(reh_df[reh_feature_cols].values, dtype=torch.float32)
                y_reh = torch.tensor(reh_df[reh_target_cols].values, dtype=torch.float32)
                reh_dataset = torch.utils.data.TensorDataset(X_reh, y_reh)
                rehearsal_loader = torch.utils.data.DataLoader(
                    reh_dataset, batch_size=config.batch_size, shuffle=True
                )
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning(
            "Rehearsal disabled due to exception during loading/normalization: %s", exc
        )
        rehearsal_loader = None

    w1, w2 = run_training_loop(
        model,
        optimizer,
        train_loader,
        val_loader,
        config,
        norm_params,
        start_epoch,
        device,
        rehearsal_loader,
        teacher_model,
    )
    # Final evaluation --------------------------------------------------------
    # The reporting helper handles metric computation, artifact persistence,
    # and logging in one place so the SageMaker job emits a consistent bundle
    # regardless of the caller (HPO, final training, fine-tuning).
    final_evaluation_and_report(
        model,
        train_df,
        val_df,
        feature_cols,
        target_cols,
        device,
        config,
        w1,
        w2,
    )


if __name__ == "__main__":
    main()
