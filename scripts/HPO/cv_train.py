#!/usr/bin/env python3
"""
cv_train.py

Cross-validation wrapper for train.py that runs K-fold CV internally and outputs average CombinedLoss.

Supports optional flags --target-speed-col, --target-dir-col, and --id-col to customize the target and grouping columns.

Author: Juan Luis Herrera Cortijo <juan.luis.herrera.cortijo@gmail.com>
Created: 2025-10-16
Disclaimer: This script is provided as-is for research workflows and must be validated before any operational deployment.
"""
import argparse
import subprocess
import re
import sys

def parse_args():
    """Parse command-line arguments controlling the K-fold experiment.

    Returns:
        argparse.Namespace: Parsed options ready to forward to each fold run.
    """
    parser = argparse.ArgumentParser(
        description="Cross-validation wrapper for train.py"
    )
    parser.add_argument(
        '--n_folds', type=int, default=5,
        help='Number of folds for cross-validation'
    )
    parser.add_argument(
        '--agg_stat', type=str, choices=['mean', 'median', 'max'], default='mean',
        help='Aggregation statistic to use'
    )
    parser.add_argument(
        '--use_mad', type=str, default='0',
        help='Whether to use MAD (1/True or 0/False)'
    )
    parser.add_argument(
        '--use_velocity_median', type=int, choices=[0, 1], default=0,
        help='Include per-station median radial velocity features (1) or skip them (0)'
    )
    parser.add_argument(
        '--hidden_layers', type=int, default=2,
        help='Number of hidden layers for MLP'
    )
    parser.add_argument(
        '--hidden_units', type=int, default=64,
        help='Number of units per hidden layer'
    )
    parser.add_argument(
        '--dropout', type=float, default=0.0,
        help='Dropout rate'
    )
    parser.add_argument(
        '--epochs', type=int, default=50,
        help='Number of epochs for training'
    )
    parser.add_argument(
        '--batch_size', type=int, default=32,
        help='Batch size for training'
    )
    parser.add_argument(
        '--lr', type=float, default=1e-3,
        help='Learning rate'
    )
    parser.add_argument(
        '--weight_decay', type=float, default=0.0,
        help='Weight decay (L2)'
    )
    parser.add_argument(
        '--range_loss_weight', type=float, default=None,
        help='Weight applied to the range classification loss term'
    )
    parser.add_argument(
        '--range_margin', type=float, default=None,
        help='Margin (m/s) around the valid wind range for masking'
    )
    parser.add_argument(
        '--range_flag_threshold', type=float, default=None,
        help='Minimum probability to emit a confident range flag'
    )
    parser.add_argument(
        '--early_stopping', type=int, choices=[0,1], default=0,
        help='Enable early stopping (1) or disable (0)'
    )
    parser.add_argument(
        '--patience', type=int, default=5,
        help='Number of epochs with no improvement before stopping'
    )
    parser.add_argument(
        '--save_error_data', type=int, choices=[0, 1], default=0,
        help='Save validation error data for analysis (1) or skip (0)'
    )
    parser.add_argument(
        '--data_path', type=str, default='',
        help='Path to the GeoParquet file or dataset directory (optional)'
    )
    parser.add_argument(
        '--norm-override', type=str, default=None,
        help=('Semicolon-separated overrides for normalization parameters '
              '(feature.param=value) using the neutral names center/scale, '
              'e.g. vila_aggregated_dist.scale=0.1;prio_aggregated_dist.center=0.5)')
    )
    parser.add_argument(
        '--model-config', type=str, default=None,
        help='Path to a YAML file describing model hyperparameters and overrides'
    )
    parser.add_argument(
        '--stations', type=str, default='',
        help=('Semicolon-separated list of station names to include for '
              'feature extraction (optional when provided via model config)')
    )
    parser.add_argument(
        '--target-speed-col', type=str, default=None,
        help='Name of the wind speed column (default: wind_speed)'
    )
    parser.add_argument(
        '--target-dir-col', type=str, default=None,
        help='Name of the wind direction column (default: wind_direction)'
    )
    parser.add_argument(
        '--id-col', type=str, default=None,
        help='Name of the column to stratify on for location-based metrics (default: location_id)'
    )
    return parser.parse_args()

def main():
    """Run train.py across folds and emit the average CombinedLoss.

    The function mirrors the SageMaker entry point, iterating over every
    requested fold, collecting the CombinedLoss printed by the training loop,
    and finally reporting their arithmetic mean for the tuning objective.
    """
    args = parse_args()
    # Accumulate the CombinedLoss reported by each fold execution.
    losses = []
    for fold in range(1, args.n_folds + 1):
        # Build the command line mirroring SageMaker's invocation but scoped to a fold.
        cmd = ['python3', 'train.py', '--fold_actual', str(fold)]
        if args.agg_stat is not None:
            cmd.extend(['--agg_stat', str(args.agg_stat)])
        if args.use_mad is not None:
            cmd.extend(['--use_mad', str(args.use_mad)])
        if args.use_velocity_median is not None:
            cmd.extend(['--use_velocity_median', str(args.use_velocity_median)])
        if args.hidden_layers is not None:
            cmd.extend(['--hidden_layers', str(args.hidden_layers)])
        if args.hidden_units is not None:
            cmd.extend(['--hidden_units', str(args.hidden_units)])
        if args.dropout is not None:
            cmd.extend(['--dropout', str(args.dropout)])
        if args.epochs is not None:
            cmd.extend(['--epochs', str(args.epochs)])
        if args.batch_size is not None:
            cmd.extend(['--batch_size', str(args.batch_size)])
        if args.lr is not None:
            cmd.extend(['--lr', str(args.lr)])
        if args.weight_decay is not None:
            cmd.extend(['--weight_decay', str(args.weight_decay)])
        if args.early_stopping is not None:
            cmd.extend(['--early_stopping', str(args.early_stopping)])
        if args.patience is not None:
            cmd.extend(['--patience', str(args.patience)])
        if args.save_error_data is not None:
            cmd.extend(['--save_error_data', str(args.save_error_data)])
        if args.range_loss_weight is not None:
            cmd.extend(['--range-loss-weight', str(args.range_loss_weight)])
        if args.range_margin is not None:
            cmd.extend(['--range-margin', str(args.range_margin)])
        if args.range_flag_threshold is not None:
            cmd.extend(['--range-flag-threshold', str(args.range_flag_threshold)])
        if args.stations:
            cmd.extend(['--stations', args.stations])
        # Include custom column parameters if provided (speed, direction, id)
        if args.target_speed_col:
            cmd.extend(['--target-speed-col', args.target_speed_col])
        if args.target_dir_col:
            cmd.extend(['--target-dir-col', args.target_dir_col])
        if args.id_col:
            cmd.extend(['--id-col', args.id_col])
        if args.data_path:
            cmd.extend(['--data_path', args.data_path])
        if args.model_config:
            cmd.extend(['--model-config', args.model_config])
        print(f"Running fold {fold}/{args.n_folds}", flush=True)
        # Capture the child process output; CombinedLoss is reported on stdout.
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error running train.py for fold {fold}", file=sys.stderr)
            print(result.stderr, file=sys.stderr)
            sys.exit(result.returncode)
        # CombinedLoss is the sole metric the tuner watches; abort if missing.
        match = re.search(r'CombinedLoss: ([0-9\.eE+\-]+)', result.stdout)
        if not match:
            print(f"CombinedLoss not found for fold {fold}", file=sys.stderr)
            sys.exit(1)
        loss = float(match.group(1))
        losses.append(loss)
    # Report the mean CombinedLoss across folds; SageMaker will minimise it.
    avg_loss = sum(losses) / len(losses)
    # Output average CombinedLoss for HPO
    print(f"CombinedLoss: {avg_loss:.6f}")

if __name__ == '__main__':
    main()
