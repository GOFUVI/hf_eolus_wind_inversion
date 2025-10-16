"""
Command-line interface for the training script.

Defines and parses all CLI arguments required for data ingestion, feature engineering,
model configuration, and training hyperparameters.

Author: Juan Luis Herrera Cortijo <juan.luis.herrera.cortijo@gmail.com>
Created: 2025-10-16
Disclaimer: This script is provided as-is for research workflows and must be validated before any operational deployment.
"""

import argparse


def get_train_args():
    """Return the parsed command-line arguments used by the training entrypoint."""
    parser = argparse.ArgumentParser()

    # Cross-validation, Aggregation, and Feature Selection Arguments
    parser.add_argument(
        '--fold_actual', type=int, required=True,
        help=(
            'Index of the current fold to use for validation (1-N). '
            'Use -1 to train on the full dataset without cross-validation'
        ),
    )
    parser.add_argument(
        '--agg_stat', type=str, default=None,
        help=(
            'Power column aggregation statistic. One of mean, median, or max; '
            'strings containing these terms will be normalized accordingly.'
        ),
    )
    parser.add_argument(
        '--use_mad', type=str, default=None,
        help='Whether to use MAD: "1"/"True" (case-insensitive) or "0"/"False"',
    )
    parser.add_argument(
        '--use_velocity_median', type=int, choices=[0, 1], default=None,
        help='Include per-station median radial velocity features (1) or skip them (0)',
    )

    # Model Architecture and Training Hyperparameters
    parser.add_argument(
        '--hidden_layers', type=int, default=None,
        help='Number of hidden layers in the MLP',
    )
    parser.add_argument(
        '--hidden_units', type=int, default=None,
        help='Number of units in each hidden layer',
    )
    parser.add_argument(
        '--dropout', type=float, default=None,
        help='Dropout rate (0 = no dropout)',
    )
    parser.add_argument(
        '--epochs', type=int, default=None,
        help='Number of training epochs',
    )
    parser.add_argument(
        '--batch_size', type=int, default=None,
        help='Batch size for training',
    )
    parser.add_argument(
        '--lr', type=float, default=None,
        help='Initial learning rate for the optimizer',
    )
    parser.add_argument(
        '--weight_decay', type=float, default=None,
        help='Weight decay (L2) for the optimizer',
    )

    # Fine-tuning specific (no effect in baseline training if unset)
    parser.add_argument(
        '--finetune-heads-lr', type=float, default=None,
        help='Learning rate for task-specific heads during fine-tuning (optional).'
    )
    parser.add_argument(
        '--finetune-backbone-lr', type=float, default=None,
        help='Learning rate for the last backbone layer during fine-tuning (optional).'
    )
    parser.add_argument(
        '--use-l2sp', type=int, choices=[0, 1], default=None,
        help='Enable L2-SP regularization anchored to the pre-fine-tuning weights (1 to enable).'
    )
    parser.add_argument(
        '--l2sp-backbone-lambda', type=float, default=None,
        help='L2-SP coefficient for backbone parameters (optional).'
    )
    parser.add_argument(
        '--l2sp-heads-lambda', type=float, default=None,
        help='L2-SP coefficient for head parameters (optional).'
    )

    # Data Path Argument
    parser.add_argument(
        '--data_path', type=str, default='',
        help='Path to the GeoParquet file or dataset directory (optional)',
    )

    # Early Stopping and Error Data Flags
    parser.add_argument(
        '--early_stopping', type=int, default=None,
        help='Activate early stopping (1) or disable (0)',
    )
    parser.add_argument(
        '--patience', type=int, default=None,
        help='Number of epochs without improvement before stopping',
    )
    parser.add_argument(
        '--save_error_data', type=int, default=None,
        help='Save raw validation data and predictions for error analysis (1) or disable (0)',
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility (optional). If omitted, random initialization is used.',
    )

    # Normalization Overrides Argument
    parser.add_argument(
        '--norm-override', type=str, default=None,
        help=(
            'Semicolon-separated overrides for normalization parameters (feature.param=value) using "center" or "scale", '
            'e.g. vila_aggregated_dist.scale=0.1;prio_aggregated_dist.center=30.5'
        ),
    )
    parser.add_argument(
        '--normalization-mode',
        type=str,
        choices=['standard', 'robust'],
        default=None,
        help='Feature normalization strategy: standard (mean/std) or robust (median/MAD with std fallback).',
    )
    parser.add_argument(
        '--model-config',
        type=str,
        default=None,
        help='Path to a YAML file describing model hyperparameters and overrides'
    )
    # Station Selection Argument
    parser.add_argument(
        '--stations', type=str, default=None,
        help=(
            'Semicolon-separated list of station names to include for feature extraction '
            '(at least two names when provided)'
        ),
    )

    # Custom Target Column Names Argument
    parser.add_argument(
        '--target-speed-col',
        type=str,
        default=None,
        help='Name of the wind speed column (default: wind_speed)',
    )
    parser.add_argument(
        '--target-dir-col',
        type=str,
        default=None,
        help='Name of the wind direction column (default: wind_direction)',
    )
    parser.add_argument(
        '--id-col',
        type=str,
        default='location_id',
        help=(
            'Name of the column to stratify on for location-based metrics '
            '(optional; default constant location_id). '
            'If the column does not exist, a new column with constant value 1 '
            'will be generated.'
        ),
    )
    parser.add_argument(
        '--range-min',
        type=float,
        default=None,
        help='Lower bound (inclusive) for valid wind speed values.',
    )
    parser.add_argument(
        '--range-max',
        type=float,
        default=None,
        help='Upper bound (inclusive) for valid wind speed values.',
    )
    parser.add_argument(
        '--range-margin',
        type=float,
        default=None,
        help='Margin (m/s) to consider predictions near the valid range boundaries.',
    )
    parser.add_argument(
        '--range-loss-weight',
        type=float,
        default=None,
        help='Weight applied to the range classification loss term.',
    )
    parser.add_argument(
        '--range-flag-threshold',
        type=float,
        default=None,
        help='Minimum class probability required to flag the predicted range as confident.',
    )

    # Optional rehearsal (source-domain replay) — only used during fine-tuning
    parser.add_argument(
        '--rehearsal-data-path',
        type=str,
        default=None,
        help='GeoParquet path (file or prefix) to source-domain data for rehearsal batches (fine-tuning only).',
    )
    parser.add_argument(
        '--rehearsal-target-speed-col',
        type=str,
        default=None,
        help='Name of the wind speed column in the rehearsal dataset.',
    )
    parser.add_argument(
        '--rehearsal-target-dir-col',
        type=str,
        default=None,
        help='Name of the wind direction column in the rehearsal dataset.',
    )
    parser.add_argument(
        '--rehearsal-fraction',
        type=float,
        default=None,
        help='Approximate fraction of rehearsal steps per epoch (e.g., 0.1 = 1 step cada 10).',
    )

    # Knowledge distillation (fine-tuning only)
    parser.add_argument(
        '--use-kd',
        type=int,
        choices=[0, 1],
        default=None,
        help='Enable knowledge distillation against the teacher checkpoint (1 to enable).',
    )
    parser.add_argument(
        '--lambda-kd',
        type=float,
        default=None,
        help='Global weight applied to the summed distillation losses (speed, direction, range).',
    )
    parser.add_argument(
        '--lambda-kd-reg',
        type=float,
        default=None,
        help='Weight applied specifically to the regression distillation loss.',
    )
    parser.add_argument(
        '--lambda-kd-cls',
        type=float,
        default=None,
        help='Weight applied specifically to the range-classification distillation loss.',
    )
    parser.add_argument(
        '--kd-temperature',
        type=float,
        default=None,
        help='Temperature τ used to soften teacher/student logits in the distillation KL term.',
    )
    parser.add_argument(
        '--teacher-checkpoint',
        type=str,
        default=None,
        help='Optional path to the teacher checkpoint (metadata only – the loader relies on the artifact snapshot).',
    )

    return parser.parse_args()
