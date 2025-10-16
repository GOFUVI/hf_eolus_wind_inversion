"""Configuration and validation for the training pipeline.

Author: Juan Luis Herrera Cortijo <juan.luis.herrera.cortijo@gmail.com>
Created: 2025-10-16
Disclaimer: This script is provided as-is for research workflows and must be validated before any operational deployment.
"""

import logging
import sys
from pathlib import Path

import json

logger = logging.getLogger(__name__)


def _load_model_config(path: str) -> dict:
    """Load and validate the JSON model configuration file."""
    model_path = Path(path)
    if not model_path.is_file():
        logger.error(f"Model config file not found: {model_path}")
        sys.exit(1)
    try:
        with model_path.open('r', encoding='utf-8') as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        logger.error(f"Failed to parse JSON config {model_path}: {exc}")
        sys.exit(1)
    if not isinstance(data, dict):
        logger.error(f"Model config {model_path} must define a mapping of settings")
        sys.exit(1)
    return data


def _format_norm_overrides(norm_cfg) -> str:
    """Serialise nested normalization overrides into the CLI-friendly format."""
    if norm_cfg is None:
        return ''
    if isinstance(norm_cfg, dict):
        parts = []
        for feature, params in norm_cfg.items():
            if isinstance(params, dict):
                for param, value in params.items():
                    if value is not None:
                        parts.append(f"{feature}.{param}={value}")
            else:
                parts.append(f"{feature}={params}")
        return ';'.join(parts)
    return str(norm_cfg)


def _apply_model_config(args, model_cfg):
    """Populate argparse arguments using values declared in the model config mapping."""
    inline_schema = None

    model_section = model_cfg.get('model', {})

    def fetch(key):
        value = None
        if isinstance(model_section, dict):
            value = model_section.get(key)
        if value is None:
            value = model_cfg.get(key)
        return value

    def set_value(attr, key, transform=lambda x: x):
        value = fetch(key)
        if value is None:
            return
        existing = getattr(args, attr, None)
        if existing is None:
            setattr(args, attr, transform(value))

    set_value('stations', 'stations', lambda v: ';'.join(v) if isinstance(v, list) else str(v))
    set_value('target_speed_col', 'target_speed_col', str)
    set_value('target_dir_col', 'target_dir_col', str)
    set_value('agg_stat', 'agg_stat', str)
    set_value('hidden_layers', 'hidden_layers', int)
    set_value('hidden_units', 'hidden_units', int)
    set_value('dropout', 'dropout', float)
    set_value('epochs', 'epochs', int)
    set_value('batch_size', 'batch_size', int)
    set_value('lr', 'lr', float)
    set_value('weight_decay', 'weight_decay', float)
    set_value('patience', 'patience', int)
    set_value('save_error_data', 'save_error_data', int)
    set_value('id_col', 'id_col', str)
    set_value('seed', 'seed', int)
    set_value('range_min', 'range_min', float)
    set_value('range_max', 'range_max', float)
    set_value('range_margin', 'range_margin', float)
    set_value('range_loss_weight', 'range_loss_weight', float)
    set_value('range_flag_threshold', 'range_flag_threshold', float)
    set_value('normalization_mode', 'normalization_mode', str)
    # Rehearsal (fine-tuning only)
    set_value('rehearsal_data_path', 'rehearsal_data_path', str)
    set_value('rehearsal_target_speed_col', 'rehearsal_target_speed_col', str)
    set_value('rehearsal_target_dir_col', 'rehearsal_target_dir_col', str)
    set_value('rehearsal_fraction', 'rehearsal_fraction', float)
    # Fine-tuning specific knobs (harmless when unused in baseline training)
    set_value('finetune_heads_lr', 'finetune_heads_lr', float)
    set_value('finetune_backbone_lr', 'finetune_backbone_lr', float)
    set_value('use_l2sp', 'use_l2sp', int)
    set_value('l2sp_backbone_lambda', 'l2sp_backbone_lambda', float)
    set_value('l2sp_heads_lambda', 'l2sp_heads_lambda', float)
    set_value('use_kd', 'use_kd', int)
    set_value('lambda_kd', 'lambda_kd', float)
    set_value('lambda_kd_reg', 'lambda_kd_reg', float)
    set_value('lambda_kd_cls', 'lambda_kd_cls', float)
    set_value('kd_temperature', 'kd_temperature', float)
    set_value('teacher_checkpoint', 'teacher_checkpoint', str)

    target_range = fetch('target_speed_range')
    if target_range is not None and isinstance(target_range, (list, tuple)) and len(target_range) == 2:
        args.target_speed_range = [float(target_range[0]), float(target_range[1])]

    use_mad_val = fetch('use_mad')
    if use_mad_val is not None:
        args.use_mad = '1' if use_mad_val else '0'
    use_velo_median_val = fetch('use_velocity_median')
    if use_velo_median_val is not None:
        args.use_velocity_median = 1 if use_velo_median_val else 0
    early_stop_val = fetch('early_stopping')
    if early_stop_val is not None:
        args.early_stopping = 1 if early_stop_val else 0

    formatted = _format_norm_overrides(model_cfg.get('norm_override'))
    if formatted:
        args.norm_override = formatted

    schema_section = model_cfg.get('schema')
    if schema_section:
        if isinstance(schema_section, dict):
            stations_map = schema_section.get('stations')
            if stations_map is None or isinstance(stations_map, dict):
                inline_schema = stations_map if stations_map is not None else schema_section
            else:
                logger.error('Model config schema section must provide a stations mapping')
                sys.exit(1)
        else:
            logger.error('Model config schema section must be a mapping')
            sys.exit(1)

    return inline_schema


def build_config(args):
    """Validate and enrich parsed args into a configuration object."""

    inline_schema = None
    model_config_path = getattr(args, 'model_config', None)
    if model_config_path:
        args.model_config = model_config_path
        model_cfg = _load_model_config(model_config_path)
        inline_schema = _apply_model_config(args, model_cfg)
        logger.info(f"Loaded model configuration from {model_config_path}")

    if inline_schema:
        args.station_schema = inline_schema
    else:
        logger.error('Model config must include an inline "schema" mapping with station definitions.')
        sys.exit(1)

    stations_raw = getattr(args, 'stations', None) or ''
    args.station_names = [s.strip() for s in stations_raw.split(';') if s.strip()]
    if len(args.station_names) < 2:
        logger.error(
            'The station configuration must provide at least two station names.'
        )
        sys.exit(1)
    logger.info(f"Station names for feature extraction: {args.station_names}")

    if not hasattr(args, 'id_col') or args.id_col in (None, ''):
        args.id_col = 'location_id'

    # Fold index logging
    if args.fold_actual < 0:
        logger.info(
            f"Starting training in full-dataset mode (no cross-validation) [fold {args.fold_actual}]"
        )
    else:
        logger.info(f"Starting training for fold {args.fold_actual}")

    # Aggregation statistic parsing
    raw_agg = args.agg_stat
    valid_aggs = ['mean', 'median', 'max']
    matched = [a for a in valid_aggs if a in raw_agg.lower()]
    if len(matched) == 1:
        args.agg = matched[0]
    else:
        raise ValueError(
            f"Unsupported --agg_stat value '{raw_agg}'. Must contain one of {valid_aggs}."
        )
    logger.info(f"Aggregation statistic set to '{args.agg}'")

    # MAD feature flag and early stopping configuration
    raw_use_mad = args.use_mad
    if raw_use_mad is None:
        raw_use_mad = '0'
    elif isinstance(raw_use_mad, bool):
        raw_use_mad = '1' if raw_use_mad else '0'
    else:
        raw_use_mad = str(raw_use_mad)
    args.use_mad = raw_use_mad.lower() in ('1', 'true', 'yes')

    raw_use_velo_median = getattr(args, 'use_velocity_median', None)
    if raw_use_velo_median is None:
        args.use_velocity_median = False
    elif isinstance(raw_use_velo_median, str):
        args.use_velocity_median = raw_use_velo_median.lower() in ('1', 'true', 'yes')
    else:
        args.use_velocity_median = bool(raw_use_velo_median)
    args.use_early_stopping = (args.early_stopping == 1)
    args.save_error_data = (args.save_error_data == 1)
    logger.info(f"Using MAD features: {args.use_mad}")
    logger.info(f"Using median radial velocity features: {args.use_velocity_median}")
    logger.info(f"Using early stopping: {args.use_early_stopping} (patience={args.patience})")
    logger.info(f"Saving error data: {args.save_error_data}")

    # Range configuration for regression masking and classification head
    target_range = getattr(args, 'target_speed_range', None)
    if target_range and isinstance(target_range, (list, tuple)) and len(target_range) == 2:
        args.range_min = float(target_range[0])
        args.range_max = float(target_range[1])

    range_min = getattr(args, 'range_min', None)
    range_max = getattr(args, 'range_max', None)
    if range_min is None or range_max is None:
        logger.error('Model configuration must specify range_min and range_max for wind speed.')
        sys.exit(1)
    args.range_min = float(range_min)
    args.range_max = float(range_max)
    if args.range_min >= args.range_max:
        logger.error(
            f"Invalid wind speed range: range_min={args.range_min} must be less than range_max={args.range_max}."
        )
        sys.exit(1)

    if getattr(args, 'range_margin', None) is None:
        args.range_margin = 0.5
    args.range_margin = max(0.0, float(args.range_margin))

    if getattr(args, 'range_loss_weight', None) is None:
        args.range_loss_weight = 1.0
    args.range_loss_weight = float(args.range_loss_weight)

    if getattr(args, 'range_flag_threshold', None) is None:
        args.range_flag_threshold = 0.5
    args.range_flag_threshold = min(max(float(args.range_flag_threshold), 0.0), 1.0)

    args.range_class_labels = ['below', 'in', 'above']
    args.range_in_class_index = 1
    logger.info(
        "Configured wind speed range: %.2f to %.2f m/s (margin %.2f m/s, classification weight %.2f)",
        args.range_min,
        args.range_max,
        args.range_margin,
        args.range_loss_weight,
    )

    # Normalization mode validation
    raw_mode = getattr(args, 'normalization_mode', None)
    if raw_mode is None:
        args.normalization_mode = 'standard'
    else:
        normalized_mode = str(raw_mode).lower()
        if normalized_mode not in ('standard', 'robust'):
            logger.error(
                "Unsupported normalization mode '%s'. Choose 'standard' or 'robust'.",
                raw_mode,
            )
            sys.exit(1)
        args.normalization_mode = normalized_mode
    logger.info(f"Feature normalization mode: {args.normalization_mode}")

    # Normalization overrides parsing
    args.center_override = {}
    args.scale_override = {}
    if args.norm_override:
        for item in args.norm_override.split(';'):
            if '=' not in item or '.' not in item.split('=', 1)[0]:
                logger.error(
                    f"Invalid format for --norm-override entry '{item}'. "
                    "Expected 'feature.center=value' or 'feature.scale=value'."
                )
                sys.exit(1)
            left, val = item.split('=', 1)
            feature, param = left.split('.', 1)
            try:
                valf = float(val)
            except ValueError:
                logger.error(
                    f"Invalid value for normalization override '{item}'. "
                    "Center or scale must be numeric."
                )
                sys.exit(1)
            if param == 'center':
                args.center_override[feature] = valf
            elif param == 'scale':
                args.scale_override[feature] = valf
            else:
                logger.error(
                    f"Invalid parameter '{param}' for override '{item}'. Must be 'center' or 'scale'."
                )
                sys.exit(1)
    logger.info(f"Normalization center overrides: {args.center_override}")
    logger.info(f"Normalization scale overrides: {args.scale_override}")
    # ID column grouping for location-based metrics
    logger.info(f"ID column grouping for location-based metrics: {args.id_col}")

    # Normalize fine-tuning toggles and defaults
    if getattr(args, 'use_l2sp', None) is None:
        args.use_l2sp = 0
    # Coefficients may be None; coerce to 0.0 if unset
    if getattr(args, 'l2sp_backbone_lambda', None) is None:
        args.l2sp_backbone_lambda = 0.0
    if getattr(args, 'l2sp_heads_lambda', None) is None:
        args.l2sp_heads_lambda = 0.0
    if getattr(args, 'use_kd', None) is None:
        args.use_kd = 0
    if getattr(args, 'lambda_kd', None) is None:
        args.lambda_kd = 0.0
    else:
        args.lambda_kd = float(args.lambda_kd)
    if getattr(args, 'lambda_kd_reg', None) is None:
        args.lambda_kd_reg = float(args.lambda_kd)
    else:
        args.lambda_kd_reg = float(args.lambda_kd_reg)
    if getattr(args, 'lambda_kd_cls', None) is None:
        args.lambda_kd_cls = float(args.lambda_kd)
    else:
        args.lambda_kd_cls = float(args.lambda_kd_cls)
    if getattr(args, 'kd_temperature', None) is None:
        args.kd_temperature = 1.0
    else:
        args.kd_temperature = float(args.kd_temperature)
    if getattr(args, 'teacher_checkpoint', None) is not None:
        args.teacher_checkpoint = str(args.teacher_checkpoint)
    logger.info(
        "Knowledge distillation enabled: %s (λ_reg=%.3f, λ_cls=%.3f, τ=%.2f)",
        bool(args.use_kd),
        float(args.lambda_kd_reg),
        float(args.lambda_kd_cls),
        float(args.kd_temperature),
    )
    # Learning rates for fine-tuning param groups are optional; leave as None when absent
    if getattr(args, 'finetune_heads_lr', None) is not None:
        args.finetune_heads_lr = float(args.finetune_heads_lr)
    if getattr(args, 'finetune_backbone_lr', None) is not None:
        args.finetune_backbone_lr = float(args.finetune_backbone_lr)
    # Rehearsal defaults
    if getattr(args, 'rehearsal_fraction', None) is None:
        args.rehearsal_fraction = 0.0
    args.rehearsal_fraction = float(args.rehearsal_fraction)

    if getattr(args, 'seed', None) is not None:
        logger.info(f"Random seed set to {args.seed}")
    else:
        logger.info("Random seed not provided; using stochastic initialization.")

    return args
