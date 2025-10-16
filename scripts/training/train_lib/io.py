"""
I/O utilities for saving normalization parameters and script arguments.

Author: Juan Luis Herrera Cortijo <juan.luis.herrera.cortijo@gmail.com>
Created: 2025-10-16
Disclaimer: This script is provided as-is for research workflows and must be validated before any operational deployment.
"""

import os
import json
import logging

logger = logging.getLogger(__name__)


def save_initial_outputs(norm_params, config):
    """
    Save normalization parameters and script arguments to the output data directory.

    Args:
        norm_params (dict): Normalization metadata payload (mode, labels, centers, scales, diagnostics).
        config (argparse.Namespace): Configuration or arguments namespace.
    """
    output_data_dir = os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output/data')
    os.makedirs(output_data_dir, exist_ok=True)
    norm_out = os.path.join(output_data_dir, 'normalization_params.json')
    with open(norm_out, 'w') as f:
        json.dump(norm_params, f)
    logger.info(f"Saved normalization parameters JSON to {norm_out}")

    args_out = os.path.join(output_data_dir, 'script_args.json')
    with open(args_out, 'w') as f:
        json.dump(vars(config), f)
    logger.info(f"Saved script arguments JSON to {args_out}")
