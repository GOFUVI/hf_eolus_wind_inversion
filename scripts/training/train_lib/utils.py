"""
Utility functions for training script, including logging configuration.

Author: Juan Luis Herrera Cortijo <juan.luis.herrera.cortijo@gmail.com>
Created: 2025-10-16
Disclaimer: This script is provided as-is for research workflows and must be validated before any operational deployment.
"""

import logging


def configure_logging():
    """
    Configure root logger and suppress verbose output from AWS SDKs.
    """
    logging.basicConfig(level=logging.DEBUG, format="%(message)s")
    # Suppress verbose logs from SageMaker toolkit and AWS SDK
    for lib in ["sagemaker-training-toolkit", "boto3", "botocore", "urllib3"]:
        logging.getLogger(lib).setLevel(logging.WARNING)