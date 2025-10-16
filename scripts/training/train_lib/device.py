"""
Device selection for training (CUDA vs CPU).

Author: Juan Luis Herrera Cortijo <juan.luis.herrera.cortijo@gmail.com>
Created: 2025-10-16
Disclaimer: This script is provided as-is for research workflows and must be validated before any operational deployment.
"""

import torch


def select_device():
    """
    Determine whether to use a CUDA-enabled GPU or fall back to CPU.

    Returns:
        torch.device: Selected device for torch operations.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')