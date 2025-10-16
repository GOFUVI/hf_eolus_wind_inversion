"""
Model definition and optimizer setup, including optional checkpoint loading for fine-tuning.

Author: Juan Luis Herrera Cortijo <juan.luis.herrera.cortijo@gmail.com>
Created: 2025-10-16
Disclaimer: This script is provided as-is for research workflows and must be validated before any operational deployment.
"""

import os
import logging

import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)


class RangeAwareMLP(nn.Module):
    """Shared MLP backbone with separate heads for regression and range classification."""

    def __init__(self, input_dim, hidden_layers, hidden_units, drop_rate=0.0, num_classes=3):
        super().__init__()
        backbone_layers = []
        in_dim = input_dim
        for _ in range(hidden_layers):
            backbone_layers.append(nn.Linear(in_dim, hidden_units))
            backbone_layers.append(nn.ReLU(inplace=True))
            if drop_rate > 0:
                backbone_layers.append(nn.Dropout(drop_rate))
            in_dim = hidden_units
        if backbone_layers:
            self.backbone = nn.Sequential(*backbone_layers)
            feature_dim = hidden_units
        else:
            self.backbone = nn.Identity()
            feature_dim = input_dim

        self.speed_head = nn.Linear(feature_dim, 1)
        self.direction_head = nn.Linear(feature_dim, 2)
        self.range_head = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        """Return regression targets (speed, sin/cos) and range logits."""
        features = self.backbone(x)
        speed = self.speed_head(features)
        direction_raw = self.direction_head(features)
        direction = torch.tanh(direction_raw)
        logits = self.range_head(features)
        regression = torch.cat([speed, direction], dim=1)
        return regression, logits


def build_model_and_optimizer(train_df, val_df, feature_cols, target_cols, config, device):
    """
    Instantiate the MLP model, optimizer, optionally load checkpoint for fine-tuning,
    and prepare DataLoaders for training and validation.

    Args:
        train_df, val_df (pd.DataFrame): Normalized feature DataFrames.
        feature_cols (list[str]): Input feature column names.
        target_cols (list[str]): Output target column names.
        config (argparse.Namespace): Configuration with hidden layers, lr, etc.
        device (torch.device): Device for model.

    Returns:
        tuple: (model, optimizer, train_loader, val_loader, start_epoch)
    """
    # Convert DataFrames to tensors
    X_train = torch.tensor(train_df[feature_cols].values, dtype=torch.float32)
    y_train = torch.tensor(train_df[target_cols].values, dtype=torch.float32)
    X_val = torch.tensor(val_df[feature_cols].values, dtype=torch.float32)
    y_val = torch.tensor(val_df[target_cols].values, dtype=torch.float32)
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False
    )

    input_dim = X_train.shape[1]
    num_classes = len(getattr(config, 'range_class_labels', ['below', 'in', 'above']))

    # Instantiate model and optional teacher
    model = RangeAwareMLP(
        input_dim,
        config.hidden_layers,
        config.hidden_units,
        config.dropout,
        num_classes=num_classes,
    ).to(device)
    logger.info(f"Neural network structure:\n{model}")
    teacher_model = None

    # Optional checkpoint loading for fine-tuning
    checkpoint_dir = '/opt/ml/checkpoints'
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pth')
    start_epoch = 1
    if os.path.exists(checkpoint_path):
        logger.info(f"Found existing checkpoint at {checkpoint_path}, loading for fine-tuning.")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
        if 'optimizer_state_dict' in checkpoint:
            # Intentionally ignore optimizer state for controlled fine-tuning
            logger.info("Ignoring optimizer state from checkpoint to apply fine-tuning param groups.")
        else:
            logger.warning(
                "Checkpoint %s does not contain optimizer state; optimizer will start from scratch",
                checkpoint_path,
            )
        start_epoch = checkpoint.get('epoch', 0) + 1
        logger.info(f"Resuming training from epoch {start_epoch}.")
        # Freeze all layers except last hidden and output for fine-tuning
        for param in model.parameters():
            param.requires_grad = False
        if isinstance(model.backbone, nn.Sequential) and model.backbone:
            linear_layers = [module for module in model.backbone if isinstance(module, nn.Linear)]
            if linear_layers:
                for param in linear_layers[-1].parameters():
                    param.requires_grad = True
        for head in (model.speed_head, model.direction_head, model.range_head):
            for param in head.parameters():
                param.requires_grad = True
        logger.info("Enabled fine-tuning for the last hidden layer and all task-specific heads.")

        # Anchor state for L2-SP regularization (kept on CPU; moved per-parameter on demand)
        try:
            from copy import deepcopy
            model._anchor_state_dict = deepcopy(model.state_dict())
            logger.info("Captured anchor weights for L2-SP regularization.")
        except Exception as exc:
            logger.warning("Unable to snapshot anchor weights for L2-SP: %s", exc)

        configured_weight_decay = float(getattr(config, 'weight_decay', 0.0) or 0.0)
        if getattr(config, 'use_l2sp', 0) == 1 and configured_weight_decay > 0.0:
            logger.info(
                "L2-SP is enabled; overriding weight decay from %.6f to 0 to keep the anchor optimum unchanged.",
                configured_weight_decay,
            )
            configured_weight_decay = 0.0

        # Build optimizer with discriminative learning rates for fine-tuning
        param_groups = []
        heads_params = list(model.speed_head.parameters()) + list(model.direction_head.parameters()) + list(model.range_head.parameters())
        heads_lr = getattr(config, 'finetune_heads_lr', None) or config.lr
        if heads_params:
            param_groups.append({'params': heads_params, 'lr': float(heads_lr), 'weight_decay': configured_weight_decay})
        last_linear = None
        if isinstance(model.backbone, nn.Sequential) and model.backbone:
            for m in reversed(model.backbone):
                if isinstance(m, nn.Linear):
                    last_linear = m
                    break
        if last_linear is not None:
            bb_lr = getattr(config, 'finetune_backbone_lr', None) or config.lr
            param_groups.append({'params': last_linear.parameters(), 'lr': float(bb_lr), 'weight_decay': configured_weight_decay})
        if not param_groups:
            # Fallback to all trainable params if grouping failed
            param_groups = [{'params': [p for p in model.parameters() if p.requires_grad], 'lr': config.lr, 'weight_decay': configured_weight_decay}]
        optimizer = optim.Adam(param_groups)
    else:
        logger.info("No checkpoint found, starting training from scratch.")
        configured_weight_decay = float(getattr(config, 'weight_decay', 0.0) or 0.0)
        if getattr(config, 'use_l2sp', 0) == 1 and configured_weight_decay > 0.0:
            logger.info(
                "L2-SP requested without checkpoint; disabling weight decay to avoid conflicting anchors."
            )
            configured_weight_decay = 0.0
        optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=configured_weight_decay)

    if int(getattr(config, 'use_kd', 0)) == 1:
        try:
            teacher_model = RangeAwareMLP(
                input_dim,
                config.hidden_layers,
                config.hidden_units,
                config.dropout,
                num_classes=num_classes,
            ).to(device)
            anchor_state = getattr(model, '_anchor_state_dict', None)
            if anchor_state is None:
                anchor_state = model.state_dict()
                logger.warning(
                    "Knowledge distillation requested but anchor weights are missing; using current model weights as teacher snapshot."
                )
            teacher_model.load_state_dict(anchor_state)
            teacher_model.eval()
            for param in teacher_model.parameters():
                param.requires_grad = False
            logger.info("Teacher model instantiated for knowledge distillation.")
        except Exception as exc:
            logger.warning("Failed to instantiate teacher model for knowledge distillation: %s", exc)
            teacher_model = None

    return model, optimizer, train_loader, val_loader, start_epoch, teacher_model


MLP = RangeAwareMLP
