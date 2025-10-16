"""
Compute evaluation metrics for wind speed and direction.

Author: Juan Luis Herrera Cortijo <juan.luis.herrera.cortijo@gmail.com>
Created: 2025-10-16
Disclaimer: This script is provided as-is for research workflows and must be validated before any operational deployment.
"""

import math
from typing import Dict, Optional, Tuple

import torch


def evaluate_metrics_full(model, data_loader, device):
    """
    Run the model on the provided data_loader and compute speed and direction metrics.

    Args:
        model (torch.nn.Module): Trained model.
        data_loader (torch.utils.data.DataLoader): DataLoader for evaluation data.
        device (torch.device): Device to perform computation on.

    Returns:
        dict: Metrics including rmse, mae_speed, corr_speed, r2_speed, bias_speed,
              si_speed, si_speed_max, eam_dir, eaam_dir, rmse_dir, compcorr_dir.
    """
    model.eval()
    # Initialize accumulators for speed metrics
    total_sq_error = 0.0
    total_abs_error = 0.0
    sum_pred = 0.0
    sum_true = 0.0
    sum_pred_sq = 0.0
    sum_true_sq = 0.0
    sum_prod = 0.0
    sum_error = 0.0
    sum_error_sq = 0.0
    # Direction metrics accumulators
    total_angle_diff = 0.0
    total_angle_error = 0.0
    total_angle_error_sq = 0.0
    real_sum = 0.0
    imag_sum = 0.0
    count = 0
    max_true_speed = float('-inf')
    # Range classification accumulators
    total_class_correct = 0.0
    total_class_samples = 0.0
    per_class_correct = None
    per_class_true = None
    per_class_pred = None
    range_labels = ['below', 'in', 'above']

    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_X)
            if isinstance(outputs, tuple):
                regression, logits = outputs
            else:
                regression, logits = outputs, None

            if logits is not None:
                true_classes = batch_y[:, 3].long()
                pred_classes = logits.argmax(dim=1)
                total_class_correct += (pred_classes == true_classes).sum().item()
                total_class_samples += len(true_classes)
                if per_class_correct is None:
                    num_classes = logits.shape[1]
                    per_class_correct = torch.zeros(num_classes, dtype=torch.float64)
                    per_class_true = torch.zeros(num_classes, dtype=torch.float64)
                    per_class_pred = torch.zeros(num_classes, dtype=torch.float64)
                for idx in range(per_class_correct.shape[0]):
                    true_mask = true_classes == idx
                    pred_mask = pred_classes == idx
                    tp = (true_mask & pred_mask).sum().item()
                    per_class_correct[idx] += tp
                    per_class_true[idx] += true_mask.sum().item()
                    per_class_pred[idx] += pred_mask.sum().item()

            mask = batch_y[:, 4] > 0.5
            if not mask.any():
                continue

            pred_speed = regression[:, 0][mask]
            pred_cos = regression[:, 1][mask]
            pred_sin = regression[:, 2][mask]
            true_speed = batch_y[:, 0][mask]
            true_cos = batch_y[:, 1][mask]
            true_sin = batch_y[:, 2][mask]

            max_true_speed = max(max_true_speed, true_speed.max().item())

            error_speed = pred_speed - true_speed
            sq_error_speed = error_speed ** 2
            abs_error_speed = torch.abs(error_speed)
            total_sq_error += sq_error_speed.sum().item()
            total_abs_error += abs_error_speed.sum().item()
            sum_pred += pred_speed.sum().item()
            sum_true += true_speed.sum().item()
            sum_pred_sq += (pred_speed ** 2).sum().item()
            sum_true_sq += (true_speed ** 2).sum().item()
            sum_prod += (pred_speed * true_speed).sum().item()
            sum_error += error_speed.sum().item()
            sum_error_sq += sq_error_speed.sum().item()

            pred_angle = torch.atan2(pred_sin, pred_cos)
            true_angle = torch.atan2(true_sin, true_cos)
            angle_diff = torch.atan2(torch.sin(pred_angle - true_angle), torch.cos(pred_angle - true_angle))
            angle_error = torch.abs(angle_diff)
            total_angle_diff += angle_diff.sum().item()
            total_angle_error += angle_error.sum().item()
            total_angle_error_sq += (angle_diff ** 2).sum().item()
            real_sum += (pred_cos * true_cos + pred_sin * true_sin).sum().item()
            imag_sum += (pred_sin * true_cos - pred_cos * true_sin).sum().item()
            count += mask.sum().item()
    metrics = {
        'rmse': 0.0,
        'mae_speed': 0.0,
        'corr_speed': 0.0,
        'r2_speed': 0.0,
        'bias_speed': 0.0,
        'si_speed': 0.0,
        'si_speed_max': 0.0,
        'eam_dir': 0.0,
        'eaam_dir': 0.0,
        'rmse_dir': 0.0,
        'compcorr_dir': 0.0,
    }

    if count > 0:
        rmse = math.sqrt(total_sq_error / count)
        mae_speed = total_abs_error / count
        mean_true = sum_true / count
        var_pred = sum_pred_sq - (sum_pred ** 2) / count
        var_true = sum_true_sq - (sum_true ** 2) / count
        cov = sum_prod - (sum_pred * sum_true) / count
        corr_speed = cov / math.sqrt(var_pred * var_true) if var_pred > 1e-12 and var_true > 1e-12 else 0.0
        ss_res = total_sq_error
        ss_tot = var_true
        r2_speed = 1 - (ss_res / ss_tot) if ss_tot > 1e-12 else (1.0 if ss_res <= 1e-12 else 0.0)
        bias_speed = sum_error / count
        mean_error_sq = total_sq_error / count
        var_error = max(mean_error_sq - (sum_error / count) ** 2, 0.0)
        std_error = math.sqrt(var_error)
        si_speed = std_error / mean_true if abs(mean_true) > 1e-12 else 0.0
        si_speed_max = std_error / max_true_speed if abs(max_true_speed) > 1e-12 else 0.0
        mean_angle_error_rad = total_angle_error / count
        mean_angle_diff_rad = total_angle_diff / count
        eam_dir = math.degrees(mean_angle_diff_rad)
        eaam_dir = math.degrees(mean_angle_error_rad)
        rmse_angle_rad = math.sqrt(total_angle_error_sq / count)
        rmse_dir = math.degrees(rmse_angle_rad)
        compcorr_dir = math.sqrt(real_sum ** 2 + imag_sum ** 2) / count

        metrics.update({
            'rmse': rmse,
            'mae_speed': mae_speed,
            'corr_speed': corr_speed,
            'r2_speed': r2_speed,
            'bias_speed': bias_speed,
            'si_speed': si_speed,
            'si_speed_max': si_speed_max,
            'eam_dir': eam_dir,
            'eaam_dir': eaam_dir,
            'rmse_dir': rmse_dir,
            'compcorr_dir': compcorr_dir,
        })

    if total_class_samples > 0 and per_class_correct is not None:
        per_class_stats = {}
        macro_precision_sum = 0.0
        macro_recall_sum = 0.0
        macro_f1_sum = 0.0
        macro_count = 0
        for idx in range(per_class_correct.shape[0]):
            label = range_labels[idx] if idx < len(range_labels) else f'class_{idx}'
            support = per_class_true[idx].item()
            predicted = per_class_pred[idx].item()
            tp = per_class_correct[idx].item()
            precision = tp / predicted if predicted > 0 else 0.0
            recall = tp / support if support > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            if support > 0:
                macro_count += 1
                macro_precision_sum += precision
                macro_recall_sum += recall
                macro_f1_sum += f1
            per_class_stats[label] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': support,
                'predicted': predicted,
                'true_positives': tp,
            }

        macro_precision = macro_precision_sum / macro_count if macro_count else 0.0
        macro_recall = macro_recall_sum / macro_count if macro_count else 0.0
        macro_f1 = macro_f1_sum / macro_count if macro_count else 0.0

        metrics['range_classification'] = {
            'accuracy': total_class_correct / total_class_samples,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'per_class': per_class_stats,
            'support': total_class_samples,
        }

    return metrics


def compute_combined_loss(
    metrics: Dict[str, float],
    w_speed: float,
    w_angle: float,
    classification_weight: float = 1.0,
) -> Tuple[float, Optional[float], float, float, float]:
    """Compute the CombinedLoss consistent with training and reporting.

    Args:
        metrics (dict): Output of :func:`evaluate_metrics_full`.
        w_speed (float): Dynamic weight applied to the speed MSE term.
        w_angle (float): Dynamic weight applied to the angle MSE term.
        classification_weight (float): Multiplier for the macro-F1 penalty.

    Returns:
        tuple: ``(combined_loss, macro_f1, penalty, speed_mse, angle_mse)`` where
        ``penalty`` equals ``classification_weight * (1 - macro_f1)`` when
        available.
    """
    rmse_speed = float(metrics.get('rmse', 0.0))
    rmse_dir_deg = float(metrics.get('rmse_dir', 0.0))
    speed_mse = rmse_speed ** 2
    angle_mse = math.radians(rmse_dir_deg) ** 2

    macro_f1: Optional[float] = None
    penalty = 0.0
    range_metrics = metrics.get('range_classification')
    if range_metrics:
        raw_macro_f1 = range_metrics.get('macro_f1')
        if raw_macro_f1 is not None:
            macro_f1 = max(0.0, min(1.0, float(raw_macro_f1)))
            weight = max(0.0, float(classification_weight))
            penalty = weight * (1.0 - macro_f1)

    combined = (float(w_speed) * speed_mse) + (float(w_angle) * angle_mse) + penalty
    return combined, macro_f1, penalty, speed_mse, angle_mse
