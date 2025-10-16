"""
Training loop with dynamic weight averaging, early stopping, and checkpoint management.

Author: Juan Luis Herrera Cortijo <juan.luis.herrera.cortijo@gmail.com>
Created: 2025-10-16
Disclaimer: This script is provided as-is for research workflows and must be validated before any operational deployment.
"""

import os
import logging
import math

import torch
import torch.nn.functional as F

from train_lib.metrics import evaluate_metrics_full, compute_combined_loss

logger = logging.getLogger(__name__)


def run_training_loop(
    model,
    optimizer,
    train_loader,
    val_loader,
    config,
    norm_params,
    start_epoch=1,
    device=None,
    rehearsal_loader=None,
    teacher_model=None,
):
    """
    Execute the training loop over epochs, applying DWA, early stopping, and checkpointing.

    Args:
        model (nn.Module): The neural network to train.
        optimizer (Optimizer): Optimizer for model parameters.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        config (argparse.Namespace): Configuration including epochs, patience, etc.
        norm_params (dict): Normalization parameters for saving.
        start_epoch (int): Epoch number to start from (for resumes).

    Returns:
        tuple: Final dynamic weights (w1, w2).
    """
    # Initialize DWA weights and history
    w1, w2 = 1.0, 1.0
    history_rmse = []
    history_angle = []

    classification_weight = float(getattr(config, 'range_loss_weight', 1.0))
    lambda_kd = float(getattr(config, 'lambda_kd', 0.0) or 0.0)
    lambda_kd_reg = float(getattr(config, 'lambda_kd_reg', None) or lambda_kd)
    lambda_kd_cls = float(getattr(config, 'lambda_kd_cls', None) or lambda_kd)
    tau = float(getattr(config, 'kd_temperature', 1.0) or 1.0)
    if tau <= 0.0:
        logger.warning(
            "Knowledge distillation temperature %.5f is non-positive; resetting to 1.0.",
            tau,
        )
        tau = 1.0
    tau_sq = tau * tau
    # Only enable distillation when explicitly requested and weighted; this
    # prevents accidental activation when users forget to specify lambda terms.
    kd_requested = (
        int(getattr(config, 'use_kd', 0)) == 1 and (lambda_kd_reg > 0.0 or lambda_kd_cls > 0.0)
    )
    kd_active = kd_requested and teacher_model is not None
    if kd_active:
        teacher_model.eval()
    elif kd_requested and teacher_model is None:
        logger.warning(
            "Knowledge distillation requested but teacher model is unavailable; skipping distillation terms."
        )
        lambda_kd_reg = 0.0
        lambda_kd_cls = 0.0
    best_combined = float('inf')
    best_epoch = 0
    no_improve_count = 0

    # Prepare directories for model and checkpoints
    model_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'model.pth')
    checkpoint_dir = '/opt/ml/checkpoints'

    # Save normalization params and script args before training
    output_data_dir = os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output/data')
    os.makedirs(output_data_dir, exist_ok=True)
    norm_out = os.path.join(output_data_dir, 'normalization_params.json')
    with open(norm_out, 'w') as f:
        import json

        json.dump(norm_params, f)
    logger.info(f"Saved normalization parameters JSON to {norm_out}")

    conditional_diag = (norm_params.get('diagnostics') or {}).get('conditional_normalization')
    if conditional_diag:
        audit_path = os.path.join(output_data_dir, 'conditional_normalization_audit.md')
        lines = ["# Conditional Normalization Audit", ""]
        mode = norm_params.get('mode', getattr(config, 'normalization_mode', 'standard'))
        lines.append(f"- Normalization mode: {mode}")
        min_samples = conditional_diag.get('min_samples')
        if min_samples is not None:
            lines.append(f"- Minimum samples per maintenance interval: {min_samples}")
        lines.append("")

        fallback_summary = conditional_diag.get('fallback_summary', {}) or {}
        def _format_summary(title, entries):
            lines.append(f"## {title}")
            if not entries:
                lines.append("- No fallbacks triggered.")
                return
            for entry in entries:
                station = entry.get('station', 'unknown')
                feature = entry.get('feature', 'unknown')
                counts = [
                    f"{key}: {value}"
                    for key, value in entry.items()
                    if key not in {'station', 'feature'} and value
                ]
                if not counts:
                    counts = ['direct: all samples']
                lines.append(f"- {station} · {feature} → {'; '.join(counts)}")
            lines.append("")

        _format_summary('Training Fallbacks', fallback_summary.get('train', []))
        _format_summary('Validation Fallbacks', fallback_summary.get('validation', []))

        stations_diag = conditional_diag.get('stations', {})
        if stations_diag:
            lines.append("## Interval Notes")
            for station in sorted(stations_diag.keys()):
                lines.append(f"### {station}")
                feature_diag = stations_diag[station]
                for feature in sorted(feature_diag.keys()):
                    details = feature_diag[feature]
                    intervals = []
                    for interval, info in sorted(details.get('intervals', {}).items()):
                        if info.get('source') == 'interval':
                            continue
                        source = info.get('source', 'global')
                        origin = info.get('source_interval')
                        descriptor = f"{interval} (count={info.get('count', 0)}) → {source}"
                        if origin:
                            descriptor += f" from {origin}"
                        intervals.append(descriptor)
                    if intervals:
                        lines.append(f"- {feature}: {', '.join(intervals)}")
                    else:
                        lines.append(f"- {feature}: all intervals rely on direct statistics")
                lines.append("")

        with open(audit_path, 'w', encoding='utf-8') as audit_file:
            audit_file.write('\n'.join(lines).strip() + '\n')
        logger.info(f"Wrote conditional normalization audit to {audit_path}")

    args_out = os.path.join(output_data_dir, 'script_args.json')
    with open(args_out, 'w') as f:
        import json

        json.dump(vars(config), f)
    logger.info(f"Saved script arguments JSON to {args_out}")

    model_config_payload = None
    model_config_path = getattr(config, 'model_config', None)
    if model_config_path and os.path.isfile(model_config_path):
        try:
            with open(model_config_path, 'r', encoding='utf-8') as handle:
                model_config_payload = handle.read()
        except OSError as exc:
            logger.warning("Unable to read model config at %s: %s", model_config_path, exc)

    # Training epochs
    # Rehearsal scheduling: maintain the configured rehearsal fraction exactly
    reh_fraction = float(getattr(config, 'rehearsal_fraction', 0.0) or 0.0)
    rehearsing_enabled = rehearsal_loader is not None and reh_fraction > 0.0
    if reh_fraction >= 1.0 and rehearsing_enabled:
        logger.warning(
            "Rehearsal fraction %.3f is >= 1.0; clamping to 0.99 to keep target batches in the schedule.",
            reh_fraction,
        )
        reh_fraction = 0.99
    reh_ratio = 0.0
    reh_accumulator = 0.0
    reh_iter = None
    if rehearsing_enabled:
        denom = max(1.0 - reh_fraction, 1e-8)
        reh_ratio = reh_fraction / denom
        reh_iter = iter(rehearsal_loader)

    for epoch in range(start_epoch, config.epochs + 1):
        model.train()
        logger.info(f"Epoch {epoch}/{config.epochs} start")
        train_speed_sq_error = 0.0
        train_angle_error = 0.0
        train_angle_sq_error = 0.0
        train_count = 0
        train_class_loss = 0.0
        train_class_correct = 0.0
        train_class_total = 0
        train_kd_reg_sum = 0.0
        train_kd_cls_sum = 0.0
        kd_reg_steps = 0
        kd_cls_steps = 0

        for step, (batch_X, batch_y) in enumerate(train_loader, start=1):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            regression, logits = model(batch_X)
            pred_speed = regression[:, 0]
            pred_cos = regression[:, 1]
            pred_sin = regression[:, 2]
            true_speed = batch_y[:, 0]
            true_cos = batch_y[:, 1]
            true_sin = batch_y[:, 2]
            true_range_class = batch_y[:, 3].long()
            in_range_mask = batch_y[:, 4] > 0.5

            mask_sum = in_range_mask.sum()
            if mask_sum.item() > 0:
                valid_speed_err = (pred_speed[in_range_mask] - true_speed[in_range_mask]) ** 2
                speed_mse_loss = valid_speed_err.mean()
                pred_angle = torch.atan2(pred_sin[in_range_mask], pred_cos[in_range_mask])
                true_angle = torch.atan2(true_sin[in_range_mask], true_cos[in_range_mask])
                angle_diff = torch.atan2(torch.sin(pred_angle - true_angle), torch.cos(pred_angle - true_angle))
                angle_loss = torch.mean(angle_diff ** 2)
            else:
                speed_mse_loss = torch.zeros(1, device=device, dtype=pred_speed.dtype)
                angle_loss = torch.zeros(1, device=device, dtype=pred_speed.dtype)
                angle_diff = None

            class_loss = F.cross_entropy(logits, true_range_class)

            kd_reg_loss = batch_X.new_tensor(0.0)
            kd_cls_loss = batch_X.new_tensor(0.0)
            if kd_active:
                with torch.no_grad():
                    teacher_reg, teacher_logits = teacher_model(batch_X)
                teacher_speed = teacher_reg[:, 0]
                teacher_cos = teacher_reg[:, 1]
                teacher_sin = teacher_reg[:, 2]
                teacher_probs_tau = torch.softmax(teacher_logits / tau, dim=1).detach()
                if mask_sum.item() > 0:
                    student_speed_valid = pred_speed[in_range_mask]
                    teacher_speed_valid = teacher_speed[in_range_mask]
                    speed_sq = (student_speed_valid - teacher_speed_valid).pow(2)
                    student_angle = torch.atan2(pred_sin[in_range_mask], pred_cos[in_range_mask])
                    teacher_angle = torch.atan2(teacher_sin[in_range_mask], teacher_cos[in_range_mask])
                    angle_gap = torch.atan2(
                        torch.sin(student_angle - teacher_angle),
                        torch.cos(student_angle - teacher_angle),
                    )
                    angle_sq = angle_gap.pow(2)
                    active_count = mask_sum.to(speed_sq.dtype).clamp_min(1.0)
                    coeff_speed = float(w1)
                    coeff_angle = float(w2)
                    kd_reg_loss = (coeff_speed * speed_sq.sum() + coeff_angle * angle_sq.sum()) / active_count
                    train_kd_reg_sum += kd_reg_loss.item()
                    kd_reg_steps += 1
                kd_cls_loss = tau_sq * F.kl_div(
                    F.log_softmax(logits / tau, dim=1),
                    teacher_probs_tau,
                    reduction='batchmean',
                )
                train_kd_cls_sum += kd_cls_loss.item()
                kd_cls_steps += 1

            total_loss = (w1 * speed_mse_loss.squeeze()) + (w2 * angle_loss.squeeze()) + (
                config.range_loss_weight * class_loss
            )
            if kd_active:
                if lambda_kd_reg > 0.0:
                    total_loss = total_loss + lambda_kd_reg * kd_reg_loss
                if lambda_kd_cls > 0.0:
                    total_loss = total_loss + lambda_kd_cls * kd_cls_loss

            # Optional L2-SP penalty (only active during fine-tuning when enabled)
            if getattr(config, 'use_l2sp', 0) == 1 and hasattr(model, '_anchor_state_dict'):
                bb_lambda = float(getattr(config, 'l2sp_backbone_lambda', 0.0) or 0.0)
                hd_lambda = float(getattr(config, 'l2sp_heads_lambda', 0.0) or 0.0)
                if bb_lambda > 0.0 or hd_lambda > 0.0:
                    anchor = model._anchor_state_dict
                    l2sp_penalty = batch_X.new_tensor(0.0)
                    for name, param in model.named_parameters():
                        if not param.requires_grad:
                            continue
                        anchor_param = anchor.get(name)
                        if anchor_param is None:
                            continue
                        # Select coefficient based on parameter group (heads vs backbone)
                        coef = hd_lambda if ('speed_head' in name or 'direction_head' in name or 'range_head' in name) else bb_lambda
                        if coef <= 0.0:
                            continue
                        diff = param - anchor_param.to(param.device)
                        l2sp_penalty = l2sp_penalty + coef * (diff.pow(2).sum())
                    total_loss = total_loss + l2sp_penalty
            # Optimise on the primary batch before optionally injecting a
            # rehearsal step; rehearsal has its own zero_grad / backward cycle
            # to keep the gradients disentangled.
            total_loss.backward()
            optimizer.step()
            # Optional rehearsal update
            if reh_iter is not None and reh_ratio > 0.0:
                reh_accumulator += reh_ratio
                while reh_accumulator >= 1.0:
                    reh_accumulator -= 1.0
                    try:
                        reh_X, reh_y = next(reh_iter)
                    except StopIteration:
                        reh_iter = iter(rehearsal_loader)
                        reh_X, reh_y = next(reh_iter)
                    reh_X, reh_y = reh_X.to(device), reh_y.to(device)
                    optimizer.zero_grad()
                    reh_reg, reh_logits = model(reh_X)
                    r_speed = reh_reg[:, 0]
                    r_cos = reh_reg[:, 1]
                    r_sin = reh_reg[:, 2]
                    t_speed = reh_y[:, 0]
                    t_cos = reh_y[:, 1]
                    t_sin = reh_y[:, 2]
                    t_class = reh_y[:, 3].long()
                    r_mask = reh_y[:, 4] > 0.5
                    r_mask_sum = r_mask.sum()
                    if r_mask_sum.item() > 0:
                        v_speed_err = (r_speed[r_mask] - t_speed[r_mask]) ** 2
                        r_speed_mse = v_speed_err.mean()
                        r_pred_ang = torch.atan2(r_sin[r_mask], r_cos[r_mask])
                        r_true_ang = torch.atan2(t_sin[r_mask], t_cos[r_mask])
                        r_ang_diff = torch.atan2(torch.sin(r_pred_ang - r_true_ang), torch.cos(r_pred_ang - r_true_ang))
                        r_ang_loss = torch.mean(r_ang_diff ** 2)
                    else:
                        r_speed_mse = torch.zeros(1, device=device, dtype=r_speed.dtype)
                        r_ang_loss = torch.zeros(1, device=device, dtype=r_speed.dtype)
                    r_class_loss = F.cross_entropy(reh_logits, t_class)
                    reh_kd_reg = reh_X.new_tensor(0.0)
                    reh_kd_cls = reh_X.new_tensor(0.0)
                    if kd_active:
                        with torch.no_grad():
                            teacher_reh_reg, teacher_reh_logits = teacher_model(reh_X)
                        if r_mask_sum.item() > 0:
                            reh_teacher_speed = teacher_reh_reg[:, 0]
                            reh_teacher_cos = teacher_reh_reg[:, 1]
                            reh_teacher_sin = teacher_reh_reg[:, 2]
                            speed_sq = (r_speed[r_mask] - reh_teacher_speed[r_mask]).pow(2)
                            student_angle = torch.atan2(r_sin[r_mask], r_cos[r_mask])
                            teacher_angle = torch.atan2(reh_teacher_sin[r_mask], reh_teacher_cos[r_mask])
                            angle_gap = torch.atan2(
                                torch.sin(student_angle - teacher_angle),
                                torch.cos(student_angle - teacher_angle),
                            )
                            angle_sq = angle_gap.pow(2)
                            active_count = r_mask_sum.to(speed_sq.dtype).clamp_min(1.0)
                            coeff_speed = float(w1)
                            coeff_angle = float(w2)
                            reh_kd_reg = (coeff_speed * speed_sq.sum() + coeff_angle * angle_sq.sum()) / active_count
                            train_kd_reg_sum += reh_kd_reg.item()
                            kd_reg_steps += 1
                        reh_teacher_probs_tau = torch.softmax(teacher_reh_logits / tau, dim=1).detach()
                        reh_kd_cls = tau_sq * F.kl_div(
                            F.log_softmax(reh_logits / tau, dim=1),
                            reh_teacher_probs_tau,
                            reduction='batchmean',
                        )
                        train_kd_cls_sum += reh_kd_cls.item()
                        kd_cls_steps += 1

                    r_total = (w1 * r_speed_mse.squeeze()) + (w2 * r_ang_loss.squeeze()) + (config.range_loss_weight * r_class_loss)
                    if kd_active:
                        if lambda_kd_reg > 0.0:
                            r_total = r_total + lambda_kd_reg * reh_kd_reg
                        if lambda_kd_cls > 0.0:
                            r_total = r_total + lambda_kd_cls * reh_kd_cls
                    # L2-SP penalty also applies on rehearsal steps if active
                    if getattr(config, 'use_l2sp', 0) == 1 and hasattr(model, '_anchor_state_dict'):
                        bb_lambda = float(getattr(config, 'l2sp_backbone_lambda', 0.0) or 0.0)
                        hd_lambda = float(getattr(config, 'l2sp_heads_lambda', 0.0) or 0.0)
                        if bb_lambda > 0.0 or hd_lambda > 0.0:
                            anchor = model._anchor_state_dict
                            l2sp_term = reh_X.new_tensor(0.0)
                            for name, param in model.named_parameters():
                                if not param.requires_grad:
                                    continue
                                a = anchor.get(name)
                                if a is None:
                                    continue
                                coef = hd_lambda if ('speed_head' in name or 'direction_head' in name or 'range_head' in name) else bb_lambda
                                if coef <= 0.0:
                                    continue
                                diff = param - a.to(param.device)
                                l2sp_term = l2sp_term + coef * (diff.pow(2).sum())
                            r_total = r_total + l2sp_term
                    r_total.backward()
                    optimizer.step()
            if mask_sum.item() > 0:
                train_speed_sq_error += valid_speed_err.sum().item()
                train_angle_error += torch.abs(angle_diff).sum().item()
                train_angle_sq_error += (angle_diff ** 2).sum().item()
                train_count += mask_sum.item()
            train_class_loss += class_loss.item() * len(batch_X)
            train_class_correct += (logits.argmax(dim=1) == true_range_class).sum().item()
            train_class_total += len(batch_X)

        train_rmse = math.sqrt(train_speed_sq_error / train_count) if train_count else 0.0
        train_mean_angle_error = (train_angle_error / train_count) if train_count else 0.0
        avg_class_loss = (train_class_loss / train_class_total) if train_class_total else 0.0
        train_class_acc = (train_class_correct / train_class_total) if train_class_total else 0.0
        avg_kd_reg = (train_kd_reg_sum / kd_reg_steps) if kd_reg_steps else 0.0
        avg_kd_cls = (train_kd_cls_sum / kd_cls_steps) if kd_cls_steps else 0.0

        if epoch % 100 == 0 or epoch == config.epochs:
            logger.info(
                "Training - Epoch %d/%d: RMSE=%.4f, MeanAngleError=%.2f°, ClassLoss=%.4f, ClassAcc=%.3f, w1=%.3f, w2=%.3f",
                epoch,
                config.epochs,
                train_rmse,
                math.degrees(train_mean_angle_error),
                avg_class_loss,
                train_class_acc,
                w1,
                w2,
            )
            if kd_active:
                logger.info(
                    "Training - Epoch %d/%d: KD_reg=%.6f (λ_reg=%.3f) | KD_cls=%.6f (λ_cls=%.3f, τ=%.2f)",
                    epoch,
                    config.epochs,
                    avg_kd_reg,
                    lambda_kd_reg,
                    avg_kd_cls,
                    lambda_kd_cls,
                    tau,
                )
        history_rmse.append(train_speed_sq_error / train_count if train_count else 0.0)
        history_angle.append(train_angle_sq_error / train_count if train_count else 0.0)

        # Early stopping based on validation
        if config.use_early_stopping:
            metrics = evaluate_metrics_full(model, val_loader, device)
            combined_val, macro_f1, penalty, val_speed_mse, val_angle_mse = compute_combined_loss(
                metrics,
                w1,
                w2,
                classification_weight,
            )
            if combined_val < best_combined:
                best_combined = combined_val
                best_epoch = epoch
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'normalization_params': norm_params,
                    'args': vars(config),
                    'model_config_payload': model_config_payload,
                }, model_path)
                macro_f1_msg = f"{macro_f1:.3f}" if macro_f1 is not None else "n/a"
                logger.info(
                    "Epoch %d: Improved CombinedLoss to %.4f (speed_mse=%.4f, angle_mse=%.4f, "
                    "macro_F1=%s, penalty=%.4f). Model saved.",
                    epoch,
                    combined_val,
                    val_speed_mse,
                    val_angle_mse,
                    macro_f1_msg,
                    penalty,
                )
                if 'range_classification' in metrics:
                    acc = metrics['range_classification'].get('accuracy', 0.0)
                    logger.info(f"Validation range classification accuracy: {acc:.3f}")
                no_improve_count = 0
            else:
                no_improve_count += 1
                logger.info(
                    f"Epoch {epoch}: No improvement ({no_improve_count}/{config.patience})."
                )
            if no_improve_count >= config.patience:
                logger.info(
                    f"Early stopping at epoch {epoch}. Best epoch was {best_epoch}."
                )
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'normalization_params': norm_params,
                    'model_config_payload': model_config_payload,
                }, os.path.join(checkpoint_dir, 'checkpoint.pth'))
                logger.info(f"Saved checkpoint to {checkpoint_dir}/checkpoint.pth")
                break

        # Update DWA weights
        if len(history_rmse) >= 2 and epoch < config.epochs:
            prev_rmse = history_rmse[-2] if history_rmse[-2] > 0 else 1e-8
            prev_angle = history_angle[-2] if history_angle[-2] > 0 else 1e-8
            r1 = history_rmse[-1] / prev_rmse
            r2 = history_angle[-1] / prev_angle
            T = 2.0
            exp1 = math.exp(r1 / T)
            exp2 = math.exp(r2 / T)
            w1 = 2 * exp1 / (exp1 + exp2)
            w2 = 2 * exp2 / (exp1 + exp2)

        # Periodic checkpoint
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'normalization_params': norm_params,
            'model_config_payload': model_config_payload,
        }, os.path.join(checkpoint_dir, 'checkpoint.pth'))
        logger.info(f"Saved checkpoint to {checkpoint_dir}/checkpoint.pth")

    return w1, w2
