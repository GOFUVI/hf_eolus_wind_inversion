"""
Final evaluation and reporting: reload best model, compute metrics, and write CSV.

Author: Juan Luis Herrera Cortijo <juan.luis.herrera.cortijo@gmail.com>
Created: 2025-10-16
Disclaimer: This script is provided as-is for research workflows and must be validated before any operational deployment.
"""

import os
import logging

import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

from train_lib.metrics import evaluate_metrics_full, compute_combined_loss

logger = logging.getLogger(__name__)


def final_evaluation_and_report(
    model,
    train_df,
    val_df,
    feature_cols,
    target_cols,
    device,
    config,
    w1,
    w2,
):
    """
    Reload best model if early stopping, evaluate metrics globally and by wind_bin,
    and by id_col and id_col+wind_bin for both validation and training sets,
    writing CSVs to the SageMaker output directory.
    Print CombinedLoss (combined validation loss) to stdout for hyperparameter
    optimization metric collection.
    If wind_bin or id_col columns are absent, add them with default value 1.

    Args:
        model (nn.Module): Trained model.
    train_df (pd.DataFrame): Training DataFrame (normalized, with optional wind_bin).
    val_df (pd.DataFrame): Validation DataFrame (normalized, with optional wind_bin).
        feature_cols (list): Feature column names.
        target_cols (list): Target column names.
        device (torch.device): Device for evaluation.
        config (argparse.Namespace): Configuration with fold_actual and use_early_stopping.
        w1 (float): Final dynamic weight for speed.
        w2 (float): Final dynamic weight for direction.
    """
    # Ensure wind_bin column exists with default value 1 if absent
    for df in (train_df, val_df):
        if 'wind_bin' not in df.columns:
            df['wind_bin'] = 1
    # Ensure id column exists with default value 1 if absent
    id_col = config.id_col
    for df in (train_df, val_df):
        if id_col not in df.columns:
            df[id_col] = 1

    # Reload best model if early stopping
    model_path = os.path.join(os.environ.get('SM_MODEL_DIR', '/opt/ml/model'), 'model.pth')
    if config.use_early_stopping:
        ckpt = torch.load(model_path, map_location=device)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            model.load_state_dict(ckpt)
        logger.info(f"Loaded best model from {model_path} for final evaluation")

    # Compute full validation metrics via a one-batch DataLoader
    X_val = torch.tensor(val_df[feature_cols].values, dtype=torch.float32)
    y_val = torch.tensor(val_df[target_cols].values, dtype=torch.float32)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=len(val_df))
    metrics_val = evaluate_metrics_full(model, val_loader, device)
    range_metrics_val = metrics_val.get('range_classification')
    classification_weight = float(getattr(config, 'range_loss_weight', 1.0))
    combined_val, macro_f1_val, penalty_val, val_speed_mse, val_angle_mse = compute_combined_loss(
        metrics_val,
        w1,
        w2,
        classification_weight,
    )
    if macro_f1_val is not None:
        logger.info(
            "Validation macro F1: %.3f, classification penalty: %.4f (weight %.2f)",
            macro_f1_val,
            penalty_val,
            classification_weight,
        )
    val_rmse = metrics_val['rmse']
    val_rmse_dir = metrics_val['rmse_dir']

    # Log results
    if config.fold_actual < 0:
        logger.info(
            f"Fold {config.fold_actual} (full-dataset) validation RMSE: {val_rmse:.4f}, RMSE_dir: {val_rmse_dir:.2f}°"
        )
    else:
        logger.info(
            f"Fold {config.fold_actual} validation RMSE: {val_rmse:.4f}, RMSE_dir: {val_rmse_dir:.2f}°"
        )

    # Write global validation metrics
    output_data_dir = os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output/data')
    os.makedirs(output_data_dir, exist_ok=True)
    val_metrics_file = os.path.join(
        output_data_dir, f"metrics_fold{config.fold_actual}.csv"
    )
    with open(val_metrics_file, 'w') as f:
        f.write(
            "fold,rmse_speed,mae_speed,corr_speed,r2_speed,bias_speed,"
            "si_speed,eam_dir,eaam_dir,rmse_dir,compcorr_dir,si_speed_max,combined_loss\n"
        )
        f.write(
            f"{config.fold_actual},{metrics_val['rmse']:.6f},"
            f"{metrics_val['mae_speed']:.6f},"
            f"{metrics_val['corr_speed']:.6f},"
            f"{metrics_val['r2_speed']:.6f},"
            f"{metrics_val['bias_speed']:.6f},"
            f"{metrics_val['si_speed']:.6f},"
            f"{metrics_val['eam_dir']:.6f},"
            f"{metrics_val['eaam_dir']:.6f},"
            f"{metrics_val['rmse_dir']:.6f},"
            f"{metrics_val['compcorr_dir']:.6f},"
            f"{metrics_val['si_speed_max']:.6f},"
            f"{combined_val:.6f}\n"
        )
    logger.info(f"Wrote global validation metrics to {val_metrics_file}")

    def write_range_metrics(report: dict, filename: str) -> None:
        if not report:
            return
        rows = []
        for label, stats in report.get('per_class', {}).items():
            rows.append({
                'class': label,
                'precision': stats.get('precision', 0.0),
                'recall': stats.get('recall', 0.0),
                'f1': stats.get('f1', 0.0),
                'support': stats.get('support', 0.0),
                'predicted': stats.get('predicted', 0.0),
                'true_positives': stats.get('true_positives', 0.0),
                'accuracy': None,
            })
        rows.append({
            'class': 'overall',
            'precision': report.get('macro_precision', 0.0),
            'recall': report.get('macro_recall', 0.0),
            'f1': report.get('macro_f1', 0.0),
            'support': report.get('support', 0.0),
            'predicted': report.get('support', 0.0),
            'true_positives': report.get('accuracy', 0.0) * report.get('support', 0.0),
            'accuracy': report.get('accuracy', 0.0),
        })
        pd.DataFrame(rows).to_csv(filename, index=False)
        logger.info(f"Wrote range classification metrics to {filename}")

    if range_metrics_val:
        write_range_metrics(
            range_metrics_val,
            os.path.join(output_data_dir, f"metrics_fold{config.fold_actual}_range_classification.csv"),
        )

    # Compute and write global training metrics

    X_train = torch.tensor(train_df[feature_cols].values, dtype=torch.float32)
    y_train = torch.tensor(train_df[target_cols].values, dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=len(train_df))
    metrics_train = evaluate_metrics_full(model, train_loader, device)
    range_metrics_train = metrics_train.get('range_classification')
    combined_train, macro_f1_train, penalty_train, train_speed_mse, train_angle_mse = compute_combined_loss(
        metrics_train,
        w1,
        w2,
        classification_weight,
    )
    if macro_f1_train is not None:
        logger.info(
            "Training macro F1: %.3f, classification penalty: %.4f (weight %.2f)",
            macro_f1_train,
            penalty_train,
            classification_weight,
        )
    train_metrics_file = os.path.join(
        output_data_dir, f"metrics_train_fold{config.fold_actual}.csv"
    )
    with open(train_metrics_file, 'w') as f:
        f.write(
            "fold,rmse_speed,mae_speed,corr_speed,r2_speed,bias_speed,"
            "si_speed,eam_dir,eaam_dir,rmse_dir,compcorr_dir,si_speed_max,combined_loss\n"
        )
        f.write(
            f"{config.fold_actual},{metrics_train['rmse']:.6f},"
            f"{metrics_train['mae_speed']:.6f},"
            f"{metrics_train['corr_speed']:.6f},"
            f"{metrics_train['r2_speed']:.6f},"
            f"{metrics_train['bias_speed']:.6f},"
            f"{metrics_train['si_speed']:.6f},"
            f"{metrics_train['eam_dir']:.6f},"
            f"{metrics_train['eaam_dir']:.6f},"
            f"{metrics_train['rmse_dir']:.6f},"
            f"{metrics_train['compcorr_dir']:.6f},"
            f"{metrics_train['si_speed_max']:.6f},"
            f"{combined_train:.6f}\n"
        )
    logger.info(f"Wrote global training metrics to {train_metrics_file}")

    if range_metrics_train:
        write_range_metrics(
            range_metrics_train,
            os.path.join(output_data_dir, f"metrics_train_fold{config.fold_actual}_range_classification.csv"),
        )

    def strip_range_metrics(metrics_dict: dict) -> dict:
        return {k: v for k, v in metrics_dict.items() if k != 'range_classification'}

    # Compute and write per-wind-bin metrics for validation and training
    for df, base in [(val_df, 'metrics_fold'), (train_df, 'metrics_train_fold')]:
        bin_list = sorted(df['wind_bin'].unique())
        rows = []
        for b in bin_list:
            subset = df[df['wind_bin'] == b]
            if subset.empty:
                continue

            X_sub = torch.tensor(subset[feature_cols].values, dtype=torch.float32)
            y_sub = torch.tensor(subset[target_cols].values, dtype=torch.float32)
            loader = DataLoader(TensorDataset(X_sub, y_sub), batch_size=len(subset))
            m = evaluate_metrics_full(model, loader, device)
            rows.append({'wind_bin': b, **strip_range_metrics(m)})
        bin_df = pd.DataFrame(rows)
        bin_file = os.path.join(
            output_data_dir,
            f"{base}{config.fold_actual}_by_wind_bin.csv"
        )
        bin_df.to_csv(bin_file, index=False)
        logger.info(f"Wrote {base} metrics by wind_bin to {bin_file}")
    # Compute and write per-id metrics for validation and training
    for df, base in [(val_df, 'metrics_fold'), (train_df, 'metrics_train_fold')]:
        id_list = sorted(df[id_col].unique())
        rows = []
        for id_val in id_list:
            subset = df[df[id_col] == id_val]
            if subset.empty:
                continue

            X_sub = torch.tensor(subset[feature_cols].values, dtype=torch.float32)
            y_sub = torch.tensor(subset[target_cols].values, dtype=torch.float32)
            loader = DataLoader(TensorDataset(X_sub, y_sub), batch_size=len(subset))
            m = evaluate_metrics_full(model, loader, device)
            rows.append({id_col: id_val, **strip_range_metrics(m)})
        id_df = pd.DataFrame(rows)
        id_file = os.path.join(output_data_dir, f"{base}{config.fold_actual}_by_{id_col}.csv")
        id_df.to_csv(id_file, index=False)
        logger.info(f"Wrote {base} metrics by {id_col} to {id_file}")

    # Compute and write per-id and wind_bin metrics for validation and training
    for df, base in [(val_df, 'metrics_fold'), (train_df, 'metrics_train_fold')]:
        id_list = sorted(df[id_col].unique())
        bin_list = sorted(df['wind_bin'].unique())
        rows = []
        for id_val in id_list:
            for b in bin_list:
                subset = df[(df[id_col] == id_val) & (df['wind_bin'] == b)]
                if subset.empty:
                    continue

                X_sub = torch.tensor(subset[feature_cols].values, dtype=torch.float32)
                y_sub = torch.tensor(subset[target_cols].values, dtype=torch.float32)
                loader = DataLoader(TensorDataset(X_sub, y_sub), batch_size=len(subset))
                m = evaluate_metrics_full(model, loader, device)
                rows.append({id_col: id_val, "wind_bin": b, **strip_range_metrics(m)})
        id_bin_df = pd.DataFrame(rows)
        id_bin_file = os.path.join(output_data_dir, f"{base}{config.fold_actual}_by_{id_col}_by_wind_bin.csv")
        id_bin_df.to_csv(id_bin_file, index=False)
        logger.info(f"Wrote {base} metrics by {id_col} and wind_bin to {id_bin_file}")
    # Print combined validation loss for hyperparameter optimization metric collection
    print(f"CombinedLoss: {combined_val:.6f}", flush=True)
