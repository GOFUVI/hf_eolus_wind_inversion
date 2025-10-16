# Feature Importance and Sensitivity Analysis

This note documents the analysis script that quantifies and visualises the relative contribution of each input feature to the HF wind inversion MLP, without modifying the training or inference pipelines.

## Scope and Principles

The analysis operates on a trained model artifact and a validation/test dataset. It reconstructs the exact inference feature pipeline (engineering and scaling), respects the physical range gating used during training, and reports three complementary views of importance:

- Permutation importance: model-agnostic increases in error when shuffling inputs.
- Local sensitivities: median absolute input–output gradients (Jacobian) over in-range samples.
- Weight-path importance: an Olden/Garson-style aggregation of absolute weights along network paths (sanity check; activation non-linearities are ignored by design).

Outputs are written to a dedicated timestamped folder under `artifacts_root/analysis/` with CSVs and, when available, PNG figures.

## Usage

Run the script from the repository root:

`python scripts/analysis/feature_importance.py --model-artifact <path/to/model.pth|model.tar.gz> --data-path <path/to/validation.parquet|dir> --grouping grouped --top-k 20`

The most relevant command-line options are:

- `--model-artifact`: Path to `model.pth`, to a `model.tar.gz` (SageMaker artifact), or to a directory containing `model.pth` and (optionally) `script_args.json`.
- `--data-path`: Parquet dataset directory, Parquet file, or CSV with the columns used in training. The script derives exact feature names from the model configuration embedded in the artifact.
- `--output-dir`: Base directory for results (default `artifacts_root/analysis`).
- `--grouping`: One of `grouped`, `feature`, or `both`. Grouping aggregates logically related inputs before shuffling (e.g., `PELI__pwr` = both Bragg peaks; `PELI__bearing` = cosine+sin components).
- `--top-k`: Number of elements to display in bar plots (CSV always contains full results).
- `--max-samples-sensitivity`: Maximum number of in-range samples used to estimate Jacobian-based sensitivities (default 512).

No training code is altered. The script reads but does not modify any existing artifacts.

## Data and Feature Pipeline

The script rebuilds features with the same logic used at inference time:

- Required HF inputs are taken from the schema in the model configuration and copied into standard names.
- Engineered features include per-station power for both Bragg peaks, optional median radial velocities, a MAD-derived proxy (`pwr_mad`), per-station distances, and bearing sine/cosine components.
- Scaling applies persisted global centers/scales and per-interval conditional normalization when present (anchored on maintenance intervals). Angular features are not scaled.

If any required column is missing, the script fails with a descriptive error, listing the expected feature names.

## Metrics and Gating

Regression metrics are computed only on samples whose true wind speed lies within the configured physical operating range `[range_min, range_max]`. Specifically:

- Speed: RMSE in m/s (or the same units as training labels), on in-range samples only.
- Direction: Mean absolute circular error in degrees (0–180), on the same in-range subset.

This respects the training objective where regression gradients are masked outside the valid band; classification behavior outside the band is not evaluated here.

## Methods

1. **Permutation importance.**  
   The script shuffles a feature (or a logical group of features) across samples and measures the increase in speed RMSE and directional MAE with respect to a baseline. Because it is model-agnostic, this estimate is robust to scale and activation details. Use grouped results to summarise station-level effects and per-feature results to drill down.

2. **Local sensitivities (Jacobian).**  
   For a random in-range subset, the script computes per-sample input–output gradients via automatic differentiation. It reports the median absolute gradient of the speed output with respect to each input, and an analogous magnitude for the direction head by combining cosine/sine gradients. Values are reported in the normalised feature space; this aligns with training and avoids confounds from heterogeneous units.

3. **Weight-path importance (Olden/Garson).**  
   A fast approximation that multiplies absolute weights along linear paths (input → hidden layers → head) and aggregates outputs. It ignores activation gating and dropout, so it should be used as a qualitative sanity check, not a quantitative ranking. Agreement in broad strokes with permutation/sensitivities is a good sign; discrepancies warrant attention to potential collinearities or saturation effects.

## Outputs

Under `artifacts_root/analysis/feature_importance_<UTC_TIMESTAMP>/` the script writes:

- `baseline_metrics.json`: In-range RMSE and directional MAE of the baseline.
- `permutation_importance_groups.csv`: ΔRMSE and ΔMAE per group (when `--grouping grouped|both`).
- `permutation_importance_features.csv`: ΔRMSE and ΔMAE per feature (when `--grouping feature|both`).
- `local_sensitivities.csv`: Median absolute gradients for speed and combined direction.
- `weight_path_importance.csv`: Olden/Garson-style importances for speed and direction heads.
- `*.png`: Bar charts for quick review (skipped if matplotlib is unavailable).
- `analysis_metadata.json`: Provenance (paths, range, seed) and the exact feature list.
- `report.txt`: A compact human-readable summary highlighting top contributors.

## Interpretation Tips

- Prefer permutation and sensitivities for quantitative ranking; use weight-path charts to spot-check whether learned readouts align with expectations.
- Examine groups first (e.g., station-level power vs distance vs bearing) before drilling into individual peaks.
- Where cross-station collinearity exists, interpret per-feature permutation deltas with caution: group-level shuffles are more faithful under strong correlation.
- For direction, remember the head predicts cosine/sine components; sensitivity is reported as a vector magnitude and may not map linearly to degrees everywhere.

## Limitations

- Jacobian-based sensitivities are local: they capture behaviour around sampled points and may not reflect global non-linearities.
- Weight-path estimates ignore ReLU/dropout state and are known to overstate importance in saturated regimes.
- If the dataset mixes maintenance intervals and conditional normalisation is active, permutation across the full sample may under- or over-estimate effects that are interval-specific.

## Reproducibility

All runs log input paths, seeds, and the complete feature list. When comparing models, prefer identical datasets and seeds. For large datasets, you can downsample via the `--max-samples-sensitivity` knob (permutation importance uses the full set by default).
