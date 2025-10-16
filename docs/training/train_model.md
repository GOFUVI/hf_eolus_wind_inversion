# SageMaker Training Pipeline

## Purpose
`scripts/training/train_model.sh` orchestrates AWS SageMaker training jobs for the wind inversion MLP, running one job per cross-validation fold (or a full-dataset pass) and aggregating their artefacts locally. It consumes a JSON model configuration to derive station schemas, hyperparameters, and range settings, builds or reuses the Docker training image, provisions IAM roles on demand, and captures logs and metrics for downstream reporting or model review.

## Introduction

### Objective
Wind inversion from HF radar refers to estimating near‑surface wind speed and direction from coastal HF‑radar backscatter. Land‑based HF radars illuminate the sea surface and receive echoes dominated by first‑order Bragg resonant waves; the magnitude and structure of these Bragg peaks, together with station geometry, carry information about wind‑forced wave energy and thus wind stress. This pipeline learns a supervised mapping from station‑wise Bragg power features (plus optional diagnostics) to collocated buoy winds, enabling reliable wind retrievals from new radar observations.

The goal of this training pipeline is to learn a range‑aware neural model that delivers accurate, physically consistent wind retrievals from HF‑radar features and that clearly communicates when a prediction is outside the reliable operating band. Practically, this means jointly regressing wind speed and direction while classifying whether the target lies below, inside, or above a configurable speed range, and doing so in a way that is reproducible, auditable, and easy to deploy. Physically, HF‑radar backscatter relates to wind via first‑order Bragg scattering only within a practical speed band: below the lower bound Bragg waves are under‑developed, and above the upper bound the wave spectrum saturates, so sensitivity to speed collapses. This motivates the range‑aware design with masked regression in‑band and a classifier trained on all samples.

To that end, the pipeline emphasizes: (1) predictive quality across stations and wind regimes, measured by established speed and direction metrics; (2) robust handling of out‑of‑range conditions via an explicit range classifier; (3) reproducibility through a single model JSON, immutable `script_args.json`, and saved normalization parameters; and (4) trustworthy evaluation via cross‑validation and detailed by‑bin/by‑ID diagnostics. The resulting artifacts and reports are suitable for hyper‑parameter tuning, model comparison, and downstream inference or fine‑tuning.

Operationally, the objectives also include: consistent feature scaling via selectable normalization modes (standard z‑score or robust median/MAD) captured in `normalization_params.json`; efficient fine‑tuning from checkpoints that preserves the learned backbone while adapting the last hidden layer and heads; clear range diagnostics at inference (per‑class probabilities, confidence and near‑boundary flags) to communicate reliability; and a principled model‑selection signal based on a DWA‑weighted CombinedLoss of the two regression terms, which is used for reporting and hyper‑parameter optimization.





### Neural network architecture

The inversion model is a range‑aware, multi‑task MLP implemented in PyTorch, reflecting the practical Bragg‑physics speed band described in the Objective.

Each observation is mapped to a compact feature vector that blends magnitude, geometry, and direction: per‑station and per‑Bragg‑peak HF power; the optional maximum deviation of power (MAD); optional median radial velocity statistics (when enabled); station‑to‑site distances; and sine/cosine encodings of station bearing angles.

These inputs pass through a configurable stack of fully connected layers—depth and width come from the model JSON—with ReLU activations and optional dropout that together form a shared backbone representation. When `hidden_layers` is set to 0, this backbone collapses to an identity mapping and the engineered features feed the two heads directly. Downstream consumption of these artefacts, including how inference jobs reuse trained checkpoints and normalization metadata, is documented in `docs/inference.md`.

From this backbone the network produces two coupled outputs. The regression pathway estimates wind speed together with the cosine and sine of wind direction; in the implementation the direction components are bounded with a tanh non‑linearity and the final angle is reconstructed via atan2(pred_sin, pred_cos). In parallel, a range‑classification pathway predicts whether the true wind speed falls below, inside, or above the physically reliable band specified in the configuration; the number and names of classes are configurable via `range_class_labels` (defaulting to `below`, `in`, `above`).

This joint design lets the shared representation focus on extracting HF‑radar structure while the classifier models outside‑of‑range behaviour without contaminating the regression targets.

#### Range‑aware labeling and mask
During feature engineering, each sample is annotated with two signals derived from the configured speed band (`range_min`, `range_max`): (1) a binary `range_mask` that is 1 for in‑range observations and 0 otherwise, and (2) a three‑class `range_class` label with values `below`, `in`, or `above` (the index of `in` is `range_in_class_index`, default 1). The regression head uses `range_mask` to compute errors only on in‑range samples, preventing out‑of‑range targets from biasing the fit, while the range‑classification head is trained on all samples so it learns how signatures behave outside the reliable band. These annotations are produced by the training loader and saved through to outputs so downstream steps can reproduce the same behaviour.


### Training losses and evaluation metrics

Training optimizes a masked multi‑task objective that reflects the physics of the problem. The regression part consists of two terms: a wind‑speed MSE and an angular MSE computed on the wrapped direction error. Crucially, both regression terms are evaluated only on samples whose observed speed lies inside the configured valid range; out‑of‑range targets do not contribute to the regression gradients and therefore cannot bias the fit. In parallel, a range‑classification term is added as a cross‑entropy over the three classes (below, in, above), and—unlike the regression terms—every sample contributes to this classifier so the model learns how HF‑radar signatures behave outside the reliable band. The overall objective combines the three components as `w_speed*MSE_speed + w_angle*MSE_angle + lambda*CE_range`. The weights `w_speed`, `w_angle` are adapted during training via Dynamic Weight Averaging (see below), and `lambda` is the `range_loss_weight` provided in the model configuration.

Why a speed band? The in‑range mask reflects the physical limits above: when winds sit below the lower threshold, Bragg waves are not sufficiently developed; when winds exceed the upper threshold, the radar spectrum saturates and stops evolving with speed. Masking the regression terms prevents the network from fitting uninformative or non‑identifiable targets, while the range classifier still learns from every sample to detect and flag those regimes.

For validation, early stopping, and reporting we use `CombinedLoss = w_speed*MSE_speed + w_angle*MSE_angle + lambda*(1 - macro_F1)`, where `macro_F1` is the macro-averaged F1 score of the range classifier on the validation split. This keeps model selection aligned with the business requirement that the range classifier must remain strong while the regression head improves. The training/reporting code prints the line `CombinedLoss: <value>` and logs the paired macro-F1 whenever a checkpoint improves, so Hyperparameter Tuning jobs and downstream tooling can capture the metric reliably.

For reporting we summarize validation performance using standard wind‑verification statistics. Speed quality is measured by RMSE and MAE, together with the Pearson correlation (r), the coefficient of determination (R²), the mean bias, and two scatter‑index variants (SI based on mean true speed, and SI_max based on the maximum observed speed). Directional skill is captured by the mean signed angular error (EAM), the mean absolute angular error (EAAM), and the directional RMSE, with the angle reconstructed from the network’s cosine/sine outputs via `atan2`. We also compute the complex vector correlation `C = |sum(exp(i*(theta_pred - theta_true)))| / N`, which quantifies the alignment between predicted and true unit‑direction vectors. Finally, because the model includes a range classifier, we report classification accuracy and macro‑averaged precision, recall, and F1, along with per‑class summaries. All metrics are averaged over the evaluation split and are also produced by wind‑direction bin, by `id_col`, and by the combination `id_col × wind_bin` for deeper diagnostics.
### Features

#### Feature normalization
Only numeric‑valued features (e.g., raw power measurements, radial velocity, deviation statistics, distance) are normalized. By default the pipeline applies standard z‑score scaling (subtract the training‑set mean, divide by the standard deviation). Setting `--normalization-mode robust` or `"normalization_mode": "robust"` in the model JSON switches to a median/MAD scheme that multiplies the MAD by 1.4826, with an automatic fallback to the sample standard deviation when the MAD collapses near zero. Angular encodings (sine/cosine) are left untouched because they already lie in [−1,1] and represent circular quantities. Both modes honour `norm_override` entries so that individual centers and scales can be overridden explicitly via `feature.center=value;feature.scale=value`.

When fine‑tuning from a checkpoint, if `/opt/ml/checkpoints/checkpoint.pth` contains `normalization_params`, those centers/scales (and the `normalization_mode`) are reloaded to ensure consistency with the pretrained model instead of recomputing from data.

When training with maintenance-enriched historical datasets the pipeline adds a
second, conditional layer of scaling so that calibration drift is removed per
station and per maintenance interval. The parameters are estimated strictly
from the records assigned to the training partition:

- site1 features `site1_aggregated__pwr_mean_*` and `site1_aggregated__pwr_mad_*`
  use the mean and standard deviation estimated for each
  `site1_maintenance_interval_id` (declare the column with
  `"maintenance_interval_column": "site1_maintenance_interval_id"`).
- site2 features `site2_aggregated__pwr_mean_*` and `site2_aggregated__pwr_mad_*`
  are normalized with the statistics tied to their
  `site2_maintenance_interval_id` (set
  `"maintenance_interval_column": "site2_maintenance_interval_id"`).

The interval-specific parameters are persisted under the `conditional` block of
`normalization_params.json` and reused verbatim when normalizing the validation
set, cross-validation folds, and every downstream inference job. Intervals with
fewer than 24 training samples (or with a degenerate scale) fall back to the
most recent interval of the same station; if no suitable predecessor exists the
global station baseline is used. The inference runner applies the same rules to
unseen intervals, meaning that brand-new maintenance windows default to the
latest calibrated statistics available in the training set. Removing the
`maintenance_interval_column` entry from a station definition disables the
conditional branch for that station, falling back to the global normalization
path.

All fallbacks are auditable. For each station and feature the diagnostics store
the intervals that required assistance as well as the source interval and the
sample count used. The training job also drops a
`conditional_normalization_audit.md` file in the output data directory
summarising the fallback counts for the training and validation splits and
highlighting the specific maintenance intervals that had to borrow parameters.
Because calibrations are treated as instantaneous events, every observation is
associated with exactly one maintenance interval and the conditional scaling is
constant inside that window. This removes slow power drift while keeping the
absolute values available for QA reviewers.

The exported `normalization_params.json` captures the chosen mode, labels the statistics used (`center_label`, `scale_label`), stores per‑feature centers/scales, and includes diagnostics. In robust mode, the diagnostics check that the normalized training data has median ≈ 0 and scaled MAD ≈ 1 within a small tolerance (≈ 0.05), emitting warnings in logs otherwise; in standard mode, training means/stds are included for inspection.

#### Feature encoding: angle conversion
To handle circular quantities, angular features such as wind direction and station bearing angles are converted into sine and cosine pairs. This dual‑component encoding preserves continuity (avoiding abrupt jumps at wrap‑around points) and enables the network to learn directional relationships naturally. The transformation is applied in `scripts/training/train_lib/features.py` for both training and inference, guaranteeing identical preprocessing across environments.


### Dynamic weighting of loss terms (DWA)

Because the two regression terms have different units and scales, we use Dynamic Weight Averaging (DWA) to balance their contributions while keeping the classification weight (`lambda`) fixed. We initialize `w_speed = w_angle = 1.0`. After each epoch from the second onward, we update only these two weights based on the relative rate at which the masked training losses decrease. If `L_speed(t)` and `L_angle(t)` are the masked MSEs on epoch `t`, we form the ratios `r_speed = L_speed(t)/L_speed(t-1)` and `r_angle = L_angle(t)/L_angle(t-1)`, then set (with temperature `T = 2.0`):

`w_speed = 2*exp(r_speed/T) / (exp(r_speed/T) + exp(r_angle/T))`

`w_angle = 2*exp(r_angle/T) / (exp(r_speed/T) + exp(r_angle/T))`

This enforces `w_speed + w_angle = 2`. Intuitively, if one regression task is improving more slowly, its weight increases so the optimizer pays more attention to it. The weights apply to the regression losses in both the training objective and model selection. The classification term is handled separately: during training the total loss is `w_speed*MSE_speed + w_angle*MSE_angle + lambda*CE_range`; for early stopping and reporting we reuse the DWA weights and add a classification penalty via `CombinedLoss = w_speed*MSE_speed + w_angle*MSE_angle + lambda*(1 - macro_F1)`.

### Fine-tuning
When a checkpoint is provided via the `--artifact-uri <s3_path>` flag, `train_model.sh` downloads and stages the archived `model.pth` under `/opt/ml/checkpoints` and passes control to `train.py`, which resumes training from the saved network state. The `model.pth` checkpoint is saved as a dictionary containing three fields:

- **`model_state_dict`**: The serialized network weights for all layers.
- **`normalization_params`**: The feature-normalization metadata, including the selected mode (`standard` or `robust`), labelled centers (mean or median) and scales (std or scaled MAD), plus the post-normalization diagnostics written by the training job.
- **`args`**: The original training configuration captured as the argument namespace (`vars(config)`), including all hyperparameters and feature options.

The script loads these fields so that train.py can apply the exact same feature scaling (`normalization_params`) and reconstruct the training configuration (`args`). Upon loading, all model layers are frozen except the final hidden and output layers—thus preserving the core feature extractor—and only these top layers are enabled for gradient updates. Training then continues at the specified learning rate and hyperparameter settings, allowing the network’s final mappings to adapt quickly to the new domain while leveraging the pretrained foundation.

Fine-tuning runs can now optionally enable knowledge distillation to stabilise the student around the frozen checkpoint. Setting `use_kd` in the model JSON (which `train_model.sh` forwards as a hyper-parameter) instantiates a teacher copy of the pre-fine-tuning network, freezes it in evaluation mode, and adds three auxiliary losses: a masked MSE on wind-speed residuals within the physical operating range, a masked MSE on the paired sine/cosine direction channels, and a soft cross-entropy term on the range-class logits. The regression components reuse the same in-range mask as the primary objective, so distillation never attempts to align out-of-range speeds without physical support. The distillation terms are evaluated on the main batches and on rehearsal batches alike, ensuring the student preserves source-domain behaviour while adapting to the target domain. The optional `teacher_checkpoint` metadata flag exists for bookkeeping; the loader derives the teacher weights from the staged checkpoint and does not require a second download. Distillation remains inactive unless `lambda_kd`, `lambda_kd_reg`, or `lambda_kd_cls` are supplied in the configuration, because all three coefficients default to zero. A dedicated discussion of these regularised fine-tuning strategies, including L2-SP anchoring, rehearsal sampling, and knowledge distillation, is provided in `docs/fine_tuning_anti_forgetting.md`.
### Inference outputs
At inference time the pipeline reloads `script_args.json` to reconstruct configuration and class labels, applies the saved normalization, and returns per‑record predictions together with range diagnostics. In addition to `pred_wind_speed`, `pred_cos_wind_dir`, `pred_sin_wind_dir`, and `pred_wind_direction`, the output includes probability columns for each range class (`prob_range_<label>`), the `pred_range_label` and its `pred_range_confidence`, and a convenience `prob_range_in` aligned to the configured `range_in_class_index`. Boundary proximity flags (`range_near_lower_margin`, `range_near_upper_margin`, `range_near_any_margin`) reflect the configurable `range_margin`. A `range_flag` indicates a confident class prediction vs `uncertain` according to `range_flag_threshold` (with `range_flag_confident` as a boolean), and `range_prediction_consistent` reports agreement between the classifier’s label and the speed‑based label relative to the configured band.



## Prerequisites
- AWS CLI v2 with permissions for SageMaker, IAM, CloudWatch Logs, ECR, and S3 in the target account.
- Docker installed locally to build and push the training image when `--image-uri` is not provided.
- `jq`, `tar`, and `bc` available on the host; the script uses them for configuration parsing, archive handling, and metric aggregation.
- An S3 bucket/prefix that will hold the training dataset (`--train-data-uri`) and receive job outputs under `--s3-prefix`.
- When overriding the container image, ensure the package still exposes the training code at `/opt/ml/code/train.py` or supplies an equivalent `SAGEMAKER_PROGRAM` entrypoint.
 - AWS CLI profile/region: pass `--profile`/`--region` or ensure defaults are set (`us-east-1` is used by default if not provided).
 - ECR privileges: ability to describe/create repositories and authenticate (`aws ecr get-login-password | docker login`), plus permission to push images to your account registry.
 - POSIX utilities: the scripts rely on standard tools (`awk`, `sed`, `grep`, `find`, `date`, `mktemp`) typically present on Unix-like systems.
 - IAM role handling: ensure your credentials can create or update `MySageMakerRole` and attach the required managed policies (SageMaker, S3, CloudWatch Logs, ECR); the current implementation always reuses that role regardless of `--role-arn`.

## Workflow Highlights
1. Validates required flags (`--train-data-uri`, `--s3-prefix`, `--model-config`) and resolves output paths relative to the invocation directory.
2. Parses the model JSON with `jq` to populate station lists, target columns, range bounds, feature toggles (MAD, velocity median), and early stopping behaviour.
3. Ensures a SageMaker execution role exists, attaching the AWS-managed policies needed for training, logging, and catalog access, and records the resulting ARN.
4. Builds and pushes the Docker image defined in `scripts/training/Dockerfile` when `--image-uri` is omitted; otherwise it reuses the supplied ECR image.
5. Builds the SageMaker channel definition pointing to the GeoParquet dataset, expands the fold list (`--folds`, `--folds-list`, `--no-cv`), and launches one `create-training-job` call per fold. Fine-tuning checkpoints supplied with `--artifact-uri` are downloaded, re-uploaded under `S3_PREFIX/fine-tuning/<job>/checkpoint.pth`, and exposed through the container via `/opt/ml/checkpoints`.
6. When `--wait-for-jobs true`, the script waits for completion, fetches CloudWatch logs, downloads each job’s `output.tar.gz`, extracts per‑fold and training metrics CSVs into `<output-dir>/metrics_results/`, and copies `normalization_params.json` and `script_args.json` into `<output-dir>/` for reproducibility. It then computes cross‑fold averages with `bc`.
7. Produces a Markdown report `<job-base-prefix>_report.md` summarizing fold metrics, global averages, and per‑bin/ID breakdowns, and embeds the extracted JSON artefacts.

8. If `--clean-logs true` (default), removes any existing local logs under `<output-dir>/logs/<job-base-prefix>*` before launching new jobs to avoid mixing runs.
9. CombinedLoss reported in logs/CSVs includes the classification penalty `lambda*(1 - macro_F1)` (with `lambda = range_loss_weight`) in addition to the DWA‑weighted regression terms.
10. Alongside global metrics, writes range‑classification CSVs per fold/train with precision/recall/F1 by class and an overall summary, to aid QA of the range head.
11. The selected normalization mode (`standard` or `robust`) and the exact centers/scales are persisted in `normalization_params.json` and reused consistently by inference and fine‑tuning.

Note: if you need to fetch metrics for a finished job later (without rerunning the pipeline), use the helper scripts `scripts/training/get_train_metrics.sh` and `scripts/training/get_bin_metrics.sh` to pull CSVs from the SageMaker output bundle. For quick monitoring you can also parse `CombinedLoss:` lines and metric summaries in `<output-dir>/logs/<TrainingJobName>/events.log`.

## Model artifact and metadata

This section documents the model artifact produced by SageMaker and the custom metadata embedded to support reproducible inference and fine‑tuning.

- Files produced
  - Model artifact tarball (SageMaker): a `model.tar.gz` containing the file `model.pth` at the root.
  - Output metadata (SageMaker): under `/opt/ml/output/data` the training job writes `normalization_params.json` and `script_args.json`, plus metrics CSVs. The pipeline copies the two JSONs into your local output directory.

- `model.pth` structure
  - Either a bare PyTorch state dict (older exports) or a dictionary with keys:
    - `model_state_dict`: learned network weights for the backbone and both heads.
    - `normalization_params`: full normalization metadata (see below) used by inference and fine‑tuning to ensure identical scaling.
    - `args`: the exact training configuration as a flat dictionary (the serialized `vars(config)`), including stations/schema, range settings, normalization mode, and hyperparameters.
    - `model_config_payload` (optional): the raw model JSON used to launch training, embedded for provenance.
  - Checkpoints (`/opt/ml/checkpoints/checkpoint.pth`) additionally include `optimizer_state_dict` and `epoch` so training can resume; the final `model.pth` in `model.tar.gz` does not carry optimizer state.

- Normalization metadata (`normalization_params`)
  - Keys:
    - `mode`: `standard` or `robust`.
    - `center_label` / `scale_label`: labels of the statistics used (e.g., `mean`/`std` for standard; `median`/`scaled_mad` for robust).
    - `centers`: mapping `feature -> float` with the per‑feature center used during training.
    - `scales`: mapping `feature -> float` with the per‑feature scale used during training.
    - `diagnostics` (optional): includes a `train` block with post‑normalization summary (means/stds; and in robust mode also medians/scaled MAD) and `fallback_to_std` when any column fell back from MAD to standard deviation.
  - Backwards compatibility: inference also accepts older payloads that use `means`/`stds` in place of `centers`/`scales`.

- Script arguments (`script_args.json` and `args` inside `model.pth`)
  - Capture the full configuration used to train the model, including:
    - Station list (`stations`) and inline schema mapping (`station_schema`) that drive column resolution.
    - Target column names (`target_speed_col`, `target_dir_col`) and ID column (`id_col`).
    - Range settings (`range_min`, `range_max`, `range_margin`, `range_loss_weight`, `range_flag_threshold`, `range_class_labels`, `range_in_class_index`).
    - Feature toggles and aggregation (`agg_stat`, `use_mad`, `use_velocity_median`).
    - Normalization mode and any overrides (`normalization_mode`, `norm_override` flattened into `feature.center/feature.scale`).
    - Model capacity and training hyperparameters (`hidden_layers`, `hidden_units`, `dropout`, `epochs`, `batch_size`, `lr`, `weight_decay`, `early_stopping`, `patience`, `seed`).
  - Inference (`scripts/inference/inference.py`) reloads these settings to rebuild feature engineering and range processing consistently with training.

- Consumption summary
  - Fine‑tuning: reloads `normalization_params` and `args` from `model.pth` (or, when resuming, from the latest checkpoint) to continue with the same scaling and configuration; restores optimizer state only when present.
  - Inference: loads `model.pth` and uses `normalization_params`/`script_args.json` to standardize inputs and emit range‑aware outputs; see `docs/inference.md` for column‑level details and the supplementary `inference_metadata.json` produced by the pipeline.

## Typical Usage
```bash
bash scripts/training/train_model.sh \
  --train-data-uri s3://<your-bucket>/analytics_db/training/stationX/geoparquet/ \
  --s3-prefix s3://<your-bucket>/analytics_db/models/stationX/ \
  --model-config artifacts_root/stationX/config/stationX_model.json \
  --profile your_profile \
  --region us-east-1 \
  --folds 5 \
  --output-dir training_metrics
```

Append `--normalization-mode robust` here (or set `"normalization_mode": "robust"` in the model JSON) whenever you need the median/MAD scaler instead of standard z-score normalization.

## Model JSON Options

Use a single JSON file to describe stations, schema, and model options. Keys may live at the top level or under a `model` section; when both exist, values under `model` take precedence (except where noted). Place `stations` and `schema` at the top level for clarity.

- Required (top level)
  - `stations` (array of strings): Station names used throughout the schema and feature engineering. Requires at least two names.
  - `schema.stations` (object): Per‑station column templates used to locate features in the GeoParquet dataset.
    - `power_pattern` (string, required): Template for Bragg power columns. Must include `{peak}` and usually `{agg}`. Example: `"site1_aggregated__pwr_{agg}_{peak}"`.
    - `mad_pattern` (string, required if `use_mad=true`): Template for MAD power columns, includes `{peak}` and optionally `{agg}`.
    - `velocity_median_pattern` (string, optional; used when `use_velocity_median=true`): Template for median radial velocity columns. If omitted, the loader tries `"<station>__velo_median_{peak}"`, `"<station>_velo_median_{peak}"`, or `"<station>_stats_velo_median_{peak}"`.
    - `bearing` (string, required): Column for station bearing (degrees). May include `{agg}`; if not present, it is used verbatim.
    - `distance` (string, required): Column for station distance. May include `{agg}`.

- Targets and data columns (prefer under `model`, fallback to top level)
  - `target_speed_col` (string): Name of wind speed column (default `wind_speed`).
  - `target_dir_col` (string): Name of wind direction column (default `wind_dir` or `wind_direction`).
  - `id_col` (string): Identifier used for grouped metrics (default `location_id`).

- Feature aggregation and toggles (under `model`)
  - `agg_stat` ("mean"|"median"|"max"): Aggregation used in schema templates (mapped to `agg`).
  - `use_mad` (bool): Include MAD‑based power feature.
  - `use_velocity_median` (bool): Include per‑station median radial velocity features.

- Normalization (under `model`, with one exception)
  - `normalization_mode` ("standard"|"robust"): Scaling strategy for numeric features. Robust uses median/MAD (×1.4826) with safe fallbacks.
  - `norm_override` (object, top‑level only): Per-feature overrides using neutral keys `center` and `scale`.
    Example: `{\"site1_aggregated_dist\": {\"scale\": 33.03}, \"site2_aggregated_dist\": {\"center\": 30.59}}`.
    The override always applies to the statistic selected by `normalization_mode`: in **standard** mode `center` maps to the mean and `scale` to the standard deviation, whereas in **robust** mode `center` targets the median and `scale` the scaled MAD (1.4826 × MAD). The supplied values are used during training and persisted into `normalization_params.json`, so inference and fine-tuning reuse the exact same custom scaling.

- Range and classification (under `model`)
  - `target_speed_range` ([min, max]) or `range_min`/`range_max` (floats): Valid wind‑speed band used to mask regression and label range classes.
  - `range_margin` (float): Margin near band edges (m/s) used in downstream diagnostics and inference. Predictions whose speed lies within this margin of `range_min` or `range_max` are flagged as "near the boundary" (see columns `range_near_lower_margin`, `range_near_upper_margin`, `range_near_any_margin` in inference outputs). This value does not change training; it helps reviewers triage borderline cases. Set `0.0` to disable, keep the default `0.5`–`0.8` m/s for typical buoy accuracies, or increase to `1.0`–`1.5` m/s if you prefer conservative gating around the valid band.
  - `range_loss_weight` (float): Weight λ applied to the range‑classification loss.
  - `range_flag_threshold` (float in [0,1]): Minimum class probability required to consider the range prediction "confident" in inference. If the top class probability falls below this threshold, the `range_flag` is set to `uncertain` and `range_flag_confident=false`; otherwise the predicted label is emitted. This value does not affect training. Use `0.5` as a balanced default, raise to `0.7`–`0.8` to reduce false flags at the expense of more `uncertain` cases, or lower for more permissive behaviour.

- Training hyperparameters (under `model`)
  - `hidden_layers` (int), `hidden_units` (int), `dropout` (float)
  - `epochs` (int), `batch_size` (int)
  - `lr` (float), `weight_decay` (float)
  - `early_stopping` (bool), `patience` (int)
  - `save_error_data` (bool)
  - `seed` (int): Random seed for reproducibility

How to add options to the JSON
- Place station names under `stations` and define their column templates under `schema.stations` as shown above. Templates interpolate `{agg}` with the normalized aggregation (`mean`/`median`/`max`) and `{peak}` with `0`/`1` for the Bragg peaks.
- Put model options inside the `model` section. Most keys also work at the top level, but keeping them under `model` improves readability. The exception is `norm_override`, which must be defined at the top level (it is flattened into CLI overrides internally).
- Choose one of `target_speed_range: [min, max]` or explicit `range_min`/`range_max`. Do not set both with conflicting values.
- When enabling `use_velocity_median`, ensure the schema either provides `velocity_median_pattern` or the dataset uses one of the auto‑detected names listed above.

Minimal example
```json
{
  "stations": ["site1_aggregated", "site2_aggregated"],
  "schema": {
    "stations": {
      "site1_aggregated": {
        "power_pattern": "site1_aggregated__pwr_{agg}_{peak}",
        "mad_pattern": "site1_aggregated__pwr_mad_{peak}",
        "bearing": "site1_bearing",
        "distance": "site1_dist_km"
      },
      "site2_aggregated": {
        "power_pattern": "site2_aggregated__pwr_{agg}_{peak}",
        "mad_pattern": "site2_aggregated__pwr_mad_{peak}",
        "bearing": "site2_bearing",
        "distance": "site2_dist_km"
      }
    }
  },
  "model": {
    "agg_stat": "mean",
    "use_mad": true,
    "use_velocity_median": false,
    "normalization_mode": "standard",
    "target_speed_col": "wind_speed",
    "target_dir_col": "wind_dir",
    "target_speed_range": [5.7, 17.8],
    "range_loss_weight": 1.0,
    "early_stopping": true,
    "patience": 20,
    "hidden_layers": 2,
    "hidden_units": 256,
    "dropout": 0.2,
    "epochs": 1000,
    "batch_size": 256,
    "lr": 0.001,
    "weight_decay": 0.001,
    "id_col": "location_id",
    "seed": 42
  },
  "norm_override": {
    "site1_aggregated_dist": {"scale": 33.03},
    "site2_aggregated_dist": {"scale": 30.59}
  }
}
```


## Data Requirements
- GeoParquet dataset with fold assignments, wind speed and direction columns, and per-station feature columns matching the patterns declared in the model JSON (`schema.stations`).
- For each station: aggregated Bragg power per peak, optional MAD or median radial velocity statistics when enabled, bearing and distance columns. The loader renames them to canonical fields like `station_pwr_0`, `station_dist`, `cos_station_bearing`.
- Targets and units: specify the target columns in the model JSON via `target_speed_col` (m/s) and `target_dir_col` (degrees clockwise from north), or use the defaults (`wind_speed` and `wind_dir`/`wind_direction`). The loader resolves these names and renames them internally to the canonical `wind_speed`/`wind_dir` for downstream processing.
- Peaks and aggregation: for each station and Bragg peak `0`/`1`, power columns must follow the `power_pattern` with placeholders for `{agg}` and `{peak}`; `{agg}` is resolved from `agg_stat` (`mean`/`median`/`max`).
- Median radial velocity (optional): if `use_velocity_median=true` and no `velocity_median_pattern` is provided, the loader attempts fallback names: `<station>__velo_median_<peak>`, `<station>_velo_median_<peak>`, or `<station>_stats_velo_median_<peak>`.
- The JSON configuration must define `range_min` / `range_max` (or `target_speed_range`) to mask regression losses and drive the range‑classification head; this band encodes where Bragg‑based retrievals are physically reliable for your instrument/setup.
- Optional `wind_bin` and `id_col` columns supply stratification for reporting. Absent columns are filled with a constant `1` before metrics are generated. The `fold` column (if present) is used only to split train/validation and is dropped from features.
- Dataset layout: the loader accepts a single Parquet file or a directory of Parquet files (Hive‑partitioned). Columns are resolved via a PyArrow dataset, then renamed to the canonical schema for feature engineering.
- Robustness: MAD features replace `-inf`/`NaN` entries and, when necessary, fall back to the row‑wise maximum of power features. In robust normalization, if a column’s MAD collapses near zero, scaling falls back to the sample standard deviation; diagnostics are recorded in `normalization_params.json`.


## Companion Scripts
- `scripts/training/train.py` delegates to `train_lib` modules: `cli.py` parses flags, `config.py` merges the model JSON, `data_load.py` performs fold splits, `features.py` builds input tensors, `normalization.py` manages scaling and reuse during fine-tuning, `model.py` defines the multi-head MLP, `train_loop.py` implements DWA/early stopping/checkpointing, and `reporting.py` emits the CSV suite together with the CombinedLoss used in downstream summaries.
- `scripts/training/get_norm_params.sh` downloads `output.tar.gz` or model artefacts for a completed job, extracts `normalization_params.json`, and writes a short Markdown summary under `normalization_params/<job>/`.
- `scripts/training/get_train_metrics.sh` retrieves `metrics_train_fold*.csv`, collates them into `training_metrics/<job>/`, and generates a Markdown synopsis of per-fold training performance.
- `scripts/training/get_bin_metrics.sh` focuses on per-bin and classification CSVs, aggregating them under `bin_metrics/<job>/` with statistics grouped by wind bin and range class.
- `scripts/training/generate_model_config_from_hpo.sh` inspects a SageMaker job (typically the best HPO trial), hydrates a fresh model JSON with the discovered hyperparameters, and keeps inline schema definitions synchronized with the repository.

## Key Options

- Required
  - `--train-data-uri S3URI`: GeoParquet dataset location (S3 file or prefix) mounted as the SageMaker `training` channel; must be readable by the execution role; directory inputs can be Hive‑partitioned.
  - `--s3-prefix S3URI`: Base S3 prefix for job outputs and artifacts (e.g., model.tar.gz); also used to stage warm‑start checkpoints under `<s3-prefix>/fine-tuning/<job>/`; must be writable by the role.
  - `--model-config PATH`: Model JSON describing stations/schema, targets, range, normalization, and hyperparameters; relative paths are resolved from the original working directory.

- Compute & environment
  - `--profile PROFILE`: AWS CLI profile to use; defaults to `$AWS_PROFILE` or `default`.
  - `--region REGION`: AWS region; defaults to `$AWS_REGION` or `us-east-1`.
  - `--role-arn ARN`: Existing SageMaker execution role. `train_model.sh` currently always provisions or reuses `MySageMakerRole` with the required AWS‑managed policies and then overwrites the ARN with that role, so providing this flag has no effect yet.
  - `--image-uri URI`: Prebuilt training image in ECR; if omitted, the script builds from `scripts/training/Dockerfile`, logs in to ECR and pushes.
  - `--ecr-repo NAME`: ECR repository name used when building the image (default `buoy_train`).
  - `--instance-type TYPE`: Instance type for the training job (default `ml.m5.large`).
  - `--volume-size GB`: EBS volume size attached to the training instance (default `10`).
  - `--max-runtime SECONDS`: Wall‑clock timeout for the training job (default `3600`).

- Cross‑validation & scheduling
  - `--folds COUNT`: Number of folds to run when `--folds-list` is not provided and cross‑validation is enabled (default `5`); folds are indexed as `1..COUNT` and must match the dataset’s `fold` values.
  - `--folds-list LIST`: Comma‑separated list of fold indices to run (e.g., `0,2,3` or `1,3,5` depending on your dataset), overrides `--folds`.
  - `--no-cv[=BOOL]`: Disable cross‑validation and train once on the full dataset (`fold_actual=-1`); useful for “final train” after tuning.
  - `--wait-for-jobs true|false`: Wait for SageMaker jobs to finish and aggregate outputs (default `true`); set `false` to submit and exit.
  - `--job-base-prefix NAME`: Prefix for job names and the report filename (default `train-fold`); jobs are named `<prefix>-<timestamp>-<fold>`.
  - `--clean-logs true|false`: Remove existing local logs under `<output-dir>/logs/<prefix>*` before launching (default `true`).
  - `--output-dir DIR`: Local directory to collect logs, extracted metrics, JSON artifacts, and the Markdown report (default `.`); relative paths are resolved from the original working directory.
  - `--seed INT`: Forwarded to `train.py` to initialize data loaders and model weights for reproducibility when supported by the hardware.

- Warm‑start & normalization
  - `--artifact-uri S3URI`: Warm‑start from an existing tarball; the script downloads it, extracts `model.pth`, re‑uploads it to `<s3-prefix>/fine-tuning/<job>/checkpoint.pth`, and configures SageMaker checkpoints at `/opt/ml/checkpoints`.
  - `--normalization-mode MODE`: Feature scaling mode forwarded to training (`standard` or `robust`); the selected mode and statistics are persisted in `normalization_params.json` and reused by inference/fine‑tuning.

## Operational Notes
- `train_model.sh` sets `SAGEMAKER_SUBMIT_DIRECTORY=${S3_PREFIX}/code.tar.gz` for compatibility with the training toolkit. The provided Dockerfile already includes the sources under `/opt/ml/code`; if you rely on the S3 package instead, upload the tarball before launching jobs.
- The IAM role helper attaches broad managed policies for convenience. In production environments you may want to pre-provision a narrower role and supply its ARN via `--role-arn`.
- The script assumes `bc` is available for floating-point arithmetic; install it when running from minimal containers or CI runners.
- CloudWatch log retrieval uses `aws logs filter-log-events`. Ensure the active profile has `logs:FilterLogEvents` permissions or the log step will fail after the training job completes.

## File References
- `scripts/training/train_model.sh`
- `scripts/training/train.py`
- `scripts/training/train_lib`
- `scripts/training/Dockerfile`
- `scripts/training/get_norm_params.sh`
- `scripts/training/get_train_metrics.sh`
- `scripts/training/get_bin_metrics.sh`
- `scripts/training/generate_model_config_from_hpo.sh`
