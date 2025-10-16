# HF-EOLUS HF-Radar Wind Inversion Toolkit for Artificial Neural Networks Training and Inference 

## Introduction

This toolkit assembles the pipelines and scripts needed to train and infer near-surface wind speed and direction from HF-Radar measurements using artificial neural networks. Supervision and validation rely on two external references: (i) Sentinel-1 Level-2 OCN products aggregated on a regular grid (for example, 20 km spacing) and (ii) in-situ winds from an oceanographic buoy located within that grid. HF-Radar inputs are aggregated within neighbourhoods whose radius typically matches half of the grid spacing. To respect the physics of the inversion, regression heads operate only within the wind-speed interval supported by the radar frequency, while a dedicated range-classification head handles the remaining samples (see [Physical Range Gating](#physical-range-gating-frequency-specific-operating-band)).

The workflow balances two complementary domains: the SAR grid captures broad spatial variability, whereas the buoy provides a high-fidelity point anchor for the HF-radar-to-wind relationship. Fine-tuning routes explore both transfer directions (SAR → buoy and buoy → SAR) to trade point accuracy for spatial coverage as needed, outlining the operational compromise between geographic reach and local precision. For script-level reference and CLI examples, consult the documentation index under [docs/README.md](docs/README.md). The sections below describe the toolkit layout, the available pipelines, and the training and inference components.

[Back to top](#introduction)

### Table of Contents

- [Background](#background)
  - [Physical Link Between HF-Radar Observables and Wind](#physical-link-between-hf-radar-observables-and-wind)
  - [Model, Inputs, and Outputs](#model-inputs-and-outputs)
    - [Architecture](#architecture)
    - [Inputs](#inputs)
    - [Outputs and Losses](#outputs-and-losses)
    - [Physical Range Gating (frequency-specific operating band)](#physical-range-gating-frequency-specific-operating-band)
    - [Trainable Parameters vs Hyperparameters](#trainable-parameters-vs-hyperparameters)
  - [Data Sources, Footprints, and Collocation](#data-sources-footprints-and-collocation)
    - [Sensor Footprints](#sensor-footprints)
    - [Sampling Asynchrony and Collocation](#sampling-asynchrony-and-collocation)
- [Workflow Stages](#workflow-stages)
  - [STAC Cataloging and Provenance](#stac-cataloging-and-provenance)
  - [Data Preparation and Schema Alignment](#data-preparation-and-schema-alignment)
    - [Bragg-Peak Pivot and Geometry Validation](#bragg-peak-pivot-and-geometry-validation)
    - [Station Bearing and Distance Features](#station-bearing-and-distance-features)
    - [Maintenance-Interval Tagging](#maintenance-interval-tagging)
    - [Join of Pivoted Stations and Domain-Specific Views](#join-of-pivoted-stations-and-domain-specific-views)
    - [Partitioning and Stratification](#partitioning-and-stratification)
    - [Partition-Specific Filtering and Source Annotation](#partition-specific-filtering-and-source-annotation)
    - [Cross-Domain Concatenation (Combined Corpus)](#cross-domain-concatenation-combined-corpus)
    - [Training-Time Schema Standardization and Feature Engineering](#training-time-schema-standardization-and-feature-engineering)
      - [Station Schema Definition and Resolution](#station-schema-definition-and-resolution)
      - [Schema Alignment and Standardization](#schema-alignment-and-standardization)
      - [Robust Dispersion Aggregates (MAD)](#robust-dispersion-aggregates-mad)
      - [Directional Encoding and Geometric Transforms](#directional-encoding-and-geometric-transforms)
      - [Feature Normalization (Maintenance-Aware)](#feature-normalization-maintenance-aware)
      - [Outputs for Downstream Stages](#outputs-for-downstream-stages)
  - [Training Execution Workflow](#training-execution-workflow)
  - [Hyperparameter Optimization (HPO)](#hyperparameter-optimization-hpo)
  - [Final Training (No-CV)](#final-training-no-cv)
  - [Fine-Tuning](#fine-tuning)
    - [Fine-Tuning Strategies](#fine-tuning-strategies)
      - [Plain Fine-Tuning](#plain-fine-tuning)
      - [L2-SP Anchoring](#l2-sp-anchoring)
      - [Knowledge Distillation (KD)](#knowledge-distillation-kd)
  - [Inference and Evaluation](#inference-and-evaluation)
    - [Evaluation Subsets](#evaluation-subsets)
    - [Metrics and Subsets](#metrics-and-subsets)
    - [Inference Coverage](#inference-coverage)
    - [Maintenance-Interval Diagnostics](#maintenance-interval-diagnostics)
- [Related Scripts (per phase)](#related-scripts-per-phase)
- [Acknowledgements](#acknowledgements)
- [Disclaimer](#disclaimer)
- [References](#references)

> **Security Notice:** This code has been tested using an AWS CLI profile with administrator permissions. For production use, restrict the profile’s permissions to the minimum necessary privileges, following the principle of least privilege (PLP).

> **Cost Warning:** Training and data preparation on AWS can incur significant costs in real-world applications.

## Background

### Physical Link Between HF‑Radar Observables and Wind
HF‑Radar measures the line‑of‑sight (radial) surface current by exploiting Bragg scattering of the transmitted radio waves by ocean waves of a specific wavelength. The Doppler shift of the first‑order Bragg peaks yields the radial component of the near‑surface current; combining radials from multiple bearings produces a horizontal current vector representative of the upper few decimeters and spatially averaged over the radar’s footprint. This current responds to wind stress on short time and space scales through the surface Ekman layer and wave‑driven Stokes drift, while also being influenced by tides, pressure‑gradient (geostrophic) flows, and coastal constraints. In addition to kinematic products (radial velocity), radar spectra and backscatter power encode sea‑state information that co‑varies with wind via wave energy and directional spread. Our feature set summarizes these signals—using robust statistics such as medians and MADs aggregated within neighborhoods commensurate with the grid spacing—to let the ANN learn a nonlinear mapping from HF‑Radar observables to 10 m wind speed and direction.

### Model, Inputs, and Outputs

#### Architecture
The model is a feed‑forward artificial neural network (ANN): a stack of fully connected layers that propagate information forward from inputs to outputs without recurrences or convolutions. This choice suits compact, tabular feature vectors derived from HF‑Radar, is efficient to train, and allows straightforward control of depth, width, and regularization (e.g., activation nonlinearities, normalization, dropout) to balance bias and variance.

In a multi‑task ANN, “heads” are specialized output branches that share a common backbone but optimize for different targets. We use three heads because the tasks have distinct statistical and physical characteristics: (i) a speed‑regression head to predict 10 m wind speed, (ii) a direction‑regression head to predict wind direction, and (iii) a range‑classification head to label each sample as below, within, or above the physically valid wind‑speed range for the operating frequency. Separating heads allows tailored loss weighting, calibration, and diagnostics per task.

The backbone refers to the stack of hidden layers that precedes the task-specific heads. It transforms the raw feature vector-power statistics, geometric descriptors, and optional dispersion metrics-into a compact latent representation that captures the shared structure useful to every downstream task. Because the heads branch from this shared representation, improvements to the backbone (depth, width, activation choice) benefit speed regression, direction regression, and range classification simultaneously, while the heads specialize the final mapping for each target.

The training loss is the objective minimized during learning; it quantifies how far model outputs are from their references. Here, the overall objective couples the two regression losses (for speed and direction) with a classification loss for the range head. Regression losses are applied only to samples inside the valid range, while the classification loss is computed on all samples. The classifier exists specifically to decide the physical range membership; operationally it gates regression usage so that speeds and directions are not interpreted outside the supported regime.

#### Inputs
Inputs are features derived from HF‑Radar observations pre‑aggregated upstream: robust statistics (e.g., medians and MADs) of radial velocity and spectral/backscatter summaries within local neighborhoods centred on the grid nodes or the buoy location. Auxiliary inputs include identifiers needed for maintenance‑aware normalization and, where applicable, categorical flags that encode the sample’s range class for the classification head.

Normalization (see [Training‑Time Schema Standardization and Feature Engineering → Feature Normalization](#feature-normalization-maintenance-aware)). The training workflow fits and applies a maintenance‑aware normalization to numeric features. Details on scope, per‑interval strategy, and exceptions are documented in that section.

#### Outputs and Losses
Outputs comprise (a) 10 m wind speed, (b) wind direction, and (c) a three‑class range flag. Let `x` denote the input features, `v` the reference wind speed, `θ` the reference wind direction, and `c ∈ {below, in, above}` the range class at the operating frequency. The model produces predictions `v̂(x)` and directional components `(ĉ(x), ŝ(x)) ~ (cos θ̂, sin θ̂)`, together with a probability vector `p(c|x)` from a softmax classifier. The predicted angle `θ̂` is recovered via `atan2(ŝ, ĉ)` when needed for analysis.

Frequency‑specific in‑range mask. Define the valid speed interval `[v_min, v_max]` dictated by the chosen operating frequency and the in‑range mask `M_in = 1(v ∈ [v_min, v_max])`. Regression losses are multiplied by `M_in` so that only physically supported samples contribute to regression training.

Speed loss. A standard squared error is used on in‑range samples: `L_speed = E[ M_in · (v̂ - v)² ]`.

Directional loss. Direction is periodic; errors are computed on the smallest signed angular difference. Let `Δθ = wrap(θ̂ - θ)` be wrapped to `[-π, π]` (or equivalently `[-180 deg, 180 deg]`). The directional loss uses the squared circular error on in‑range samples: `L_dir = E[ M_in · (Δθ)² ]`.

Range classification loss. The range head is trained on all samples with a cross‑entropy objective `L_range = E[ CE( p(c|x), c* ) ]`, where `c*` denotes the target class. The configuration includes a “range loss weight” `λ_range` that scales this term relative to regression. Note: the configured range margin `m` (see [Hyperparameter Optimization (HPO)](#hyperparameter-optimization-hpo) and [Fine-Tuning Strategies](#fine-tuning-strategies)) is used for inference‑time diagnostics and flags (e.g., proximity to bounds), not for softening training labels.

Total multi‑task loss. The overall objective is a weighted sum
`L_total = w_speed · L_speed + w_dir · L_dir + λ_range · L_range (+ regularization)`,
where the optional regularization bundle aggregates penalties such as L2 weight decay, dropout, and (during fine-tuning) L2-SP anchors configured through the training manifests (see [Hyperparameter Optimization (HPO)](#hyperparameter-optimization-hpo) and [Fine-Tuning Strategies](#fine-tuning-strategies)). With `w_speed`, `w_dir`, and `λ_range` tuned/controlled alongside architecture and optimization hyperparameters, only the classifier is responsible for deciding physical range membership; during inference, its outputs gate the use of the regression heads outside the valid interval.

Dynamic weighting. The regression weights `w_speed` and `w_dir` are updated epoch‑wise using Dynamic Weight Averaging (DWA) with temperature `T = 2.0`, based on the relative change of each task’s recent losses. This adapts the speed‑vs‑direction balance during training without manual retuning. In practice, DWA raises the weight of the task whose loss is improving more slowly relative to its previous epoch and lowers it for the task improving faster. This compensates for differing learning rates between speed and direction, reducing the need to hand‑tune static weights and mitigating domination of the total loss by either task early or late in training. The effect is a smoother training trajectory and a more stable compromise between speed RMSE and directional error.

Evaluation. Reports include in‑range regression metrics (RMSE, MAE, bias, correlation, R2, directional RMSE/EAAM) and range‑classification metrics (accuracy, precision/recall/F1, macro‑F1) on all samples. “Within range + class match” subsets further restrict to samples where the predicted class matches the true range.

#### Physical Range Gating (frequency‑specific operating band)
Regression targets are trained and evaluated exclusively within the wind‑speed regime where the calibration and forward assumptions are physically valid for the operating frequency. Within typical HF-Radar bands, the Bragg-resonant ocean waves sustain a nonlinear but interpretable coupling to wind stress only once the friction velocity exceeds the threshold at which resonant gravity-capillary waves are generated; below that threshold the wind fails to raise those Bragg waves, the coherent echo vanishes, and the HF-Radar loses deterministic sensitivity to wind. Conversely, once the short-wave spectrum saturates, the assumed balance between Bragg power and 10 m winds is disrupted. The bounds adopted in this repository follow the operating limits reported for HF-radar wind retrievals by Emery & Kirincich (2021), who derive the interval from the inversion study of Shen et al. (2011). Samples outside this range are not used to assess regression skill; instead, the dedicated range‑classification head provides the operational gate and supports post‑hoc filtering during inference, while still letting us monitor how often the model ventures beyond the physically defensible regime.

#### Trainable Parameters vs Hyperparameters
Neural networks distinguish between trainable parameters and hyperparameters. Trainable parameters are the weights and biases optimized by gradient‑based learning to minimize the training objective. Hyperparameters are configuration choices fixed before training that control model capacity, regularization, optimization, and feature usage; they are not learned directly from the loss and are selected by validation.

Scope in this work. Hyperparameters include architectural choices (number of layers and hidden units; activation; dropout rate), optimization controls (learning rate and schedule; weight decay; batch size), multi‑task loss controls (relative weights for speed and direction; range‑loss weight and margin), and feature toggles (use of MAD‑based dispersion; inclusion of per‑station median radial velocity). Maintenance‑aware normalization is configured (e.g., per‑interval vs global), but its statistics are estimated from the data. By contrast, the frequency‑specific range bounds used for gating are physical constraints, not hyperparameters.

Selection. Hyperparameters are selected via cross‑validated HPO on the training split: we assign F folds within train (default 5), train on F-1 folds, validate on the held‑out fold, and cycle folds. Selection prioritizes the composite multi‑task loss together with in‑range speed and direction metrics and macro‑F1 for range classification, while monitoring fold‑to‑fold variance. The final configuration is retrained on the full training set and evaluated once on the fixed test set. See [Hyperparameter Optimization (HPO)](#hyperparameter-optimization-hpo) for the search procedure, and `scripts/training/train_lib/config.py` plus `artifacts_root/*/config/*_model.json` for how hyperparameters are declared and resolved.

### Data Sources, Footprints, and Collocation
#### Sensor Footprints
The sensor geometry assumed throughout this repository matches the upstream aggregation workflow released in Herrera Cortijo et al. (2025, Zenodo DOI: 10.5281/zenodo.17115413). HF‑Radar and SAR share the same nominal spatial support: a grid spacing realised by aggregating observations within sub-grid neighbourhoods large enough to avoid excessive overlap yet compact enough to preserve spatial detail. For HF-Radars, the node spacing is driven by the deployed geometry and operating frequency; aggregation footprints are typically set to roughly half that spacing. The buoy is point-scale; for inversion features, HF-Radar is also aggregated within a comparable neighbourhood around the buoy, which locally breaks the non-overlap rule. The number of usable samples depends strongly on that aggregation radius: tighter radii make it harder to meet the requirement of seeing both Bragg peaks at the two contributing stations. Experiments with a minimal search radius—which matches the radar cell geometry—showed a dramatic drop in available training data, underscoring the need to balance spatial resolution against sample support.

| Source | Spatial footprint | Collocation rule | Comment |
| --- | --- | --- | --- |
| HF‑Radar | Grid of application-specific resolution with sub-grid aggregation | Neighbourhood matching the aggregation footprint around each grid node and around the buoy | Same nominal support as SAR; node‑to‑node non‑overlap by design; buoy exception |
| SAR (Sentinel‑1 L2 OCN) | Grid mirroring HF‑Radar spacing with sub-grid aggregation | Neighbourhood matching the aggregation footprint around the center of each grid cell | Aggregation chosen to mirror HF‑Radar’s non‑overlapping nodes |
| Buoy station | Point measurement | Located inside the grid footprint; not necessarily on a grid node | Point‑scale reference target; HF‑Radar echoes used at the buoy may partially overlap with those used for nearest nodes |

Leakage considerations. Node‑to‑node leakage is minimized by keeping the aggregation footprint sufficiently smaller than the grid spacing for both HF‑Radar and SAR. The exception is the buoy node, where the buoy‑centred HF aggregation can partially reuse echoes also contributing to nearby grid nodes. Any cross‑sensor leakage into SAR‑supervised samples is expected to be minimal given temporal decorrelation and the multi‑day revisit of Sentinel‑1 relative to the hourly buoy sampling. A formal audit of echo‑set overlap is advisable for fine‑grained comparative conclusions.

#### Sampling Asynchrony and Collocation
Supervising sources operate at different cadences: in-situ buoys typically sample a location far more frequently than the satellite sensor, while HF‑Radar provides near‑continuous coverage. The data‑preparation pipeline harmonizes schemas, aggregates HF‑Radar and SAR over the same sub-grid neighbourhoods, aligns sensors spatially to the shared grid and the buoy location, and materializes the train/test partitions used here. Evaluation uses these collocations.

[Back to top](#introduction)

## Workflow Stages
The repository root ships with copy-and-paste runnable entrypoints (`run_*_pipeline_example.sh`) that mirror the stages described in this section. Each script demonstrates how to invoke the corresponding workflow with canonical arguments while wiring logging, configuration paths, and output directories. They are intended as reproducible templates—you can execute them as-is for smoke tests or adapt them to your infrastructure by tweaking the inline options. The examples currently cover data preparation, grid-only (SAR) training, buoy-only training, joint grid+buoy training, grid-offset benchmarking, and a composite ANN orchestration run.

Three main pipelines are evaluated:

- Grid (SAR) pipeline: trains on grid-referenced samples and evaluates both in-domain and on buoy-based partitions. Its fine-tuning stages adapt the grid model toward buoy supervision.
- Buoy pipeline: trains on buoy observations and evaluates both on the buoy-derived test partition (in-domain) and on grid data (cross-domain), with analogous fine-tuning stages.
- Combined grid+buoy pipeline: merges pre-split grid and buoy corpora into a unified training/test set to learn a single model across domains.
- Grid-offset evaluation pipeline: mirrors the pivot-and-join preparation on the offset grid (shifted by roughly one aggregation radius eastward and northward) and runs inference for the final, plain fine-tuning, L2-SP, and L2-SP+KD checkpoints of both grid and buoy models, as well as the joint model, to quantify cross-domain generalization on a shared footprint.
- Full orchestration: the composite ANN example chains the data-prep, grid, and buoy pipelines to refresh every stage in one go.

The workflows below are presented in canonical order. Notes in parentheses clarify when a stage is specific to one class of pipelines or skipped entirely.
- Data preparation, schema alignment, and stratified partitioning *(data-prep pipeline + calls to train_model.sh in other pipelines)*
- Hyperparameter optimization (HPO) over depth/width, learning rates, range-loss weights/margins, and robust features *(training pipelines only)*
- Final no-CV training with the best configuration *(training pipelines only)*
- Plain fine-tuning (no anchoring) to adapt the model to the target supervision *(training pipelines only except the combined grid+buoy pipeline)*
- Fine-tuning via L2-SP (anchoring to source weights with rehearsal) *(training pipelines only except the combined grid+buoy pipeline)*
- Optional knowledge distillation (KD) on top of L2-SP *(training pipelines only except the combined grid+buoy pipeline)*
- Inference and evaluation, including maintenance-interval diagnostics *(not in data-prep)*

[Back to top](#introduction)

### STAC Cataloging and Provenance

This is a cross‑cutting process applied whenever a stage materializes tabular geospatial outputs. After each materialization (e.g., station pivots, maintenance‑enriched tables, joined pivots, stratified train/test splits, cross‑domain concatenations, and evaluation artifacts), datasets are consolidated to GeoParquet and registered into STAC catalogs. The helpers `scripts/geo_utils/finalize_geoparquet.sh` and `scripts/geo_utils/build_stac_catalog.sh` are invoked at multiple points in the run scripts to keep catalog metadata synchronized with produced assets.

Derived tables, configurations, and evaluation artifacts can be materialized as GeoParquet and registered in STAC catalogs (for example under a `catalogs/` root and the pipeline-specific `artifacts_root/*/stac_config` tree). Logs and metrics are typically organized by stage under `artifacts_root/*/logs` and `artifacts_root/*/inference_metrics`. This layout supports end‑to‑end reproducibility and post‑hoc audits, but downstream projects may adapt the exact directory structure.

Metadata alignment and registration. To maximize interoperability:
- Geo metadata: GeoParquet footers stamp CRS (CRS84), geometry type, and dataset bounding boxes; small Parquet parts are consolidated to improve IO. See `scripts/geo_utils/finalize_geoparquet.sh` for consolidation and metadata injection.
- Table catalog: AWS Glue table schemas are kept in sync with the harmonized column set and partition layout so querying and downstream scripts remain deterministic.
- STAC: collections and items encode provenance (source URIs, time ranges, spatial footprint) to anchor train/test artifacts in a machine‑readable catalog. See `scripts/geo_utils/build_stac_catalog.sh` and `artifacts_root/*/stac_config/*`.

Exceptions - steps that do not produce GeoParquet/STAC entries:
- Final Training and Fine‑Tuning checkpoints (model weight files) stored under `artifacts_root/*/{final_training,fine_tuning*}`.
- Plots and markdown reports (e.g., `*_review.md`) under `artifacts_root/*/reports/`.
- Stage logs under `artifacts_root/*/logs/` and tuner logs under HPO subfolders.
- Per‑run configuration snapshots under `artifacts_root/*/config/` (tracked alongside runs, not cataloged as STAC items).


[Back to top](#introduction)

### Data Preparation and Schema Alignment
This stage assumes the upstream, pre‑aggregated inputs described in “Data Sources, Footprints, and Collocation,” matching the schema published in Herrera Cortijo et al. (2025, Zenodo DOI: 10.5281/zenodo.17115413). Concretely, the pipeline aligns source schemas and metadata across HF-Radar, SAR, and buoy tables produced by that aggregation workflow; attaches maintenance-interval identifiers to enable interval-aware diagnostics and normalization; and materializes pivoted and joined tables that downstream steps consume.

Preparation steps (chronological, excluding GeoParquet/STAC):
- Pivot aggregated station tables to wide form by Bragg peak (two peaks), validating unique geometry per node.
- Add station bearing and distance features to the pivoted tables (per station coordinates to grid cell).
- Attach maintenance‑interval identifiers per station to enable interval‑stratified normalization downstream.
- Join pivoted station tables and derive domain‑specific pivot views (e.g., SAR‑linked pivots; buoy‑linked pivots) under a unified schema.
- Stratify pivots into train/test (and optional CV folds) with deterministic hashing, preserving range‑class and wind‑bin distributions.
- Create partition‑specific filtered views for “valid wind” and annotate per‑partition source metadata where applicable.
- Concatenate cross‑domain partitions (e.g., SAR ∪ buoy) for combined training/testing, harmonising target names (`wind_speed`, `wind_direction`).
- Hand‑off to training (outside the data‑preparation pipeline): subsequent schema normalization, feature engineering, and maintenance‑aware normalization are executed within `scripts/training/train_model.sh`.



#### Bragg-Peak Pivot and Geometry Validation
Station-level HF-Radar aggregations arrive in a long layout with one record per combination of timestamp, node identifier, geometry, and Bragg-peak index. The `pos_bragg` field takes values in {0, 1} and indexes the negative and positive first-order Bragg peaks resolved by the spectral processor. The pivot step reshapes these inputs into a wide, model-ready table by expanding every non-key measurement into a pair of columns, one per Bragg peak. The result is a fixed-width design matrix in which each row represents a unique grid node at a given timestamp and every feature has an explicit `_0`/`_1` counterpart for `pos_bragg=0` and `pos_bragg=1`. When multiple stations are present, an optional table-name prefix is applied to output columns to avoid collisions, preserving station identity while maintaining a consistent naming scheme across sources.

The pivot preserves the geospatial keys verbatim and groups by `timestamp`, `node_id`, and `geometry`. To enforce semantic completeness and keep the feature space strictly rectangular, only groups that contain both peaks are retained; groups missing either peak are excluded rather than implicitly treated as missing data. This conservative policy prevents hidden imputation, simplifies normalization, and keeps the learning problem well-posed across stations and time.

Empirical studies show that the Bragg peak power ratio (e.g., P1/P0, or P1-P0 in dB) encodes wind/sea‑state directionality, motivating the requirement that both peaks be present at each station so the ratio can act as a station‑internal directional cue (Long & Trizna, 1973). Also, a 180 deg ambiguity can arise in wind‑direction retrievals when relying on single‑station information, and disambiguation is achieved by pairing power measurements at two stations (Gurgel et al., 2006). Enforcing both peaks at every timestamp preserves these Bragg‑ratio features and stabilizes the coupling between HF‑derived sea state and the wind vector across the multi‑station configuration.

Because spatial representativeness is anchored in the node geometry, the pipeline validates a one-to-one mapping between `node_id` and `geometry` prior to materialization. The check counts distinct geometries per node and reports any nodes mapping to more than one geometry-typically symptomatic of upstream rounding inconsistencies or rare redefinitions across maintenance windows. The stage can be configured to either warn (and list offenders) or fail fast; enforcing geometry uniqueness guarantees that subsequent unions and joins remain unambiguous and that GeoParquet metadata stays self-consistent.

Outputs of this phase are materialized as Parquet-backed Athena external tables at configured S3 prefixes, replacing any existing data and Glue table metadata at those locations. Operationally, the pivot is orchestrated by `scripts/aggregation/pivot_tables.sh`. The subsequent consolidation across stations and optional attachment of SAR or buoy references is performed by `scripts/aggregation/join_pivoted_tables.sh`.

#### Station Bearing and Distance Features
To encode the relative station-node geometry, the pivoted tables are augmented with, for each station, the great‑circle distance (in kilometres) and the azimuthal bearing (in degrees clockwise from geographic north) from the station to the grid‑cell location represented by the row’s geometry. These quantities capture two key aspects of the HF‑Radar measurement geometry: attenuation and directional sensitivity with range, and the orientation of the station with respect to the target location. They later support feature construction through distance scalars and sine/cosine encodings of bearings.

Distances and bearings are computed under WGS84 (CRS84) by referencing the station latitude/longitude and the point‑on‑surface of the row geometry (centroid when polygons are present). Distances follow a great‑circle approximation consistent with geodesic practice at coastal scales; bearings are normalized into [0 deg, 360 deg). Null geometries propagate to null distances and bearings by design, aligning with the conservative handling of missing spatial support.

This step is implemented as an Athena view layered on top of the pivoted table, created by `scripts/geo_utils/add_station_bearing_distance_view.sh`. The view appends two columns per station, `<station>_bearing` and `<station>_dist_km`, using station metadata provided at invocation time. Because geometry uniqueness per node is validated upstream, bearing and distance are well‑defined for each node across maintenance windows, preventing ambiguous station-node relationships during later unions and joins.

#### Maintenance-Interval Tagging
Instrumental drifts in HF‑Radar are slow and the calibration windows are brief relative to analysis horizons. Within each maintenance interval, it is physically reasonable to assume that the central tendency and dispersion of station‑specific power features remain approximately stationary. To enable maintenance‑aware preprocessing downstream, each station’s pivoted observations are tagged with the most recent maintenance interval in effect at the observation time.

For a given station, maintenance metadata are provided via the repository‑root CSV `calibrations.csv`, which lists `station_id`, an optional `event_type` (interpreted as maintenance type), and `effective_start` timestamps in ISO‑8601. The pipeline attaches to every row four station‑prefixed fields: a canonical interval identifier (`<station>_maintenance_interval_id`, built from the ISO start timestamp), the maintenance type (`<station>_maintenance_type` when available), the formatted start time (`<station>_maintenance_start`), and a continuous covariate with the elapsed hours since the last calibration (`<station>_hours_since_last_calibration`). The mapping uses an as‑of policy: for each observation timestamp, the latest interval start at or before that timestamp is selected; if none exists (observations predating the first recorded start), all fields remain null.

Downstream, interval identifiers gate the computation of normalization parameters for drift‑sensitive inputs, chiefly the per‑station Bragg‑peak power features and, when enabled, their dispersion summaries. Minimum support per interval is enforced; if a specific interval lacks sufficient samples, normalization falls back first to the most recent previous interval in chronological order and otherwise to global parameters, avoiding target leakage while preserving continuity across calibration windows.

This stage is executed per station by `scripts/aggregation/attach_station_maintenance_table.sh`, which materializes a Parquet‑backed Athena table enriched with the maintenance fields using a CTAS query. The resulting station‑enriched tables are then consolidated with `scripts/aggregation/join_pivoted_tables.sh` alongside other per‑station augmentations, maintaining a unified schema ready for partitioning and model ingestion.

Maintenance events are ingested from a user-provided catalog (for example, `calibrations.csv` at the repository root) that records each station’s identifier, maintenance type, and effective start time in ISO‑8601 format. During preprocessing the pipeline translates every record into a deterministic interval identifier following the `<station>_YYYYMMDDThhmmssZ` convention and attaches it to the corresponding observations. Downstream reports and partitions rely on this identifier to enforce maintenance-aware normalization and to surface interval-level diagnostics. A representative catalog would contain rows such as:

| Station | Maintenance type | Effective start (ISO‑8601) | Derived interval_id |
| --- | --- | --- | --- |
| `station_a` | calibration | `2019-01-15T00:00:00Z` | `station_a_20190115T000000Z` |
| `station_b` | hardware_replacement | `2020-06-02T08:30:45Z` | `station_b_20200602T083045Z` |

#### Join of Pivoted Stations and Domain-Specific Views
This stage consolidates the per-station pivot tables into a single, station-complete feature table and derives domain-specific variants linked to SAR products and buoy observations. Consolidation proceeds by inner-joining the pivoted station tables on the shared geospatial keys-`timestamp`, `node_id`, and `geometry`-so only rows with simultaneous coverage from all required stations are retained. This preserves a rectangular feature space with no implicit imputation across stations and ensures that every sample carries the full set of Bragg-peak features and station-relative geometry attributes introduced upstream.

To avoid column collisions and retain provenance, all non-key columns are carried forward with sanitized table-name prefixes. As a result, features appear under stable namespaces (for example, `<station>_aggregated__…` for station signals), keeping the schema unambiguous for downstream modeling and reporting. Geometry uniqueness validated earlier guarantees that the spatial join remains one-to-one at the node level across maintenance windows.

After consolidating the station pivots (the joined pivots), the pipeline optionally attaches external supervision sources to produce domain-specific pivot views:
- SAR-linked pivots: join the joined station table with the SAR aggregation table. The join condition uses `timestamp` and `node_id`, and includes `geometry` if present in the SAR table. Columns are prefixed under the `sar__` namespace to prevent clashes.
- Buoy-linked pivots: join the joined station table with the buoy observations, likewise on `timestamp` and `node_id` (and `geometry` if available). Columns are prefixed under the `buoy__` namespace.

Because SAR acquisitions and buoy measurements are asynchronous relative to HF-Radar snapshots, these joins naturally yield sparser matched sets than the station-only consolidation. This is intentional: it ensures that evaluation on SAR- or buoy-referenced targets is based on temporally co-located samples rather than on interpolations. Subsequent steps create filtered “valid wind” views and later harmonize target names and units for the combined SAR+buoy training/testing workflow.

Operationally, `scripts/aggregation/join_pivoted_tables.sh` implements both the station consolidation (producing the joined station table) and the domain-specific attachments (producing SAR- and buoy-linked pivot tables) as Athena CTAS operations that overwrite prior outputs at the configured S3 prefixes and refresh the corresponding Glue tables.

#### Partitioning and Stratification
Motivation. By partitioning we mean dividing the dataset into disjoint subsets-typically a training set for parameter estimation and a test set reserved for unbiased evaluation, with optional cross‑validation folds within training-such that membership is fixed and non‑overlapping. Partitioning establishes independent training and test sets that are statistically comparable and free of information leakage, enabling an unbiased assessment of generalization within and across domains (SAR, buoy, combined). By stratification we mean defining categorical strata (here, wind bins built from direction bins and optional speed strata, optionally crossed with an ID) and allocating samples so that each stratum’s relative frequency is approximately preserved between train and test, with explicit guarantees of minimum test coverage. Without stratification, rare regimes-specific directional sectors or in‑range speed intervals-can be under‑represented in the test set, inflating metrics and obscuring domain shifts. We therefore enforce distribution preservation by wind bins (and, when provided, by IDs), alongside deterministic rules that make splits reproducible across reruns.

Cross‑validation and folds. Cross‑validation (CV) estimates out‑of‑sample performance by repeatedly training on subsets of the training data and validating on the held‑out remainder. We define F folds (default 5) within the training split; at each CV iteration, one fold plays the role of validation while the remaining F-1 folds provide fit data. This procedure leaves the external test set untouched, reduces variance in hyperparameter decisions, and favors configurations that generalize beyond any single validation subset. Fold membership respects the same stratification by wind bins (and IDs when provided) and is deterministic under the global seed.

Method. Each record is assigned a stable hash via CRC32 of the concatenation of its `timestamp`, a stratification identifier (`location_id` by default or an explicit `--id-column`), and a fixed seed. The split rule uses this hash modulo 100 against the target train fraction (default 85%). Two guardrails force representation in the test set: the first record per ID and the first record per wind bin are always assigned to test. Training folds are then assigned cyclically within each wind bin to `F` folds (default 5), based on the cumulative count of training rows ordered by the same hash, ensuring balanced per‑bin fold allocations. Note that `wind_bin` does not enter into the CRC32 hash; bin awareness is introduced by the guardrails and by per‑bin fold assignment using row numbers computed within each `wind_bin` ordered by `hash_val`.

Wind bins. Stratification uses angular direction bins and optional speed strata combined into a single label `wind_bin`. Direction bins are computed as `floor(direction / (360 / N))`, where `N` is the chosen count of directional sectors. Common defaults are 4 bins, or 8 when data volume supports finer angular resolution. When speed strata are provided (for example, using the lower and upper limits of the admissible wind-speed band to define three buckets), speed bins are formed accordingly and paired with direction bins into labels of the form `speedBin.directionBin`.

Run configuration. Direction-bin counts, seeds, and train fractions should be selected to reflect the data volume and reproducibility requirements of each corpus. Typical defaults strike a balance between sufficient angular coverage (for example, 4 or 8 direction bins) and a generous train fraction (≈0.8–0.9), with distinct seeds per domain to avoid correlated splits. Fold indices (default F=5) remain present on training rows and are used during HPO. When required, specific IDs can be held out from training via `--exclude-ids`, which forces all their samples into test while keeping the rest of the stratification intact.

Operational tooling. `scripts/partition/partition.sh` orchestrates the entire partitioning workflow: it applies the hashing policy, enforces the guardrails, materialises the train/test tables, and writes the accompanying Markdown report with per-bin diagnostics. Helper scripts such as `scripts/partition/concat_tables_view.sh` and `scripts/partition/materialize_view.sh` support cross-domain concatenation and CTAS materialisation when combined datasets are required.

Artifacts. The partitioner materializes Parquet‑backed Athena tables named `<TABLE>_train` and `<TABLE>_test` under the provided S3 prefixes, together with Markdown reports summarizing per‑set distributions by wind bin and by ID, plus optional fold‑level descriptive statistics for selected numeric columns. In this workflow, reports are written under `artifacts_root/{grid_domain,buoy_domain}/reports/partition/` (one subfolder per observation domain) and tables registered under the `ann_training` database (for example, `PIVOTS_GRID_VALID_train/test` and `PIVOTS_BUOY_VALID_train/test`). Calibration windows are excluded by upstream “valid wind” filters; extremely sparse regimes (e.g., beyond the admissible band) are acknowledged but not used for regression assessment.

#### Partition‑Specific Filtering and Source Annotation
Following the station‑join stage and prior to modeling, domain‑specific “valid wind” filters are applied and, after partitioning, per‑partition provenance annotations are added. The goal is to ensure that target references are complete and to attach source metadata so that train/test analyses can attribute metrics to their origin.

Valid‑wind filters (domain specific). The grid-linked and buoy-linked pivot views are first narrowed to rows with complete wind references. Buoy rows must contain non-null wind speed and direction (with any ingestion-specific sentinel values removed), while grid/SAR aggregates require valid mean wind statistics. These filters do not apply frequency-specific range gating; range gating is enforced later at training/inference time. In practice the filtering logic is implemented via Athena SQL templates invoked by `scripts/aggregation/create_filtered_view.sh`, which materialises domain-specific “valid wind” views for downstream consumption.

Per-partition source annotation. After stratified splitting, each partition (train/test) is annotated with provenance fields so downstream reporting can separate grid- and buoy-referenced evaluations. The annotations add a categorical `wind_source` (for example, `grid` or `buoy`) and a deterministic `node_source_id` formed by concatenating the `node_id` with the source tag. The same helper (`create_filtered_view.sh`) applies the appropriate SQL templates under `artifacts_root/pivot_and_join/sql/` to produce the partition-specific annotated views. These tags are carried through cross-domain concatenation and used for per-source breakdowns.

These filtered and annotated tables feed into the subsequent “[Training‑Time Schema Standardization and Feature Engineering](#training-time-schema-standardization-and-feature-engineering)” stage, which aligns station‑specific columns under a unified schema for deterministic feature construction.

#### Cross‑Domain Concatenation (Combined Corpus)
When training a combined model across domains, the domain‑specific, filtered partitions are vertically concatenated under a shared schema. As part of this step, target columns are harmonised to common names (`wind_speed`, `wind_direction`) and the source annotations are retained (`wind_source`, `node_source_id`). The resulting train/test views serve as the logical input for the combined training pipeline, inheriting split membership from the source partitions without additional resampling.

Target harmonisation. Source‑specific target columns are renamed into the common pair expected by downstream training: for SAR, `sar__owiwindspeed_mean → wind_speed` and `sar__owiwinddirection_mean → wind_direction`; for the buoy, `buoy__wind_speed → wind_speed` and `buoy__wind_dir → wind_direction`. Unit conversions are not performed here because the preparation pipeline is assumed to have already aligned units and frames; this step focuses solely on renaming columns to unify the schema.

Operational notes. The concatenation is implemented as an Athena view using `scripts/partition/concat_tables_view.sh`, which accepts explicit rename maps per source to realise the harmonisation. These views are then materialised into Parquet‑backed tables with `scripts/partition/materialize_view.sh`, producing `PIVOTS_SAR_BUOY_WIND_train` and `PIVOTS_SAR_BUOY_WIND_test` under the `ann_training` database and their corresponding S3 prefixes. Set membership is inherited from the domain‑specific partitions; no additional hashing or resampling is performed in this stage.


#### Training‑Time Schema Standardization and Feature Engineering
This stage runs inside the training workflow, executed by `scripts/training/train_model.sh` and its library modules under `scripts/training/train_lib`. It is not part of the data‑preparation pipeline. The training script consumes the partitioned, filtered (and, when applicable, concatenated) tables produced upstream, resolves station‑specific schemas, and standardizes columns in‑memory before feature construction and normalization.

This section covers the following training‑time steps in detail:
- Station schema definition and resolution
- Schema alignment and standardization (targets/IDs/units + station features)
- Robust dispersion aggregates (MAD)
- Directional encoding and geometric transforms
- Feature normalization (maintenance‑aware)
- Outputs for downstream stages

##### Station Schema Definition and Resolution
Each station declares its inputs through an explicit schema provided in the model configuration JSON (e.g., files under `artifacts_root/*/config/*_model.json`, and persisted in `script_args.json` alongside checkpoints). The schema is given as a `station_schema` mapping with one entry per station, using pattern strings that the training workflow resolves into canonical column names. The common fields are:

- `power_pattern`: template for Bragg‑peak power columns (two peaks per station). Placeholders like `{agg}` and `{peak}` allow the same template to match, for example, `station_aggregated__pwr_mean_0` and `..._1`.
- `mad_pattern` (optional): template for per‑peak MAD‑based power dispersion columns, used when robust dispersion features are enabled.
- `velocity_median_pattern` (optional): template for per‑peak median radial‑velocity columns, used only when velocity‑median features are enabled.
- `bearing`: column (or template) for the station’s bearing angle to the target grid cell.
- `distance`: column (or template) for the station‑to‑target distance.
- `maintenance_interval_column` (optional): name of the column identifying maintenance intervals for that station.

Schema resolution. Within training, the station pattern strings from the model config are resolved to actual source columns and validated; missing or inconsistent mappings are surfaced as explicit errors so that subsequent steps always operate on a stable, deterministic column set. The resolution produces an ordered column list and a rename map consumed by the standardization step below. Specific canonical names are detailed in “Schema Alignment and Standardization”.

##### Schema Alignment and Standardization
This step unifies naming for targets, identifiers, and timestamps, and standardizes station feature names before feature engineering. Wind speed and direction are aligned under `wind_speed` and `wind_dir` (later encoded to cosine/sine for modeling). When present, the configured grouping column (default `location_id`) is preserved for reporting and diagnostics, while optional `fold` annotations are retained for cross‑validation within training and pruned from features thereafter. Units and temporal reference frames are assumed to have been harmonized upstream in the data‑preparation steps; the training code does not perform unit conversions or timestamp normalization.

Using the resolved rename map, station‑derived columns are standardized in a deterministic way. Bragg‑peak power columns become `<station>_pwr_0` and `<station>_pwr_1`; distances are staged as `<station>_dist_source` (then renamed to `<station>_dist` during feature engineering); and bearing angles are staged as `<station>_bearing_source` for subsequent angular encoding. When median radial‑velocity features are enabled (`use_velocity_median`), per‑station, per‑peak medians are standardized to `<station>_velo_median_<peak>` and used as additional station features. This convention ensures station‑agnostic feature construction across domains.

##### Robust Dispersion Aggregates (MAD)
When robust dispersion is enabled, per‑station MAD‑based power dispersion is read (one column per peak) and collapsed into a single aggregate `pwr_mad` via row‑wise maxima across stations/peaks. To avoid propagating missing values, any NaNs/Infs in the dispersion inputs trigger a fallback to the corresponding maxima of the standardized power columns for that row.

##### Directional Encoding and Geometric Transforms
Station bearing angles are encoded as cosine and sine pairs to preserve circular geometry and avoid discontinuities at 0/360 deg. Specifically, the training code constructs `cos_<station>_bearing` and `sin_<station>_bearing` from `<station>_bearing_source`, while distances are retained as scalar inputs after standardization. In addition, the wind‑direction target `wind_dir` is encoded to `cos_wind_dir` and `sin_wind_dir` and the raw angle dropped, ensuring circular handling in losses and metrics.

##### Feature Normalization (Maintenance‑Aware)
Prior to training, numeric features are normalized to reduce scale disparities and stabilize optimization. We adopt maintenance‑aware normalization: centers and scales are estimated per maintenance interval to absorb slow, documented changes in radar power and processing without conflating them with geophysical variability. In standard mode, features are centered and scaled using mean/std; in robust mode, median/MAD are used with safe fallbacks to mean/std where necessary. Normalization parameters are fitted on the training split and then applied consistently to validation, fine‑tuning rehearsal, and inference data via the saved parameter set. The goal is instrument consistency; normalization does not attempt to remove environmental signals.

What is normalized. All non‑angular, numeric input features are normalized. Concretely, this includes station Bragg‑peak power features (per station and peak), optional robust dispersion features (e.g., the aggregated MAD‑based proxy) and, when configured, per‑station median radial‑velocity features; geometric scalars such as station‑to‑target distance are also centered and scaled to keep the feature space well conditioned.

Maintenance‑aware scope. When station metadata includes a maintenance‑interval column, per‑interval centers/scales are computed for the station’s Bragg‑peak power features (`<station>_pwr_0`, `<station>_pwr_1`). When robust dispersion features are enabled, the per‑peak MAD inputs are ingested only to construct the aggregated `pwr_mad` feature; consequently, `pwr_mad` is normalized globally rather than per interval. Other numeric inputs (e.g., distances and per‑station median radial‑velocity features when used) also follow the global normalization path and are not normalized per interval. A minimum support per interval is enforced (24 samples by default). If an interval lacks sufficient support, normalization falls back first to the most recent previous interval in chronological order, and otherwise to the global feature parameters, preserving continuity without target leakage.

What is not normalized. Angle‑encoded inputs are deliberately left unscaled: all cosine/sine features for station bearings remain within [-1, 1] to preserve the geometry of the unit circle. Identifiers, maintenance keys, or any auxiliary columns used solely for stratification/diagnostics are not part of the model inputs and are not scaled. Targets are never normalized as inputs; they are handled by the loss functions and evaluation pipeline.

##### Outputs for Downstream Stages
The alignment stage emits station‑standardized tables in memory that expose a deterministic set of input columns for feature construction. The training code then builds `feature_cols` by concatenating standardized power columns (two per station), optional per‑station median radial‑velocity features (when configured), the aggregated `pwr_mad` (when enabled), one distance per station, and the cosine/sine bearing pairs per station. Targets are provided as wind speed and direction, with direction encoded later as cosine/sine for loss computation.

[Back to top](#introduction)

### Training Execution Workflow
A single execution harness underpins hyperparameter searches, definitive no‑CV fits, and fine‑tuning runs. `scripts/training/train_model.sh` provisions the SageMaker infrastructure (IAM role, container image, input channels), resolves the tuned manifest (station schema, normalization mode, range thresholds, and optional rehearsal/teacher settings), and submits one job per requested fold-or a single full-dataset job when `--no-cv` is engaged. Every job consumes the stratified GeoParquet partition staged during [Partitioning and Stratification](#partitioning-and-stratification) and propagates the same manifest so that configuration parity is guaranteed across HPO, final training, and fine‑tuning experiments. Detailed CLI options and implementation notes for `train_model.sh` are documented in `docs/training/train_model.md`.

The training workflow persists *checkpoints* at strategic points. In this README a checkpoint denotes the serialized state of the neural network (weights and biases) plus the metadata required to resume or audit a run-most notably the normalization statistics (`normalization_params.json`) and the CLI arguments resolved into `script_args.json`. During training checkpoints are written under `/opt/ml/model` and `/opt/ml/checkpoints` inside the SageMaker container; downstream scripts copy the relevant snapshots to `artifacts_root/*/{final_training,fine_tuning*/}` so that subsequent fine‑tuning, inference, and reporting stages can reuse the exact model state without rerunning the entire pipeline.

Optimization is handled by an Adam solver tailored to the manifest. Adam is a stochastic-gradient algorithm that maintains per-parameter first and second moment estimates (the running mean and uncentered variance of the gradients). These moments allow the optimizer to adapt the effective learning rate for each weight, accelerating convergence on poorly scaled problems while damping oscillations. The learning rate (`lr`) sets the step size of every gradient update: larger values speed up progress but risk overshooting minima, whereas smaller values stabilise training at the cost of longer runs. The learning rate and weight decay (`weight_decay`) are selected by the HPO stage and passed to `build_model_and_optimizer`; in fine‑tuning scenarios, Adam operates on discriminative parameter groups so that backbone layers and task-specific heads can use different learning rates, sharpening control over adaptation.

The inner optimization cycle alternates *forward* and *backward* propagation. The forward pass feeds each mini-batch through the shared backbone and task heads to produce wind-speed estimates, directional sine/cosine components, and range logits; the losses described in [Outputs and Losses](#outputs-and-losses) are then evaluated with the in-range mask enforcing the frequency-dependent guardrail. The backward pass-backpropagation-computes gradients of the total loss with respect to every trainable weight and applies the Adam update step. Optional penalties-knowledge distillation, L2‑SP anchoring, rehearsal-induced replay-are injected into this loss only when the fine-tuning configuration enables them (see [Fine‑Tuning Strategies](#fine-tuning-strategies)).

Training proceeds in mini-batches: compact subsets of the stratified dataset (typically 32-256 samples, matching the tuned `batch_size`) that are shuffled every epoch. Mini-batches offer a compromise between full-batch gradient descent (stable but expensive) and pure stochastic updates (fast but noisy), delivering smoother convergence while keeping memory usage tractable for highly dimensional feature vectors. A full pass over all mini-batches constitutes one epoch, after which the data order is reshuffled and the optimization loop repeats.

To avoid overfitting and needless computation, the workflow employs early stopping with a configurable patience. After each epoch the validation metrics are compared against the best historical score; if the combined loss improves, the model checkpoint is updated and the patience counter resets. If no improvement occurs for a number of consecutive epochs equal to the tuned `patience`, training halts and the best-performing checkpoint (stored earlier) is retained. This mechanism prevents the network from continuing to train once validation performance plateaus or degrades, stabilizing generalization while shortening runs when convergence is reached early.

Inside SageMaker the training routine expands into the following stages:

1. Deterministic setup and data ingestion. `train.py` activates deterministic seeds (CPU/GPU, cuDNN), resolves the GeoParquet path via `load_data`, and materializes the fold-specific train/validation split dictated by the manifest’s `fold` column; when fine-tuning augments the workflow, the handling of additional rehearsal inputs follows the policies described in [Fine‑Tuning Strategies](#fine-tuning-strategies).
2. Feature engineering and target encoding. `engineer_features` standardizes station-derived columns exactly as documented in [Training‑Time Schema Standardization and Feature Engineering](#training-time-schema-standardization-and-feature-engineering): power features (`<station>_pwr_<peak>`), optional MAD aggregates, velocity medians, cosine/sine bearings, and station distances. Wind direction targets are converted to cosine/sine, and range masks/classes are derived using the tuned `[v_min, v_max]` thresholds.
3. Maintenance-aware normalization. `normalize_features` applies the strategy detailed in [Feature Normalization (Maintenance‑Aware)](#feature-normalization-maintenance-aware), computing per-interval parameters when maintenance metadata are present, honouring robust vs. standard mode, and reusing checkpointed stats when fine-tuning resumes from an anchor.
4. Model and optimizer construction. `build_model_and_optimizer` instantiates the multitask architecture from [Architecture](#architecture), applies the learning-rate and weight-decay schedule surfaced by HPO, and constructs an Adam optimizer with discriminative parameter groups when fine-tuning is active (pattern outlined in [Fine‑Tuning Strategies](#fine-tuning-strategies)); otherwise the default Adam configuration (learning rate, weight decay) is used for a scratch run.
5. Epoch orchestration with guardrails. `run_training_loop` iterates over epochs, maintaining Dynamic Weight Averaging (temperature 2.0) to balance speed and direction losses, applying the in-range mask so that regression residuals outside the admissible wind-speed band are ignored, and logging training diagnostics (RMSE, angular error, range accuracy) at regular intervals.
6. Validation, checkpointing, and early stopping. `evaluate_metrics_full` computes held-out metrics; `compute_combined_loss` tracks the tuned objective; the best epoch is checkpointed alongside normalization parameters and manifest arguments; patience-driven early stopping halts stagnant runs and writes restart metadata under `/opt/ml/checkpoints`.

Upon completion `train_model.sh` (when `--wait-for-jobs=true`) retrieves `output.tar.gz`, extracting `normalization_params.json`, `script_args.json`, and per-fold metrics into `artifacts_root/*/{final_training,train_metrics}`. `scripts/training/record_final_job.sh` persists the job identifier, and auxiliary helpers (`get_train_metrics.sh`, `get_norm_params.sh`, `get_bin_metrics.sh`) populate reproducible artefacts reused by downstream inference, diagnostics, and documentation.

[Back to top](#introduction)

### Hyperparameter Optimization (HPO)

Hyperparameter optimization tunes the architectural and training choices that the ANN cannot infer from data alone. In this project it is the mechanism that balances the coupled objectives of in‑range wind regression, range classification, maintenance‑aware normalization, and cross‑domain generalization across the available corpora. A systematic search replaces ad‑hoc experimentation and compels every corpus to extract as much skill as possible from a limited, heterogeneous dataset.

Each tuning wave targets a specific corpus (SAR, buoy, or the combined stack) and invokes the training execution workflow described above, toggling `train_model.sh` into fold-based mode so SageMaker can drive cross-validation. The current release executes two sequential campaigns per corpus: an initial Bayesian sweep followed by a warm‑started refinement. The warm start simply instructs SageMaker to reuse the first campaign as a parent so the second one begins from the already explored regions instead of sampling blindly again. Earlier exploratory jobs remain archived for traceability but the paired campaigns capture the definitive searches referenced in this report.

Every campaign is launched through `scripts/HPO/run_hpo.sh`, which resolves the dataset‑specific configuration (`artifacts_root/sar/config/sar_hpo.json`, `artifacts_root/vilano/config/vilano_hpo.json`, `artifacts_root/sar_vilano/config/sar_vilano_hpo.json`) and dispatches up to fifty trials on Amazon SageMaker. Each trial invokes the cross‑validation driver `scripts/HPO/cv_train.py`, performing five‑fold training on the stratified training split and emitting the fold‑averaged `CombinedLoss` (see [Outputs and Losses](#outputs-and-losses)) together with wind‑speed RMSE, wind‑direction RMSE, and range‑classification macro‑F1. The tuner minimizes this combined loss while the orchestration scripts retain complete fold‑level diagnostics to guard against configurations that only perform well on a subset of folds.

Reporting is automated in two layers. Immediately after a campaign completes, `scripts/HPO/hpo_metrics_report.sh` generates a per‑job Markdown summary in `artifacts_root/<corpus>/reports/hpo/` (for example `artifacts_root/sar/reports/hpo/sar-hpo-job-i-2_hpo_report.md`). These reports tabulate every trial with its hyperparameters, objective value, and fold statistics retrieved from the compressed SageMaker artefacts. When multiple campaigns exist for the same corpus, `scripts/HPO/integrate_hpo_reports.sh` aggregates the individual reports into a global view (for example `artifacts_root/sar/reports/hpo/sar-hpo_all_hpo_report.md`), prepending the job identifier and sorting the combined table by `CombinedLoss`. This layered structure makes it straightforward to audit a single campaign or compare several generations side by side without duplicating information.

All three corpora share the same search space. Continuous ranges are explored on logarithmic scales where appropriate to cover orders of magnitude without biasing towards extremes; integer and categorical choices reflect architectural and feature toggles that have discrete interpretations in the modeling code. Because every HPO fold reuses the training workflow above, each trial produces the same artefact bundle (`normalization_params.json`, per-fold metrics, logs), enabling like-for-like comparisons. The table below summarizes the parameters exposed to HPO.

| Hyperparameter | Type | Interval / Options | Modeling role |
| --- | --- | --- | --- |
| `lr` | Continuous (log) | 1x10-⁴ - 1x10-¹ | Step size for Adam optimizer controlling convergence speed and stability |
| `range_loss_weight` | Continuous (log) | 0.25 - 4.0 | Relative weight of the range‑classification loss within the multi‑task objective |
| `range_margin` | Continuous (linear) | 0.2 - 2.0 m/s | Margin applied to the in‑range band when forming range targets |
| `epochs` | Integer | 50 - 1000 | Upper bound on training epochs per fold before early stopping |
| `hidden_layers` | Integer | 2 - 3 | Depth of the fully connected backbone |
| `hidden_units` | Categorical | {64, 128, 256, 512, 768} | Width of each hidden layer (shared across layers) |
| `dropout` | Categorical | {0.0, 0.2, 0.5} | Dropout applied between hidden layers for regularization |
| `weight_decay` | Categorical | {0, 1x10-⁴, 1x10-³, 1x10-²} | L2 penalty applied through the optimizer |
| `batch_size` | Categorical | {32, 64, 128, 256} | Mini‑batch size; influences stochasticity and memory footprint |
| `patience` | Categorical | {3, 5, 7, 10, 15, 20, 25, 30, 35, 45} | Early‑stopping patience in epochs |
| `agg_stat` | Categorical | {mean, median, max} | Aggregation statistic used when resolving per‑station power features |
| `use_mad` | Binary (0/1) | {0, 1} | Toggles robust dispersion inputs based on median absolute deviation |
| `use_velocity_median` | Binary (0/1) | {0, 1} | Toggles inclusion of per‑station radial‑velocity medians |
| `range_flag_threshold` | Categorical | {0.5, 0.6, 0.7} | Threshold separating range classes from classifier logits |

For each completed job the orchestration scripts (`scripts/HPO/integrate_hpo_reports.sh` and `scripts/HPO/select_best_hpo_job.sh`) aggregate metrics, filter trials that violate training or validation sanity checks, and export the winning configuration to the repository (`*_hpo_final_model.json` or `*_hpo_best_model.json`). Those manifests capture the finalized hyperparameters, the resolved station schema, and the identifier of the SageMaker tuning job, ensuring that downstream training and inference runs can be reproduced exactly. Quantitative summaries of the top trials are collated in the per‑pipeline HPO reports under `artifacts_root/*/reports/hpo` and are discussed in the Results section.

[Back to top](#introduction)

### Final Training (No‑CV)
Using the best HPO configuration, each corpus undergoes a definitive no‑CV fit executed exactly as described in Training Execution Workflow, with `train_model.sh --no-cv` consuming the tuned manifest without modification. Immediately beforehand, `run_*_pipeline.sh` materializes that manifest through `scripts/training/generate_model_config_from_hpo.sh`, locking in the station schema, aggregation statistic, range thresholds, and normalization mode endorsed by the selected trial. This guarantees that station coverage, normalization, regularization weights, and gating behaviour match the validated search outcome rather than an ad hoc reconfiguration.

The resulting artifacts-checkpoint identifiers, maintenance-aware normalization bundles, aggregated training diagnostics, and bin-level metrics-are preserved under `artifacts_root/*/{final_training,train_metrics}` and later drive fine-tuning experiments and inference reports. Because the no‑CV run shares the same harness as the HPO folds, its artefact layout mirrors the per-fold outputs, easing reproducibility and comparison. Training aggregates are reproduced in the Results section for sanity checking, but only held-out evaluations are used to claim generalization skill.

[Back to top](#introduction)

### Fine-Tuning
Fine-tuning resumes training of a pre-trained neural network on a target supervision corpus while retaining the representation learned on the source domain. Within this toolkit we fine-tune the SAR-only backbone and the buoy-only backbone so that each model can adapt to the complementary validation sets and the offset-grid aggregates. The joint SAR+buoy backbone is not fine-tuned: its mixed-domain training already captures both corpora, and further adaptation would erode the balance established during joint optimization. These workflows rely on the SAR corpus to encode broad spatial variability across the shared grid, whereas the buoy station, with its higher-fidelity anemometer measurements, anchors the HF-radar-to-wind relationship with precise point references.

Viewed through this lens, the two fine-tuning routes deliver complementary transfers. The SAR → buoy path takes a model accustomed to spatial gradients and re-centres it on the most reliable point measurements, seeking sharper buoy skill while tolerating some loss on the original SAR grid. The buoy → SAR path starts from a locally precise baseline and exposes it to a broader mosaic of conditions, trading a sliver of buoy fidelity for improved performance across the spatially extensive grid. Together they bracket the practical compromise between geographic coverage and pointwise accuracy that an operational deployment would need to balance.

Across all variants, the physical guardrails described in [Physical Range Gating](#physical-range-gating-frequency-specific-operating-band) remain in force: regression heads operate strictly inside the admissible wind-speed band, while the range-classification head polices samples outside it. Evaluation guardrails, including macro-F1 monitoring and maintenance-window diagnostics, continue to apply so that adaptations are judged on comparable grounds.

[Back to top](#introduction)

#### Fine-Tuning Strategies
The fine-tuning framework is implemented through three progressively constrained strategies that trade raw target-domain adaptation for better retention of source behaviour. Each variant reuses the same manifests and orchestration scripts but toggles additional regularizers or rehearsal inputs to modulate how aggressively the pre-trained backbone is altered. The following subsections summarise the configuration and expected behaviour of each route.

For a comprehensive derivation of the fine-tuning formulations (losses, regularisers, and guardrails), see the extended discussion in [docs/training/fine_tuning_anti_forgetting.md](docs/training/fine_tuning_anti_forgetting.md).

[Back to top](#introduction)

##### Plain Fine-Tuning
Serves as a control experiment that reveals how far the checkpoint can move toward the new supervision when only the upper layers are allowed to adapt. The aim is to measure the raw adaptability of the output heads and final projection while the learned representation remains frozen, providing a baseline against which the more regularized strategies can be compared.

Reloads the baseline checkpoint and keeps the feature extractor unchanged while allowing only the final projection layer and the task-specific heads to adapt. “Freezing” here means that gradients are not propagated through the earlier backbone weights, so the representation learned on the source domain is preserved verbatim. The trainable portion-the last backbone block plus the speed, direction, and range heads-acts as a slim adaptor that learns how to map the frozen features onto the new supervision. The objective mirrors the original multi-task loss (range classification plus the two regression heads), and the optimizer maintains two learning-rate groups: one for the adaptor block, another for the heads. This split lets the adaptor take smaller, cautious steps to avoid distorting the frozen features, while the heads adjust more freely to re-calibrate the outputs. Manifest knobs such as `finetune_backbone_lr` or `finetune_heads_lr` override the default step sizes when tighter or looser control is desired. This setup offers a minimal-variance baseline: it guards the shared representation from drift while still permitting moderate re-calibration of the outputs to the target corpus.

[Back to top](#introduction)

##### L2-SP Anchoring
Seeks to adapt the model to the target supervision without erasing the behavior validated on the source domain. The method constrains the shared representation to stay near its source optimum, while allowing the final layers to re-align with the new data through carefully rationed updates.

Anchors the backbone to the source optimum while mixing in a rehearsal fraction of source-domain batches. “Anchoring” the backbone means discouraging the shared feature extractor from drifting away from the representation that worked on the source domain: the network is nudged to start every update from its earlier “starting point” rather than learning a brand-new embedding. We use “rehearsal”-experience replay-to intermittently inject previously seen examples so the optimizer never loses sight of the original distribution during adaptation. The anchor is implemented through “L2-SP” (L2-Starting Point), which augments the loss with a quadratic penalty `λ‖θ-θ₀‖²` measuring the squared distance between the current backbone weights (`θ`) and the snapshot captured when fine-tuning starts (`θ₀`). The coefficient `lambda_l2sp`, stored in the fine-tuning manifest, controls how hard the optimizer pulls current backbone weights back toward their initial values, whereas the task-specific heads are exempt so they can fully re-target the outputs.

As with plain fine-tuning, all backbone layers remain frozen except for the last linear block, and the three task-specific heads are the only components that receive gradient updates. The anchor therefore operates on a largely frozen representation, steering the unfrozen adaptor layer with L2-SP while the heads recalibrate to the target supervision.

In practice, rehearsal samples a calibrated fraction of each epoch (typically 15-25%, set through `rehearsal_fraction`) from the original corpus, interleaving those batches with the target batches and applying the same normalization statistics used during baseline training. This pairing ensures that the L2-SP term acts on fresh gradients rather than stale memories and that the model continues to see the source manifold during adaptation. Learning rates follow a discriminative split-smaller on the backbone, larger on the heads-to prioritize gentle adjustments to the shared representation while letting the adaptor layers absorb residual bias. Anchor plus rehearsal therefore provide a controlled way to adapt to the target supervision without erasing the behaviour that justified the original deployment.

[Back to top](#introduction)

##### Knowledge Distillation (KD)
Targets the situations where we must transfer to the new supervision yet keep the student tightly aligned with the decision boundaries and directional behaviour of the source model. Distillation introduces a teacher-student pairing: the fine-tuned network (student) is trained not only on ground-truth labels, but also to mimic the probabilistic outputs of a frozen, high-confidence teacher checkpoint drawn from the source domain.

Adds a teacher-student penalty so the student remains close to a frozen teacher checkpoint, improving retention of source behavior while adapting to the target. The distillation term blends the target cross-entropy on range logits and the mean-squared error on regression heads with soft targets emitted by the teacher, scaled by `lambda_kd` and an adjustable temperature that controls how sharp the teacher distributions are. Teacher checkpoints are exported from the best baseline model on the source domain and frozen; their descriptors, manifest hash, and evaluation metrics are logged under `fine_tune_model/metadata.json` for auditability.

KD runs retain the L2-SP anchor and rehearsal stream by default, combining three mechanisms: structural regularization, experiential replay, and soft supervision that conveys how the source model distributes probability mass. As in the other strategies, only the last backbone block and the three task-specific heads remain trainable; the distillation loss acts on these unfrozen layers while the frozen teacher guides their updates. The design goal is to keep the source-domain decision surface intact while the heads adapt to the new supervision, accepting a slower adaptation pace and the risk of under-fitting whenever the teacher’s guidance conflicts with the target data. KD is therefore best suited to deployments that prioritize continuity of classification behaviour or directional stability inherited from the source model.

[Back to top](#introduction)

### Inference and Evaluation
Inference is the process of taking a trained checkpoint-baseline or fine‑tuned-and applying it to unseen data to obtain wind-speed, wind-direction, and range predictions. In practical terms, the corresponding pipeline materialises the held-out partitions, runs the model forward pass to generate predictions, and hands those outputs to a deterministic reporting workflow. That workflow computes metrics, applies the project’s guardrails, and assembles maintenance-aware diagnostics so each model variant can be compared on equal footing.

[Back to top](#introduction)

#### Evaluation Subsets
Because the physics underpinning the inversion hold only within a frequency-specific wind-speed band, the evaluation distinguishes between “in-range” samples (within that admissible band for the reference configuration) and the full set of observations. Regression metrics are meaningful only on the in-range subset, whereas range-classification metrics consider every sample to monitor the gating behaviour. In addition, we derive a conservative “within range + class match” view by filtering to samples whose range label is both valid and correctly classified. The reporting workflow therefore delivers complementary perspectives: in-range statistics anchored in physical validity, all-sample statistics that track the classifier’s policing role, and the class-match subset that approximates operational usability.

In addition, the evaluation always references two external supervisors—areal SAR retrievals aggregated within the same sub-grid footprint around the grid centres and in-situ buoy winds sampled hourly near the grid-cell centre—which reduces domain bias, reveals cross-sensor transfer gaps, and anchors in-range regression to validated targets. Each report surfaces per-source (SAR vs buoy) comparisons, train/test partition summaries, and maintenance-interval cohorts; the domain and partition tables quantify cross-domain transfer, while the maintenance view highlights calibration-window stability. Full tables live in the Results section.


[Back to top](#introduction)

#### Metrics and Subsets
- RMSE / MAE (in-range). Capture absolute dispersion of wind-speed errors; computed only on samples whose true wind speed lies within the frequency-specific physical gate.
- Bias (in-range). Measures systematic over- or underestimation of wind speed; helps detect persistent offsets.
- Correlation / R2 (in-range, speed only). Quantify linear association and explained variance for wind-speed predictions; directional agreement is captured instead through the angular metrics below.
- Speed SI / Speed SI max (in-range). The scatter index normalises the RMSE by the mean observed wind speed to express relative dispersion; the “SI max” column reports the maximum SI encountered across maintenance intervals and is useful to spot localised degradation even when the global SI remains acceptable.
- Directional RMSE/EAAM (in-range). Evaluates angular accuracy in degrees, using wrapped differences derived from sine/cosine direction encodings to stay robust around 0 deg/360 deg.
- Accuracy / precision / recall / F1 / macro-F1 (all samples). Accuracy measures the share of correctly classified range labels. Precision and recall are computed per class (below/within/above) to capture false-positive and false-negative tendencies; their harmonic mean yields the class-wise F1, and macro-F1 is the unweighted average of those F1 scores, providing a guardrail-friendly summary of overall classification balance. These quantities are tabulated separately from the regression metrics to keep the physical (in-range) statistics distinct from the gate-keeping performance of the classifier.

[Back to top](#introduction)

#### Inference Coverage
We run held-out evaluations for every checkpoint produced in the pipelines, covering both baseline models and their fine-tuned variants. Each evaluation emits the reports described above. The roster of inference jobs is:

- SAR baseline on the SAR test partition (own-domain) and on the buoy test partition (cross-domain).
- SAR fine-tuned (L2-SP) on the buoy test partition to quantify adaptation from SAR to buoy supervision, plus optional evaluation back on the SAR test set for retention checks.
- SAR fine-tuned (L2-SP + KD) on both SAR and buoy test partitions when knowledge distillation is enabled, maintaining SAR behaviour while adapting to the buoy corpus.
- Buoy baseline on the buoy test partition (own-domain) and on the SAR test partition (cross-domain).
- Buoy fine-tuned (L2-SP) on the SAR test partition with optional buoy retention assessment.
- Buoy fine-tuned (L2-SP + KD) on both buoy and SAR test partitions to balance buoy fidelity and SAR transfer.
- Joint SAR+buoy baseline on the combined held-out partition, including per-source breakdowns (SAR-labelled vs buoy-labelled samples).
Detailed metrics for every run are tabulated in the Results section.

[Back to top](#introduction)

#### Maintenance-Interval Diagnostics
For every calibration interval we recompute the evaluation metrics using samples from that window only, compare the results against the guardrail thresholds, and annotate them with contextual metadata (maintenance type, hours since calibration). Because HF-Radar power drifts slowly between brief calibration windows, the evaluation harness applies maintenance-aware normalization, excludes calibration windows from assessment, and tags every report with interval identifiers. Intervals breaching the thresholds are surfaced in the Results section together with suggested actions such as normalization refinements or calibration checks.
## Related Scripts (per phase)
- Data preparation (Athena/Glue): `scripts/aggregation/pivot_tables.sh`, `scripts/geo_utils/add_station_bearing_distance_view.sh`, `scripts/aggregation/attach_station_maintenance_table.sh`, `scripts/aggregation/join_pivoted_tables.sh`, and `scripts/aggregation/create_filtered_view.sh` produce the pivoted, joined, filtered, and maintenance‑aware tables.
- Partitioning and materialization: `scripts/partition/partition.sh` (stratified splits), `scripts/partition/concat_tables_view.sh` (cross‑domain stacking with renames), and `scripts/partition/materialize_view.sh` (persist views as Parquet and register Glue tables).
- GeoParquet and STAC (cross‑cutting): `scripts/geo_utils/finalize_geoparquet.sh` consolidates parts and injects GeoParquet metadata; `scripts/geo_utils/build_stac_catalog.sh` publishes STAC collections/items using `artifacts_root/*/stac_config`.
- Training and fine‑tuning: `scripts/training/train_model.sh` drives training; core preprocessing lives in `scripts/training/train_lib/features.py` (feature engineering) and `scripts/training/train_lib/normalization.py` (maintenance‑aware normalization). Fine‑tuning helpers include `scripts/training/prepare_finetune_config.sh` and `scripts/training/record_final_job.sh`.
- HPO: `scripts/HPO/run_hpo.sh`, `scripts/HPO/cv_train.py`, `scripts/HPO/integrate_hpo_reports.sh`, `scripts/HPO/select_best_hpo_job.sh`, and `scripts/HPO/extract_best_hpo_metrics.sh` coordinate cross‑validated searches and report integration.
- Inference and metrics: `scripts/inference/run_inference.sh`, `scripts/inference/inference.py`, `scripts/inference/compute_inference_metrics.sh`, and `scripts/inference/compute_inference_metrics.py` generate predictions and consolidated evaluation reports, now including wind-bin stratified tables (`inference_metrics_by_wind_bin*.csv`) that mirror the training-time bin diagnostics.


[Back to top](#introduction)

## Acknowledgements

This work has been funded by the HF-EOLUS project (TED2021-129551B-I00), financed by MICIU/AEI /10.13039/501100011033 and by the European Union NextGenerationEU/PRTR - BDNS 598843 - Component 17 - Investment I3. Members of the Marine Research Centre (CIM) of the University of Vigo have participated in the development of this repository.


[Back to top](#introduction)

## Disclaimer

This software is provided "as is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages, or other liability, whether in an action of contract, tort, or otherwise, arising from, out of, or in connection with the software or the use or other dealings in the software.


[Back to top](#introduction)

## References
- Emery, B. & Kirincich, A. (2021). Ocean Remote Sensing Technologies: High frequency, marine and GNSS-based radar. In *Ocean Remote Sensing Technologies* (pp. 191–216). Institution of Engineering and Technology. https://doi.org/10.1049/sbra537e_ch8

- Gurgel, K.-W., Essen, H.-H., & Schlick, T. (2006). An empirical method to derive ocean waves from second-order Bragg scattering: prospects and limitations. *IEEE Journal of Oceanic Engineering*, 31, 804–811.

- Herrera Cortijo, J. L., Fernández-Baladrón, A., Rosón, G., Gil Coto, M., Dubert, J., & Varela Benvenuto, R. (2025). *Project HF-EOLUS. Task 2. Aggregated SAR and HF-Radar Radial Metrics for Wind-Inversion Model Training* [Data set]. Zenodo. https://doi.org/10.5281/zenodo.17115413

- Long, A., & Trizna, D. (1973). Mapping of North Atlantic winds by HF radar sea backscatter interpretation. *IEEE Transactions on Antennas and Propagation*, 21, 680–685.

- Shen, W., Gurgel, K.-W., Voulgaris, G., Schlick, T., & Stammer, D. (2011). Wind-speed inversion from HF radar first-order backscatter signal. *Ocean Dynamics*, 62, 105–121. https://doi.org/10.1007/s10236-010-0359-9





---
<p align="center">
  <a href="https://next-generation-eu.europa.eu/">
    <img src="logos/EN_Funded_by_the_European_Union_RGB_POS.png" alt="Funded by the European Union" height="80"/>
  </a>
  <a href="https://planderecuperacion.gob.es/">
    <img src="logos/LOGO%20COLOR.png" alt="Logo Color" height="80"/>
  </a>
  <a href="https://www.aei.gob.es/">
    <img src="logos/logo_aei.png" alt="AEI Logo" height="80"/>
  </a>
  <a href="https://www.ciencia.gob.es/">
    <img src="logos/MCIU_header.svg" alt="MCIU Header" height="80"/>
  </a>
  <a href="https://cim.uvigo.gal">
    <img src="logos/Logotipo_CIM_original.png" alt="CIM logo" height="80"/>
  </a>
  <a href="https://www.iim.csic.es/">
    <img src="logos/IIM.svg" alt="IIM logo" height="80"/>
  </a>

  
</p>

[Back to top](#introduction)
