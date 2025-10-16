# Stratified Partitioning Workflow

## Purpose
`scripts/partition/partition.sh` orchestrates stratified train/test (and optional cross-validation folds) for the pivoted datasets stored in Athena. It bins wind direction (and optionally wind speed) to preserve distributional balance, assigns per-ID holdouts when requested, and writes Parquet-backed Athena tables for the `train` and `test` splits under a shared S3 prefix. Fold membership is materialised as a column within the training table rather than as separate per-fold relations. All splits are reproducible: the assignment derives from deterministic hashing of timestamp, location, and a user-provided seed, so rerunning the script with the same parameters yields byte-identical partitions.

## Stratified Cross-Validation Rationale
Cross-validation estimates generalisation performance by training on `K−1` folds and validating on the remaining fold, cycling through all folds and averaging the metrics. Naïve K-fold splits assume that samples are i.i.d. and that the response distribution is homogeneous; neither assumption holds for HF radar wind retrievals. Wind direction exhibits marked anisotropy—certain sectors dominate because of coastline geometry and prevailing weather systems—and the learnable signal depends on wave conditions, which in turn are governed by fetch (the distance travelled by wind over water). Two observations with identical wind speed but different azimuth can produce radically different radar backscatter intensities. If folds over-represent a particular direction band, model evaluation becomes biased toward the fetch regime seen in that fold.

Stratified cross-validation addresses this by enforcing proportional representation of the discretised target in every fold. Direction alone is not sufficient: radar cross-sections also depend on the energy available in the wave field, which is largely a function of wind speed. The empirical wind-speed distribution in coastal deployments is highly non-uniform (calm conditions dominate while gales remain sparse), so naïve sampling tends to overweight the most common magnitude regime. In addition, HF-radar wind inversion is governed by theoretical lower and upper speed limits: below ~3–4 m/s the Bragg signal falls below the noise floor, while above ~20 m/s sea-state decorrelation and wave-breaking degrade the retrieval. If these boundary regimes are under-represented during training or evaluation, error statistics will mischaracterise real-world performance. To mitigate this, the script optionally combines angular bins with user-defined speed thresholds (`--speed-strata`). This separates calm, moderate, and storm-driven observations, preventing the model from overfitting to low-variance segments of the data or under-exposing rare high-speed cases that dominate operational risk assessments.

The workflow implemented by `partition.sh` mirrors the following theoretical steps:

1. **Discretise the continuous target.** Divide the 360° circle into `--direction-bin-count` equal sectors. Optionally combine them with speed strata to produce composite bins that capture both direction and magnitude influences on radar response. Typical thresholds mirror the Beaufort scale (e.g., 5.7 m/s for moderate breeze, 12 m/s for near-gale) so that each bin spans a distinct flow regime with consistent signal-to-noise characteristics.
2. **Label each observation.** Assign every row the index of its angular (or composite) bin so folds can treat the problem as a stratified sampling exercise over categorical bins.
3. **Apply stratified splitting.** Ensure that the proportion of samples per bin is preserved across the `K` folds. The script still computes a deterministic hash to derive a reproducible ordering, but the acceptance test now operates on wind-bin quotas: forced hold-outs are honoured first, then each bin receives its target number of test samples, and only the remaining rows are admitted into the training cohort. The surviving training rows are enumerated per `wind_bin`, and folds are assigned by cycling through that ordering so every bin distributes its quota evenly across folds.
4. **Respect grouping constraints.** When the user passes `--id-column`, the splitter must honour group boundaries (e.g., individual grid nodes on the HF radar footprint or specific buoy moorings) while still keeping the per-bin proportions balanced. Adjacent nodes share calibration artefacts and viewing geometries, so letting samples from the same physical point fall into both train and test tends to inflate accuracy. The script couples stratification labels with ID-aware hashing, reserves the first occurrence per ID and per wind bin for the test set, and only then applies the round-robin training assignment so no fold or holdout set violates these group-based leakage requirements.

The resulting folds offer three guarantees: (1) every fold contains representative samples from rare direction/speed combinations, including high-energy storms, (2) averaged performance has lower variance because folds are comparable, and (3) validation is fair—no fold disproportionately favours a specific fetch or wind-intensity regime. When the training corpus mixes multiple grid nodes or buoy stations, keeping groups intact is equally important: neighbouring radar cells share hardware biases, and buoy deployments experience micro-climatic quirks. Optional location stratification (`--id-column`) prevents these correlated clusters from leaking between train and test while still enforcing directional/speed balance. Together, these properties align the statistical evaluation with the physical processes driving HF radar measurements, leading to more reliable model selection and downstream error estimates.

## Prerequisites
- AWS CLI v2 with permissions to execute Athena CTAS queries, query results, and manage Glue tables.
- `jq`, `bc`, and `awk` available locally for query inspection and report generation.
- An upstream dataset (typically from `create_filtered_view.sh`) that exposes at minimum:
  - `timestamp` (the script hashes on this field—rename upstream via a view if the source column differs).
  - A wind-direction column (default `wind_direction`).
  - Optionally a wind-speed column (default `wind_speed`) when using `--speed-strata`.
  - An ID column if stratifying by location or platform; otherwise the script injects a constant `location_id = 1` for balancing purposes.

## Key Options
- `--input-db <db>` / `--table <table>`: Source Athena database and table to partition.
- `--output-db <db>`: Destination database to host the generated partitions.
- `--s3-prefix <s3://bucket/prefix>`: Target prefix for the Parquet outputs.
- `--profile` / `--region`: AWS CLI configuration controlling Athena execution.
- `--results-s3 <s3://...>`: Optional scratch bucket for query results.
- `--folds <k>`: Number of cross-validation folds (default `5`).
- `--seed <value>`: Deterministic seed for hashing rows into folds (default `20250510`).
- `--direction-bin-count <n>`: Number of equal-width wind direction bins (default `8`).
- `--speed-strata <limits>`: Comma-separated wind speed thresholds to create combined `wind_bin = speedBin.directionBin` bins.
- `--direction-column <name>` / `--speed-column <name>`: Column names to use when the defaults (`wind_direction`, `wind_speed`) differ.
- `--id-column <name>`: Optional identifier used to guarantee representation across partitions; a constant placeholder is used when omitted.
- `--exclude-ids <list>`: Force listed IDs into the test set.
- `--train-fraction <frac>`: Training fraction (default `0.85`); the remainder becomes the holdout test set.
- `--report-file <path>` / `--local-dir <dir>`: Control where Markdown summary reports and logs are saved.

## Outputs
- Athena tables `<table>_train`, `<table>_test`, and `<table>_fold_<k>` stored in `--output-db`, each backed by Parquet files under `<s3-prefix>/train`, `<s3-prefix>/test`, and `<s3-prefix>/fold_<k>`.
- Markdown report describing distribution by set, wind bins, IDs, combined strata, and a fold-balance diagnostic section (written alongside logs in `--local-dir`).
- Console logs with Athena query identifiers for traceability.
- Each record carries two new attributes: `set_type` (one of `train`, `test`) and `fold`. Training rows receive fold numbers `1..K`; holdout rows are kept with `fold = 0` so downstream tooling can ignore them when iterating folds.

## Execution Flow
1. **Parse options and validate inputs.** Missing `--input-db`, `--table`, `--output-db`, or `--s3-prefix` immediately abort the run. The script materialises a log file `${OUTPUT_DIR}/partition.log` capturing every AWS CLI command.
2. **Resolve stratification identity.** If `--id-column` is provided, that column is used directly and later used to enforce minimum coverage per site. This is essential when working with gridded HF radar nodes or buoy stations: neighbouring grid points share beam geometry and calibration, and buoy moorings experience local bathymetry and sheltering effects. Keeping each location intact prevents artificially optimistic validation. When no ID is given, the script adds `location_id = 1` in the CTAS query so every row belongs to the same synthetic group, ensuring the downstream row numbering logic still works.
3. **Derive bin geometry.** `--direction-bin-count` determines `bin_width = 360 / bins`. When `--speed-strata` is supplied (comma-separated thresholds), a CASE expression assigns an integer `speed_bin`; the final `wind_bin` becomes `"speed_bin.direction_bin"`. Without speed strata, `wind_bin` is the integer direction bin.
4. **Hash rows deterministically.** A base CTE computes `hash_val = crc32(CONCAT(CAST(timestamp AS VARCHAR), '#', id_value, '#', seed))`. CRC32 is a fast checksum that maps each row to an integer in `[0, 2³²)`. Because the inputs (timestamp, ID, seed) are fixed, the hash outcome is repeatable: the same dataset and seed will always produce identical partitions. Switching the seed simply rotates the assignment in a controlled way, giving you alternative partitions without sacrificing reproducibility.
5. **Guarantee coverage.** A second CTE applies two window functions: `row_number()` over `(id)` and over `(wind_bin)`. The first row per ID and per wind bin are reserved for the test set (see “Coverage guarantees”), guaranteeing that every location contributes at least one holdout sample while also providing the running order that will drive the fold assignment.
6. **Assign `set_type` and `fold`.** Rows with `row_id = 1`, `row_wind_bin = 1`, or matching `--exclude-ids` are labelled `test`. The remaining rows compare `mod(hash_val, 100)` against the requested training percentage (`--train-fraction`) to set an `is_train` flag. Training rows receive a cumulative count per `wind_bin` (`train_row_number`) and folds are assigned in round-robin order via `fold = ((train_row_number - 1) % K) + 1`. Non-training rows retain `fold = 0`.
7. **Materialise CTAS outputs.** For each of `train` and `test`, the script drops any existing Glue table, clears the corresponding S3 prefix, and issues an Athena CTAS statement to write compressed Parquet files. Query scratch data is stored in `${S3_PREFIX}/athena-results/` unless overridden by `--results-s3`. Each CTAS embeds the `set_type` and `fold` expressions so that every row emerges from Athena already labelled.
8. **Generate per-fold tables (optional).** If cross-validation is requested (`--folds > 1`), individual fold tables are produced as part of the CTAS outputs and inherit the same `set_type`/`fold` layout.
9. **Build the Markdown report.** Secondary Athena queries count rows for: (a) overall set totals (`SELECT set_type, COUNT(*)`), (b) fold × `wind_bin` matrices, (c) marginal `wind_bin` histograms, (d) ID distributions, and (e) ID × `wind_bin` matrices. When the user provides `--fold-stat-columns`, the script inspects each nominated feature in the output schema: numeric types trigger an additional query (f) that computes per-fold descriptive statistics—counts, means, standard deviations, minima, and maxima—while categorical or timestamp-like fields yield query (g), which ranks the ten most frequent values per fold (flagging nulls explicitly) and narrates the dominant categories. The script post-processes the fold/bucket matrix to emit a “Fold Balance Checks” block that highlights bins with empty folds or large max/min ratios, writing warnings both to the Markdown file and to the log. The results are rendered into `${OUTPUT_DIR}/${OUTPUT_DB}_partition_report.md` (or a custom `--report-file`).

## Typical Usage
```bash
./scripts/partition/partition.sh \
  --input-db <source_db> \
  --table <source_table> \
  --s3-prefix s3://<bucket>/<training_prefix> \
  --profile <aws_profile> \
  --output-db <target_db> \
  --direction-column <direction_column> \
  --speed-column <speed_column> \
  --speed-strata "5.7,8,10,12,14,17.8" \
  --report-file <local_report_path> \
  [--fold-stat-category <categorical_column>] \
  --seed 20250522
```
This template partitions any table that exposes compatible wind speed and direction columns; adjust the placeholders to mirror your storage layout and naming scheme.

## Behaviour Notes
- Stratification combines angular bins with optional speed strata to form the `wind_bin` column stored in each output table. The bin schema is echoed in the Markdown report.
- Hash-based assignment guarantees reproducible splits given the same seed and dataset.
- When `--exclude-ids` is supplied, excluded identifiers bypass the training selection and are added to the test set while still contributing to fold statistics.
- Glue databases referenced in `--output-db` are created as needed.
- `--fold-stat-columns` inspects the warehouse data type of every nominated feature: numeric fields produce per-fold descriptive statistics, whereas categorical or timestamp-like columns render top-ten frequency tables (nulls included) with narrative highlights, making maintenance-driven skews immediately visible.
- `--fold-stat-category <col>` (repeatable) adds per-fold descriptive tables for each nominated categorical feature. When multiple categories are supplied, the report also emits a combined breakdown using the Cartesian grouping of all categories, helping analysts contrast regimes such as maintenance intervals, radials, or quality flags.

## Balanced Partitioning Mechanics
The script embeds the full stratification machinery inside the Athena CTAS queries so every run is deterministic and self-documenting. Key elements are:

- **Wind-direction bins**: `wind_bin` is computed as `CAST(FLOOR(<direction_column> / bin_width) AS INTEGER)`, where `bin_width = 360 / --direction-bin-count`. When `--speed-strata` is present, the direction bin is concatenated with the speed stratum index (`speedBin.directionBin`) to preserve both angular and velocity balance.
- **ID-aware hashing**: Each row is assigned a deterministic 32-bit hash `crc32(CONCAT_WS('#', <timestamp>, <id>, <seed>))` (see `README.md`, “Stratified Cross-Validation for Data Partitioning”). Although the hash acts like a random number generator, it is entirely reproducible: with the same inputs you always get the same integer, and therefore the same partition. When `--id-column` is provided, the ID becomes part of the hash input so that entire locations shift together between training and test when seeds change, avoiding partial leakage of site-specific artefacts.
- **Train/test quotas**: Each wind bin computes a baseline number of test samples by rounding `wind_bin_total × (1 - train_fraction)` to the nearest integer. The script protects the hold-out guarantees by tracking `forced_test` rows (mandatory ID/bin representatives and any `--exclude-ids`) and bumps the quota to at least the count of forced rows. Within the remaining population, rows are ranked by the deterministic hash and admitted into the test set until the quota is met; the rest become training samples.
- **Fold balancing**: Training rows derive their fold from a per-bin running count rather than directly from the hash. Specifically, `train_row_number` counts accepted training rows per `wind_bin`, and the fold becomes `((train_row_number - 1) % --folds) + 1`. This rotation keeps the training distribution balanced even when some bins expend more rows satisfying the test quota. Test rows keep `fold = 0` and remain available for final evaluation.
- **Coverage guarantees**: Window functions `row_number() OVER (PARTITION BY id ORDER BY hash_val)` and `row_number() OVER (PARTITION BY wind_bin ORDER BY hash_val)` identify the first occurrence per group. Those rows are tagged as `forced_test`, ensuring every ID and every populated wind bin contributes at least one observation to the holdout split. If `--exclude-ids` is provided, the filter short-circuits both assignments and sends the entire ID to the test cohort. This guarantees that each grid node or buoy location’s calibration biases and environmental conditions are measured against unseen data instead of being absorbed silently into the training set.
- **Report validation**: After the partitions are written, supplemental Athena queries populate the Markdown report (`<output_db>_partition_report.md`). The report includes per-set totals, wind-bin histograms, ID distributions, and combined ID/bin matrices, allowing you to verify empirically that the balancing logic behaves as described.

### CTAS SQL Skeleton
For each set, the script emits a CTAS query whose CTE chain mirrors the following structure (simplified for clarity):

```sql
WITH base AS (
  SELECT
    t.*,
    wind_bin_expression AS wind_bin,
    crc32(CAST(CONCAT(
      CAST(timestamp AS VARCHAR),
      COALESCE(CAST(id_value AS VARCHAR), '1'),
      'SEED'
    ) AS VARBINARY)) AS hash_val
  FROM source_table AS t
),
ranked AS (
  SELECT
    base.*,
    row_number() OVER (PARTITION BY id_value ORDER BY hash_val) AS row_id,
    row_number() OVER (PARTITION BY wind_bin ORDER BY hash_val) AS row_wind_bin,
    COUNT(*)     OVER (PARTITION BY wind_bin) AS wind_bin_total
  FROM base
),
annotated AS (
  SELECT
    ranked.*,
    CASE
      WHEN id_value IN (<exclude_ids>) THEN TRUE
      WHEN row_id = 1 THEN TRUE
      WHEN row_wind_bin = 1 THEN TRUE
      ELSE FALSE
    END AS forced_test
  FROM ranked
),
quota AS (
  SELECT
    annotated.*,
    CAST(ROUND(wind_bin_total * (1 - train_fraction)) AS INTEGER) AS base_test_quota,
    SUM(CASE WHEN forced_test THEN 1 ELSE 0 END) OVER (PARTITION BY wind_bin) AS forced_test_count,
    SUM(CASE WHEN forced_test THEN 1 ELSE 0 END) OVER (
      PARTITION BY wind_bin ORDER BY hash_val
      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS forced_test_cume
  FROM annotated
),
labeled AS (
  SELECT
    quota.*,
    CASE
      WHEN forced_test THEN FALSE
      WHEN (row_wind_bin - forced_test_cume) <= GREATEST(base_test_quota - forced_test_count, 0)
        THEN FALSE
      ELSE TRUE
    END AS is_train
  FROM quota
),
augmented AS (
  SELECT
    labeled.*,
    SUM(CASE WHEN is_train THEN 1 ELSE 0 END) OVER (
      PARTITION BY wind_bin ORDER BY hash_val
      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS train_row_number
  FROM labeled
)
SELECT
  augmented.*,
  CASE WHEN is_train THEN ((train_row_number - 1) % folds) + 1 ELSE 0 END AS fold,
  CASE WHEN is_train THEN 'train' ELSE 'test' END AS set_type
FROM augmented
WHERE set_type = 'train' -- or 'test' depending on the CTAS invocation
```

All uppercase tokens are parameterised at runtime. When no ID column is supplied, the shell script injects a constant `location_id = 1` so the window functions have a deterministic grouping key.

## Reproducibility Guarantees

- **Seed-controlled randomness**: The only stochastic-looking element is the CRC32 hash, yet its inputs are deterministic. Using the same source table and `--seed` always yields byte-identical training/test splits. Rotating the seed gives alternative splits without breaking reproducibility.
- **Explicit assignment formulas**:
  - `hash_val = crc32(CAST(CONCAT(CAST(timestamp AS VARCHAR), COALESCE(CAST(id_value AS VARCHAR), '1'), '<seed>') AS VARBINARY))`
  - `forced_test = (row_id = 1) OR (row_wind_bin = 1) OR (id_value IN <exclude_ids>)`
  - `is_train = NOT forced_test AND (non_forced_rank > residual_test_quota)`
  - `train_row_number = SUM(CASE WHEN is_train THEN 1 ELSE 0 END) OVER (PARTITION BY wind_bin ORDER BY hash_val ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)`
  - `fold = CASE WHEN is_train THEN ((train_row_number - 1) % folds) + 1 ELSE 0 END`
  - `set_type = CASE WHEN is_train THEN 'train' ELSE 'test' END`
  The shell script materialises these expressions inside the CTAS query, so every run recomputes the assignment from raw columns rather than reusing cached metadata.
  Here `non_forced_rank = row_wind_bin - forced_test_cume`, while `residual_test_quota = GREATEST(target_test_quota - forced_test_count, 0)` derives from the quota-balancing CTEs described above.
- **Logged configuration**: Every execution logs the full option set (`bins`, `seed`, `train_frac`, `speed strata`, `id column`, etc.) to `${OUTPUT_DIR}/partition.log`, making it straightforward to replay the run later.
- **Explicit CTAS outputs**: The log captures the opening portion of each generated CTAS statement (the first twenty lines) alongside the configuration. Combine this snippet with the script version and parameters to reproduce the exact SQL if full archival is required.

## Maintenance Interval Diagnostics

Feeding maintenance-enriched pivot tables into the partitioner introduces four descriptors per radar (`*_maintenance_interval_id`, `*_maintenance_type`, `*_maintenance_start`, and `*_hours_since_last_calibration`). The Markdown report now surfaces these fields in two complementary ways. First, the numeric recency indicators (`*_hours_since_last_calibration`) appear in the fold-level descriptive statistics table so that drifts in maintenance recency or abrupt gaps become visible. Second, the categorical siblings generate a dedicated “fold-level categorical distributions” section where the ten most frequent interval identifiers, maintenance types, and start timestamps are ranked for every fold along with short textual highlights.

These diagnostics are designed to answer two audit questions: (1) do all folds observe comparable maintenance intervals, avoiding a scenario where a single calibration window monopolises the validation set, and (2) are there folds whose recent maintenance episodes differ so starkly from the rest that model performance might hinge on radar health rather than generalisable signal? When reviewing a freshly generated report, confirm that the top interval identifier in each fold captures a similar fraction of samples. As a rule of thumb drawn from coastal HF-radar deployments, any fold exceeding ~35% share for a single interval merits a closer look at the upstream pivot process. Likewise, verify that the mean and standard deviation of `*_hours_since_last_calibration` do not diverge by more than a few dozen hours across folds; larger gaps usually point to missing maintenance metadata or an imbalanced ingestion of calibration logs.

The quickest manual check after running your partition workflow is therefore:
1. Regenerate the partition report (for example, `<local_report_path>` from the command above).
2. Inspect the “fold-level categorical distributions” section and note the highlighted dominant intervals for each fold.
3. Cross-reference those folds with the numeric recency statistics to ensure that none of them align exclusively with either brand-new or long-unserviced radar states. When one or more `--fold-stat-category` entries are provided, review the additional tables—including the combined breakdown—and confirm that each category retains representative samples across folds and that their descriptive statistics stay aligned with the global aggregates.
Documenting the outcome of this review keeps the wind-bin stratification narrative intact while guaranteeing that maintenance metadata does not undermine the fold balance achieved by the hashing scheme.

## Composing cross-domain splits without re-partitioning

When the SAR-only and stationX-only pipelines already provide vetted train/test splits, it is often preferable to reuse those labels instead of triggering another round of stratification on the concatenated corpus. The recommended workflow is:

1. Annotate each partitioned table with deterministic provenance fields (`wind_source`, `node_source_id`) via `create_filtered_view.sh`, mirroring the approach follow for the full-domain views.
2. Harmonise the per-domain train and test tables separately using `concat_tables_view.sh`, renaming wind speed/direction columns to a shared schema while preserving the source annotations.
3. Materialise each harmonised view (`..._TRAIN` and `..._TEST`) with `materialize_view.sh`, which issues an Athena CTAS and refreshes the S3-backed Parquet datasets without altering the original, domain-specific partitions.

The resulting combined splits inherit the original sampling decisions, guaranteeing comparability with single-domain baselines and eliminating the stochastic variability that a fresh split would introduce. Because the concatenation is purely vertical, any gating or physical-range enforcement already encoded in the individual partitions remains intact.

## Troubleshooting
- *Athena query failures*: Inspect the generated log file (e.g., `partition_<table>.log`) in `--local-dir` for the failing SQL and AWS error messages.
- *Missing wind columns*: Supply `--direction-column` / `--speed-column` flags that match the source schema after prefixing (e.g., `sar__owiwinddirection_mean`).
- *Uneven distributions*: Adjust `--speed-strata`, `--direction-bin-count`, or the training fraction to better match the source data characteristics described in `README.md`.

## File Reference
- `scripts/partition/partition.sh`
