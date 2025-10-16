# Anti‑Forgetting Fine‑Tuning for Range‑Aware Wind Inversion

## Overview

This document describes the design and mathematical formulation of a fine‑tuning procedure that preserves the pre‑trained calibration of a range‑aware neural model while adapting to a new domain (for example, transferring from SAR retrievals to buoy winds or the reverse). The method addresses the catastrophic forgetting observed when simply unfreezing the backbone and continuing training on the target domain. Crucially, nothing in this design alters the baseline training pipeline; the anti‑forgetting mechanics engage only in fine‑tuning runs initialized from an existing checkpoint.

## Physical Gating and Task Formulation

The model is range‑aware by construction. It jointly learns (i) a regression of wind speed and direction and (ii) a 3‑class range classifier that flags whether the true wind speed lies below, inside, or above the physically valid interval specified by the configuration. The regression terms are evaluated only for samples inside the valid range, reflecting that the physical mapping from HF‑Radar signatures to wind components is not calibrated outside that interval. The range classifier trains on all samples and provides useful context to the regressor at the operational boundaries.

Let yi denote the targets for sample i: speed si and direction θi. Direction is encoded via a 2‑vector (cos θi, sin θi). Let ŷi denote the model predictions (ŝi, ĉi, ŝi^{sin}), where ĉi and ŝi^{sin} are the tanh‑bounded cosine and sine logits output by the network, and let ri ∈ {below, in, above} be the ground‑truth range class. Define an in‑range mask Mi ∈ {0,1} that equals 1 when si is in the configured interval [Smin, Smax] (with potential safety margin), and 0 otherwise. Denote the current mini‑batch by \mathcal{B}. The batch task loss implemented in training is

$$
\mathcal{L}_{\text{task}}(\mathcal{B}) =
\frac{w_S \sum_{i \in \mathcal{B}} M_i (\hat{s}_i - s_i)^2 + w_D \sum_{i \in \mathcal{B}} M_i \Delta \theta_i^2}{\max \left( 1, \sum_{i \in \mathcal{B}} M_i \right)}
+ \frac{\lambda_{\mathrm{R}}}{\lvert \mathcal{B} \rvert} \sum_{i \in \mathcal{B}} CE(r_i, \hat{p}_i).
$$

where CE is the cross‑entropy of the range classifier, \( \hat{p}_i \) are the predicted class probabilities, and \(w_S\) and \(w_D\) coincide with the dynamic regression weights (`wS`, `wD`) exposed by the training loop. The wrapped angular error reproduces the exact computation in the training loop:

$$
\hat{\theta}_i = \operatorname{atan2}(\hat{s}^{\sin}_i, \hat{c}_i), \qquad
\Delta \theta_i = \operatorname{atan2}\left(\sin(\hat{\theta}_i - \theta_i), \cos(\hat{\theta}_i - \theta_i)\right).
$$

Because the directional logits are not renormalised before atan2, the denominator guards against division by zero whenever the entire batch falls outside the calibrated range (∑ Mi = 0). The classification term, instead, is always averaged over the |\mathcal{B}| samples, preserving the global context supplied by the range predictions.

## Anchored Regularization (L2‑SP)

To mitigate forgetting, the fine‑tuning stage penalizes deviations from the pre‑trained weights W0 captured at the start of the adaptation. The L2‑SP penalty is a trust‑region style quadratic around W0:

$$
\Omega_{\mathrm{L2SP}}(W; W_0) = \lambda_{\mathrm{bb}} \sum_{j \in J_{\mathrm{bb}}} \lVert W_j - W_{0,j} \rVert^2 + \lambda_{\mathrm{hd}} \sum_{k \in J_{\mathrm{hd}}} \lVert W_k - W_{0,k} \rVert^2.
$$

where Jbb collects the backbone parameters that remain trainable during fine‑tuning (in this design, only the last linear block), and Jhd collects the task‑specific heads (speed, direction, range). Separate coefficients λbb and λhd allow stronger anchoring on the backbone than on the heads, biasing adaptation to occur primarily through the readouts. Parameters that remain frozen carry no gradients and therefore do not contribute to the penalty. In practice, the penalty is injected only when the configuration sets `use_l2sp = 1`; both λbb and λhd default to zero so the baseline training path stays untouched unless non‑zero coefficients are provided.
Whenever ΩL2SP is active the optimizer disables ordinary weight decay on the fine‑tuned parameter groups. This keeps the minimiser of the regularised objective at W0 instead of nudging it towards the origin through the decay term.

The fine‑tuning objective becomes

$$
\mathcal{L}_{\mathrm{FT}} = \mathcal{L}_{\text{task}} + \Omega_{\mathrm{L2SP}}.
$$

minimized by stochastic gradient descent over the subset of trainable parameters defined by the fine‑tuning schedule described below. ΩL2SP only adds curvature for trainable tensors with positive λ values, keeping the optimisation identical to the baseline regime otherwise.
Since ΩL2SP depends exclusively on W and W0, it contributes the same value to every mini‑batch and therefore acts as a deterministic regularizer rather than a data‑dependent term.

## Discriminative Learning Rates and Partial Freezing

Fine‑tuning adopts discriminative learning rates and a conservative unfreezing schedule:

1) All backbone layers are frozen except the last linear layer; the three task heads remain trainable. This preserves the pre‑trained feature extractor while exposing a narrow adaptation interface.

2) The heads are optimized with a learning rate ηhead, while the last backbone layer uses a smaller rate ηbb (typically ηhead ≫ ηbb). Earlier backbone layers remain frozen (η = 0). Weight decay follows the global configuration for scratch training; when L2-SP is enabled the implementation automatically sets the decay of the fine-tuning parameter groups to zero so that the anchor optimum remains at W0.

This design concentrates updates where they best serve domain adaptation (task heads), while allowing a slow readjustment of the final feature projection (last backbone linear) under the L2‑SP trust region. When `finetune_heads_lr` or `finetune_backbone_lr` are omitted, both groups inherit the base learning rate (ηhead = ηbb = η), so discriminative scaling only emerges once explicit overrides are supplied.

## Mathematical Summary of the Fine‑Tuning Step

Given a mini‑batch \mathcal{B}, the model computes ŷi and p̂i for each i ∈ \mathcal{B}. Using the wrapped angular difference defined earlier, the batch loss implemented during fine‑tuning is

$$
\mathcal{L}_{\mathrm{FT}}(\mathcal{B}) =
\frac{w_S \sum_{i \in \mathcal{B}} M_i (\hat{s}_i - s_i)^2 + w_D \sum_{i \in \mathcal{B}} M_i \Delta \theta_i^2}{\max \left( 1, \sum_{i \in \mathcal{B}} M_i \right)}
+ \frac{\lambda_{\mathrm{R}}}{\lvert \mathcal{B} \rvert} \sum_{i \in \mathcal{B}} CE(r_i, \hat{p}_i)
+ \lambda_{\mathrm{bb}} \sum_{j \in J_{\mathrm{bb}}} \lVert W_j - W_{0,j} \rVert^2
+ \lambda_{\mathrm{hd}} \sum_{k \in J_{\mathrm{hd}}} \lVert W_k - W_{0,k} \rVert^2.
$$

The optimizer applies discriminative learning rates {ηbb, ηhead} to the respective parameter groups. Dynamic weights wS and wD follow the implementation in the training loop and evolve over epochs based on the relative descent of each regression component.

## Rehearsal Mixing (Source‑Domain Replay)

To further stabilize the representation, fine‑tuning includes rehearsal steps drawn from the source domain. Let PT denote the target‑domain data distribution and PS the source‑domain distribution used during pre‑training. The effective training signal becomes a mixture with weight α ∈ [0, 1] for the source component. Practically, the training loop accumulates a ratio ρ = α/(1 − α) after every target update and fires as many rehearsal updates as needed to keep the cumulative share of rehearsal steps equal to α over the epoch. Guard rails clamp α below 1 to prevent starving target batches. The rehearsal step reuses the same range‑aware objective and, when enabled, the same L2‑SP penalty anchored to W0.

The expected objective can be understood as an empirical mixture:

$$
\mathbb{E}[\mathcal{L}_{\mathrm{FT}}] \approx (1 - \alpha)\, \mathbb{E}_{P_T}[\mathcal{L}_{\text{task}}] + \alpha\, \mathbb{E}_{P_S}[\mathcal{L}_{\text{task}}] + \Omega_{\mathrm{L2SP}}.
$$

where the masking Mi and the classification loss operate identically in both domains, preserving the physical range gating. The normalization applied to the rehearsal batches matches that of the target domain: per‑feature centers and scales are reused, and conditional per‑interval normalization is applied consistently. This ensures the model receives rehearsal gradients in the same normalized space where adaptation occurs.

## Knowledge Distillation Mechanics

When fine‑tuning needs an additional constraint beyond rehearsal and L2‑SP, the training script supports a distillation scheme governed by `use_kd`. Distillation becomes effective only when at least one of `lambda_kd_reg` or `lambda_kd_cls` is strictly positive. The legacy `lambda_kd` flag remains for backward compatibility and seeds both coefficients when the specialised weights are not provided. The logit softening temperature is exposed through `kd_temperature` (τ defaults to 1). As soon as the checkpointed model is loaded, the implementation snapshots a frozen teacher network with the same architecture and the anchor weights W0. The teacher is kept in evaluation mode with gradients disabled, so it preserves the original calibration throughout the fine‑tuning session.

For every target mini‑batch, the student compares its predictions against the teacher in two channels:
1. **Regression guidance.** The in‑range mask Mi defines the active subset (Ni = ∑ Mi). For those samples the student speed is matched to the teacher speed, and the wrapped angular discrepancy between student and teacher directions is evaluated through atan2. The resulting errors are combined with the same dynamic weights wS = w1 and wD = w2 used by the primary loss, producing  
   kdreg = [wS ∑ Mi (ŝi − ŝTi)² + wD ∑ Mi (ΔθiT,S)²] / max(1, Ni).
2. **Range classification guidance.** Teacher and student logits are softened with τ, and the Kullback–Leibler divergence τ²·KL(pTτ‖pSτ) (implemented with `reduction="batchmean"`) encourages the student to reproduce the teacher’s range probabilities under the same temperature.

The two components enter the batch loss independently: LFT ← LFT + λkd_reg·kdreg + λkd_cls·kdcls. The same procedure applies verbatim to rehearsal updates, ensuring that replayed batches reinforce the teacher guidance in tandem with the target batches.

Operationally, if anchoring fails (for example, when fine‑tuning resumes from a checkpoint that did not capture W0), the code falls back to cloning the current student weights, thereby performing self‑distillation with initial loss zero. The `teacher_checkpoint` field remains dormant until loader support for external teacher artefacts is implemented.

## Activation Scope and Invariance of Baseline Training

The anti‑forgetting mechanisms are gated by the presence of a pre‑trained checkpoint at fine‑tuning start. If training starts from scratch (no checkpoint), the model follows the baseline regime: all parameters are trainable; the optimizer uses a single learning rate; and ΩL2SP is inactive. This ensures the standard training behavior and performance characteristics remain unchanged.

## Configuration and Pipelines

Two configuration files exemplify the recommended settings (station schema, range, architecture, and fine‑tuning knobs):

• artifacts_root/stationX/config/stationX_finetune_l2sp_rehearsal.json
• artifacts_root/sar/config/sar_finetune_l2sp_rehearsal.json

Key fields of the model section include:

• finetune_heads_lr: learning rate for the task heads during fine‑tuning.
• finetune_backbone_lr: learning rate for the last backbone linear layer.
• use_l2sp: binary switch to activate the L2‑SP penalty.
• l2sp_backbone_lambda and l2sp_heads_lambda: penalty coefficients (λbb, λhd).
• use_kd together with λkd_reg, λkd_cls, and kd_temperature: enable distillation, control the relative strength of the regression and classification guidance, and adjust the logit softening temperature τ (default τ = 1).
• target_speed_range, range_margin, range_loss_weight: range‑awareness controls that maintain the physical gating.

Rehearsal‑specific configuration fields are provided at the training interface to indicate the source dataset and its target labels, and to control the mixing schedule:

• rehearsal_data_path: path or prefix to the source‑domain GeoParquet used for replay.
• rehearsal_target_speed_col and rehearsal_target_dir_col: column names for speed and direction in the rehearsal dataset.
• rehearsal_fraction: approximate fraction of rehearsal steps per epoch (e.g., 0.10 executes about one rehearsal update every ten target updates).

To streamline execution without affecting other pipeline stages, the repository includes a dedicated fine‑tuning script:

• run_cross_domain_finetune.sh

An orchestration script can mirror the original pipeline by chaining four stages: (1) materialise the target-domain training split and download the source-domain checkpoint; (2) launch a fine-tuning job with L2-SP and optional rehearsal/distillation enabled; (3) run inference on native and cross-domain test sets, post-processing each result into GeoParquet; and (4) compute inference metrics and update the STAC catalog. Keeping every step in a single script ensures reproducibility—the entire adaptation can be replayed with one command while still allowing analysts to copy-paste individual blocks when they need ad-hoc experimentation.

## Practical Guidance and Expected Effects

• Preservation of source‑domain calibration: With moderate λbb and smaller λhd, the L2‑SP term stabilizes the feature space, reducing the drift that typically degrades the pre‑trained domain’s metrics.

• Adaptation leverage through heads: Discriminative learning rates encourage the heads to absorb most domain shift. Increasing ηhead relative to ηbb accelerates adaptation without compromising the backbone.

• Range‑consistent evaluation: Regression diagnostics must be interpreted strictly within the valid range; outside‑range samples are handled by the classifier and excluded from regression error aggregation.

Limitations and Extensions

• Rehearsal: Replaying a minority of source‑domain batches counteracts drift in the shared representation. A practical starting point is α ≈ 0.10. The current implementation samples rehearsal mini‑batches by simple shuffling, so any maintenance‑interval stratification must be enforced when materialising the replay dataset.

• Knowledge distillation: Enabling `use_kd = 1` snapshots the anchored checkpoint as a teacher and penalises discrepancies in speed/angle (weighted by wS, wD) and range probabilities during both target and rehearsal updates. The coefficients `lambda_kd_reg` and `lambda_kd_cls` control the relative strength of each channel (with `lambda_kd` serving as a legacy shortcut), while `kd_temperature` exposes the logit softening τ (the τ² scaling is already in place). If both coefficients are zero the extra terms are skipped, effectively reverting to the pure anti‑forgetting regime.

• Interval‑specific normalization: The model already supports conditional normalization by maintenance interval for power and dispersion features. Interval‑conditioned affine adaptation (e.g., FiLM) could be layered on the heads to absorb slow drifts more explicitly.

## Implementation References

• scripts/training/train_lib/model.py — checkpoint loading, partial freezing schedule, parameter groups for discriminative learning rates, and snapshot of anchor weights.
• scripts/training/train_lib/train_loop.py — range‑aware objective with L2‑SP; rehearsal scheduling and updates mixed with target batches.
• scripts/training/train_lib/data_load.py — ingestion and feature engineering for the rehearsal dataset using its own target column names.
• scripts/training/train_model.sh — propagation of rehearsal configuration to the training job in SageMaker hyper‑parameters.
• scripts/training/train_lib/config.py and scripts/training/train_lib/cli.py — configuration knobs for fine‑tuning; defaults keep baseline training unaffected.
