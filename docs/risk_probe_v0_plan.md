# Risk probe v0: do the verifier's hidden states carry downstream risk beyond validity?

Date: 2026-10-13. Experiment ID: `risk_probe_v0`. Predecessors: §21.18 (input conventions), `factorized_step_labels_v1` (risk labels).

## 1. Why

`factorized_step_labels_v1` measured downstream risk on 531 forks:

```
risk_diff = err(A) - mean err(C)
```

Here err(A) is the continuation error rate after the real step, and err(C) is the rate after the model's own resampled alternatives at the same point.

Risk is related to validity, but only partly (Spearman 0.37):

* 13% of wrong steps are bypassed: every continuation still reaches the right answer.
* A few correct-labeled steps are risky.

For answer selection and rejection, risk is the quantity that matters. Labelling it costs 72 generations per step, so the question is whether a hidden-state probe can supply it cheaply.

This experiment asks three questions, in order:

1. **Existing signal.** Does the current validity probe's score already track risk *within* a validity class?
2. **Learnable signal.** Can a probe trained on risk labels predict risk from the same states, beyond what validity explains?
3. **Combination preview.** Does adding a risk signal to the validity score predict whether continuing from a step fails better than the validity score alone?

## 2. Conditions

**States.** All readouts use the §21.18 setting:

* causal backbone context (`full` view: the problem and steps 0..i, v1 text format);
* `hidden_states[35]`;
* the `local` arm, which reads only step i's tokens.

**Fork views.** Fork views are built with the `build_context_views.py` code itself, from the v2 manifest steps. The fork step's text is checked against the factorized-label split.

**Validity probes.**

| probe | trained on | relation to the forks |
|---|---|---|
| `xfit_v2ds` (primary) | v2_train `full` views minus every fork problem, with v2_dev minus pilot problems for early stopping; DeepSeek labels; seeds 42/43/44; v1 HP (lr 1e-4, wd 0.1) | clean: no fork problem seen |
| `v2ds` (existing §21.18 fits) | the capped v2_train | 161 of 531 fork traces were training traces; reported on the 370 out-of-sample forks |
| `v1human` (existing §21.18 fits) | PRM800K human labels | no fork trace, 291 fork problems seen through other (human-written) solutions |

**Reproduction gate.** The existing `v2ds` and `v1human` checkpoints rescore 600 `rp_test` views through this job's store and scoring code. The logits must match the saved §21.18 predictions:

* median absolute difference in P(incorrect) ≤ 0.002;
* maximum ≤ 0.02.

These are the project's established tolerances for bf16 batching noise, on the probability scale. The saved logits are bf16, so a one-unit rounding flip alone moves a logit by about 0.03. If the gate fails, no probe score from this job is interpreted.

**Risk probes.** These are cross-validated linear probes on pooled L35 step states (mean over the step's tokens, and the last token) of the 531 forks:

* ridge regression on `risk_diff`;
* L2 logistic regression on `harmful` (`risk_diff` > 2 SE);
* 10 folds, repeated 5 times, with the inner ridge or C selected on the training folds only.

A permutation null (200 label shuffles) gives the chance level for 531 samples in 4,096 dimensions.

## 3. Sources and evaluation

* **Forks.** 501 forks from v2 `train`, one per problem (300 incorrect, 201 correct), plus the 30 pilot forks from v2 `dev`. Labels come from `results/factorized_step_labels_v1/{full,pilot}`.
* **Validity labels.** DeepSeek and self labels agree on every fork step by construction.
* **Gate set.** 600 `rp_test` views, the first by sha1 of the view id, from the frozen context_ablation_v3 `full` manifest.

## 4. Metrics and contrasts

**Q1, existing signal** (seed-mean logit; primary `xfit_v2ds`):

* Sanity: validity AUROC on the forks. It should be well above 0.5.
* Within incorrect forks:
  * Spearman(score, `risk_diff`);
  * AUROC of `harmful` against not harmful;
  * AUROC of bypassed (err(A) = 0) against the rest.
* Within correct forks: Spearman(score, `risk_diff`), plus the ranks of the risky correct forks (descriptive, n ≈ 7).
* All forks: partial Spearman(score, `risk_diff` | validity label, log continuation length, relative position).

**Q2, learnable signal:**

* Out-of-fold Spearman(prediction, `risk_diff`) and AUROC(`harmful`), against the permutation null.
* The same on the residual of `risk_diff` after regressing out validity, log length and relative position. This is the "beyond validity" test.

**Q3, combination preview.** Out-of-fold logistic regression for two targets: `fail` = err(A) ≥ 0.5, and `harmful`. Two feature sets:

* (a) validity probe score;
* (b) validity probe score plus the out-of-fold risk-probe prediction.

The contrast is ΔAUROC (b) − (a), with a fork-bootstrap 95% interval.

**Prespecified reading.**

* **Risk is readable from these states beyond validity** if both hold:
  * the Q2 residual Spearman exceeds the 95th percentile of its permutation null;
  * the Q3 ΔAUROC for `harmful` has a 95% interval above 0.
* **The current probe already carries some of it** if the Q1 within-incorrect Spearman has a 95% interval above 0.
* **Otherwise**, with 531 forks, the result is "not detectable at this size". It is not "absent". The next step would then be more risk labels, not a different probe.

**Caveats.**

* `fail` depends on problem difficulty, so it is not a pure risk target. `harmful` is the risk target.
* This is a feasibility test, not a deployment result.

## 5. Execution

1. `scripts/factorized_step_labels/build_risk_probe_views.py` runs on the TamIA login node (tokenizer only). It writes a frozen view manifest with four splits: `v2_train_xfit`, `v2_dev_xfit`, `fsl_forks` and `gate_rp_test`.
2. One H100:4 job (`slurm/bidirectional_token_probe_train_tamia.sh`, unchanged) does three things:
   * encodes the manifest node-locally at L35;
   * runs the 3 `xfit_v2ds` fits, which score `fsl_forks` and `gate_rp_test`;
   * through `EXTRA_STEPS`, rescores the 6 existing checkpoints (`scripts/factorized_step_labels/score_with_checkpoints.py`) and dumps pooled fork features (`scripts/factorized_step_labels/dump_fork_features.py`).
3. `scripts/factorized_step_labels/analyze_risk_probe.py` runs locally (CPU) and writes `results/risk_probe_v0/summary.{json,md}`.

## 6. Out of scope

* New risk labels. Scaling the labels is the follow-up if Q2 or Q3 is positive or underpowered.
* A deep risk probe. 531 forks only support linear readouts.
* Layers other than 35.
* Downstream TTS or rejection runs.
