# Risk probe v1: two ways to read downstream risk from the verifier's states

Date: 2026-10-14. Experiment ID: `risk_probe_v1`. Predecessors: `risk_probe_v0` (§21.20, the questions and the 531 test forks), `step_value_labels_v1` (the training labels).

## 1. Why

risk_probe_v0 asked whether Qwen3-8B's L35 states carry downstream risk beyond validity, with 531 labeled forks:

* A linear readout learned risk beyond validity above a permutation null (residual Spearman 0.195, null 95th percentile 0.055).
* Adding it to the validity score did not improve `harmful` detection: +0.019 AUROC, 95% CI [-0.008, 0.045].

The plan's reading was "not detectable at this size", and the next step was more labels. `step_value_labels_v1` supplies them. In the partial run (5,394 of 5,874 traces):

* 29,708 labeled steps, against 531 forks;
* 1,322 `harmful` steps, against 98, of which about 300 are expected from noise alone (rollout-shuffle null);
* 537 harmful DeepSeek first errors.

The labels also showed what a probe can and cannot be asked to fit:

* **Step risk is a noisy target.** It is a difference of two 16-rollout error rates. Split-half reliability is 0.43 at 8 rollouts, so 0.60 at 16 (Spearman-Brown). A perfect predictor would correlate with it at about 0.78 at most.
* **The error rate at a cut is a much cleaner target.** The two traces of a problem give two independent estimates of err(0), and they correlate at 0.95 over 2,055 problems. Part of that comes from problems differing a lot in difficulty.
* **Validity explains 2.3% of the true risk variance** (variance net of sampling noise), so a risk readout has room to add information.

This experiment trains risk readouts on the new labels in two ways and tests them on the same 531 held-out forks as v0.

## 2. Arms

Both arms read the same states (section 3) and are fit on the same problem folds.

**Arm A, direct risk (v0's design).**

* Ridge regression of `risk` on the step's state.
* L2 logistic regression of `harmful` on the same state.
* Inputs: the step's last-token row (primary) and the mean over the step's token rows (secondary), as in v0.

**Arm B, value difference.**

* A value probe V predicts the policy's error rate at a cut point from the state at that point.
* It is a binomial logistic regression with L2 on (wrong rollouts, k) counts at each cut, so its predictions stay in [0, 1].
* Every labeled step gives two training rows:
  * its **before row**, the token just ahead of the step's first token, whose target is err(s);
  * its **after row**, the step's last token, whose target is err(s + 1).
* The step's predicted risk is V(after row) − V(before row). Both rows come from one forward pass over one view, so the difference isolates what the step's own tokens changed.

**Combination A + B.** This exists only in Q3 (section 5): the validity score plus both readouts as features. It asks whether the two readouts add to each other. No joint model is trained.

Regularization (ridge alpha, logistic C) is chosen by inner cross-validation on the training folds only, from the same grids as v0.

## 3. States and labels

**Views.**

* Every readout uses §21.18's setting, as in v0: the `full` context view (`Problem:` header, then steps 0..i, joined with blank lines), `hidden_states[35]`, and the step's own token rows.
* Views are built with `build_context_views.py`'s code (`Tok`, `view_steps`) from the v2 manifest steps. The view id is `<trace_id>#<claim index>`, the same id as `step_values.jsonl`.
* The before row of step i is the last token ahead of it: the end of `Solution:\n` for step 0, otherwise the end of the blank-line separator after step i − 1.
* Each label step's view text must equal its ReProbe step line after `strip_step`. Mismatches are dropped and counted.

**Training labels.** These come from `step_value_labels_v1/label/summary/step_values.jsonl`, frozen with its sha256 before any fit:

* `risk`, `se`, `harmful`;
* err before and after, with the 16 rollout outcomes per cut;
* DeepSeek and self validity, phase, and `finished`.

**Test labels.** These are the 531 forks of v0 (501 full-run plus 30 pilot), with their factorized_step_labels_v1 labels: `risk_diff`, `harmful`, err(A) and err(C).

* The label set shares no problem with them, by construction.
* On the forks, err(C) estimates err(s) and err(A) estimates err(s + 1). Arm B's two predictions are compared with them directly.

**Validity scores.** Q3 needs a validity score that has not seen the evaluated step.

* **On the forks:** v0's `xfit_v2ds` (v2_train minus every fork problem, 3 seeds, seed-mean logit). This is unchanged, so the v0 baseline row reproduces.
* **On the label set:** `xfit_v2ds` cannot be used.
  * 2,865 of the 5,874 label traces (2,545 of 3,304 problems) are in its training views.
  * The job therefore cross-fits a validity probe, `xfit5`. It uses 5 problem folds over the union of the v2_train `full` views (minus fork problems) and the label-set views, with §21.18's v1 hyperparameters, seed 42, and `v2_dev_xfit` for early stopping.
  * Each label step is scored by the fold model that never saw its problem.
  * Fold membership is the same as the probe-training folds of section 4.
  * The trainer takes one `--train_split` today; it gets a list form for this.

## 4. Evaluation sets

**E1, held-out forks (primary).**

* Arms A and B are fit on all label-set steps, then applied once to the 531 forks.
* The fork evaluation (Q2 and Q3 below) is v0's code path, with v0's metrics and fork bootstrap. Only the risk readout's training data changes.

**E2, label set, 5-fold cross-validation by problem (secondary).**

* The folds are fixed by sha1(problem_key) before any fit.
* Every prediction is out of fold, and intervals come from a problem bootstrap (2,000 resamples).
* E2 has about 1,300 harmful steps against 98 in E1, so its intervals are much narrower. Its distribution differs from the forks:
  * every step of a trace, not one fork per trace;
  * 68% of traces (3,977 of 5,874) cut off at ReProbe's 256-token cap;
  * DeepSeek validity from the all-step labels.

**Subsets reported on E2.** All steps; first errors; steps with headroom (0 < err(s) < 1); finished traces only.

## 5. Questions and metrics

**Q1, the existing validity score (E2 only).**

* Within DeepSeek-incorrect steps: Spearman(xfit5 score, risk) and AUROC for `harmful`.
* This repeats v0's Q1 on 20 times more incorrect steps. The fork version is unchanged from v0 and is not rerun.

**Q2, learnable risk (E1 and E2, per arm).**

* Spearman(predicted risk, risk). On E1 the comparison label is `risk_diff`.
* The same on the residual of risk after regressing out validity, log continuation length and relative position. These are v0's covariates; the continuation length is the original trace's remaining reply.
* AUROC for `harmful`.
* **Null:** 200 refits on labels shuffled across traces within problem folds, which keeps each trace's steps together.
* **Arm B only, calibration:** on E1, Spearman(V(before), err(C)) and Spearman(V(after), err(A)).

**Q3, does a risk readout add to validity (E1 and E2).**

* Out-of-fold logistic regression of `harmful` on:
  * (a) the validity score, log length and relative position;
  * (b) (a) plus arm A's prediction;
  * (c) (a) plus arm B's prediction;
  * (d) (a) plus both.
* Contrasts: ΔAUROC (b) − (a), (c) − (a), (d) − (a), with fork (E1) or problem (E2) bootstrap intervals. On E1 the combination is fit by 10-fold CV over the forks, as in v0.
* `fail` = err(A) ≥ 0.5 is reported as in v0, secondary.

**A against B (E1 and E2).** A paired bootstrap of the Q2 Spearman difference, B − A, on the same steps.

## 6. Prespecified reading

Two primary arms are tested, so the primary intervals are 97.5% (Bonferroni over A and B).

* **Risk adds to validity at this scale** if, for arm A or arm B, Q3 ΔAUROC (harmful) has a 97.5% interval above 0 on E1 and a 95% interval above 0 on E2.
* **E2 alone positive.** The readout helps on the label distribution, but the fork test is underpowered or the distributions differ. At 98 harmful forks, E1's half-width was about 0.03 AUROC in v0. This is reported as such, not as a pass.
* **B is the better target** if the B − A Spearman difference has a 95% interval above 0 on E2 and a positive point estimate on E1. The reverse reading is symmetric.
* **Otherwise** the result is a null at roughly 30,000 training steps. It closes the linear-readout question at L35 for this view, and does not change the label work.

Q1 and the residual Q2 are descriptive under this rule.

## 7. Gates

These run before any probe is interpreted.

1. **Labels.** All 32 label shards are complete (5,874 traces). `step_values.jsonl` is frozen with its sha256.
2. **Reproduction.** The re-encoded `fsl_forks` views reproduce v0's dumped pooled features (`results/risk_probe_v0/run/features`): median cosine ≥ 0.9999. `xfit_v2ds` rescores them within v0's tolerances (median |ΔP(incorrect)| ≤ 0.002, max ≤ 0.02). If this fails, nothing on E1 is interpreted.
3. **Alignment.** At least 99% of label steps map to a view whose step text matches. The cut offsets of each trace must match the view's step order.
4. **Leakage.** No label-set problem is among the 531 fork problems. Folds are disjoint by problem_key. Each xfit5 fold model is checked against the problems it scores.

## 8. Execution

1. **Labels finish:** `summarize_values.py --trace_set` writes `step_values.jsonl`, and its sha256 goes into this plan's run log.
2. **`scripts/risk_probe_v1/build_views.py`** runs on the TamIA login node (tokenizer only) and writes a frozen manifest. Splits:
   * `lab_steps`: about 32,300 label-step views;
   * `xf0`..`xf4`: the v2_train views minus fork problems, plus `lab_steps`, assigned by problem fold;
   * `v2_dev_xfit`;
   * `fsl_forks`;
   * `gate_rp_test`.
3. **One H100:4 job** (`slurm/bidirectional_token_probe_train_tamia.sh`):
   * encodes the manifest at L35;
   * runs the 5 xfit5 fits, each scoring its held-out fold and `fsl_forks`;
   * rescores `xfit_v2ds` on `fsl_forks` and `gate_rp_test`;
   * dumps, per view, the before row, the after row and the step mean (float16, about 0.8 GB).

   v0's job encoded 47k views and ran 3 fits in 9 minutes. This one has about 65k views (43k xfit, 17k new label-step views, dev, forks and gate) and 5 fits, so a 2 h limit should be enough; it will be set from a smoke run.
4. **Locally (CPU):**
   * `scripts/risk_probe_v1/fit_readouts.py` fits arms A and B on E2 folds and on the full set;
   * `scripts/risk_probe_v1/analyze.py` writes `results/risk_probe_v1/summary.{json,md}`.

Tests come first for the row indexing (before and after rows against a known view), the binomial value fit on synthetic counts, and the fold and leakage checks.

## 9. Caveats

* **Different readers.** The states come from the verifier view (the `Problem / Solution` text), not the policy's own reply context that the rollouts were sampled from. This keeps continuity with v0 and the validity probe, but a value probe on the policy's context is the more direct reading. It is listed in section 10.
* **Neighboring steps share a cut,** so their labels are correlated. All resampling is by problem.
* **Headroom.** 25 to 30% of first errors sit at err(s) = 1 and cannot be harmful. Arm A trains on them as zero-risk steps. Arm B sees them as two cuts at 1.
* **Different mixes.** The label set is mostly capped, harder traces. The forks are finished first errors and correct steps, so E1 is the test of transfer.
* **Grading noise** is not a concern for the training labels: a lenient re-grade changed 0.35% of `harmful` labels.

## 10. Out of scope

* Layers other than 35, and nonlinear readouts. Both follow only if a linear arm passes.
* A value probe on the policy's own context (ReProbe prompt plus reply ids, the exact states the rollouts continued from). It is the first follow-up if arm B fails on the verifier view.
* Downstream selection or rejection runs.

## 11. Implementation notes (2026-10-14, before any readout was fit)

These details are fixed in code before any result exists:

* **Folds.** `fold_of(problem_key) = int(sha1(problem_key)) mod 5`; inner folds use the same hash with the salt `inner:` and 3 folds.
* **Solvers.** Ridge is closed form from one float64 eigendecomposition of X'X per training set. Logistic regression is L-BFGS in torch on binomial counts. Both match sklearn's Ridge and LogisticRegression to 1e-8 and 1e-4 in tests.
* **Inner selection.** Ridge alpha minimizes held-out MSE. The `harmful` C maximizes held-out AUROC. Arm B's C minimizes held-out binomial deviance.
* **Null.**
  * Permutations move whole label vectors between traces with the same (fold, number of steps). Each refit uses the real fit's hyperparameters.
  * 200 permutations are split over 4 GPUs.
* **Residual risk** uses DeepSeek validity. The 0.3% of steps without a DeepSeek label take their self label in the residualization only; residual metrics are computed on DeepSeek-labeled steps.
* **E2's Q3 combination** is a LogisticRegressionCV (v0's grid and inner 5-fold AUROC selection), fit on four problem folds and applied to the fifth.
* **Scripts.**
  * `scripts/risk_probe_v1/build_views.py` (run 2026-10-14 22:38: 32,316 of 32,316 label steps aligned);
  * `dump_rows.py`, `fit_readouts.py`, `analyze.py`;
  * `slurm/submit_risk_probe_v1.sh` (smoke, then full: the encode/fit job, then the readouts job, afterok).
* **Amendment, 23:30, after the smoke run and before any full-run fit or any metric was read.** On the partial label set the inner CV chose the edge of v0's grids in every fold: ridge alpha 3e5 to 1e6 (grid maximum 1e6), C 1e-4 (grid minimum) for every logistic arm. Both grids are extended by two decades on the regularized side with the same half-decade spacing: alpha in logspace(0, 8, 17), C in logspace(-6, 2, 17). v0's grid points are a subset.
