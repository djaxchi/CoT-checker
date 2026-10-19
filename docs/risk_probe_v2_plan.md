# Risk probe v2: the leaderboard's top architecture trained on risk labels

Date: 2026-10-15. Experiment ID: `risk_probe_v2`. Predecessors: `step_value_labels_v1` (§21.23, the labels), `risk_probe_v1` (§21.24, linear readouts), `instruct_leaderboard_v1` (§21.12, the architecture and the validity probe).

## 1. Why

risk_probe_v1 read risk with linear models on one or two rows of the step. The validity score it was added to came from a token transformer. Two checks on the v1 label set (5 problem folds) bear on the training:

* **Class weighting does not matter for the linear readout.** A balanced `harmful` logistic gives AUROC 0.710 against 0.709 unweighted.
* **The readout is still data-limited.** Training on 1/8, 1/4, 1/2 and all of each fold gives AUROC 0.616, 0.649, 0.664 and 0.682 for `harmful`.

This run trains the top cell of the Instruct leaderboard on the same labels, then asks two questions on ProcessBench:

1. How well does the risk probe alone detect first errors?
2. Does adding it to the leaderboard's validity probe raise F1_PB?

## 2. Model and training protocol

The cell is `step_tokens × transformer:d512,l2,f2048,h8` (8,665,089 parameters), which ranks first on the Instruct leaderboard (calib-20 F1_PB 0.6107, PRM800K test AUROC 0.9128, §21.12).

* Training uses the leaderboard trainer's own functions (`scripts/train_rep_learner_cell.py`: `build_learner`, `SpanLoader`, `train_one`, `score_all`). Its protocol is unchanged:
  * AdamW with binary cross-entropy;
  * batch 256, at most 30 epochs, patience 3 on validation loss;
  * dropout 0.1, token cap 512, raw inputs (`rescale none`), length-bucketed batches.
* **Hyperparameters.** The grid is the leaderboard's (lr {1e-3, 3e-4, 1e-4} × weight decay {0, 0.01}).
  * The search runs once per target, on fold 0 at seed 42.
  * The winner is chosen by validation AUROC against hard `harmful`, then reused for every fold and seed.

## 3. Inputs

**Backbone and template.** Qwen3-8B (Instruct), `hidden_states[35]`, with the leaderboard's verifier template:

```
Problem:\n{problem}\n\nPrevious reasoning:\n{prefix}\n\nCurrent step:\n{step}
```

The prefix holds the earlier steps joined by blank lines.

**Encoding.** The leaderboard's span-only encoder (`scripts/encode_prm800k_token_store.py --span_only`) stores the pre-step boundary row and the step's tokens.

* **Label set:** the 32,316 label steps.
  * Problem text and step texts come from risk_probe_v1's aligned views (`views/inputs/xf*.jsonl`, `lab_views.jsonl`).
  * They are written as one jsonl by `scripts/risk_probe_v2/build_label_jsonl.py`.
* **ProcessBench:** the existing Instruct store (`qwen3_8b_instruct_v1/repstore/pb_step_spans`), the one the leaderboard cell was scored on. Its fingerprints are recorded.

## 4. Targets

* **Soft harmful (primary).** q = P(θ_after > θ_before) for each step, where θ_before and θ_after are the policy's error rates at the two cuts.
  * Each rate has a Beta(1 + wrong, 1 + right) posterior from its 16 rollouts.
  * The target keeps the strength of the evidence that the hard 2-SE cut throws away. It is trained with BCE against q.
* **Hard harmful (secondary).** risk > 2 SE, as in v1. This is the leaderboard loss on a binary label.

## 5. Splits

* **Outer folds.** v1's five problem folds (`fold_of`).
* **Validation.** Inside every training set, problems with sha1("val:" + problem_key) mod 10 = 0 (about 10%) form the validation set, used for early stopping and hyperparameter selection.
* **Out-of-fold fits (E2).** For each outer fold k, train on the other folds minus validation, score fold k. Seed 42, both targets.
* **Full fits.** All label steps minus validation, seeds 42, 43 and 44, both targets. These score ProcessBench.

## 6. Evaluation

**Label set (E2).** Out-of-fold AUROC for `harmful`, overall and within DeepSeek-incorrect steps. v1's linear `harmful` readout gives 0.709 and 0.679.

**ProcessBench, risk alone.** For each seed's full fit:

* first-error F1_PB at calib-20 (four-subset mean, `merge_rep_grid_leaderboard.calib20_subset`);
* oracle F1_PB;
* AUROC over steps with a known label.

Seed means are reported against the validity cell's 0.6107.

**ProcessBench, validity plus risk.** The validity probe is the leaderboard cell itself: the three saved d512 seeds. Seed s of validity pairs with seed s of risk.

* **Primary combination.** A logistic regression of DeepSeek invalidity on [validity logit, risk logit].
  * It is fit on the label set. Validity there comes from the d512 checkpoints rescored on the label-set store; risk is the out-of-fold score.
  * It is then applied to ProcessBench. No ProcessBench label is used.
* **Secondary combination.** calib-20 with a weight. In each of the 20 calibration draws, the weight w (score = z_validity + w · z_risk, standardized on the label set, w in {0, 1/8, 1/4, 1/2, 1, 2}) and the threshold are chosen together on the 20 calibration traces, then scored on the rest.
* **Contrast.** ΔF1_PB calib-20 = combination - validity alone.
  * Computed per seed, then averaged over seeds.
  * The interval is a paired bootstrap over traces within subsets (1,000 resamples).
  * AUROC on known steps is reported alongside.

## 7. Gates

1. **Alignment.** Every label step's decoded step tokens equal its step text (at least 99%), and the label-set store holds all 32,316 steps.
2. **Validity reproduction.** The d512 checkpoints rescore ProcessBench through this job's loader within a median |ΔP| of 0.002 of their saved `pb_step_scores`.
3. **Leakage.** No label-set problem text matches a ProcessBench problem text (normalized whitespace).
4. **Completeness.** Every label step has an out-of-fold score for each target.

## 8. Prespecified reading

* **Risk adds to validity on ProcessBench** if the primary ΔF1_PB calib-20 (seed mean) has a 95% interval above 0.
* **Risk alone** is descriptive.
  * ProcessBench labels the first invalid step, and on the label set even the true risk label separates first errors from earlier steps at only AUROC 0.61.
  * A risk-alone F1_PB well below validity is the expected result and does not count against the probe.
* **Soft against hard** is descriptive: the E2 AUROC difference with a problem-bootstrap interval.

## 9. Execution

1. `build_label_jsonl.py` on the TamIA login node writes `label.jsonl`, with its sha256.
2. One H100:4 job:
   * encodes the label set (four shards, about 5 min);
   * scores the label set and ProcessBench with the three d512 validity checkpoints (gate 2);
   * runs the hyperparameter search (2 targets × 6 settings), the out-of-fold fits (2 × 5) and the full fits (2 × 3);
   * runs `scripts/risk_probe_v2/analyze.py`.
3. A smoke run with 2 epochs and capped rows comes first. The walltime is set from it.

## 10. Caveats

* The combination is fit to DeepSeek labels on Qwen3-8B's own traces and applied to ProcessBench's human labels on other models' solutions.
* The label set is mostly traces cut off at 256 tokens. ProcessBench solutions are complete.
* Validity and risk read the same template and layer, but the validity probe saw PRM800K human labels and the risk probe saw Qwen3-8B rollouts.
