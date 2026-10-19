# sae_localization_v0: does step correctness localize to a few SAE features?

Frozen 2026-10-19, before any result was read.

## Why

The project's opening question (REPORT §1) asks whether SAE features carry the
step-correctness signal. The August rep grid on Qwen3-8B-Base (layer
`hidden_states[35]`, Qwen-Scope `layer34` TopK SAE, k=50, FVU 0.327 on the
PRM800K step store) answered a weaker question: the full 65,536-wide code keeps
the signal (`sae_mean x linear` test AUROC 0.867 against `step_mean x linear`
0.869, three seeds, `cot-checker-results/rep_grid_q3`). Keeping it is not
localizing it. Section 15 found the dense signal spread over hundreds of
coordinates (235 of 3,584 for full accuracy at Qwen2.5-7B L20). An SAE earns an
interpretability claim only if a handful of its features carry what hundreds of
dense coordinates need.

## Design

`scripts/analysis/sae_feature_localization.py`. Inputs already on TamIA:
`$SCRATCH/cot_mech/qwen3_8b_v1/sae_reps/sae_mean*` and the dense twin
`cache/grid_vectors/step_mean*`. Same rows, same labels (asserted).

* Rank features by |Cohen's d| (incorrect vs correct) on `probe_train_full`.
* For k in 1, 2, 4, ..., 1024: logistic regression on the top-k columns, fit on a
  200,000-row subsample of train, C from {0.001, 0.01, 0.1, 1} by val_5k AUROC.
* Identical ladder on dense `step_mean` coordinates.
* Report PRM800K test_2k AUROC, ProcessBench known-step AUROC and calib-20
  F1_PB (four-subset means).
* Dump the top 40 SAE features with class frequencies and most-activating
  training rows.

**GO:** at k=16, SAE test AUROC >= 0.80 and at least 0.05 above dense top-16.
Otherwise NO-GO: the dictionary does not isolate the signal at this layer.

## One-class companion

`scripts/analysis/one_class_correct.py`, same dense caches, `step_mean` and
`last_token`. A shrunk Gaussian fit to correct steps only, scored by
Mahalanobis distance, against two-class LDA, a mirror fit on incorrect steps
and the unwhitened distance to the correct centroid.

**GO:** one-class within 0.02 of LDA on test AUROC and no more than 0.02 below
it on ProcessBench calib-20, or above LDA on calib-20 by at least 0.02.

## Cost

Two CPU jobs, no GPU, no new encoding.
