# rejection_value_offline_v0: which score should decide what to resample?

Frozen 2026-10-19, before any result was read.

## Why

Resampling step s from its own prefix replaces a continuation whose error rate
is err(s+1) with one whose error rate is, to first order, err(s). The expected
reduction in final-answer error from one blind redraw of s is therefore
risk(s) = err(s+1) - err(s), the quantity step_value_labels_v1 measures for
32,316 steps (REPORT §21.23). A rejection controller flags the steps whose score
exceeds a threshold, so its offline value at a flag rate q is the mean risk of
the steps it flags. The validity probe is near chance at ranking risk within
incorrect steps (§21.20, §21.24); a linear risk readout is not (0.65 to 0.68).
Neither has been evaluated on the quantity a controller spends its budget on.

## Design

`scripts/analysis/rejection_value_offline.py` on local files only:
`results/step_value_labels_v1/label_summary/step_values.jsonl` (risk, labels)
and `results/risk_probe_v1/analysis_prod/step_scores.jsonl` (out-of-fold
scores: `xfit5` validity log-odds, `A_last` risk ridge, `A_last_h` harmful
logistic). Scorers: validity, A_last, A_last_h, and their unfitted combination
z(validity) + z(A_last_h), against random. For q in {0.05, 0.10, 0.20, 0.30}
over all steps: mean risk of flagged steps, and the share of all positive risk
they capture. Paired bootstrap over problem keys (2,000 resamples).

**GO:** at q = 0.10 the combination's mean flagged risk is at least 25% above
validity alone, with the paired 95% interval of the difference above 0.

Limits: first-order (one redraw, no interaction between flagged steps of one
trace); risk is defined for this policy (Qwen3-8B, T=1.0) and these traces (68%
capped at 256 tokens).
