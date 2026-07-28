# Unified Harness 7B — Data Setup

Frozen experimental spine. Only the representation (and learner) varies across
leaderboard rows; everything below is held constant.

## Backbone

- **Qwen2.5-7B base** (not instruct), last transformer layer, hidden dim 3584.
- Weights cached at `/project/aip-azouaq/dchikhi/hf_cache` (offline load).

## In-domain data: PRM800K

Built with `scripts/build_prm800k_full.py` (seed 42, max_seq_len 2048). Labels:
rating +1 -> correct (0), rating -1 -> incorrect (1); rating 0 dropped; pos/neg
balanced 50/50. Splits are **problem-id disjoint** (train / val / test share no
problem).

| split | stem | steps | balance |
|---|---|---|---|
| train | `probe_train_full` | 513,810 | 50/50 |
| val (threshold selection) | `val_5k` | 5,000 | 50/50 |
| in-domain test | `test_2k` | 2,000 | 50/50 |

Available pool after length filter: 677,904 correct / 261,511 incorrect. "Full"
uses all balanced pairs minus a 1,000/class safety margin (`--full_safety_margin`).

Token stats (full sequence = question + prior steps + candidate step): mean 283,
median 233, p95 645, max 2047 tokens/step. Total train tokens 145.4M.

## Out-of-domain test: ProcessBench

Raw at `/scratch/d/dchikhi/cot-checker/processbench`. Encoded per subset, then a
combined view (ids namespaced `<subset>::<id>`).

| subset | traces | error traces | correct traces |
|---|---|---|---|
| gsm8k | 400 | 207 | 193 |
| math | 1,000 | 594 | 406 |
| combined | 1,400 | 801 | 599 |

## Threshold protocol

- **val-selected** (deployable-but-untuned): threshold maximizing balanced
  accuracy on the balanced PRM800K val. Picks ~0.5, under-tuned for ProcessBench's
  correct-skew.
- **calib-20** (deployable headline): hold out 20 ProcessBench traces per subset
  (stratified error/correct), pick the threshold that maximizes F1_PB there, apply
  to the remaining traces. Mean over 20 random splits.
  (`scripts/analysis/pb_threshold_calibration.py`.)
- **oracle** (ceiling, NOT deployable): grid-max F1_PB over the full test.

## Metrics

- **In-domain (PRM800K test):** step-level AUROC (rank-based, positive = incorrect)
  and macro-F1 at fixed 0.5 / val-selected / oracle.
- **OOD (ProcessBench):** first-error F1_PB = harmonic mean of Acc(correct
  solutions predicted all-correct) and Acc(error solutions with exact first-error
  index). Threshold grid step 0.01.

## Representation store

Full last-layer token states of every step live in a ragged store under
`$SCRATCH/.../repstore/` (`src/repstore/`, `kind=token_seq`): packed `h.npy` +
`lengths.npy` + `y.npy` + `meta.jsonl` + `spec.json`. Per-step meta carries the
indexing offsets (step-token span start, pre-step boundary, prior-step
boundaries) so every representation is an offline slice/reduce:

- `dense_last` = last row of the step span
- `delta` = last step row minus pre-step boundary row
- pooled = mean/max/attention over the step span
- `trajectory` = the prior-step boundary rows

Master token store lives in `$SCRATCH` (regenerable, purge-after-60-days);
derived vector reps and results are kept in `$STORE`.

## Provenance

- Run root: `$SCRATCH/cot_mech/dense_full_7b_v1`; results copied to
  `$STORE/results/dense_full_7b_v1`.
- Slurm: `slurm/{build,encode_prm800k_full,encode_processbench_full,train_harness}_7b_tamia.sh`.
- Harness: `scripts/train_easy_probe_method.py`; calibration:
  `scripts/analysis/pb_threshold_calibration.py`.
