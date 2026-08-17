# Unified Harness 7B: Representation Leaderboard

Backbone **Qwen2.5-7B base**. Every representation is trained on the same frozen
PRM800K split, evaluated in-domain on a held-out balanced PRM800K test, then
transferred to all four ProcessBench subsets. Only the representation and the
learner vary. See `data_setup.md` for splits, sizes, and metrics.

Deployable OOD headline is **F1_PB @ calib-20**: the first-error threshold is
calibrated on 20 held-out ProcessBench traces per subset (stratified), applied to
the rest, mean over 20 splits. `val-selected` (t~0.5) and `oracle` (peeks at the
full test) are recorded in the run artifacts for context.

## Naming

Rows are named for **what the representation is**, never for the paper that
happens to use it. The representation (what vector the learner sees) and the
learner (what reads it) are separate axes, and two rows can share a
representation while differing in learner. Where a row reproduces a published
system, that is noted in the description and in `related_work.md`, but it does
not name the row.

Run directories and score files on disk still carry the original short keys, so
this table is the mapping:

| row | representation | learner | artifact key |
|---|---|---|---|
| `step_tokens` x transformer | all step token states | 2-layer transformer | `reprobe` |
| `step_tokens` x attn-query | all step token states | learned attention query | `attn_pool` |
| `step_stats` x linear | 5-statistic pool of step tokens | linear | `multistat` |
| `last_token` x linear | final token state | linear | `dense_last` |
| `step_delta` x linear | S_t minus S_{t-1} | linear | `delta` |
| readout ensemble (3) | mean of the three best readouts | none (post-hoc) | `ens3` |

## Representations: what is fed, and what is probed

Every entry reads from the same frozen Qwen2.5-7B base last-layer states. They
differ only in how a step's tokens are turned into the vector the learner sees;
the label being probed is always the step's binary correctness (0 correct,
1 incorrect).

- **step_tokens**: the raw *set* of all last-layer token states of the step
  (variable length, capped at 128 tokens). No pooling is applied, so the learner
  sees the sequence and decides what to read. Two learners sit on it:
  - *transformer*: a small (~2.5M-param) sequence model (project 3584 to 256,
    learned positions, 2 encoder layers, masked mean-pool, linear head). This is
    the maximal learner on this axis and asks how much of the OOD gap is detector
    capacity rather than representation. It also reproduces ReProbe (Ni et al.,
    2025), restricted to the last-layer subset we store.
  - *attn-query*: a single learned attention query that softmax-weights the tokens
    into one 3584-dim vector, then a linear head. Asks whether learning *which*
    tokens to read beats a fixed pooling rule, at a fraction of the capacity.
- **step_stats**: a fixed 5-statistic summary of all of a step's last-layer token
  states, concat[mean, max, min, std, last-token], giving 5 x 3584 = 17,920 dims,
  read by a single linear probe. Asks whether step correctness is linearly
  readable from a permutation-invariant summary of the whole step rather than from
  its final token alone.
- **last_token**: a single 3584-dim vector, the last-layer state at the step's
  last token, read by a linear probe. This is the point-readout baseline every
  other representation is measured against.
- **step_delta**: the 3584-dim transition vector S_t minus S_{t-1}, i.e. the state
  at the step's last token minus the state at the pre-step boundary token, read by
  a linear probe. Asks whether the *change* a step makes to the residual stream
  carries the correctness signal, rather than the absolute state it lands in.
- **readout ensemble (3)**: not a new representation. The unweighted mean of the
  predicted probabilities of the three best readouts above (`step_tokens` x
  transformer, `step_tokens` x attn-query, `step_stats` x linear), computed
  post-hoc from the saved per-step scores. It bounds how much of each readout's
  error is idiosyncratic rather than shared. Note the members are *not*
  independent: all three read the same frozen last-layer token states, so this is
  diversity in the readout, not in the representation.

## In-domain: PRM800K test (balanced, 2,000 steps)

| representation | learner | AUROC | macro-F1 (val) | macro-F1 (oracle) |
|---|---|---|---|---|
| step_tokens | transformer | **0.874** | 0.790 | n/a |
| step_stats | linear | 0.866 | 0.783 | 0.788 |
| step_tokens | attn-query | 0.860 | 0.778 | n/a |
| last_token | linear | 0.828 | 0.754 | 0.760 |
| step_delta | linear | 0.817 | 0.740 | 0.743 |
| readout ensemble (3) | n/a | not computed | | |

(The ensemble's in-domain number is missing because only aggregate in-domain
metrics were retained per row, not per-step in-domain scores. It needs one rerun
that saves in-domain scores to fill in.)

## Out-of-domain: ProcessBench first-error F1_PB @ calib-20

| representation x learner | gsm8k | math | olympiadbench | omnimath | **avg (4)** |
|---|---|---|---|---|---|
| **readout ensemble (3)** | 0.591 | 0.553 | 0.500 | 0.484 | **0.532** |
| **step_tokens x transformer** | 0.568 | 0.558 | 0.492 | 0.469 | **0.522** |
| **step_tokens x attn-query** | 0.495 | 0.533 | 0.486 | 0.486 | **0.500** |
| **step_stats x linear** | 0.544 | 0.473 | 0.445 | 0.477 | **0.485** |
| **last_token x linear** | 0.459 | 0.414 | 0.347 | 0.357 | **0.394** |
| **step_delta x linear** | 0.345 | 0.409 | 0.344 | 0.346 | **0.361** |

(calib-20 std ~0.03 to 0.08 per subset; calib-20 recovers 85 to 95% of per-subset
oracle for every row. Per-subset oracle for `last_token`: gsm8k 0.501, math 0.472,
olympiadbench 0.391, omnimath 0.395. For the ensemble: 0.628 / 0.597 / 0.558 /
0.532, avg 0.579, with calib-20 recovering 94.2 / 92.7 / 89.7 / 90.9% of it.)

The ensemble is a post-hoc combination computed with
`scripts/analysis/pb_threshold_calibration.py` over the saved per-step scores of
its three members; it involved no training and no re-encoding. Its identity was
verified by matching the stored `ens3` scores against every candidate combination
rule (exact match, max per-step absolute difference 0.0, for the unweighted
probability mean of the three).

## External reference systems (ProcessBench, reported elsewhere)

Same benchmark, same first-error F1 as our F1_PB, on the same human-labeled
ProcessBench solutions, so directly comparable on the 4-subset average. These are
fully fine-tuned 7B+ PRMs (vs our frozen-state readouts) with their own
threshold, so a reference ceiling, not an apples-to-apples training comparison.
Sourcing in `related_work.md`.

| system | gsm8k | math | olympiad | omnimath | **avg (4)** |
|---|---|---|---|---|---|
| **readout ensemble (3)** (ours, frozen states, calib-20) | 59.1 | 55.3 | 50.0 | 48.4 | **53.2** |
| **step_tokens x transformer** (ours, frozen states, calib-20) | 56.8 | 55.8 | 49.2 | 46.9 | **52.2** |
| **step_tokens x attn-query** (ours, frozen states, calib-20) | 49.5 | 53.3 | 48.6 | 48.6 | **50.0** |
| **step_stats x linear** (ours, frozen states, calib-20) | 54.4 | 47.3 | 44.5 | 47.7 | **48.5** |
| **last_token x linear** (ours, frozen states, calib-20) | 45.9 | 41.4 | 34.7 | 35.7 | **39.4** |
| **step_delta x linear** (ours, frozen states, calib-20) | 34.5 | 40.9 | 34.4 | 34.6 | **36.1** |
| Math-Shepherd-PRM-7B | 47.9 | 29.5 | 24.8 | 23.8 | 31.5 |
| Skywork-PRM-7B | 70.8 | 53.6 | 22.9 | 21.0 | 42.1 |
| Qwen2.5-Math-7B-PRM800K | 68.2 | 62.6 | 50.7 | 44.3 | 56.5 |
| ThinkPRM-14B | n/a | n/a | 87.3 | 85.7 | n/a |

**Reading.** Because every row shares the same frozen 7B spine, the
representation and the learner effects separate cleanly, and the naming above
makes the comparison direct.

*Representation effect (learner held fixed at linear).* Going from `last_token` to
`step_stats`, i.e. from the step's final token to a fixed 5-statistic pool of all
its tokens, moves the 4-subset average from **39.4** to **48.5**. That is **+9.1
F1_PB for feeding the whole step**, with no change in learner and no added
capacity beyond the wider input.

*Learner effect (representation held fixed at `step_tokens`).* Swapping the
attention-query readout for the ~2.5M-param transformer moves **50.0** to
**52.2**, i.e. **+2.2**.

So the representation change is worth roughly 4x the learner change. Most of the
OOD gain comes from having the step's tokens at all, not from the sequence model
that reads them. `step_tokens` x transformer is the best single row on every axis
(in-domain AUROC **0.874**, OOD avg **52.2**); averaging the three best readouts
adds a further **+1.0** to **53.2** without any training, which says a meaningful
slice of each readout's error is idiosyncratic rather than a shared property of
the frozen states.

Against external systems, the frozen-state ensemble clears Skywork-PRM-7B (42.1)
outright and closes most of the gap to the fully fine-tuned
Qwen2.5-Math-7B-PRM800K (56.5), with no backbone fine-tuning at all. The gains
concentrate on the hard subsets: the ensemble scores 50.0 / 48.4 on OlympiadBench
/ OmniMath where `last_token` sat at 34.7 / 35.7 and the fine-tuned
Math-Shepherd and Skywork models collapse to the low 20s.

`step_delta` remains the weakest. It trails `last_token` in domain (0.817 vs 0.828
AUROC) and on gsm8k, though it is close on the harder subsets: the transition
geometry does not beat the boundary state at step granularity, which echoes the
transition-operator result in REPORT.md and localizes CLUE's trace-level finding.

## Systems reproducible within this framework

Each is one representation x learner under the same protocol (see
`related_work.md`), named here by the paper only to record the correspondence:
**ReProbe = `step_tokens` x transformer, DONE** (`scripts/train_reprobe_probe.py`,
~2.5M-param transformer, last-layer token subset); CLUE = `step_delta` x
nearest-centroid; SSAE = sparse latents x linear;
Hidden-States-as-Early-Signals = `last_token` x MLP. All rows read from the same
7B token store (`data_setup.md`): `last_token` / `step_delta` / `step_stats`
derive offline with no re-encoding, while the `step_tokens` learners train
directly on the stored token spans.
