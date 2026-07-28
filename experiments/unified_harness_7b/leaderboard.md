# Unified Harness 7B — Representation Leaderboard

Backbone **Qwen2.5-7B base**. Every representation is trained on the same frozen
PRM800K split, evaluated in-domain on a held-out balanced PRM800K test, then
transferred to ProcessBench. Only the representation (and, later, the learner)
varies. See `data_setup.md` for splits, sizes, and metric definitions.

Deployable headline for the OOD transfer is **F1_PB @ calib-20**: the first-error
threshold is calibrated on a small held-out ProcessBench subset (20 traces per
subset, stratified), not on the balanced PRM800K val. `fixed` (t=0.5) and
`oracle` (peeks at the full test, not deployable) are shown for context.

## In-domain: PRM800K test (balanced, 2,000 steps)

| representation | learner | AUROC | macro-F1 (val) | macro-F1 (oracle) |
|---|---|---|---|---|
| dense_last | linear | **0.828** | 0.754 | 0.760 |

## Out-of-domain: ProcessBench first-error F1_PB

| representation | learner | subset | fixed t=0.5 | val-selected | **calib-20** | oracle |
|---|---|---|---|---|---|---|
| dense_last | linear | gsm8k | 0.406 | 0.406 | **0.459 ± 0.036** | 0.501 |
| dense_last | linear | math | 0.228 | 0.228 | **0.414 ± 0.060** | 0.472 |
| dense_last | linear | combined | 0.287 | 0.287 | **0.407 ± 0.056** | 0.466 |

## External reference systems (ProcessBench, reported elsewhere)

Same benchmark, same first-error F1 as our F1_PB, on the same human-labeled
ProcessBench solutions — directly comparable metric and task. These are fully
fine-tuned 7B+ PRMs (vs our frozen-state probe) and use their own threshold, so
they are a reference ceiling, not an apples-to-apples training comparison. See
`related_work.md` for sourcing.

| system | GSM8K F1 | MATH F1 | avg (4 subsets) |
|---|---|---|---|
| **dense_last (ours, frozen linear, calib-20)** | **45.9** | **41.4** | n/a (2 subsets) |
| Math-Shepherd-PRM-7B | 47.9 | 29.5 | 31.5 |
| Skywork-PRM-7B | 70.8 | 53.6 | 42.1 |
| Qwen2.5-Math-7B-PRM800K | 68.2 | 62.6 | 56.5 |
| ThinkPRM-14B | — | — | Olympiad 87.3 / Omni 85.7 |

Reading: on GSM8K our frozen linear probe (45.9) is on par with
Math-Shepherd-PRM-7B (47.9); on MATH it (41.4) **exceeds** Math-Shepherd-PRM-7B
(29.5), from a hidden state with no fine-tuning. It trails the PRM800K-fine-tuned
PRM (68.2 / 62.6) and the generative SOTA. We cover GSM8K + MATH only, so the
4-subset average is not yet comparable.

## Systems reproducible within this framework

Each is one representation x learner under the same protocol (see
`related_work.md`): ReProbe = `token_store` x transformer; CLUE = `delta` x
nearest-centroid; SSAE = `sparse` x linear; Hidden-States-as-Early-Signals =
`dense_last` x MLP.

## Notes

- `dense_last` = last-layer hidden state at the last token of the step (the
  post-step boundary). It is the baseline point representation.
- calib-20 recovers ~88–92% of oracle F1_PB and roughly doubles the balanced-val
  number on math (0.228 -> 0.414); the val/oracle gap was threshold
  miscalibration, not a detection failure.
- Pending representations (same protocol): `delta` (S_t - S_{t-1}),
  `step_tokens` pooled/attended, `step_trajectory`. All derive offline from the
  full last-layer token store (`data_setup.md`).
