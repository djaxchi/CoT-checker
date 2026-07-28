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

Reference PRMs (ProcessBench paper, gsm8k F1): Math-Shepherd-PRM-7B 47.9,
Qwen2.5-Math-7B-PRM800K 68.2. The dense linear probe at calib-20 (45.9) is in the
neighborhood of Math-Shepherd-PRM-7B, from a frozen hidden state, no fine-tuning.

## Notes

- `dense_last` = last-layer hidden state at the last token of the step (the
  post-step boundary). It is the baseline point representation.
- calib-20 recovers ~88–92% of oracle F1_PB and roughly doubles the balanced-val
  number on math (0.228 -> 0.414); the val/oracle gap was threshold
  miscalibration, not a detection failure.
- Pending representations (same protocol): `delta` (S_t - S_{t-1}),
  `step_tokens` pooled/attended, `step_trajectory`. All derive offline from the
  full last-layer token store (`data_setup.md`).
