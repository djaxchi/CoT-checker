# Unified Harness 7B — Representation Leaderboard

Backbone **Qwen2.5-7B base**. Every representation is trained on the same frozen
PRM800K split, evaluated in-domain on a held-out balanced PRM800K test, then
transferred to all four ProcessBench subsets. Only the representation (and,
later, the learner) varies. See `data_setup.md` for splits, sizes, and metrics.

Deployable OOD headline is **F1_PB @ calib-20**: the first-error threshold is
calibrated on 20 held-out ProcessBench traces per subset (stratified), applied to
the rest, mean over 20 splits. `val-selected` (t~0.5) and `oracle` (peeks at the
full test) are recorded in the run artifacts for context.

## In-domain: PRM800K test (balanced, 2,000 steps)

| representation | learner | AUROC | macro-F1 (val) | macro-F1 (oracle) |
|---|---|---|---|---|
| dense_last | linear | **0.828** | 0.754 | 0.760 |
| delta (S_t − S_{t-1}) | linear | 0.817 | 0.740 | 0.743 |

## Out-of-domain: ProcessBench first-error F1_PB @ calib-20

| representation | gsm8k | math | olympiadbench | omnimath | **avg (4)** |
|---|---|---|---|---|---|
| **dense_last** | 0.459 | 0.414 | 0.347 | 0.357 | **0.394** |
| **delta** | 0.345 | 0.409 | 0.344 | 0.346 | **0.361** |

(calib-20 std ≈ 0.04–0.06 per subset. Per-subset oracle for dense_last:
gsm8k 0.501, math 0.472, olympiadbench 0.391, omnimath 0.395.)

## External reference systems (ProcessBench, reported elsewhere)

Same benchmark, same first-error F1 as our F1_PB, on the same human-labeled
ProcessBench solutions — directly comparable, now on the full 4-subset average.
These are fully fine-tuned 7B+ PRMs (vs our frozen-state linear probe) with their
own threshold, so a reference ceiling, not an apples-to-apples training
comparison. Sourcing in `related_work.md`.

| system | gsm8k | math | olympiad | omnimath | **avg (4)** |
|---|---|---|---|---|---|
| **dense_last (ours, frozen linear, calib-20)** | 45.9 | 41.4 | 34.7 | 35.7 | **39.4** |
| **delta (ours, frozen linear, calib-20)** | 34.5 | 40.9 | 34.4 | 34.6 | **36.1** |
| Math-Shepherd-PRM-7B | 47.9 | 29.5 | 24.8 | 23.8 | 31.5 |
| Skywork-PRM-7B | 70.8 | 53.6 | 22.9 | 21.0 | 42.1 |
| Qwen2.5-Math-7B-PRM800K | 68.2 | 62.6 | 50.7 | 44.3 | 56.5 |
| ThinkPRM-14B | — | — | 87.3 | 85.7 | — |

**Reading.** Our frozen `dense_last` linear probe averages **39.4** across all
four ProcessBench subsets, **above fine-tuned Math-Shepherd-PRM-7B (31.5)**, below
Skywork-PRM-7B (42.1) and the PRM800K-fine-tuned PRM (56.5). It is strongest on
gsm8k/math and holds up on the harder OlympiadBench/OmniMath, where it beats
Math-Shepherd and Skywork (which collapse there). **delta** trails dense_last in
domain (0.817 vs 0.828 AUROC) and on gsm8k, but is close on math and the harder
subsets — the transition geometry does not beat the boundary state at step
granularity, echoing the S18 transition-operator result and localizing CLUE's
trace-level finding.

## Systems reproducible within this framework

Each is one representation x learner under the same protocol (see
`related_work.md`): ReProbe = `token_store` x transformer; CLUE = `delta` x
nearest-centroid; SSAE = `sparse` x linear; Hidden-States-as-Early-Signals =
`dense_last` x MLP. Both dense_last and delta above derive offline from the same
7B token store (`data_setup.md`), no re-encoding.
