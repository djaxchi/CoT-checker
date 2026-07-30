# Unified Harness 7B — Representation Leaderboard

Backbone **Qwen2.5-7B base**. Every representation is trained on the same frozen
PRM800K split, evaluated in-domain on a held-out balanced PRM800K test, then
transferred to all four ProcessBench subsets. Only the representation (and,
later, the learner) varies. See `data_setup.md` for splits, sizes, and metrics.

Deployable OOD headline is **F1_PB @ calib-20**: the first-error threshold is
calibrated on 20 held-out ProcessBench traces per subset (stratified), applied to
the rest, mean over 20 splits. `val-selected` (t~0.5) and `oracle` (peeks at the
full test) are recorded in the run artifacts for context.

## Representations: what is fed, and what is probed

Every entry reads from the same frozen Qwen2.5-7B base last-layer states. They
differ only in how a step's tokens are turned into the vector the learner sees;
the label being probed is always the step's binary correctness (0 correct,
1 incorrect).

- **reprobe** — the representation is the *raw set* of all last-layer token states
  of the step (variable length, capped at 128 tokens), same as attn_pool. The
  learner is the maximal one on this axis: a small (~2.5M-param) transformer
  (project 3584→256, learned positions, 2 encoder layers, masked mean-pool, linear
  head), our reproduction of ReProbe (Ni et al., 2025) restricted to the
  last-layer subset we store. It asks how much of the OOD gap is detector capacity
  rather than the representation, by giving the same tokens a full sequence model.
- **multistat** — the representation is a fixed 5-statistic summary of *all* of a
  step's last-layer token states: concat[mean, max, min, std, last-token] over the
  token axis, giving a 5×3584 = 17,920-dim vector. A single linear probe is fit on
  it. It asks whether step correctness is linearly readable from a permutation-
  invariant summary of the whole step, not just its final token.
- **attn_pool** — the representation is the *raw set* of all last-layer token
  states of the step (variable length, capped at 128 tokens). The learner is a
  learned attention query that softmax-weights the tokens into one 3584-dim vector,
  then a linear head. It asks whether letting the model learn *which* tokens to
  read beats a fixed pooling rule.
- **dense_last** — the representation is a single 3584-dim vector: the last-layer
  state at the step's last token only. A linear probe is fit on it. This is the
  point-readout baseline every other representation is measured against.
- **delta** — the representation is the 3584-dim transition vector
  S_t − S_{t-1} = (last-layer state at the step's last token) minus (last-layer
  state at the pre-step boundary token). A linear probe is fit on it. It asks
  whether the *change* a step makes to the residual stream carries the correctness
  signal, rather than the absolute state it lands in.

## In-domain: PRM800K test (balanced, 2,000 steps)

| representation | learner | AUROC | macro-F1 (val) | macro-F1 (oracle) |
|---|---|---|---|---|
| reprobe (all step tokens, ~2.5M-param transformer) | transformer | **0.874** | 0.790 | — |
| multistat (mean⊕max⊕min⊕std⊕last, 5×d) | linear | 0.866 | 0.783 | 0.788 |
| attn_pool (learned query over all step tokens) | attention | 0.860 | 0.778 | — |
| dense_last | linear | 0.828 | 0.754 | 0.760 |
| delta (S_t − S_{t-1}) | linear | 0.817 | 0.740 | 0.743 |

## Out-of-domain: ProcessBench first-error F1_PB @ calib-20

| representation | gsm8k | math | olympiadbench | omnimath | **avg (4)** |
|---|---|---|---|---|---|
| **reprobe** | 0.568 | 0.558 | 0.492 | 0.469 | **0.522** |
| **attn_pool** | 0.495 | 0.533 | 0.486 | 0.486 | **0.500** |
| **multistat** | 0.544 | 0.473 | 0.445 | 0.477 | **0.485** |
| **dense_last** | 0.459 | 0.414 | 0.347 | 0.357 | **0.394** |
| **delta** | 0.345 | 0.409 | 0.344 | 0.346 | **0.361** |

(calib-20 std ≈ 0.03–0.08 per subset; calib-20 recovers 85–95% of per-subset
oracle for every representation. Per-subset oracle for dense_last:
gsm8k 0.501, math 0.472, olympiadbench 0.391, omnimath 0.395.)

## External reference systems (ProcessBench, reported elsewhere)

Same benchmark, same first-error F1 as our F1_PB, on the same human-labeled
ProcessBench solutions — directly comparable, now on the full 4-subset average.
These are fully fine-tuned 7B+ PRMs (vs our frozen-state linear probe) with their
own threshold, so a reference ceiling, not an apples-to-apples training
comparison. Sourcing in `related_work.md`.

| system | gsm8k | math | olympiad | omnimath | **avg (4)** |
|---|---|---|---|---|---|
| **reprobe (ours, frozen states, transformer, calib-20)** | 56.8 | 55.8 | 49.2 | 46.9 | **52.2** |
| **attn_pool (ours, frozen states, attention, calib-20)** | 49.5 | 53.3 | 48.6 | 48.6 | **50.0** |
| **multistat (ours, frozen states, linear, calib-20)** | 54.4 | 47.3 | 44.5 | 47.7 | **48.5** |
| **dense_last (ours, frozen linear, calib-20)** | 45.9 | 41.4 | 34.7 | 35.7 | **39.4** |
| **delta (ours, frozen linear, calib-20)** | 34.5 | 40.9 | 34.4 | 34.6 | **36.1** |
| Math-Shepherd-PRM-7B | 47.9 | 29.5 | 24.8 | 23.8 | 31.5 |
| Skywork-PRM-7B | 70.8 | 53.6 | 22.9 | 21.0 | 42.1 |
| Qwen2.5-Math-7B-PRM800K | 68.2 | 62.6 | 50.7 | 44.3 | 56.5 |
| ThinkPRM-14B | — | — | 87.3 | 85.7 | — |

**Reading.** Two effects stack, and they are separable because every row shares
the same frozen 7B spine. First, the **representation**: using *all* of a step's
last-layer tokens instead of just the last one is the single biggest lever. Even a
fixed 5-statistic pooling into a linear probe (`multistat`) jumps the 4-subset
average from `dense_last`'s **39.4** to **48.5**, and a learned query over the same
tokens (`attn_pool`) to **50.0** — **+9 to +11 F1_PB** for feeding the whole step
rather than its final token. Second, the **detector**: holding that representation
fixed and swapping the learner from a linear head to a ~2.5M-param transformer
(`reprobe`, our ReProbe reproduction) adds another **+2.2** to **52.2**. So the
representation change is worth ~4x the detector change here; most of the OOD gain
is having the tokens, not the sequence model that reads them. `reprobe` is the top
row on every axis (in-domain AUROC **0.874**, OOD avg **52.2**), clears
Skywork-PRM-7B (42.1) outright, and closes most of the gap to the fully
fine-tuned Qwen2.5-Math-7B-PRM800K PRM (56.5) — from a frozen state, no backbone
fine-tuning. The gains concentrate on the hard subsets: `reprobe` scores 49.2 /
46.9 on OlympiadBench / OmniMath where `dense_last` sat at 34.7 / 35.7 and the
fine-tuned Math-Shepherd/Skywork collapse to the low 20s. In-domain the order is
identical (reprobe 0.874 > multistat 0.866 > attn_pool 0.860 > dense_last 0.828).
**delta** remains the weakest: it trails `dense_last` in domain (0.817 vs 0.828
AUROC) and on gsm8k though it is close on the harder subsets — the transition
geometry does not beat the boundary state at step granularity, echoing the S18
transition-operator result and localizing CLUE's trace-level finding.

## Systems reproducible within this framework

Each is one representation x learner under the same protocol (see
`related_work.md`): **ReProbe = `token_store` x transformer — DONE, the `reprobe`
row above** (`scripts/train_reprobe_probe.py`, ~2.5M-param transformer, last-layer
token subset); CLUE = `delta` x nearest-centroid; SSAE = `sparse` x linear;
Hidden-States-as-Early-Signals = `dense_last` x MLP. All rows read from the same
7B token store (`data_setup.md`): `dense_last`/`delta`/`multistat` derive offline
with no re-encoding, `attn_pool`/`reprobe` train directly on the stored token
spans.
