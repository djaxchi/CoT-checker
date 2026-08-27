# Representation x Learner Leaderboard

Which part of a reasoning step's activations carries its correctness, and how much
of the answer is the detector rather than the representation?

Every row is a **pair**: a representation (which rows of the forward pass survive
into the vector the learner sees) and a learner (what reads it). Everything else
is pinned. Two neighbouring cells differ in exactly one coordinate, which is what
the earlier version of this file could not claim.

Two runs are recorded, on different backbones:

| run | backbone | layer read | dim | cells |
|---|---|---|---|---|
| **v1** | Qwen2.5-7B base | `hidden_states[-1]`, post-final-RMSNorm | 3,584 | 57 |
| **v2** | Qwen3-8B-Base | `hidden_states[35]` = `resid_post` of block 34 | 4,096 | 57 |

Both train on the same frozen problem-disjoint PRM800K split (513,810 / 5,000 /
2,000), transfer to all four ProcessBench subsets, and use one trainer, one
hyperparameter protocol, and three seeds per cell. v2 reads a genuine `resid_post`
rather than the post-norm final state because that is what the public Qwen-Scope
SAEs are trained on, so one store carries both the dense representations and the
forthcoming SAE cell at the same layer.

---

## The headline: the ranking survives a change of backbone

The point of a controlled grid is that its ordering should be a property of the
representations, not of the model it was measured on. That is testable, and it
holds.

```
19 shared cells, both runs scored under the same metric

  ProcessBench F1_PB @ calib-20    Spearman +0.919   Kendall +0.782   p < 1e-6
  In-domain PRM800K AUROC          Spearman +0.839   Kendall +0.689   p < 1e-5

  14 of 19 cells move by at most 2 ranks
```

That is across a simultaneous change of **backbone** (Qwen2.5-7B to Qwen3-8B-Base),
**hidden dimension** (3,584 to 4,096), and **layer** (post-final-norm last state to
block 34's `resid_post`). Absolute scores rise about 0.04, which is expected from a
stronger backbone and says nothing on its own; the ordering is the claim.

| cell | v1 | v2 | rank |
|---|---|---|---|
| `step_tokens` x transformer d512 | 0.534 | **0.566** | 1 -> 1 |
| `step_tokens` x attn_query | 0.506 | 0.558 | 4 -> 2 |
| `step_tokens` x transformer d128 | 0.528 | 0.554 | 3 -> 3 |
| `boundary_stats` x mlp:h1024 | 0.480 | 0.540 | 7 -> 4 |
| `step_stats` x mlp:h1024 | 0.494 | 0.540 | 5 -> 5 |
| `boundary_stats` x mlp:h1024x2 | 0.473 | 0.536 | 10 -> 6 |
| `step_stats` x mlp:h1024x2 | 0.482 | 0.535 | 6 -> 7 |
| `step_tokens` x transformer d256 | 0.532 | 0.531 | 2 -> 8 |
| `step_stats` x linear | 0.474 | 0.511 | 9 -> 9 |
| `boundary_stats` x linear | 0.476 | 0.509 | 8 -> 10 |
| `step_mean` x mlp:h1024 | 0.446 | 0.495 | 11 -> 11 |
| `step_mean` x mlp:h1024x2 | 0.429 | 0.481 | 13 -> 12 |
| `step_mean` x linear | 0.443 | 0.469 | 12 -> 13 |
| `step_delta` x mlp:h1024 | 0.384 | 0.440 | 16 -> 14 |
| `step_delta` x mlp:h1024x2 | 0.369 | 0.436 | 18 -> 15 |
| `last_token` x mlp:h1024 | 0.408 | 0.422 | 15 -> 16 |
| `last_token` x mlp:h1024x2 | 0.414 | 0.422 | 14 -> 17 |
| `last_token` x linear | 0.381 | 0.419 | 17 -> 18 |
| `step_delta` x linear | 0.351 | **0.395** | 19 -> 19 |

**What holds exactly.** All four `step_tokens` cells occupy the top three plus one
in both runs. `step_delta x linear` is last in both and `last_token x linear`
second-to-last in both. The three `step_mean` cells sit in the same block of the
table in both. The families do not interleave.

**The one large move is a tie, not a disagreement.** `transformer d256` falls six
places, but on v2 the three transformer capacities score 0.566 +- 0.026, 0.554 +-
0.021 and 0.531 +- 0.020 — mutually overlapping. Ordering statistically
indistinguishable cells is not information, and the same is true of the four
`step_stats` / `boundary_stats` MLP cells that shuffle among ranks 4 to 7.

**Which contrasts replicate**, at matched dimension and matched parameter count
where noted:

| contrast | v1 | v2 |
|---|---|---|
| `last_token` -> `step_mean` (identical dim, identical params) | +0.062 | +0.050 |
| `step_delta` -> `last_token` | +0.030 | +0.024 |
| `step_mean` -> `step_stats` (5x wider input) | +0.031 | +0.042 |
| pooling, then learned pooling (`step_mean` lin -> `attn_query`) | +0.063 | +0.089 |

---

## v2: Qwen3-8B-Base, `resid_post` of block 34

ProcessBench first-error F1_PB at calib-20, four-subset mean, mean +- sd over 3 seeds.

| representation | dim | linear | mlp:h1024 | mlp:h1024x2 | attn_query | tf d128 | tf d256 | tf d512 |
|---|---|---|---|---|---|---|---|---|
| `step_tokens` | 4096 | — | — | — | 0.558 ± 0.004 | 0.554 ± 0.021 | 0.531 ± 0.020 | **0.566 ± 0.026** |
| `step_stats` | 20480 | 0.511 ± 0.005 | 0.540 ± 0.006 | 0.535 ± 0.002 | — | — | — | — |
| `boundary_stats` | 24576 | 0.509 ± 0.002 | 0.540 ± 0.003 | 0.536 ± 0.009 | — | — | — | — |
| `step_mean` | 4096 | 0.469 ± 0.009 | 0.495 ± 0.006 | 0.481 ± 0.011 | — | — | — | — |
| `step_delta` | 4096 | 0.395 ± 0.006 | 0.440 ± 0.006 | 0.436 ± 0.013 | — | — | — | — |
| `last_token` | 4096 | 0.419 ± 0.009 | 0.422 ± 0.036 | 0.422 ± 0.025 | — | — | — | — |

In-domain PRM800K test AUROC:

| representation | dim | linear | mlp:h1024 | mlp:h1024x2 | attn_query | tf d128 | tf d256 | tf d512 |
|---|---|---|---|---|---|---|---|---|
| `step_tokens` | 4096 | — | — | — | 0.885 ± 0.002 | 0.887 ± 0.014 | 0.888 ± 0.009 | **0.895 ± 0.005** |
| `step_stats` | 20480 | 0.874 ± 0.004 | 0.893 ± 0.002 | 0.887 ± 0.002 | — | — | — | — |
| `boundary_stats` | 24576 | 0.866 ± 0.003 | 0.891 ± 0.002 | 0.888 ± 0.002 | — | — | — | — |
| `step_mean` | 4096 | 0.869 ± 0.002 | 0.895 ± 0.002 | 0.892 ± 0.005 | — | — | — | — |
| `step_delta` | 4096 | 0.831 ± 0.009 | 0.860 ± 0.003 | 0.859 ± 0.004 | — | — | — | — |
| `last_token` | 4096 | 0.849 ± 0.002 | 0.866 ± 0.002 | 0.869 ± 0.003 | — | — | — | — |

Capacity, for the F1_PB-against-parameters view:

| representation x learner | params | F1_PB @ calib-20 |
|---|---|---|
| `last_token` x linear | 4,097 | 0.419 ± 0.009 |
| `step_mean` x linear | 4,097 | 0.469 ± 0.009 |
| `step_tokens` x attn_query | 8,193 | 0.558 ± 0.004 |
| `step_stats` x linear | 20,481 | 0.511 ± 0.005 |
| `boundary_stats` x linear | 24,577 | 0.509 ± 0.002 |
| `step_tokens` x transformer d128 | 788,353 | 0.554 ± 0.021 |
| `step_tokens` x transformer d256 | 2,759,681 | 0.531 ± 0.020 |
| `last_token` x mlp:h1024 | 4,196,353 | 0.422 ± 0.036 |
| `step_tokens` x transformer d512 | 8,665,089 | 0.566 ± 0.026 |
| `step_stats` x mlp:h1024 | 20,973,569 | 0.540 ± 0.006 |
| `boundary_stats` x mlp:h1024 | 25,167,873 | 0.540 ± 0.003 |

An 8,193-parameter learned pooling beats every fixed-vector cell in the grid,
including MLPs three thousand times larger. Learning *which* of a step's tokens to
read is worth more than any amount of capacity applied to a pooled vector.

---

## v1: Qwen2.5-7B base, post-final-norm last state

Preserved. These are the numbers behind `REPORT.md` section 19 and the sprint-6
write-up, restated under the corrected calib-20 metric (see Protocol) so the two
runs are comparable. The originals, computed on a uniform threshold grid, differ
by 0.00 to 0.02; the conclusions drawn from them are unchanged.

| representation | dim | linear | mlp:h1024 | mlp:h1024x2 | attn_query | tf d128 | tf d256 | tf d512 |
|---|---|---|---|---|---|---|---|---|
| `step_tokens` | 3584 | — | — | — | 0.506 ± 0.005 | 0.528 ± 0.008 | 0.532 ± 0.022 | **0.534 ± 0.021** |
| `step_stats` | 17920 | 0.474 ± 0.005 | 0.494 ± 0.005 | 0.482 ± 0.002 | — | — | — | — |
| `boundary_stats` | 21504 | 0.476 ± 0.004 | 0.480 ± 0.002 | 0.473 ± 0.001 | — | — | — | — |
| `step_mean` | 3584 | 0.443 ± 0.010 | 0.446 ± 0.017 | 0.429 ± 0.007 | — | — | — | — |
| `step_delta` | 3584 | 0.351 ± 0.007 | 0.384 ± 0.012 | 0.369 ± 0.014 | — | — | — | — |
| `last_token` | 3584 | 0.381 ± 0.005 | 0.408 ± 0.012 | 0.414 ± 0.001 | — | — | — | — |

---

## Representations: what each keeps

All read the same frozen activations, one causal pass over question + prior steps
+ current step. The past is inside every vector through attention; they differ
only in which rows survive.

- **`last_token`** — the single state at the step's final token. The point-readout
  baseline everything else is measured against.
- **`step_delta`** — final-token state minus the pre-step boundary state: the
  *change* the step makes rather than the state it lands in. Weakest in both runs,
  which localises CLUE's trace-level finding to the step level and answers it
  negatively.
- **`step_mean`** — the mean over every token state of the step. The load-bearing
  addition: the only whole-step representation at the *same dimension and same
  parameter count* as `last_token`, so pooling can be measured without width
  moving underneath it.
- **`step_stats`** — concat[mean, max, min, std, last], 5x dim.
- **`boundary_stats`** — the above with the pre-step boundary state prepended, 6x dim.
- **`step_tokens`** — no reduction: the full variable-length sequence.

## Learners

Vector learners read one fixed vector: `linear`, and MLPs at one and two hidden
layers of 1,024. Sequence learners read the padded token sequence: `attn_query`
(one learned query pools the tokens, then a linear head) and transformer encoders
at three capacities.

The grid cannot be fully crossed — a linear head cannot consume a variable-length
sequence, and a transformer has nothing to attend over on one vector. `step_mean`
is the bridge: mean-pooling a sequence and reading it with a linear head *is*
`step_mean` x `linear`, which anchors the two families to a shared axis.

Note an MLP's parameter count scales with the representation, so "the same
learner" is not the same capacity across a row: `mlp:h1024` is 4.2M parameters on
`last_token` and 25.2M on `boundary_stats`. The capacity table reports true
per-cell counts.

## Protocol

Held fixed for every cell in both runs: all 513,810 PRM800K training steps (a cap
is recorded in the results and can never be silently compared against an uncapped
row); one AdamW + BCE trainer with the same epoch budget and early-stopping rule
for a linear head and an 8.7M-parameter transformer alike; the same learning-rate
x weight-decay grid, selected on validation AUROC, searched **once per cell** and
reused across its seeds; three seeds; and a fingerprint of every data split each
cell reads, which the merge script checks before rendering anything.

**calib-20**, the headline metric: 20 held-out ProcessBench traces per subset
(stratified) pick the first-error threshold, applied to the rest, averaged over 20
splits, meaned over the four subsets.

**Threshold candidates are score quantiles, not a uniform probability grid.** A
uniform grid assumes scores spread over [0,1]. On v2 the wide representations are
overconfident, piling 54 to 66% of scores below 0.01 and 10 to 30% above 0.99, so
a 0.01-step grid had almost no resolution where the decision boundary sat and
threshold selection from 20 traces became near-random: one seed of
`boundary_stats x linear` scored 0.248 against 0.498 and 0.494 for its siblings,
while its in-domain AUROC was 0.869, in line with theirs. Quantiles put the
candidates where the scores are. Broken cells recover (0.248 -> 0.510) and healthy
cells move by 0.000 to 0.005. Each split's grid comes from its own calibration
traces, never the evaluation traces.

The saturation is a consequence of the v2 layer change: post-final-norm states
have std 4.4, the pre-norm `resid_post` states have std 22.6, and five times
larger inputs push logits into saturation. v1 never hit it.

## External reference systems

Fully fine-tuned 7B+ PRMs on the same benchmark and metric. A reference ceiling,
not a matched training comparison: ours are small readouts on frozen states.

| system | added verifier | 4-subset avg |
|---|---|---|
| Qwen2.5-Math-7B-PRM800K | 7B fine-tuned | 56.5 |
| **`step_tokens` x transformer d512** (v2, frozen states) | 8.7M params | **56.6** |
| **`step_tokens` x attn_query** (v2, frozen states) | 8,193 params | **55.8** |
| Skywork-PRM-7B | 7B fine-tuned | 42.1 |
| Math-Shepherd-PRM-7B | 7B fine-tuned | 31.5 |
| ThinkPRM-14B | 14B generative | olympiad 87.3 / omnimath 85.7 |

Read with care: the external numbers come from the ProcessBench paper under those
systems' own threshold protocols, while ours use calib-20, which assumes 20
labelled target-domain traces. The comparison is indicative, not matched.

## Limits

- **One layer per run.** Nothing here separates "late layer" from "final token" as
  the limiting factor, and section 15 found layer 20 above layer 28 on Qwen2.5.
  The layer axis and the pooling axis have never been crossed.
- **Off-policy throughout.** PRM800K solutions were written by a GPT-4 fine-tune;
  we re-encode them with a model that would not produce that text. Every
  comparable internal-state paper reads the states of the model that generated the
  reasoning, and off-policy training measurably degrades probe generalisation
  (arXiv:2511.17408). Human step labels only exist for that generator's output, so
  this is a forced trade, not an oversight — and it is the subject of the on-policy
  arm.
- **calib-20 assumes 20 labelled target-domain traces.** The val-selected and
  oracle thresholds are recorded alongside in every cell's results.
- **Three seeds bound noise, they do not remove it.** Differences under about 0.02
  in this grid should not be ranked.
- **Readouts, not mechanisms.** Section 15.8 found steering along the decoded
  direction to be null; decodability is not causal relevance.
