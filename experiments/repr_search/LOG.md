# Representation search log

Screened with `scripts/screen_representation.py`, ranked by ProcessBench step
AUROC (Spearman 0.934 against the full grid's calib-20 over 31 evaluated cells).
Numbers are comparable **within** a run, not across runs: the sample size and
source cache differ, so each iteration carries its own `mean` baseline.

Resolution is about 0.01. Anything closer is a tie.

---

## Iteration 1 — alternative poolings (job 433578)

Hypothesis: the grid says *which rows you pool* dominates and every compression
tried ties or loses, so the room left is the pooling rule. Lead candidate from
this project's probe-anatomy result — the signal is **direction, not magnitude** —
so pooling L2-normalised tokens should beat plain mean, which lets a high-norm
token dominate (verified: one 50x-norm token drags `mean` to cosine 0.551 while
`mean_l2` is unmoved).

| pooling | PB step AUROC | vs mean |
|---|---|---|
| centered (mean ++ dev) | 0.7534 | +0.007 |
| quantiles (10/50/90) | 0.7475 | +0.001 |
| **mean** (baseline) | **0.7462** | — |
| diffs (within-step dynamics) | 0.7363 | −0.010 |
| first_last | 0.7006 | −0.046 |
| dir (pooled direction only) | 0.6613 | −0.085 |
| **mean_l2** (lead hypothesis) | **0.6542** | **−0.092** |
| dev (internal spread only) | 0.6509 | −0.095 |

**The lead hypothesis is refuted, decisively.** Normalising tokens before pooling
costs 0.092. Discarding the pooled magnitude (`dir`) costs 0.085.

**What that means.** Magnitude is not a nuisance here, it is doing work. Plain
`mean` weights each token by its own norm, which is an implicit importance
weighting, and it is a good one — equalising the tokens lets filler and
punctuation vote as loudly as the tokens carrying the reasoning. "Direction not
magnitude" was established for a single-token readout on the previous backbone;
it does not survive the move to a pooled representation.

Nothing beat the baseline outside the noise floor. Robust statistics, within-step
dynamics and endpoint concatenation all lose.

**Next.** If the norm acts as an importance weight, its exponent is a free
parameter nobody has tuned: pool `||h||^a * (h/||h||)`, where a=0 is `mean_l2`
(bad) and a=1 is `mean`. If the trend is monotone in a, a>1 should be better.
That also gives a cheap fixed approximation of what learned pooling — the biggest
single win in the grid, +0.089 — might be doing.
