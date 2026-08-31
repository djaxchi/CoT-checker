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

---

## Iteration 2 — norm-exponent sweep and fixed attention stand-ins (job 433586)

Hypothesis: iteration 1 showed a=0 (`mean_l2`) costs 0.092 against a=1 (`mean`),
so the norm is an importance weight; extrapolating, a>1 should be better. Also:
if the learned attention query's +0.089 is mostly "attend to high-norm tokens", a
fixed norm-softmax should recover much of it.

| pooling | PB step AUROC |
|---|---|
| centered | 0.7534 |
| normpow a=0.5 | 0.7476 |
| quantiles | 0.7475 |
| **mean (a=1.0)** | **0.7462** |
| typical | 0.7459 |
| normpow a=1.5 | 0.7445 |
| normpow a=2.0 | 0.7428 |
| atypical | 0.7408 |
| normpow a=3.0 | 0.7389 |
| softnorm t=2.0 | 0.7275 |
| softnorm t=0.5 | 0.6804 |
| mean_l2 (a=0.0) | 0.6542 |

**The exponent is not monotone and `mean` is already at its optimum.** The curve
climbs steeply from a=0 to a≈0.5, is flat between 0.5 and 1.0 (0.7476 vs 0.7462,
inside the noise floor), then declines steadily. Extrapolating past a=1 was wrong.

**The more useful result is the softnorm failure.** An explicit norm-based
attention scores 0.728 and 0.680, well below plain mean. So whatever the learned
attention query is doing to earn +0.089, **it is not primarily attending to
high-norm tokens.** Nor to typicality: weighting by cosine to the step's gist, or
against it, both land on the baseline. The learned query must be selecting on
*content* — a specific direction — which no fixed rule reproduces.

**Where this leaves fixed pooling.** Sixteen rules over two iterations; none beats
`mean` outside the noise floor. The pooling axis looks exhausted for
label-free rules.

**Next.** Stop varying the pooling and target the transfer gap directly. A
measured fact nobody has used: PRM800K steps average 38.8 tokens while
ProcessBench steps run 56 to 94. If the representation encodes step length, the
probe learns a length-dependent boundary that misfires on the longer domain — and
the screen measures exactly that transfer. Test by regressing length out of the
representation (fit on train, applied unchanged to ProcessBench), with the
opposite control of explicitly *adding* length: if adding it helps in domain and
hurts transfer, length is a transfer confound worth removing.

---

## Side result — full whitening as the grid protocol (job 433522, 45 cells)

Hypothesis (the user's): correctness is a low-variance direction, so equalising
every direction's variance should stop variance-ordered methods discriminating
against it. `zscore` divides each position by its own swing; `whiten` removes the
correlations too.

| | none | zscore | whiten |
|---|---|---|---|
| best cell | 0.540 | 0.540 | **0.494** |
| mean change vs zscore | | — | **−0.037** |
| Spearman vs zscore | | — | **+0.681** |

**Whitening loses, and reorders.** It costs 0.037 on average, drops the ceiling by
0.046, and disagrees with the zscore ranking at Spearman 0.681 — so it is not a
uniform shift, it changes which representation looks best.

The damage is concentrated in the MLP cells (−0.077 to −0.086) while the linear
cells are near-flat (−0.034 to +0.010). That fits the mechanism: whitening
amplifies every low-variance direction including noise (fitted covariance
condition numbers 2.6e5 to 8.4e6, and 0.05 shrinkage was not enough), and a
linear probe can partly undo a fixed rotation while an MLP's first layer cannot.

This is the **second** independent negative for the idea: the bottleneck screen
already put `recon_white` last of eight at 0.694. Equalising variance helps a
*distance metric* for classification, as the conicity work found; it does not
help either a *reconstruction target* or a *probe input*.

`zscore` stays the protocol.
