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

## Iteration 4b: the screen was broken, and the surface baseline is the story

Iteration 4's surface baseline came back at 0.2967 PB step AUROC for
`surface_length_poly` and 0.2961 for `surface_length`, both far below chance,
while the same job printed the statistics that make that impossible:

```
train      mean length  correct 32.8   incorrect 45.3
procbench  mean length  not-first-error 79.7   first-error 118.6
```

Longer steps are more often wrong in both domains, so a monotone probe on log
length cannot score under 0.5. Reproduced on matched synthetic data:

```
AUROC of the raw feature (no probe at all):  train 0.7077   pb 0.7363
  fitted probe epochs=8   lr=0.001 -> pb 0.2637   weight -0.1773
  fitted probe epochs=8   lr=0.1   -> pb 0.7363   weight +1.1137
  fitted probe epochs=60  lr=0.1   -> pb 0.7363   weight +1.1618
```

The probe never left its random initialisation sign. One learning rate cannot
serve both a 4,096-dim activation block (per-position swing around 22) and a
single log-length scalar (swing around 0.3). Fix: standardise inputs from train
statistics inside the screen, the same lesson `--rescale zscore` taught the grid.

The real number needs no probe at all. AUROC of raw step length as a score, on
the actual cached vectors:

| split | AUROC of length alone |
|---|---|
| PRM800K val (in domain) | 0.6142 |
| ProcessBench gsm8k | 0.7158 |
| ProcessBench math | 0.6744 |
| ProcessBench olympiadbench | 0.7151 |
| ProcessBench omnimath | 0.7101 |
| **ProcessBench mean** | **0.7039** |

Base `step_mean` scores 0.7707 on the same screen. So a single scalar, available
without running the model at all, recovers most of the transfer score, and the
gap the whole grid is competing over is about 0.067 wide, not 0.27. Every
representation claim has to be restated net of length. Whether the activations
carry anything beyond length is what `mean_residual` and `mean_pluslen` answer;
re-screened in job 433702.

Note also that length is a much weaker cue in domain (0.614) than on
ProcessBench (0.704). The transfer benchmark is the one where the shortcut pays
best, which is the opposite of what a robustness benchmark should do.

## Iteration 5: length is a shortcut on the benchmark, but the representations are not taking it

Iteration 4b left an open question: if a token count scores 0.7039 and the best
representation 0.7700, is the grid mostly ranking length? Scoring inside
equal-count length bins answers it. Inside a bin the steps are the same length,
so length carries nothing and what survives is the representation's own.

Bin count is not free, so the length control runs through the identical
procedure: it goes 0.7039 plain, 0.5359 at 10 bins, 0.5097 at 50. At 50 bins
length is gone.

| representation | PB plain | PB within-length (50 bins) | cost |
|---|---|---|---|
| dir | 0.7673 | **0.7307** | 0.0366 |
| atypical | 0.7633 | 0.7287 | 0.0347 |
| mean_l2 | 0.7646 | 0.7268 | 0.0378 |
| normpow_0.5 | 0.7629 | 0.7260 | 0.0370 |
| mean | 0.7619 | 0.7252 | 0.0367 |
| typical | 0.7614 | 0.7248 | 0.0366 |
| centered | 0.7622 | 0.7216 | 0.0406 |
| diffs | 0.7600 | 0.7106 | 0.0494 |
| quantiles | 0.7478 | 0.7040 | 0.0438 |
| dev | 0.7332 | 0.6978 | 0.0354 |
| first_last | 0.7477 | 0.6964 | 0.0512 |
| softnorm_0.5 | 0.6952 | 0.6660 | 0.0292 |
| **[control] length itself** | **0.7039** | **0.5097** | **0.1941** |

The representations lose about 0.037 while the length baseline loses 0.194. So
the activations are not riding the shortcut: their separation is almost entirely
length-independent, and the margin over length is much larger than the plain
numbers suggest.

    plain          dir 0.7673 vs length 0.7039   gap 0.063
    within-length  dir 0.7307 vs length 0.5097   gap 0.221

This is the opposite of the worry that motivated the baseline, and it is a better
result than the plain leaderboard could show. It also says what the plain
ProcessBench number is worth as a ranking signal: roughly 0.20 of every
representation's 0.76 is a token count that every row gets equally, which
compresses the visible spread between representations by about a third. The
within-length column separates `dir` from `first_last` by 0.034 where the plain
column separates them by 0.020.

Two things follow for the search. `dir` and `atypical`, the two poolings that
discard magnitude, lead the within-length column, which is consistent with the
project's earlier "direction not magnitude" result and is the first evidence for
it that survives pooling. And the honest headroom for a new representation is
measured against 0.7307, not against a length-inflated 0.767.

Caveat: the residual, withlen and surface files were skipped in this pass because
neither deriver copied the length arrays into its output. Fixed and tested; those
rows land in the next pass.

## Iteration 6: multi-layer stacking, and a discrepancy that blocks reading it

Layer 26 finished encoding, so the one hypothesis that ever bought a large gain
in this project (multi-layer stacking, about +0.05 AUC on Qwen2.5) is testable on
Qwen3. Layer 26 alone was screened alongside the stack so a gain could be
attributed to reading two layers rather than to 26 being the better single layer.

The screen liked it:

| representation | PB step AUROC | in-domain |
|---|---|---|
| dir_L26L35 | **0.7771** | 0.8653 |
| mean_L26L35 | 0.7700 | 0.8644 |
| mean_l2_L26L35 | 0.7679 | 0.8651 |
| atypical_L26L35 | 0.7662 | 0.8627 |
| dir_L26only | 0.7619 | 0.8618 |
| dir | 0.7603 | 0.8573 |
| mean | 0.7575 | 0.8561 |
| mean_L26only | 0.7531 | 0.8614 |

Every stack beats both its own layers, which is the shape the hypothesis
predicts, and dir_L26L35 at 0.7771 is the best number the screen has produced.

The stratified script, run on the same files in the same job, does not agree:

| representation | screen PB | stratified PB | gap | dim |
|---|---|---|---|---|
| dir_L26L35 | 0.7771 | 0.7563 | **-0.0208** | 8192 |
| mean_L26L35 | 0.7700 | 0.7502 | -0.0198 | 8192 |
| mean_l2_L26L35 | 0.7679 | 0.7466 | -0.0213 | 8192 |
| dir_L26only | 0.7619 | 0.7655 | +0.0036 | 4096 |
| dir | 0.7603 | 0.7673 | +0.0070 | 4096 |
| mean | 0.7575 | 0.7619 | +0.0044 | 4096 |

Both fit the same probe with the same standardisation, epochs, learning rate,
batch and seed. The only difference was 50,000 training rows against 60,000. The
gap is about +0.005 at 4,096 dims and about -0.021 at 8,192, so it tracks width,
not sampling. That is what an undertrained probe looks like: at a fixed epoch
count a wider representation sits further from convergence.

Under the stratified column stacking LOSES: dir_L26L35 at 0.7142 within strata
against plain dir at 0.7307. So the two columns do not merely differ in level,
they disagree about the hypothesis.

Stacking doubles the width, which is exactly the axis the discrepancy tracks, so
the stacking result cannot be read at all until the budget is swept. No verdict
recorded. Job 433791 runs epochs {8, 25, 60} against lr {1e-3, 1e-2} on eight
representations spanning 4,096 to 12,288 dims. The outcome to look for is not the
best cell but whether the ORDER is the same in every cell.

Also fixed: stratified_auroc now uses the screen's n_train, with a test that both
scripts keep matching n_train, epochs and lr defaults.

### The derived representations, now that they carry their lengths

The rows skipped last pass, at 50 bins:

| representation | PB plain | PB within-length | cost |
|---|---|---|---|
| mean_pluslen | 0.7915 | 0.7332 | 0.0583 |
| mean_withlen | 0.7755 | 0.7291 | 0.0464 |
| dir | 0.7673 | 0.7307 | 0.0366 |
| mean | 0.7619 | 0.7252 | 0.0367 |
| mean_residual | 0.7347 | **0.7256** | **0.0091** |
| surface_length | 0.7039 | 0.5097 | 0.1941 |

Two things worth keeping. Bolting length onto `mean` buys 0.0296 plain
(0.7619 -> 0.7915) and 0.0080 within strata (0.7252 -> 0.7332), so about
three quarters of what the length feature adds to the plain score is the
shortcut rather than new information.

And `mean_residual`, which has length regressed out of every position, scores
0.7256 within strata against plain `mean`'s 0.7252, while paying only 0.0091 to
stratification against everyone else's 0.037. So removing length costs nothing
once length is not scoreable, and it is the only representation that is already
close to length-free. Its plain score of 0.7347 is the honest one; the other
rows are carrying about 0.03 of length that stratification removes.
