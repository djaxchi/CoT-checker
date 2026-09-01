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

## Iteration 7: the screen was ranking its own budget

The width-tracking discrepancy from iteration 6 turned out to be the small half of
the problem. Sweeping epochs and learning rate over eight representations:

| representation | e8 lr1e-3 | e8 lr1e-2 | e25 lr1e-3 | e25 lr1e-2 | e60 lr1e-3 | e60 lr1e-2 |
|---|---|---|---|---|---|---|
| dir | **0.7603** | 0.7288 | 0.7438 | 0.7239 | 0.7267 | 0.7076 |
| dir_L26only | 0.7619 | 0.7262 | **0.7560** | 0.7390 | 0.7397 | 0.7244 |
| dir_L26L35 | **0.7771** | 0.7290 | 0.7259 | 0.7164 | 0.7256 | 0.6951 |
| mean | 0.7575 | 0.7190 | 0.7398 | 0.7165 | 0.7190 | 0.7047 |
| mean_L26only | 0.7531 | 0.7162 | 0.7475 | 0.7321 | 0.7317 | 0.7197 |
| mean_L26L35 | 0.7700 | 0.7168 | 0.7203 | 0.7096 | 0.7157 | 0.6941 |
| quantiles | 0.7628 | 0.7047 | 0.7239 | 0.7028 | 0.6954 | 0.6524 |
| first_last | 0.7356 | 0.7315 | 0.7226 | 0.7043 | 0.7119 | 0.6867 |

Two things, both bad for everything recorded before this point.

**Transfer decays monotonically with training.** Every single representation
scores worse the more the probe is fitted. `dir` goes 0.7603 -> 0.7267 from 8 to
60 epochs at the same learning rate, and raising the learning rate at fixed
epochs does the same thing. The probe is overfitting PRM800K, and ProcessBench
pays for it.

**The ranking at the incumbent budget anti-correlates with every other budget.**
Spearman against epochs 8 / lr 1e-3:

    epochs   8 lr 0.01     -0.143    best: first_last
    epochs  25 lr 0.001    -0.238    best: dir_L26only
    epochs  25 lr 0.01     -0.190    best: dir_L26only
    epochs  60 lr 0.001    -0.071    best: dir_L26only
    epochs  60 lr 0.01     -0.190    best: dir_L26only

So "which representation is better" was being decided by where an unregularised
SGD run happened to be stopped, and the answer at the screen's own budget is
close to the reverse of the answer everywhere else.

**Stacking was an early-stopping artifact.** `dir_L26L35` is the best cell in the
whole table at 8 epochs (0.7771) and the worst at 60 (0.6951). A wider
representation is further from convergence at a fixed budget, so it looked good
for exactly the reason it was going to look bad later. Recorded as refuted. This
also matches the instinct that concatenating two layers is a bigger vector rather
than a better idea.

**One real finding survives.** Layer 26 alone beats layer 35 alone at every budget
except the incumbent one, and the margin grows as the probe converges: at 60
epochs `dir_L26only` 0.7397 against `dir` 0.7267, and `mean_L26only` 0.7317
against `mean` 0.7190. That is consistent with the project's earlier Qwen2.5
result that L20 beat L28, now reproduced on Qwen3 at the corresponding depth.

At the converged budget, within length strata, the whole field sits at 0.63 to
0.69 against the length control's 0.5097, so the content signal is still real; it
is the absolute numbers and the ordering that were budget artifacts.

### What replaces the screen

Ridge regression on +/-1 labels, closed form: no epochs, no seed, no learning
rate, one eigendecomposition serving the whole penalty path. The penalty is the
interesting knob rather than a nuisance, because its two ends are the two rules
the conicity study compared: as lambda goes to zero the solution approaches LDA,
which whitens by the data covariance (the study's 0.82), and as lambda grows it
approaches the mean-difference direction (the study's 0.63). Sweeping it lets
each representation be read at its own best point on that path.

Reported as val-selected lambda, chosen on in-domain validation without ever
looking at ProcessBench, plus an oracle ceiling, with in-domain alongside so the
overfitting above can be read directly.

## Iteration 8: relational representations, read with the ridge probe

First results from a probe with no budget in it. The path is well behaved: for
almost every representation the validation-selected penalty is also the one that
maximises ProcessBench, so val-selected and oracle agree to the fourth decimal.
In-domain and transfer now rise and fall together along the penalty path rather
than trading off, which is what the SGD screen's decay was hiding.

| representation | val-sel PB | in-domain | within-length | dim |
|---|---|---|---|---|
| dir_geom_layer | **0.7776** | 0.8616 | 0.7335 | 4129 |
| dir_geom | 0.7757 | 0.8612 | 0.7332 | 4117 |
| contribution_geom | 0.7670 | 0.8524 | 0.7191 | 4117 |
| dir_L26only | 0.7563 | 0.8633 | 0.7259 | 4096 |
| dir | 0.7522 | 0.8592 | 0.7241 | 4096 |
| mean_L26only | 0.7503 | 0.8632 | 0.7210 | 4096 |
| mean | 0.7470 | 0.8583 | 0.7194 | 4096 |
| contribution | 0.7406 | 0.8479 | 0.7106 | 4096 |
| contribution_dir | 0.7379 | 0.8474 | 0.7081 | 4096 |
| mean_residual | 0.7282 | 0.8423 | **0.7306** | 4096 |
| surface_length | 0.7039 | 0.6142 | 0.5097 | 1 |
| geom | 0.7000 | 0.6468 | 0.5596 | 21 |
| geom_layer | 0.6987 | 0.6692 | 0.5737 | 33 |
| layer_angle | 0.6376 | 0.6422 | 0.5468 | 12 |
| geom_nolen | 0.5182 | 0.6069 | 0.4675 | 20 |
| boundary | 0.5035 | 0.7412 | 0.5091 | 4096 |

**Pure geometry is refuted.** `geom` scores 0.7000, which looks respectable until
`geom_nolen` scores 0.5182, and 0.4675 within length strata, below chance. All of
`geom`'s apparent signal is the log token count it carries. The conicity
hypothesis, that correct steps cone tightly and incorrect ones do not, does not
survive as a transferable step-level signal: cone tightness, turn angles, norm
spread and the prefix cosine together separate the classes at 0.52 out of domain.
Worth stating plainly because the geometry was measured directly and without a
metric, which was the escape hatch the conicity study left open when it blamed
its 0.63-vs-0.82 gap on the metric.

**The prefix state alone is a strong in-domain predictor that transfers at
chance.** `boundary`, the model's state just before the step, with none of the
step's own tokens, scores **0.7412 in domain and 0.5035 on ProcessBench**. So a
large part of what an in-domain probe reads is not the step at all, it is where
in the trace the step sits and what came before. That fraction is entirely
domain-specific and vanishes on transfer. This is the cleanest evidence yet that
in-domain AUROC on PRM800K overstates step-level detection, and it comes almost
free: it is one of the two things the store already held.

**Contribution does not beat content.** `contribution` (0.7406) and
`contribution_dir` (0.7379) both sit slightly BELOW plain `mean` (0.7470) and
`dir` (0.7522). Subtracting the prefix state removes something the probe was
using, which is consistent with the `boundary` row: part of the in-domain signal
lives in the prefix, and the contribution deliberately throws it away. The S4
framing does not carry over to this benchmark. Refuted.

**Cross-layer revision is weak on its own and adds nothing on top.**
`layer_angle`, twelve numbers of disagreement between layers 26 and 35, scores
0.6376 alone, above chance and interesting in isolation. But `dir_geom_layer`
beats `dir_geom` by 0.0019, which is nothing. How much the late blocks rewrote a
step is not independent information about whether the step is wrong.

**The remaining question is whether dir_geom's +0.0235 is anything but length.**
Within strata the gain shrinks to 0.0091, about what bolting bare length on was
already worth (0.0080 for `mean_pluslen`). Controls launched as job 434230:
`dir_geom_nolen` and `dir_pluslen`.

**One row deserves attention for the opposite reason.** `mean_residual`, with
length regressed out of every position, has the WORST plain score of the dense
representations (0.7282) and the BEST within-length score of all of them
(0.7306), higher than `dir_geom`'s 0.7332 is above `dir`'s 0.7241. It is the only
representation whose plain and stratified numbers nearly coincide. If the goal is
a detector that is not reading step length, it is currently the best one, and the
plain leaderboard ranks it thirteenth.

## Iteration 10: the winner holds on the real metric

`lengthfree_geom` ran the full grid under the frozen protocol: 18 cells, the full
513,810-step PRM800K train split, three seeds, the same lr x weight-decay search
and trainer, `step_mean` rerun inside the same job so the comparison is paired
rather than read across runs. All seven input fingerprints matched.

**F1_PB at calib-20, four-subset mean:**

| representation | dim | linear | mlp:h1024 | mlp:h1024x2 |
|---|---|---|---|---|
| `lengthfree_geom` | 4116 | 0.510 ± 0.004 | **0.523 ± 0.007** | 0.518 ± 0.013 |
| `step_mean` | 4096 | 0.480 ± 0.007 | 0.497 ± 0.005 | 0.490 ± 0.012 |

The gain is +0.026 to +0.030 at every learner and exceeds the seed spread in
every cell. Oracle threshold agrees: 0.565 / 0.569 / 0.563 against 0.541 / 0.546 /
0.537. The screen predicted this correctly, which is the second confirmation of
its calibration.

**The mechanism check passes.** In-domain PRM800K AUROC is 0.872 / 0.892 / 0.894
against 0.869 / 0.890 / 0.891, which is a tie. The representation buys transfer
without buying in-domain fit. That is exactly what removing a domain-shifted
shortcut should look like, and it would not look like this if the 20 geometry
features were simply adding capacity.

**A reproducibility check came free.** This job's `step_mean` mlp:h1024 cell scored
0.497 ± 0.005; the v2 leaderboard's independent run scored 0.495 ± 0.006.

**Where it sits honestly.** It does not beat the top cell. Against the v2 board:

| representation | dim | best cell |
|---|---|---|
| `step_tokens` x transformer d512 | 4096 seq | 0.566 ± 0.026 |
| `step_stats` | 20480 | 0.540 ± 0.006 |
| `boundary_stats` | 24576 | 0.540 ± 0.003 |
| **`lengthfree_geom`** | **4116** | **0.523 ± 0.007** |
| `step_mean` | 4096 | 0.495 ± 0.006 |

So it is the best representation at its width, and closes about 60% of the gap
from `step_mean` to `step_stats` using a fifth of the dimensions. Sequence
representations read by a pooling learner remain ahead, and `step_tokens` x
attn_query reaches 0.558 with 8,193 parameters, which is still the most
efficient row on the board.

**One column disagrees and should not be buried.** At the val-selected threshold
`lengthfree_geom` linear scores 0.347 ± 0.027 against `step_mean`'s 0.403 ± 0.019.
That column carries spreads of 0.027 to 0.052 across three seeds, against 0.004
to 0.013 at calib-20, because a threshold picked on PRM800K does not transfer to
ProcessBench. calib-20 exists for that reason. It is still a real caveat: the
representation improves the ranking, not the calibration.

## Iteration 11: trace-relative coordinates and between-step dynamics, both refuted

The prediction was in-domain down, transfer up. The opposite happened, cleanly.

| representation | dim | val-sel PB | in-domain | within-length |
|---|---|---|---|---|
| `winner` (recomputed on these rows) | 4116 | **0.7885** | 0.8575 | 0.7310 |
| `winner_dyn` | 4126 | 0.7749 | 0.9076 | 0.7129 |
| `winner_trace` | 8212 | 0.7560 | 0.8784 | 0.7004 |
| `winner_trace_dyn` | 8222 | 0.7523 | **0.9101** | 0.6915 |
| `mean` | 4096 | 0.7467 | 0.8540 | 0.7130 |
| `trace_centered_causal` | 4096 | 0.6957 | 0.8194 | 0.7015 |
| `trace_centered_all` | 4096 | 0.5899 | 0.8254 | 0.5968 |
| `trace_z` | 4096 | 0.5788 | 0.8186 | 0.5974 |
| `dyn` | 10 | 0.5107 | 0.8248 | 0.4870 |
| `pos` (control) | 2 | **0.3649** | 0.5846 | 0.4198 |

Nothing trace-relative helps, and everything trace-relative hurts. Every variant
loses to `winner` in 1000 out of 1000 paired resamples.

**The in-domain column explains it.** `winner_trace_dyn` posts 0.9101 in domain,
the highest number this project has ever recorded, and 0.7523 on transfer. The
trace-relative features are strongly informative in domain and actively harmful
out of it, which is the signature of a feature whose MEANING differs between the
two datasets.

The likely reason is a labelling asymmetry nobody had to invent: in PRM800K a
solution can contain many incorrect steps scattered anywhere, while ProcessBench
is built so that every step before the first error is correct. So "this step
relative to its siblings" is computed against a mixed reference in training and
against an almost-entirely-correct reference at test time. The same arithmetic
means two different things. This is worth stating because GeoReason
(arXiv:2605.13772) reports its trace-normalised teacher transfers stably while
its distilled student "collapses under shift"; our result says the normalisation
itself is a transfer liability when the trace composition differs, which is a
concrete mechanism for that collapse.

`dyn` at 0.5107 also settles the between-step dynamics question. Speed relative
to the trace, directional persistence, acceleration and typicality carry
essentially nothing about step correctness out of domain, despite in-domain 0.8248.

**The control earned its place.** `pos`, which is nothing but relative position
and trace length, scores 0.5846 in domain and **0.3649** on ProcessBench, far
BELOW chance. Position is not merely a useless shortcut here, it is an actively
inverted one: what a probe learns about where errors sit in a PRM800K solution is
wrong about where they sit in a ProcessBench solution. Any representation that
encodes position pays for it on transfer, and this is the first time the search
has been able to say that with a number.

## Iteration 12: what the 20 geometry features actually are

The block is worth +0.0614 on top of length-free content (0.7283 -> 0.7897).
Ablating it one feature at a time, with the penalty selected on validation:

| feature | add-one-in | leave-one-out |
|---|---|---|
| cone_tightness_ratio | **+0.0458** | +0.0005 |
| cone_cos_mean | **+0.0457** | +0.0005 |
| cone_cos_p50 | +0.0426 | +0.0008 |
| log_norm_mean_vec | +0.0357 | -0.0001 |
| cone_cos_p90 | +0.0355 | +0.0007 |
| cone_cos_p10 | +0.0353 | +0.0005 |
| turn_cos_mean | +0.0334 | +0.0002 |
| lognorm_p10 | +0.0278 | -0.0000 |
| lognorm_mean / lognorm_std | +0.0257 | ~0 |
| log_norm_ratio | +0.0238 | +0.0000 |
| cone_cos_std | +0.0220 | -0.0001 |
| cos_stepdir_boundary | +0.0168 | +0.0001 |
| cos_last_boundary | +0.0054 | **+0.0016** |
| lognorm_p90 | +0.0033 | -0.0001 |
| cos_first_last | +0.0008 | -0.0000 |
| cos_first_boundary / log_norm_boundary | -0.0000 | -0.0000 |

Two things, and the second is the interesting one.

**The block is one measurement, not twenty.** Any single one of about eight
features recovers three quarters of the block's value, and NO feature is
individually necessary: the largest leave-one-out in the table is 0.0016. The two
leaders, `cone_tightness_ratio` (||mean|| / mean||token||) and `cone_cos_mean`
(mean cosine of each token to the step's own mean direction), are two ways of
writing the same quantity and score within 0.0001 of each other. The five cosine
quantiles and the four norm quantiles are quantiles of two distributions. So the
honest description of the block is: **how tightly the step's tokens agree in
direction**, measured nine different ways plus some dead weight.

Three features are dead on both criteria and can be dropped: `cos_first_boundary`,
`cos_first_last`, `log_norm_boundary`.

**This vindicates the conicity thread, but relocates the claim.** The earlier
conicity work found correct steps form a tight cone and incorrect ones do not,
scored the centroid rule at 0.63 against a whitened 0.82, and concluded the gap
was the metric. The result here is sharper and stranger: cone tightness is worth
+0.046 as the single addition to a content probe, and 0.5182 as a detector on its
own, 0.4675 inside length strata. It is not a weak detector, it is not a detector
at all. It is worth a great deal only in the company of the content direction.

That combination has a name in regression. A variable that correlates weakly with
the outcome and strongly with the other predictors, and earns its place by
removing variance from them that the outcome does not explain, is a **suppressor**.
The reading predicts three numbers: near-zero correlation with the label, clear
correlation with the content probe's score, and a partial correlation given that
score which is clearly larger than the raw one. Job 434852 measures all three, and
reports the correlation with log length in the same table so that "cone tightness
is a nonlinear stand-in for step length" is settled with a number rather than
inferred from the block scoring 0.5182 where length alone scores 0.7039.

If it holds, the mechanism sentence for the whole result becomes concrete: the
pooled step direction is a noisy readout of correctness, the noise is largest
when the step's tokens disagree with each other, and cone tightness tells the
probe how much to trust the direction it just read.
