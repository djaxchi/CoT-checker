# CoT-Checker: full project context

*2026-09-22. Written as paper material and as a complete brief for anyone picking
the work up. Supersedes `docs/sprint_6_7_summary.md`. Every number traces to
`REPORT.md`; section numbers below refer to it.*

---

## 1. The arc in one page

The project began as a reproduction: can sparse autoencoder features detect
incorrect reasoning steps. It ends somewhere else, and the path is the argument.

1. **The SAE premise failed, twice, cleanly.** Sparse codes never beat the hidden
   states they were built from, at matched or even 16x advantaged probe capacity.
   Reconstruction-trained features are not where correctness lives.
2. **The signal is real, and it is in the dense residual stream.** A probe on
   frozen activations detects step incorrectness well above chance, the ability
   scales monotonically with backbone size, and it survives every confound audit
   run against it.
3. **The signal is a readout, not a lever.** Steering along the probe direction
   moves nothing causally. This closed the interpretability-as-intervention door
   and redirected the work toward use.
4. **How you read a step matters about four times more than what reads it.** This
   is the finding the verifier is built on, and it is stable across backbones.
5. **A detection benchmark orders representations correctly but exaggerates what
   the ordering is worth.** Rank transfers downstream; magnitude does not.
6. **In the decision setting the verifier is useful in one specific regime:**
   small sample budgets, and only when forbidden from overruling the vote.

The paper's contribution is (4), (5) and (6). (1) through (3) are what licenses
them and what keeps the work from being a benchmark-chasing exercise.

---

## 2. Origin and the negative result that redirected it

The starting point was reproducing a step-level sparse autoencoder (Miaow-Lab,
arXiv:2603.03031): a Qwen2.5-0.5B backbone, GSM8K-Aug, with the claim that a
linear probe on the step latent `h_c` predicts step correctness. The appeal was
that the SSAE is trained for **reconstruction, not classification**, so any
correctness signal in its latent would be emergent rather than a supervised
artifact.

The reproduction succeeded and the premise did not survive it. In the controlled
grid (§19, and repeated on Qwen3-8B-Base), the sparse code loses to the hidden
state it was built from on **every** representation, with the probe held linear on
both sides, and still loses when the sparse side is given sixteen times more
parameters. The interpretation is simple: an SAE is trained to rebuild the hidden
state, and correctness is not most of what the hidden state is made of.

This is reported as a negative result and the thread is closed. It matters to the
paper only as provenance and as the reason the representation axis, rather than
the feature-interpretability axis, became the object of study.

---

## 3. What the correctness signal is

These results constrain everything after them and are what make this an
interpretability paper rather than a verifier paper. All on PRM800K with
problem-id-disjoint splits.

**It exists and it scales.** Dense probes on residual activations detect step
incorrectness on the full natural PRM800K test (L28, final token): **AUC 0.776 at
1.5B rising to 0.828 at 32B**, monotonic, with diminishing returns past 14B.

**It is distributed, not sparse or low-rank.** The signal sits in a
low-variance subspace that is invisible to UMAP and to the top principal
components. It is carried by **direction, not magnitude**. The final token of a
step beats the first.

**It is diffuse across tokens and is not surprise.** Per-token probe trajectories
show no single firing token where incorrectness appears; peakiness is the same for
correct and incorrect steps. The probe correlates with per-token entropy at
**−0.20**, so it is not reading the model's own uncertainty.

**Geometrically, correct steps form a tight cone and incorrect ones do not.** A
centroid rule is weak (0.63) where a whitened rule reaches 0.82; the gap is the
metric, not the direction.

**It is a readout, not a causal lever.** Additive steering along the probe weight
vector is null: the probe direction moves no better than random while the
downstream readout it is supposed to control does move. The correctness direction
is something the model exposes, not something it uses.

**It is content, not surface.** A matched-fork audit (same question, same golden
prefix, one correct and one incorrect continuation) finds the supervised
correctness margin survives residualising out length and position, survives a
minimal-edit control, and scales with backbone size. Length is a minor confound,
not the effect.

The consequence for the paper: because the direction is a readout and not a lever,
the only way to convert it into value is **selection**, not intervention. That is
why the work moves to test-time scaling rather than to steering or editing.

---

## 4. The verifier

### 4.1 Representation

Backbone **Qwen3-8B-Base**, frozen, no fine-tuning anywhere. The signal is read
from `resid_post` of **block 34** of 36, hidden dimension **4,096**.

The representation under study is `step_tokens`: **the residual state of every
token of the step**, kept as a ragged sequence rather than reduced to a fixed
vector. The alternatives, all read from the identical states, are `last_token`
(what prior work in this line used), `mean` over the step's tokens, `step_stats`
(five statistics over them), `boundary_stats` (those plus the pre-step boundary
state), and `step_delta` (the change the step makes).

The enabling piece is a ragged **token store** holding the full state of every
token of every step, so each representation is an offline slice or reduce of one
identical encoding pass rather than a separate encode. Nothing in the grid differs
by backbone, split or protocol, which is what makes the representation effect and
the learner effect separable instead of confounded.

### 4.2 Probe architectures

Two probes read a sequence; the rest read a fixed vector.

**`AttnQueryPool`**, one learned query over the step's tokens, **8,193 parameters
in total**:

```
att_i  = (q · h_i) / sqrt(d),   masked to the step
z      = sum_i softmax(att)_i · h_i
logit  = w · z + b                       q, w in R^4096
```

The smallest probe in the grid that makes the pooling rule learned rather than
fixed, and a strong leaderboard row, which is the point of including it.

**`TransformerPool`**, the maximal learner on the sequence axis:

```
x_i    = W_proj h_i + p_i      W_proj: 4096 -> d_model, p learned, t_max 512
z      = MaskedMean( TransformerEncoder(x) )
logit  = w · z + b
```

Pre-norm encoder layers (`norm_first=True`), dropout 0.1. Pooling is a
**mask-aware mean over the encoded step tokens**, not a CLS token and not the last
position, so the probe cannot recover "final token only" by degenerate attention.
Three sizes:

| name | d_model | layers | heads | FFN | parameters |
|---|---|---|---|---|---|
| S | 128 | 1 | 4 | 512 | 986,625 |
| M | 256 | 2 | 4 | 1,024 | **2,759,681** |
| L | 512 | 2 | 8 | 2,048 | 8,665,089 |

### 4.3 Training

PRM800K step-correctness labels, problem-id-disjoint, balanced 50/50: **513,810
train steps**, 5,000 val, 2,000 in-domain test. Binary cross-entropy, one probe
per (representation, learner, seed) cell, three seeds (42, 43, 44). Transfer is
evaluated on all four ProcessBench subsets (gsm8k 400, math 1,000, olympiadbench
1,000, omnimath 1,000 traces), unseen in training.

**Open point that must be resolved before publication.** Two probes of identical
architecture exist: the PRM800K-trained cell above, and a second trained on
**43,837 judge-labelled steps over 842 problems** of Qwen3-8B-Base's own
trajectories, built so that training distribution is the only thing that differs.
`REPORT.md` §21 does not state which of the two produced the test-time-scaling
scores, and the staged artifacts and the launch script point at different
directories. The transfer claim depends on this, so it has to be pinned down and
stated explicitly.

---

## 5. What the detection leaderboard says

ProcessBench F1, Qwen3-8B-Base, best probe per representation:

| representation | best probe | F1 |
|---|---|---|
| every token of the step | transformer L (8.67M) | **0.566** |
| every token of the step | attention query (8,193) | 0.558 |
| every token of the step | transformer M (2.76M) | 0.531 |
| five statistics over the tokens | MLP | 0.540 |
| five statistics, plus the state before | MLP | 0.540 |
| the mean over the tokens | MLP | 0.495 |
| the change the step makes | MLP | 0.440 |
| the final token only | MLP | 0.422 |

Against externally reported systems, 4-subset average:

| system | what it costs | avg F1 |
|---|---|---|
| o1-mini | proprietary reasoning model | 0.879 |
| QwQ-32B-Preview | 32B reasoning model | 0.715 |
| GPT-4o-0806 | proprietary, prompted | 0.619 |
| **ours, step tokens + transformer L** | **8.7M probe on cached states** | **0.566** |
| Qwen2.5-Math-7B-PRM800K | 7B fine-tuned | 0.565 |
| **ours, step tokens + attention query** | **8,193 parameters** | **0.558** |
| Skywork-PRM-7B | 7B fine-tuned | 0.421 |
| Math-Shepherd-PRM-7B | 7B fine-tuned | 0.315 |

Every system above ours runs a large model over every step. Ours runs a probe over
states the policy already computed while generating.

### Two findings

**The representation is worth about four times the learner.** Holding the learner
at linear and moving `last_token` to `step_stats` is a pure representation change
worth **+9.1 F1**. Holding the representation at `step_tokens` and swapping the
attention query for the transformer is a pure learner change worth **+2.2**. The
ordering across representations is stable across backbones (Spearman **+0.919**
against the same grid on Qwen2.5-7B), so it describes the representations rather
than the model they were measured on.

**Step length lives inside the representation and costs transfer.** PRM800K steps
average 39 words; ProcessBench steps 80 to 119. Removing a per-dimension linear
length fit, fitted once on PRM800K and applied unchanged, and adding 20
length-invariant shape measurements moves ProcessBench F1 from 0.495 to 0.523 and
changes nothing in domain. A representation that travels better, not a stronger
one.

Also on the record: averaging the three best readouts adds a further +1.0 F1 with
no training at all, so a real slice of each readout's error is idiosyncratic
rather than a shared property of the frozen states.

---

## 6. Does the benchmark predict usefulness

The leaderboard is measured entirely **off-policy**: trained on PRM800K, whose
solutions a GPT-4 fine-tune wrote, evaluated on ProcessBench, whose solutions other
models wrote. Nobody deploys a verifier that way. In deployment the model being
verified is the model that wrote the reasoning, which is also the whole point of
reading internal states.

Pointing the frozen verifiers at Qwen3-8B-Base's own solutions gives the answer,
bootstrapped over problems:

| benchmark | vs on-policy within-problem AUROC | vs best-of-N |
|---|---|---|
| ProcessBench | +0.823 | +0.661 [+0.099, +0.790] |
| PRM800K | +0.658 | +0.525 [−0.017, +0.711] |

But the effect sizes tell the other half:

| contrast | Δ ProcessBench | Δ best-of-10 |
|---|---|---|
| `last_token` → `step_mean` | +0.050 | +0.009 |
| `step_mean` → `step_stats` | +0.042 | +0.011 |
| fixed → learned pooling | +0.090 | −0.006 |

**Order transfers. Magnitude does not.** The benchmark's gaps are five to ten
times the downstream ones and the largest one reverses sign. The defensible claim
is that ProcessBench preserves the ranking while exaggerating what the ranking is
worth. That is weaker than "ProcessBench predicts downstream usefulness" and it is
more useful, because it is what the data supports.

---

## 7. The metric has to change before the decision question can be asked

ProcessBench F1 needs human step labels and a decision threshold. On solutions the
policy writes itself, neither exists, and the threshold does not transfer between
datasets. Replacing it with **AUROC**, which is threshold-free and is the quantity
a selection rule actually consumes, separates two things the leaderboard had
conflated:

> The same cells sit at **AUROC 0.83 to 0.90** where their **F1 is 0.395 to
> 0.566**, and two cells sharing AUROC 0.895 differ by 0.071 in F1.

Most of the leaderboard's spread was **threshold placement**, not ranking ability.
The cell carried downstream has in-domain AUROC **0.872**. This is the single
methodological move that licenses everything after it.

---

## 8. The decision experiment

**Data.** Qwen3-8B-Base, 4-shot chain-of-thought, temperature 1.0, wrote **10
solutions each** for the full GSM8K test set (1,319 problems) and MATH-500:
**18,188 gradeable traces**. Each trace carries its per-step verifier scores, its
token-confidence statistics, and its generated token count.

Prompting had to be fixed first. A zero-shot prompt truncated 21.2% of solutions
at the token cap. Raising the cap is the expensive fix, since a cap costs nothing
for a trace that terminates and its whole price falls on traces that never do. The
cap was not the cause: the prompt ended `"Solution:\n"` and showed a base model
nothing about finishing. Four-shot exemplars that terminate, a stop string on the
next-problem delimiter, and an answer-line cut took truncation to **0.19% on GSM8K
and 0.94% on MATH-500**, cut the median solution from 337 tokens to 103 and 132,
and raised the boxed-answer rate from 79.3% to over 99%.

**Design.** Nothing aggregates during generation. Selection rules are applied
afterwards to saved per-trace atoms, replaying each problem's candidates in **32
random draw orders** at each budget N in {1,2,3,4,5,6,8,10}. Every rule therefore
sees **identical draws on identical problems**, so all contrasts are paired by
construction, and a rule invented later costs a CPU rerun rather than a GPU job.

**The three rules**, which must be kept distinct:

| rule | the verifier's authority |
|---|---|
| **majority vote** (self-consistency) | none; answers are counted |
| **tie-break** | chooses only among answers **tied** at the top count; may never overrule a plurality |
| **rerank** (best-of-N) | chooses outright; the count is ignored |

**Statistics.** Because every rule sees the same draws, the per-curve confidence
interval is the wrong uncertainty. Differences are bootstrapped directly, paired,
**resampling question text rather than problem id**, since near-duplicate
questions otherwise inflate the effective sample size. Draw orders are averaged
within a problem before the bootstrap, so order-to-order variance is not mistaken
for problem-to-problem variance.

---

## 9. Result

Lift over self-consistency in accuracy points, paired cluster bootstrap, 95%:

| | GSM8K | MATH-500 |
|---|---|---|
| **N = 2** | **+5.66** [+5.19, +6.15] | **+6.02** [+5.17, +6.86] |
| N = 4 | +1.37 [+1.10, +1.67] | +3.18 [+2.44, +3.93] |
| N = 10 | +0.48 [+0.11, +0.90] | +0.07 [−0.96, +1.12] |

Rerank, the rule the literature uses, goes the other way. Tie-break minus rerank:
+2.36 at N=4 and **+3.71** at N=10 on GSM8K; **+6.46** at N=10 on MATH-500, where
rerank's accuracy **falls** as N grows past 6.

Sweeping the whole rule family makes the ordering mechanical rather than
anecdotal. A tempered vote with weights `w_i ∝ exp(beta · z_i)` interpolates the
endpoints: beta = 0 is the plain count, large beta is rerank. On GSM8K at N=10
accuracy falls monotonically in beta: 0.9247, 0.9257, 0.9143, 0.9083, 0.9007,
0.8939, against 0.8923 for rerank. The literature's reward-weighted vote and
DeepConf's filter-then-vote both sit inside this family and **neither beats the
tie-break anywhere with an interval clear of zero.**

---

## 10. Interpretation

**The verifier's usable information is in its ranking inside a tie, not in its
magnitude across candidates.** Every rule that lets the score move votes the count
had already decided loses accuracy, and the loss grows with the authority given.
Rerank is not a badly chosen rule; it is the far endpoint of a dial whose optimum
sits at the other end.

The corollary explains the N dependence without appeal to anything else. At N = 2
the vote carries almost no information, every draw is a tie, and the verifier
decides everything. As N grows the plurality becomes unique more often, the
verifier is consulted less, and its contribution decays to the residual tie rate.

Lightman et al. (2023) report the same negative finding for weighted voting with a
trained PRM, which makes this a replication with a different verifier class and an
interval on it, rather than an isolated observation.

---

## 11. Where this sits in the field

The positioning that matters, all quoted from the cited papers:

- **Beating majority voting is the bar, not a formality.** In the best-of-8 table
  of *The Lessons of Developing PRMs* (2501.07301), **six of seven published PRMs
  lose to maj@8 on average**. Only Qwen2.5-Math-PRM-7B clears it, by 1.4 points.
- **The strongest published lift at N=10 is +0.2 points.** ReProbe's Table 3 on
  Qwen3-8B, GSM8K: pass@1 95.6, majority voting 97.6, Qwen2.5-Math-PRM-7B 97.8,
  ReProbe 97.8.
- **The small-N regime is empty.** Every comparable paper reports at N >= 8 and
  most at N = 64 or beyond. None reports N = 2 to 4, which is where our whole
  effect lives.
- **The limits literature explains the small effect sizes.** Brown et al.
  (2407.21787): coverage is log-linear in N but selection methods plateau past a
  few hundred samples, so the verifier is the bottleneck. Stroebl et al.
  (2411.17501): an imperfect verifier imposes a hard accuracy ceiling independent
  of budget. Huang et al. (2503.21878): best-of-N at large N provably suffers
  reward hacking, which is the theory behind our rerank degrading in N.

Full map in `docs/tts_related_work_v1.md` and `docs/tts_reading_list_v1.md`. The
structural observation worth making in the review: **the hidden-state probing
literature and the test-time-scaling literature barely cite each other**, and the
papers that bridge them are all 2025 or later.

---

## 12. What the result is not

- **The verifier has never met a real PRM.** Every comparator so far is a free
  signal computed from states or logits the sampler already produced. Whether
  "weighting never pays" is a fact about weighting or about a weak weight is
  unresolved until a 7B process reward model scores the same traces.
- **One scorer drove the downstream arm**, because the online scoring path refuses
  pooled readouts. The representation axis exists offline and not online.
- **The policy is a base model at temperature 1.0.** MATH-500 pass@1 is 45.0 where
  Qwen3-8B-Base is reported at 67 to 69 under greedy decoding, a gap temperature
  alone does not explain and which must be resolved before any absolute number is
  quoted next to someone else's.
- **Base, not Instruct, is deliberate.** Instruction tuning leaves artifacts in the
  activations that a correctness probe can latch onto instead of step content. An
  Instruct arm is allowed only as a matched retrain under an identical protocol,
  behind an activation-artifact audit, so the question becomes measurable rather
  than assumed.
- **The claim is about a regime, not about a better verifier**, and should be
  written that way.

---

## 13. Standing methodological commitments

These are the rules the project runs by, and several of them exist because
breaking them once produced a wrong number that looked right.

1. **Freeze splits to disk before any multi-job extraction**, and have every job
   read the frozen split.
2. **Separate the representation effect from the learner effect** by construction,
   never by regression after the fact.
3. **Report the paired difference, not two overlapping per-curve intervals**, for
   any comparison where the arms see the same draws.
4. **Cluster the bootstrap on question text**, not problem id.
5. **Quote AUROC, not F1, across datasets with different prevalence**, and never
   compare F1 across them at all.
6. **Every claim carries the exact number or quote that supports it.** No
   unsupported "confirmed".
7. **Gate before reading any number.** If an online scoring path reconstructs
   spans even slightly differently from the encoder, every downstream number is
   wrong without looking wrong, so the reproduction check runs first and the job
   aborts on failure.
8. **Price the control.** A rejection loop is compared against the same loop with
   the decision made by a coin at the measured rejection rate, so the machinery
   itself cannot be credited with the effect.
