# onpolicy_tiebreak_v2: a confirmatory study of the small-N tie-break

Status: draft preregistration. Nothing in this document may change after the
first confirmatory trajectory is generated. Written 2026-09-13, before any new
generation.

## 1. The claim under test

REPORT.md §20.14 found the only surviving positive result of the on-policy arm:
a dense hidden-state step verifier, used **only** to break ties in a majority
vote over finished solutions, is worth +0.064 [+0.053, +0.076] final accuracy at
N=2 and reduces the samples needed to reach a given accuracy by 25 to 40 percent
at N=3 to 5. It is worth nothing as a reranker at N=10 (§20.2), nothing as a
stopping rule (§20.13, §20.14), and nothing as an abandonment rule on traces
this short (§20.15).

That result rests on 234 problems, 228 distinct question texts, one policy, one
benchmark, an average over 228 verifier cells, and a comparison against free
rules that does **not** include token confidence, because the saved
trajectories carry no logprobs (§20.11). This study is designed to either
confirm it under conditions the published literature would accept, or kill it.

The mechanism the claim implies is arithmetic and is the thing actually being
tested:

    gain(N) = P(vote is tied at N) x [ P(correct | verifier breaks tie) - P(correct | chance breaks tie) ]

Both factors move with task difficulty, in opposite directions, so the gain is
predicted to be **non-monotonic in difficulty** with a maximum at intermediate
accuracy. An easy benchmark has almost no ties; a benchmark near the floor has
ties with no correct candidate to find. Predicting where the effect disappears
is the strongest available evidence that the mechanism is understood rather
than fitted.

## 2. Literature anchors, and what each one pins down

Read before the design was fixed. Each anchor contributes a different piece of
the protocol, and we adopt that piece verbatim rather than reinventing it.

| anchor | what it pins down | what we take |
|---|---|---|
| ReProbe, arXiv:2511.06209 | the incumbent hidden-state verifier, and the one published result that directly contradicts ours | evaluation datasets and sampling settings |
| DeepConf, arXiv:2508.15260 | the strongest free competitor, with precise confidence definitions | the baseline family, implemented to their equations |
| ESC (Li et al.) and Adaptive-Consistency, arXiv:2305.11860 | the small-budget sampling regime and its standard windows | budget grid and stopping-rule controls |
| CHSC, doi:10.1007/978-981-92-3520-9_27 | the nearest hidden-state prior art | the contrast we must draw |
| "When Self-Consistency Backfires", arXiv:2608.11403 | why token confidence may fail at within-question routing | a preregistered-design template and one prediction |

### 2.1 ReProbe: the head-to-head we already owe the reader

ReProbe trains probes on the frozen internal states of a reasoning LLM to score
step credibility, which is our setting. Its protocol:

- reasoning LLMs: **Qwen3-8B** and Phi-4
- training data: PRM800K, 10.8K problems x 3 chains, about 32K samples
- sampling: nucleus, **max 256 new tokens, top-k 50, top-p 0.95, T=1.0**
- offline aggregation: **minimum step score**, which ranks identically to the
  `worst_step` rule already primary here (§20.6)
- best-of-N: **N=10** for math and QA, N=5 for planning
- beam search: N=5 at **T=1.5**
- in-domain eval: MATH, GSM8K, ProofNet; OOD: StrategyQA, ScienceQA, and three
  planning sets

Our pool already matches their sampling settings except the token budget (we
used 768, they use 256). Their Table 3 reports majority voting at 97.6 on
GSM8K, 86.6 on StrategyQA and 92.5 on ScienceQA, with the best ReProbe variant
at 97.8, 88.6 and 97.1: a win over counting of +0.2, +2.0 and +4.6. §20.4
records that this is the opposite of what we find and offers difficulty as the
candidate reconciliation. **Running their evaluation datasets is therefore not a
generality check, it is the direct test of a live disagreement.**

### 2.2 DeepConf: the baseline whose absence currently invalidates the claim

DeepConf filters reasoning traces by model-internal confidence, needs no
training, and is the method a reviewer or a director will name first. Its
protocol:

- models include **Qwen3-8B** (T=0.6, top-p 0.95, top-k 20, 32k tokens) and
  Qwen3-32B, DeepSeek-8B, GPT-OSS-20B/120B
- datasets: AIME24, AIME25, BRUMO25, HMMT25, GPQA-Diamond, about 1,500 problems
- pool of 4,096 traces per problem, working sets resampled at **K in {32, 64,
  128, 256, 512}**
- all metrics averaged over **64 independent runs** with fresh resampling
- baselines: pass@1 and cons@K, unweighted majority voting
- reported jointly as accuracy **and total generated tokens**

Confidence definitions to be implemented to their equations:

    token confidence        negative mean log-prob of the top-k tokens at a position
    group confidence        mean token confidence over a sliding n-token window
    bottom-10% group conf   mean of the lowest 10% of group confidences
    lowest group conf       min over groups
    tail confidence         mean over the final 2,048 tokens

**Two deviations are forced and must be recorded.** Their windows (1024, 2048)
and tail (2048) exceed our entire trace, which averages 470 generation tokens
over 9.3 steps. We therefore run each confidence statistic at a window scaled
to our traces (32 and 64 tokens) **and** at per-step groups aggregated by the
same minimum-over-steps rule the verifier uses, so that the verifier and its
competitor differ only in the signal and not in the aggregation. Their K grid
starts at 32, which is above the regime where our effect lives; we run N = 1 to
16 and add K = 32 so the two curves overlap and the crossover is measured.

### 2.3 CHSC: the nearest prior art, and how this differs

CHSC augments early-stopping self-consistency with a hidden-state consistency
score contrasting shallow and deep layers, on Qwen2.5-7B-Instruct over GSM8K
(1,319) and MATH Level 4-5 (500), and recovers about 52% (GSM8K) and 29% (MATH)
of the accuracy lost to false consensus. It reports the shallow-layer effect at
Cohen's d = 0.60 at layer 3 on MATH.

Three differences, all of which this study must make explicit rather than
assume:

1. CHSC's signal is **unsupervised** layer-coherence; ours is a **supervised
   step-level probe** trained on correctness labels.
2. CHSC gates a **stopping** rule; §20.13 and §20.14 found stopping rules do not
   pay for us while the **answer** rule does. This is a direct empirical
   contrast, and running a CHSC-style layer-coherence score as an additional
   baseline is cheap because it needs no labels.
3. CHSC targets false consensus, meaning a confident wrong majority. Our effect
   lives at margin 0-1, meaning **no** consensus. §20.14's margin table shows the
   headroom is entirely in the low-margin band (majority 0.219 against oracle
   0.459) and that the high-margin band has nothing to win (0.976 against 0.988).

### 2.4 Why the token-confidence baseline might lose, and why that is a prediction

"When Self-Consistency Backfires" reports that on GPQA-Diamond, samples emitting
an answer that contradicts the plurality still do so with a **median margin of
20.52 nats, 75.7% above 10 nats**, and concludes that log-probability margins
carry limited discriminative value for within-question routing. That is the same
decision our tie-break makes. It is therefore a preregistered directional
prediction of this study, not a hope: **answer-token margin will not reproduce
the verifier's tie-break gain**, while trace-level DeepConf confidence is
genuinely uncertain and may.

That paper also preregistered two hypotheses and tested them on 69 previously
unseen problems. This study follows the same shape.

## 3. Design

### 3.1 Policy, and the base-versus-instruct question

Primary: **Qwen3-8B-Base**, the policy the existing verifiers were trained
against, which keeps every result in §20 comparable and makes this a
confirmation rather than a new experiment.

One prompting change is required for comparability. Our pool ran zero-shot at
T=1.0 with a 768-token budget and reached pass@1 0.375 on PRM800K test, where
the Qwen3 technical report puts Qwen3-8B-Base at **GSM8K 89.84 and MATH 60.80
(4-shot CoT)** and GPQA 44.44 (5-shot CoT). A director comparing our absolute
numbers against the published ones will see a gap that is prompting, not
capability. The confirmatory run therefore uses the published few-shot CoT
prompting so the single-sample accuracy lands in a recognisable place, with the
existing zero-shot setting kept as one arm for continuity with §20.

**Second policy: Qwen3-8B instruct, retrained under the identical protocol.**
Both ReProbe and DeepConf evaluate the instruct model, so matching the protocol
without matching the model leaves the comparison one step short. The base model
was not chosen by accident: instruction tuning leaves structure in the
activations that has nothing to do with the content of a reasoning step, and a
correctness probe fitted on those states can read formatting instead. That
concern is the reason to run the arm carefully, not the reason to skip it,
because retraining under the same protocol turns it from an assumption into a
measurement.

A probe cannot be transferred across backbones, so this is not one probe read
on two models. It is two complete pipelines, each fitted on its own policy,
compared end to end on the same benchmarks.

**The artifact audit, named in advance.** Before the instruct arm's tie-break
number is believed, four checks run at the read layer, each with a base-model
reference value from the same pipeline:

1. Outlier-dimension mass: the share of hidden-state norm carried by the top few
   residual dimensions, base against instruct. Instruct models concentrate more,
   and a probe that loads on those dimensions is a flag.
2. Attention-sink and template mass: how much of the step representation is
   drawn from token 0 and from chat-template tokens rather than the step itself.
   `attention_routing_v0` already found Qwen's token-0 sink overlapping
   question mass, and that measurement is reused here.
3. Probe weight localisation: whether the fitted direction puts its mass on
   template positions. If the probe scores a step as well with the step text
   removed, it is reading the template.
4. The confound residualisation already standard in this project (§15.3,
   §20.11): length, step position, and trace position regressed out before the
   margin is reported.

**Decoupling the two changes.** Switching to instruct changes both the
representation the probe reads and the distribution of solutions being verified.
The default instruct arm therefore runs in **non-thinking mode with the same
few-shot prompting**, so the generation change is as small as it can be made and
any difference is attributable to the backbone. Thinking mode at DeepConf's
published settings (T=0.6, top-p 0.95, top-k 20) is a separate arm, run only if
the confirmatory endpoint in Phase 3 has already landed.

### 3.2 Datasets: the four-set core

The first run uses four datasets. ReProbe's non-math out-of-distribution sets
(StrategyQA, ScienceQA) are deferred: they make the head-to-head literal but
need an answer grader this project does not have, and they are not worth
blocking the first run on.

| dataset | n | source anchor | Qwen3-8B-Base reference | predicted gain |
|---|---|---|---|---|
| GSM8K test | 1,319 | ReProbe, ESC, CHSC | 89.84 (4-shot CoT) | near zero, too few ties |
| MATH500 | 500 | ESC, Adaptive-Consistency | 60.80 on MATH (4-shot CoT) | moderate |
| PRM800K test (incumbent) | 2,000 | this project, §20 | 0.375 pass@1 zero-shot | large, this is the replication |
| GPQA-Diamond | 198 | DeepConf, backfire paper | 44.44 (5-shot CoT) | moderate, and non-math |

**What the four-set core can and cannot test.** It spans single-sample accuracy
from about 0.90 down to about 0.375, which covers the descending arm of the
predicted inverted U and includes the point where the effect should vanish for
lack of ties. It does **not** reach the floor, so the other half of the
prediction, that ties stop paying when no candidate is correct, is not tested by
the dataset axis alone.

That half is recovered for free by **stratifying within the sets we already
generate**. MATH500 and PRM800K test both carry subject and difficulty level,
so the level-5 stratum (134 problems in MATH500) and the hardest PRM800K
stratum give low-accuracy points at no extra generation cost. Power per stratum
is low, so this is registered as a secondary analysis with its own Holm family,
and a floor-level dataset (AIME24/25, per DeepConf) is the first addition if the
stratified version is suggestive but underpowered.

GPQA-Diamond earns its place twice: it is DeepConf's dataset, and being
four-way multiple choice it has a tie structure unlike the open-numeric sets,
which is a stress test of the mechanism rather than a repetition of it.

### 3.3 Samples and budget grid

16 samples per problem on all four sets, so the frontier runs N = 1 to 16 by
subsampling orders, plus 32 samples on PRM800K test and GPQA-Diamond to overlap
DeepConf's smallest K. Following DeepConf, every curve is averaged over **64
independent resampling runs**, replacing the 100 sampling orders used in
§20.14.

### 3.4 The verifier, named in advance

Primary: **the single `last_token x linear` frozen cell**, the cheapest head in
the grid, which §20.14 already reports standalone at +0.0439 at N=2 with
equivalent N of 2.8. It is named primary because it is the least selected-upon
and the most deployable, not because it is the best.

Secondary: the 57-cell log-odds ensemble used for §20.14's headline, and the
retrained on-policy cell. Both are reported, neither is the primary endpoint.

Aggregation is **minimum over steps of P(correct)**, fixed, matching ReProbe's
`Q_offline` and §20.6.

### 3.5 Baselines, in three tiers

Tier 0, free orderings already tested in §20.12: shortest, longest, fewest
steps, most steps, first sampled, random.

Tier 1, token confidence, the tier that currently does not exist:
DeepConf bottom-10% group, lowest group, tail, mean token confidence, all at
both window scales; per-step minimum token confidence under the verifier's own
aggregation; answer-token log-probability margin; predictive entropy.

Tier 2, unsupervised internal states: a CHSC-style shallow-versus-deep layer
coherence score, which needs no labels and isolates what the supervision buys.

Every tier is scored on the identical draws, so all contrasts are paired.

## 4. Preregistered endpoints and analysis

**Primary endpoint.** Difference in final-answer accuracy between
`majority + verifier tie-break` and `majority + random tie-break` at **N=4** on
the PRM800K-test confirmatory pool, using the primary verifier cell. One number,
one test. N=4 is chosen in advance as the middle of the deployable regime, not
N=2 where the effect is largest.

**Co-primary.** The same difference against the **best Tier-1 confidence rule**,
selected on the exploratory pool and frozen before the confirmatory pool is
touched.

**Secondary, Holm-corrected across the family.** N in {2, 3, 5, 8, 16, 32};
equivalent-N with its own interval; the per-dataset gain; the margin-band
decomposition; the tie-rate decomposition of the gain into its two factors.

**Statistics.** Cluster bootstrap by question text with 10,000 resamples, since
the 300 ids in §20 were only 284 questions. The unit of analysis is the
problem after averaging over resampling runs, so run-to-run variance is never
mistaken for problem variance. Paired McNemar within matched draws for the
binary contrasts. Absolute accuracy and the oracle@N ceiling are reported beside
every delta, never a delta alone.

**Power, and which cells can carry a claim.** §20.14 measured +0.064 with a CI
width of 0.023 on 234 problems, once per-problem values were averaged over
sampling orders. Scaled to the core sets, PRM800K test (2,000) and GSM8K
(1,319) can each carry a per-dataset claim on their own; MATH500 (500) is
marginal for an effect of +0.03; **GPQA-Diamond (198) cannot**, and is
registered as a mechanism stress test and a pooled contributor, never as a
standalone per-dataset result. The exploratory/confirmatory split halves each of
these, which is why the primary endpoint is defined on the PRM800K pool alone
and the per-dataset numbers are secondary.

**Exploratory / confirmatory split.** Problems are partitioned by question text
before anything runs. All rule selection, window tuning and cell choice happen
on the exploratory half. The confirmatory half is scored once.

**Cost accounting.** Every accuracy is reported against measured generated
tokens, per DeepConf, and against measured verifier wall-clock. The "nearly
free" claim of §20.14 (1.5 to 1.9 candidates scored per problem rather than N)
must be confirmed by instrumenting the generation-template path to show the
hidden states are the ones the sampler already computed, not a second forward
pass.

## 5. Phases and gates

**Phase 0.** Freeze this document. Materialise every split to JSON, including
the exploratory/confirmatory partition by question text and the difficulty
strata.

**Phase 1, the gate that decides the study.** Add logprob capture to generation
(`generate_batch` in `scripts/causal_graph/cg_stage2_fg.py:98` currently calls
`model.generate` with no `return_dict_in_generate`/`output_scores`), regenerate
the incumbent PRM800K-test pool, implement Tier 1, and run the head-to-head on
the exploratory half. If the best confidence rule matches the verifier within
the interval, the honest result is that hidden states add nothing over free
logprobs, and the study reports that and stops.

**Phase 2.** The four-set core at 16 samples, Qwen3-8B-Base, exploratory half
only. Checks the descending arm of the inverted U, runs the difficulty
stratification, and fixes every free parameter: window sizes, the Tier-1 rule
that becomes the co-primary competitor, and the aggregation variants.

**Phase 3.** The confirmatory run: one scoring pass on the held-out half,
primary and co-primary endpoints, no further tuning.

**Phase 4, the instruct arm.** Retrain the verifier on Qwen3-8B instruct under
the identical protocol, run the §3.1 artifact audit **before** reading the
tie-break number, then the same four-set evaluation in non-thinking mode. The
audit is a gate, not a footnote: if the probe's margin does not survive
residualisation or the probe scores steps as well with the step text removed,
the instruct arm reports a formatting artifact and the base-model result stands
alone.

**Phase 5.** Report section and figures.

**Deferred, in priority order if the first run justifies more:** AIME24/25 as a
floor-level dataset; ReProbe's StrategyQA and ScienceQA, which need an answer
grader; Qwen3-8B instruct in thinking mode at DeepConf's published settings.

## 6. Outcomes, all of which are reportable

- **Confirm.** The verifier beats both chance and the best token-confidence rule
  at N=4 on held-out problems, and the inverted-U holds. The claim becomes: a
  supervised readout of the generator's own states is worth 25 to 40 percent of
  the sampling budget in the low-budget regime, and published best-of-N
  comparisons at large N measure the regime where it is weakest.
- **Confidence wins.** Hidden states buy nothing over free logprobs. Reported as
  such, and it is a useful negative given how much of this literature assumes
  otherwise.
- **No replication.** The §20.14 interval was optimistic and the effect does not
  survive held-out problems. Reported as such.
- **Mechanism wrong.** The gain does not track tie rate and headroom. The effect
  may be real but is not understood, and no scaling claim is made.

## 7. Decisions taken (2026-09-13)

1. **Breadth: the four-set core.** GSM8K, MATH500, PRM800K test,
   GPQA-Diamond. The floor of the difficulty ladder is recovered by
   stratification within these sets rather than by a new dataset.
2. **The instruct policy is in, as Phase 4, with an audit gate.** The base model
   was chosen because instruction tuning leaves artifacts in the activations;
   retraining under the identical protocol makes that testable instead of
   assumed, and the §3.1 artifact audit runs before the arm's number is read.
3. **ReProbe's non-math OOD sets are out of the first run.** Worth having, not
   worth blocking on, and they need an answer grader that does not exist yet.

**Blocker found 2026-09-13, before Phase 0 froze.** Dataset availability was
checked on the TamIA login node rather than assumed. GSM8K (1,319 test rows)
and MATH500 (500 rows, carrying `level` and `subject`, so the difficulty
stratification of §3.2 is available) both download and are now cached.
**GPQA-Diamond is gated**: `Idavidrein/gpqa` returns `DatasetNotFoundError:
is a gated dataset on the Hub. You must be authenticated to access it.`
Unblocking it needs the account holder to accept the terms on the Hub and put
an `HF_TOKEN` on the cluster; neither can be done from this side. Until then
Phase 2 runs as a **three-set core** (GSM8K, MATH500, PRM800K test), which
still spans 0.90 to 0.375 single-sample accuracy and still carries the
replication. GPQA re-enters as the non-math stress test the moment the token
exists.

Remaining to close before Phase 0 is frozen: the exact few-shot prompt used for
the published-prompting arm, and the read layer for the instruct backbone, which
§15 fixed at L20/L28 for the base model and which the artifact audit may move.


---

## 8. Phase 1 result (2026-09-13): the gate fired, and the reason was not confidence

Job 462216 recovered logprobs for 2,995 trajectories in 42 seconds of GPU time,
2:03 wall. Step-span coverage 2,860/2,995 = 0.955, above the 0.90 gate. The
pipeline reproduces §20.12's verifier-versus-random number exactly, +0.0488
[+0.0076, +0.0940] on 57 clean ties, which is what makes the rest readable.

**Against DeepConf proper, the verifier neither wins nor loses.** On the 57
ties, `bottom10_group_w32` and `verifier_worst_step` both score 0.1930 against
a random baseline of 0.1279. The paired difference is +0.0000 [-0.1071,
+0.1071]. Every other DeepConf statistic lands between 0.1754 and 0.1930, and
the windowed rules at the paper's own 1024/2048 collapse onto the trace mean
exactly as §2.2 predicted they would.

**The rule that beat the verifier was not a confidence rule.**
`answer_token_margin` scored 0.2807, and in the full 228-cell race beat the
verifier by -0.1040 [-0.1901, -0.0152]. Decomposing it
(`scripts/analysis/onpolicy_format_confound.py`) shows the margin is not what
is working:

```
random                              0.1279
verifier_worst_step                 0.1930   +0.0651 [-0.0206, +0.1573]
deepconf_bottom10_w32               0.1930   +0.0651 [-0.0193, +0.1557]
has_boxed  (no logprobs at all)     0.2105   +0.0826 [+0.0062, +0.1646]
answer_margin                       0.2807   +0.1528 [+0.0721, +0.2411]
has_boxed then verifier             0.2632   +0.1352 [+0.0473, +0.2310]
```

48 of the 57 tied blocs contain at least one candidate with no `\boxed{}`
answer, so the margin is nan there and always loses. The indicator alone,
which costs no logprobs, no forward pass and no model, already carries
+0.0826. On the 9 blocs where every candidate boxed an answer, the margin is
worth +0.0370 [-0.2037, +0.2778], which is nothing on that n.

**And `has_boxed` is a truncation detector.** The pool was generated with a
768-token cap:

```
share of traces at the cap                    21.2%
P(no boxed answer | truncated)                77.8%
P(truncated | no boxed answer)                81.7%
accuracy | truncated                          0.055
accuracy | not truncated                      0.458
mean generated tokens, boxed vs unboxed       388 vs 680
```

A trace that runs out of budget never reaches its box, the grader's fallbacks
still parse an answer so it still enters the vote, and it is correct 5.5% of
the time. Any rule that deprioritises those picks up a large free gain that has
nothing to do with reasoning.

**This reaches back into §20.** The verifier's own score is higher, meaning more
suspicious, on unboxed traces: 0.8990 against 0.8193, point-biserial
r = -0.201. So some unknown part of §20.12's +0.0488 and §20.14's +0.064 is the
verifier detecting truncation rather than detecting wrong reasoning. That is a
confound §20.11's audit did not test and §20.12's free rules could not have
caught, because it tested orderings by length and step count and this is
neither.

### What this changes

1. §20.12's "no cheap ordering reproduces it" is **false as stated**. It is true
   of length orderings and false of `has_boxed`, which is cheaper still.
2. Phase 2 cannot run on a 768-token budget. The budget has to be raised until
   the truncation share is near zero, or truncation has to be handled explicitly,
   or every tie-break number is partly a budget measurement. ReProbe's own 256
   tokens would be worse, not better.
3. The Phase 1 gate as written says a confidence rule matching the verifier means
   the study reports that and stops. DeepConf matched it at +0.0000 with an
   interval of ±0.107 on 57 problems, which is not a demonstrated equivalence,
   it is no power. The gate fired on a number that cannot distinguish the
   hypotheses, and that is a flaw in how the gate was written, recorded here
   rather than quietly reinterpreted.
4. The decision now needs a human: the scientifically honest next step is a
   re-run at a token budget that removes the confound, which is new compute, not
   a re-reading of what is on disk.
