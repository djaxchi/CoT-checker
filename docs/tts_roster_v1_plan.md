# tts_roster_v1: what should you do with a token budget?

Preregistration. Frozen before generation. Written 2026-09-15.

## 1. The question

Given a fixed budget of generated tokens, which is worth more: drawing more
samples and voting, or spending the same tokens letting a checker rewrite steps
as they are written? And does the answer depend on who the checker is?

§20.17 answered half of it with one checker on one pool: step-level rejection
beat plain sampling by +0.098 [+0.030, +0.162] at 2.30x the tokens, with the
blind control null at +0.024 (p=0.49). What it never asked is which scorer makes
rejection work, and it never measured the matched-budget comparison it quotes,
which came from interpolating an offline curve. This run asks both, on two
datasets, with a roster.

## 2. What the run must produce: atoms, not aggregates

The deliverable is not a table. It is a dataset an explorer can slice: pick a
metric, pick one hyperparameter for the x-axis, fix the rest. That only works if
nothing is aggregated at write time, so the rule for this run is that **every
row is an atom and every derived number is computed downstream**.

Three tidy files, plus a manifest.

### `traces.jsonl`, one row per generated solution

```
run_id  dataset  problem_id  question_hash  split(expl|conf)
protocol(plain|reject|reject_blind)  scorer_id  q  max_retries  seed  sample_idx
solution  pred  correct  gradeable
n_gen_tokens  n_steps  hit_token_cap  has_boxed  wall_ms
```

### `steps.jsonl`, one row per step of every trace

```
trace_uid  step_idx  n_tokens  token_span
scores { scorer_id -> value }        <- EVERY scorer, not just the driving one
attempt_idx  n_attempts  was_rejected  draft_scores[]  kept_attempt
rejected_draft_texts[]
```

The line that matters is `scores` carrying **every** scorer on **every** step,
including in arms driven by a different one. That is what lets the explorer
answer counterfactuals this plan did not anticipate: "which steps would the PRM
have rejected here", "do the scorers condemn the same steps", "what would q=0.6
have done", all without a rerun. It is also what makes Figure 4 possible.

One limitation, recorded rather than hidden. The probe scorers are cheap because
they read hidden states the sampler already computed, so scoring a step with all
of them costs almost nothing extra. The PRM is a separate 7B forward pass, so in
the online arms it runs **once per completed trace** rather than once per draft.
PRM scores therefore exist for kept steps and not for rejected drafts.

### `decisions.jsonl` is deliberately absent

Selection outcomes (which sample a rule picks at N=k, under aggregation a, with
tie-break t) are a pure function of `traces.jsonl` and `steps.jsonl`. Computing
them downstream means the explorer can add a rule later. Writing them now would
freeze the rule set.

### `manifest.json`

git commit, model snapshot hash, dataset checksums, every seed, every calibrated
tau with the distribution it came from, token caps, the exact prompt strings,
and per-arm wall-clock. Enough to reproduce or to discover the run is not
comparable to another.

## 3. Data

| set | problems | N | token cap | role |
|---|---|---|---|---|
| GSM8K test | 1,319 (full) | 10 | 2,048 | comparability anchor, near-saturated |
| MATH500 | 500 (full) | 10 | 2,048 | headroom set |
| online subset | 300 (200 GSM8K + 100 MATH500) | n/a | as above | the rejection arms |

Qwen3-8B-Base throughout, **4-shot CoT prompting with an explicit stop string on
the next-problem delimiter**, so single-sample accuracy lands near the published
89.84 GSM8K / 60.80 MATH. Splits by question text, materialised to JSON before
any GPU job.

**On the token cap.** §20.16 found 21.2% of the previous pool ran to its 768
cap, and those traces are correct 5.5% of the time against 45.8% for the rest,
which handed every cheap baseline a budget signal disguised as a correctness
signal. Raising the cap is the obvious fix and it is the expensive one: a cap
costs nothing for a trace that terminates, so the whole price of a higher cap
falls on traces that never do. Worse, the cap was probably not the cause. The
old prompt was zero-shot and ended "Solution:\n", showing a base model nothing
about what finishing looks like, so it had no reason to stop.

The fix is therefore three things, cheapest first: few-shot exemplars that end,
a stop string on the delimiter that begins the next problem, and only then a
generous 2,048 cap as insurance rather than as the mechanism. Truncation is
measured by a smoke run **before** the real jobs, and the gate is 2% rather than
5%, because with a stop string anything above that means the prompt is still
wrong and no amount of cap will fix it.

`hit_token_cap` is recorded per trace and means what it should: the budget ran
out *without* the model signalling an end. A trace cut at the delimiter did end,
and is not counted as truncated.

## 4. The roster

| id | what | cost per step |
|---|---|---|
| `probe_last_linear` | our verifier, `last_token x linear`, cheapest cell | reuses sampler states |
| `probe_steptok_tf` | our verifier, `step_tokens x transformer`, top of the leaderboard | reuses sampler states |
| `prm_qwen25_math_7b` | Qwen2.5-Math-PRM-7B, off the shelf | separate 7B pass |
| `conf_bottom10_w32` | DeepConf bottom-10% group confidence | free, from logprobs |

Offline only, additionally: majority vote, length, `has_boxed`, first-sampled,
random, and the oracle ceiling.

Two probes rather than one because they differ in representation, not just in
seed, so "the verifier" is not one architecture's quirk. The PRM because it is
what a practitioner would actually reach for and because ReProbe's own baseline
table is PRMs. Confidence because §20.16 could not separate it from our probe.

## 5. The threshold protocol

§20.17 used q ∈ {0.70, 0.85}, picked by hand. q=0.70 won and q=0.85's interval
touched zero, which says the operating point is at or below 0.70 and that nobody
looked. Here the threshold is an axis, not a choice.

- **Swept** over q ∈ {0.50, 0.65, 0.80}, so the run traces a cost/accuracy
  frontier and the threshold is never a tuned parameter to defend.
- **Calibrated per scorer on the target dataset**, not on PRM800K. Each scorer's
  score distribution has its own shape, and GSM8K's differs from PRM800K's, so
  each arm's tau is the q-quantile of that scorer's own scores on a held-out
  calibration slice of the same dataset. This also makes the online job
  self-contained, which is what lets it run beside the offline job rather than
  after it.
- **Matched rejection rate across scorers is the fair comparison.** A scorer that
  rejects 45% where another rejects 30% has bought more compute, not shown more
  skill. Quantile calibration matches nominal rates by construction; the
  *realised* rate is recorded per arm and any residual mismatch is reported.

## 6. Arms

**Offline job.** Plain sampling at N=10 on both sets, one pool. Every selection
rule is then computed downstream at N = 1, 2, 3, 5, 8, 10: majority vote, each
scorer as reranker, each scorer as tie-breaker, score-weighted vote, and each of
the free rules. Aggregations worst / mean / last for every scorer.

**Online job.** On the 300-problem subset:

```
plain                                        1 arm
reject  x 4 scorers x 3 quantiles           12 arms
reject_blind                                 1 arm
```

`max_retries = 2` for the sweep, with one extra arm at `max_retries = 4` on the
best scorer, because §20.17's worked example exhausted its retries at 0.54, 0.46,
0.60 and kept the least condemned, so the cap binds and nobody has tested
raising it.

**Fixes carried in from §20.17.** The blind arm's units bug is corrected: the
launcher passed a per-step extra-draw rate as a per-draw probability, giving
p + p² = 1.05 extra draws against reject's 0.54. It now solves p + p² = rate.
Rejected draft text is saved, which the previous run did not do.

**Composition (rejection then vote) is out of v1.** It doubles the online cost
and it only becomes interesting if rejection wins here. It is the first follow-up.

## 7. Job split and hardware

Two independent jobs, one node each, four H100s each, sharded one process per GPU
by `CUDA_VISIBLE_DEVICES`, striding the problem list so shards finish together.

| job | contents | est. node-hours |
|---|---|---|
| A, offline | generation both sets N=10, hidden-state encode, all probe cells scored, PRM scored, confidence recovered | ~5 |
| B, online | per-scorer tau calibration, then 14 rejection arms on 300 problems | ~7 |

They are independent because job B calibrates its own thresholds. Run
concurrently that is **about 8 GPUs for 7 hours**, against ~12 hours serialised.
The estimate is anchored on §20.17's measured 4:31:38 for 296 problems across
four arms, which is 13.8 seconds per problem-arm on a four-GPU node, derated for
GSM8K's shorter solutions.

## 8. Gates, checked before any number is read

1. **Live-vs-offline score agreement**, the §20.17 gate: median |delta| below
   0.002 and max below 0.02, with the bf16 batch-shape floor measured first.
2. **Truncation share below 2%** on both datasets, measured by the smoke run
   before the full jobs are submitted. Above that, the token cap is confounding
   the free baselines again and the prompt, not the cap, is what needs fixing.
3. **`reject_blind` null**, within [-0.04, +0.04] of plain. If the blind loop
   moves accuracy, the retry loop is not distribution-preserving and the run is
   void rather than interesting.
4. **Single-sample accuracy within 0.05 of published**, 89.84 GSM8K and 60.80
   MATH at 4-shot CoT, or the prompting is wrong and no absolute number is
   comparable to anything.
5. **Step-span coverage above 0.90**, or per-step scores are misaligned.

## 9. Outputs

**Figure 1.** Accuracy against measured generated tokens, panels for GSM8K and
MATH500. Curves: majority vote, each scorer as reranker, the online rejection
frontier, with pass@1 and oracle@10 as horizontal bounds. The crossover between
voting and rejecting is the headline.

**Figure 2.** Gain over majority vote against N, one curve per method, tie rate
overlaid. Where verifier value decays, and whether rejection's small-N advantage
is what the tie-break decay predicts.

**Figure 3.** Accuracy against realised rejection rate, one curve per scorer,
token cost as point size. Replaces "we picked q=0.70".

**Figure 4.** Do the scorers condemn the same steps? Pairwise overlap and a
complementarity test. §20.16 found our probe and answer-token confidence at
phi = +0.715 on the tie-break, so redundancy is the expectation to beat.

**Table 1.** Every method at a matched 2.0x-plain token budget: accuracy, CI,
McNemar against plain and against majority vote.
**Table 2.** Cost accounting: generation tokens, scoring forward passes,
wall-clock. Where a probe reading existing states should beat a 7B PRM.
**Table 3.** The §20.17 follow-ups: length control, retry sweep, corrected blind
null.

**The explorer.** An HTML page over the three JSONL files: pick a metric, pick
the x-axis hyperparameter, fix the rest, and it redraws. Hyperparameters
available: dataset, scorer, protocol, N, q, max_retries, aggregation, seed,
truncation filter on or off.

## 9b. The smoke that gates the launch

60 problems, 30 per dataset, 4 samples each, 4-shot with the stop string at a
2,048 cap. Five minutes of GPU. It reports per dataset: truncation share,
single-sample accuracy against the published number, boxed-answer rate, and the
median / p95 / max token spend. The last of those is what turns the cost of the
higher cap from a guess into a measurement before 12 node-hours are committed.

`slurm/tts_smoke_tamia.sh`. The smoke set is drawn from the far end of the
question-hash order so it is disjoint from the online subset.

## 10. What a null looks like

If rejection does not beat matched-budget voting on either set, that is the
answer and it is worth reporting, because §20.17's +0.055 was interpolated and
this run is the first to measure it. If it wins on MATH500 and not GSM8K, the
finding is that the mechanism pays in proportion to how often the policy is
wrong, which is the same shape §20.14 found for the tie-break and would be the
second independent sighting of it.


## 11. Scope correction before launch (2026-09-15)

**The online arm runs one scorer, not two.** `online_bon.Checker` rebuilds a
per-step sequence head and raises on a pooled readout: "online decoding needs a
per-step sequence head; cell rep is 'last_token'. Pooled readouts score a step
too, but this script has only been verified for step_tokens." So
`last_token x linear`, the cheapest cell and the one §20.12 and §20.14 used for
the tie-break, cannot drive the rejection loop until Checker grows pooled
support. Caught by reading the class before the job ran rather than by the job
failing at stage 0.

It stays in the **offline** arm, where scoring is a batch pass and the same
restriction does not apply, so the representation comparison survives for every
offline selection rule and is missing only for online rejection.

**The PRM is deferred out of v1.** Qwen2.5-Math-PRM-7B is downloading, but it
needs a scoring adapter that does not exist, and its cost is the one large
enough to move the token axis. Wiring it half-tested overnight would spend a
node for numbers nobody should quote.

Both are the first two follow-ups, in that order.
