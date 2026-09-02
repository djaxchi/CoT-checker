# onpolicy_v1: does the representation leaderboard transfer to on-policy use?

Status (2026-09-02): **T2 ran and is the headline.** Job 434763, 19 verifiers on
2,873 self-generated solutions to 300 problems, no step labels used anywhere.
Baselines: random pick 0.375, self-consistency 0.560, any-of-10 0.700.

```
best verifier at best-of-10        0.503   (boundary_stats x mlp:h1024)
verifiers beating self-consistency  0 / 19
best score-weighted vote          0.561   against 0.560 unweighted
pooled trajectory AUROC     up to 0.833   (length alone: 0.561)
```

The signal is real and not a length artifact: keeping the shortest solution
scores 0.350, worse than random. Benchmark rank predicts downstream rank
(Spearman +0.66 to +0.84 for best-of-N, +0.76 to +0.83 for the AUROC metrics)
but the gaps compress by an order of magnitude: `last_token -> step_mean` is
+0.050 F1_PB and +0.009 best-of-10, and `fixed -> learned pooling` is +0.090
F1_PB and -0.006 best-of-10.

Two things must be settled before this is written as a result, and TamIA went
into maintenance before either could run.

1. **Did the encoder drop steps?** It skips any step whose prompt exceeds 2,048
   tokens, and on-policy solutions run long (one had 27 steps). Truncated score
   lists would break a max-over-steps aggregation exactly where it matters.
2. **Pooled versus within-problem AUROC.** A pooled 0.833 next to a failing
   best-of-N is the signature of a score that ranks *problems* by difficulty
   rather than *solutions* by correctness: pooled AUROC compares solutions
   across different problems, best-of-N compares solutions to the same one. If
   the within-problem number sits near 0.5, the result is explained and the
   explanation is the finding.

Power, stated honestly: at 300 problems the per-problem binomial SE is 0.029, so
the 0.057 gap to self-consistency is about 2 SE and the whole spread across
nineteen representations is 1.9 SE. The AUROC numbers are tight; the best-of-N
numbers are not. Job 434822 (2,000 problems) was submitted to fix that and its
fate is unknown until the cluster returns.

A design difference that keeps this compatible with ReProbe rather than
contradicting it: ReProbe trains its probe **on-policy**, labelling the target
model's own steps with R1. These nineteen verifiers were trained on PRM800K and
have never seen a Qwen-generated step, so this is a transfer test by
construction, and the result is an argument for running the on-policy training
arm rather than evidence against ReProbe.

Status (2026-08-31): Stage 0 done. **Stage 1 ran and passed every gate** (job
433635, 21 minutes). Stage 2's first judge bake-off ran (job 433640) and came
back too weak to use, which forked the labelling question; see "Stage 2: what
happened" below. Two follow-ups are queued: the same judges with reasoning
enabled (433685) and a rollout labeller gated on human fork annotations (433686).
Design frozen for Stage 0 implementation (2026-08-31). Stages 1-3 are the
minimal arm and each carries a kill gate; Stages 4-6 are the full-scale arm and are
not started until Stage 3 reports. Context and the settled parts of the setting are
in `experiments/onpolicy_v1/HANDOFF.md`, which this plan does not repeat.

Two questions, taken from the handoff:

**T1 (rank transfer).** The 19-cell leaderboard in
`experiments/unified_harness_7b/leaderboard.md` was measured entirely off-policy.
Does its *ordering* survive when the verifier reads states of steps that
Qwen3-8B-Base itself wrote? The claim is about rank; absolute F1 is not comparable
across the two arms and is never compared.

**T2 (predictive validity).** Does step-classification rank predict downstream
test-time-scaling gain, measured on the same 19 verifiers? This is the more
valuable question and it is the one the field has not asked of process verifiers.

---

## The design in one frame

The experiment is a 2x2 over *who wrote the text* and *what the verifier was
trained on*, plus a third axis that has to be split out because the codebase
currently conflates it.

|  | eval text off-policy (PRM800K/ProcessBench) | eval text on-policy (Qwen3-8B-Base) |
|---|---|---|
| **trained off-policy** | the existing leaderboard (done) | **Stage 4: T1-transfer** |
| **trained on-policy** | Stage 6 (reverse transfer) | Stage 6 (matched-size on-policy) |

T1-transfer is the cell that answers the question with no retraining at all: the
same 19 saved models, the same weights, pointed at a different text distribution.
It is by a wide margin the cheapest cell and it is also the deployment-realistic
one, since a practitioner trains a verifier on PRM800K and applies it to their own
model's output. Stage 6 exists to separate a policy effect from a training-data
effect, and only runs if Stage 4 shows the ranking moving.

### The third axis: which context the states are read under

`scripts/generate_onpolicy_steps.py` generates with

```
Problem:\n{problem}\n\nSolve the problem step by step. ... \n\nSolution:\n
```

while `scripts/encode_prm800k_token_store.py` encodes every step under
`build_prompt_prefix`:

```
Problem:\n{problem}\n\nPrevious reasoning:\n{prefix}\n\nCurrent step:\n
```

If we encode on-policy steps under the verifier template, the activations are not
the states the model held while writing the step; they are a re-read of its own
text under a different context. That is a defensible setting, and it is the one
that keeps the encoding protocol pinned to the off-policy arm, but it is not
"on-policy states" and must not be described as such.

Decision: both, and report the difference.

- **`verifier` prompt style is primary for T1.** It changes exactly one thing
  against the off-policy arm, the text distribution. That is the controlled
  comparison the rank claim needs.
- **`generation` prompt style is the second perturbation.** The prefix is
  reconstructed as `build_prompt(problem) + "\n\n".join(steps[:k]) + "\n\n"`,
  which is byte-identical to the context the model actually had at that point in
  its own sample, so the step's token states are the generative ones (teacher
  forcing over its own text reproduces them exactly).
- Stage 3 encodes one slice under both and reports the AUROC delta and the rank
  delta between them. If the two agree, the primary arm carries the paper and the
  other is a robustness line. If they disagree, that disagreement is itself a
  result, and it is a result nobody has reported for process verifiers.

Implementation note: rather than teaching the encoder a second template, the
adapter writes an explicit `prompt_prefix` string per item and the encoder gets a
`--prefix_field` flag that uses it verbatim. One flag, no template logic
duplicated, and the manifest records which style produced the store.

---

## Labels: outcome labels are not first-error labels

The generator currently inherits the trajectory's outcome onto every one of its
steps (`label = 0 if traj_correct else 1`). That is the correct label for the S3
distribution control it was written for, and it is the wrong label here. F1_PB
scores *first-error localisation*: the early steps of a wrong trajectory are
usually fine, and a metric that calls them all wrong measures nothing. The
on-policy eval set needs, per trajectory, either -1 (no error) or the index of
the first bad step, which is what `evaluate_processbench` consumes via
`{id, step_idx, label, n_steps}`.

No internet on compute nodes, so the judge is local. Rather than choosing by
argument, Stage 2 runs a bake-off and reports the number:

1. **Self-supervised**: Qwen3-8B-Base judges its own steps. Free, on-node, and a
   published setting (ReProbe) rather than an improvisation.
2. **Qwen2.5-32B-Instruct** from `$STORE/hf_cache`, if a snapshot is present.
   On-node, stronger, and not the generator, so its errors are not correlated
   with the generator's.

Both are given the question, the steps, and **the ground-truth answer**, following
Zheng et al. and ReProbe. Certification, before any label is used:

- **Agreement against human labels.** Run both judges over a held-out slice of
  PRM800K traces that carry human first-error labels (traces excluded from probe
  training). Report trace-level exact agreement and error-vs-correct confusion.
  ReProbe reports 95% on PRM800K and ~90% on a 1,000-step manual set. That is the
  number to state alongside ours, not a bar we assume we clear.
- **Judge-judge agreement** on the same slice. S3 found Opus and Haiku agreeing
  only 44% on failure-mode labels, so two judges agreeing is evidence and two
  judges disagreeing is a caveat that has to be carried into the results.
- **False-alarm rate on the on-policy distribution.** For on-policy trajectories
  that reached the *correct* final answer, the judge should return -1. The rate at
  which it does not is a free calibration measured on exactly the distribution we
  care about, needing no human labels at all. It is also the sharper test, because
  the PRM800K certification is itself off-policy.

Gate: primary judge must reach >=85% trace-level agreement with human labels and
<=15% false-alarm on correct on-policy trajectories. Below that, the labels are
reported as a noise ceiling on every downstream number, and T1's rank claim is
restricted to the metrics that do not need first-error labels (step AUROC under
outcome labels, and every T2 simulation, which needs only within-problem ordering).

A useful property of T2 worth stating early: **the downstream simulations need no
judge at all.** Best-of-N, weighted voting, early termination and pruning are
scored against the trajectory's own final-answer correctness, which
`src/eval/math_grade.py` already gives us for free. If the judge arm degrades, T2
survives intact.

---

## Stage plan

Each stage is one job or less. Gates are stated before the stage runs.

### Stage 0: plumbing (local, no GPU) - DONE

Shipped, 20 new tests in `tests/onpolicy/`, full suite 707 passed. One change of
plan against the list below: the on-policy split is encoded by the *ProcessBench*
encoder rather than the PRM800K one, because that encoder already takes a traces
file and already writes the `id` / `step_idx` / `label` / `n_steps` meta rows
`evaluate_processbench` groups on. So the adapter emits ProcessBench-shaped
traces, the on-policy split becomes another `--pb_subsets` entry, and no
evaluation code changes at all.

  - `src/onpolicy/prompts.py` holds both templates, with a test pinning them
    byte-identical to the sampler's prompt and to `build_prompt_prefix`.
  - `scripts/generate_onpolicy_steps.py`: `--shard_idx/--num_shards`, explicit
    `--top_k`, prompt delegated.
  - `scripts/onpolicy/build_pb_traces.py`: trajectories + judge labels ->
    ProcessBench traces + the T2 outcomes sidecar + conflict counts.
  - `scripts/onpolicy/pilot_gates.py`: the Stage 1 gates, headroom included.
  - `scripts/encode_processbench_token_store.py`: `--prompt_style`, recorded in
    the meta rows and in a new defaulted `RepSpec.prompt_style`.
  - `scripts/onpolicy/score_cells_on_split.py`: rebuild a trained cell and score
    any split, verified by a test that reproduces the cell's own ProcessBench
    scores exactly, rescaled cells included.
  - `slurm/onpolicy_generate_qwen3_tamia.sh`, `slurm/onpolicy_encode_span_store_qwen3_tamia.sh`.

The original list, for the record:

1. **`scripts/onpolicy/build_encoder_items.py`**: generator items -> encoder
   schema. Fills `uid`, `problem_id`, `solution_id`, `step_idx`, `n_steps`,
   `label`, and both `prompt_prefix` variants; carries `traj_uid` through as the
   trace `id` the ProcessBench evaluator groups on. Emits a sidecar
   `*_traces.jsonl` holding per-trajectory outcome, so the T2 simulator has one
   file to read.
2. **`--shard_idx/--num_shards` on the generator.** The pilot's 1,700
   trajectories/hour was one GPU: the script does `model.to(device)` and never
   shards, so three H100s idled through it. Sharding by problem index the way the
   encoder does is a 4x speedup for a few lines and is the difference between a
   12-hour full-scale generation job and a 3-hour one.
3. **`--prefix_field` on `encode_prm800k_token_store.py`**, per the prompt-style
   decision above.
4. **`scripts/onpolicy/score_cells_on_split.py`**: rebuild each trained cell from
   its `results.json` + `model.pt` and score an arbitrary store split, writing
   `pb_step_scores_<split>.jsonl` in the exact layout
   `scripts/analysis/pb_threshold_calibration.py` already consumes, so calib-20 is
   the same offline computation on both arms. The zscore statistics are not saved
   by the cell, but refitting them is exact and deterministic (vector path:
   `rs.fit` over the full cached train vectors; sequence path: the same strided
   subsample), so the script refits and asserts the refit matches the cell's
   recorded describe-line before scoring.
5. Tests in `tests/harness/` mirroring the existing style: schema round-trip, the
   generation-prefix reconstruction being byte-identical to the sampled context,
   shard partition being a partition, and the refit-stats assertion firing when
   handed the wrong store.

### Stage 1: scaled pilot, 300 problems x 10 samples (1 job, ~1h)

PRM800K test problems, ReProbe sampling (top-k 50, top-p 0.95, T=1.0), N=10 for
BoN comparability. Gates, in order of what would kill the arm:

- trajectory accuracy in 0.20-0.80 (the pilot's 0.435 at T=0.8 says this holds)
- gradeable rate >=0.95 at `max_new_tokens` 768; if truncation eats more than
  that, raise it and re-gate
- **step length in tokens, not words**, against PRM800K's 38.8. The pilot's ~24
  words is the flagged confound and words are the wrong unit. `n_tokens` is
  already written into store meta, so length-stratified reporting is free later;
  this gate only needs the distribution to overlap, not to match
- **BoN headroom: oracle@10 minus mean single-sample accuracy >= 0.10.** This gate
  is new and it is the one that decides whether T2 is answerable at all. If any
  sample of 10 is right almost exactly when a random one is, no reranker can
  separate from any other and every T2 correlation is measuring noise. If headroom
  is thin, raise N or move to harder problems before spending anything else.

### Stage 1: what happened (job 433635, 21 minutes, GO)

300 PRM800K test problems x 10 samples at T=1.0, four shards.

```
trajectories 3,000   gradeable 2,995 (0.998)   trajectory accuracy 0.372
steps per solution   median 7, mean 9.0        single-step 0.041
step length          median 32 tokens, mean 49.4
  off-policy ref     median 33 tokens, mean 37.9
pass@1 0.372    self-consistency 0.563    oracle@10 0.710
```

Two of these matter more than the rest.

**The step-length confound is much smaller than the handoff feared.** Measured in
tokens rather than words, the medians are 32 on-policy against 33 off-policy. The
"~24 words against 38.8 tokens" that looked like a distribution shift was a unit
mismatch. The means still differ (49.4 against 37.9), so the on-policy
distribution has a heavier right tail, and the length-matched rerun stays in the
protocol, but it is now a robustness check rather than a threat.

**T2 has room.** Oracle@10 is 0.710 against 0.372 for a single sample and 0.563
for self-consistency. A reranker has 0.147 to win over the baseline it actually
has to beat, which is enough for verifiers to separate. Had this been near zero,
every T2 correlation would have been measuring sampling noise.

### Stage 2: what happened, and the fork it forced

The first bake-off (job 433640) scored three local judges on 400 human-labelled
ProcessBench traces, at their natural prevalence, all answering the same
questions, with the degenerate strategies scored underneath them.

```
judge                 F1_PB  Acc_err  Acc_cor   exact  parsefail
qwen25_32b            0.435    0.294    0.834   0.522      0.000
qwen3_8b_base         0.421    0.294    0.740   0.482      0.002
qwen25_7b_instruct    0.009    0.004    0.994   0.422      0.000

always_no_error       0.000    0.000    1.000   0.422
always_last_step      0.000    0.048    0.000   0.028
always_first_step     0.000    0.130    0.000   0.075
```

Read against the leaderboard rather than in isolation: **the best representation
cell scores 0.566 on this same metric**, so a linear-probe-scale verifier on
internal states beats an 8B and a 32B model asked the question in words. That is
worth reporting on its own, with the caveat that these are base models answering
on their first generated token, and not an R1-class judge.

For this arm it is a problem: labels at Acc_error 0.294 are noisier than the
signal they would be used to measure. Qwen2.5-7B-Instruct is worse than a
problem, it is degenerate, answering "no error" on 99.4% of traces. Its exact
match of 0.422 is precisely the always-no-error baseline's, which is why that row
is in the table.

Neither judge shows the position bias the baselines were there to catch: both
point 0.45 of the way through a trace where the errors sit at 0.38, so they are
reading something, just not enough of it.

**The full local table** (job 433685 added the reasoning arms). Reasoning has
opposite effects depending on whether the model was instruction-tuned:

```
                        F1_PB  Acc_err  Acc_cor  parsefail
qwen25_7b_instruct_cot  0.481    0.346    0.787      0.000
qwen25_32b              0.435    0.294    0.834      0.000
qwen3_8b_base           0.421    0.294    0.740      0.002
qwen3_8b_base_cot       0.312    0.186    0.964      0.002
qwen25_32b_cot          0.236    0.134    0.982      0.000
qwen25_7b_instruct      0.009    0.004    0.994      0.000
always_no_error         0.000    0.000    1.000
```

It rescued the instruct model, which was degenerate without it, and hurt both
base models, which became more conservative rather than more accurate
(Acc_correct to 0.98, Acc_error to 0.13) and drifted later into the trace.
Pairwise agreement between these judges runs 0.47 to 0.81, which is the S3
lesson again: two judges agreeing is evidence and two disagreeing is a caveat.
The best of them still sits below the 0.566 of the representation it would be
supervising.

**DeepSeek-R1 over the API settles it** (the user's suggestion: compute nodes
have no network, but the labelling does not have to happen on them). Stratified
across all four subsets, 126 traces, no parse failures:

```
                        F1_PB  Acc_err  Acc_cor   exact
deepseek-r1 (API)       0.785    0.789    0.782   0.786
always_no_error         0.000    0.000    1.000   0.437

  gsm8k          n=57   0.913
  math           n=24   0.697
  olympiadbench  n=23   0.615
  omnimath       n=22   0.609
```

It points 0.39 of the way through a trace where the true errors sit at 0.39, so
it carries none of the late-drift the local judges show. At 0.785 against 0.566
for the best representation, its labels are cleaner than the signal they will be
used to measure, which is the condition this arm needed and did not have.

Two lessons paid for on the way, both now enforced in code. The certification set
was written one subset after another, so the first stopped run scored 0.807 on 63
traces that were GSM8K to the last one; the file is interleaved now, and the
subset table above shows how much that mattered. And 15.9% of the first run's
traces came back empty because R1 spent its whole 4,096-token budget reasoning;
those were being scored as "no error", penalising the judge for a budget setting.
An empty reply now buys one retry at double the budget, and only the failures pay
for the bigger ceiling.

Two follow-ups, run as controlled changes rather than a redesign:

1. **Reasoning before the verdict** (job 433685). All three answered on the first
   generated token. The same judges get an ordered check of the steps and then a
   final answer line, on the same 400 traces.
2. **A rollout labeller** (job 433686), which needs no judge at all. From each
   prefix, sample K continuations and grade them against the known answer; the
   first step after which the model cannot recover is the first error. This is
   Math-Shepherd's hard estimation and it is on-policy by construction.

The rollout labeller measures something different from what a human annotator
marks: a human marks a step that is *wrong*, a rollout marks a step after which
the model cannot *recover*, and those come apart both ways. So it is gated on
human annotations before its labels are used, by a paired test on PRM800K matched
forks: one prefix, one step humans rated +1, one they rated -1, roll out from
both, and ask whether the value is lower after the step humans called wrong.
Chance is 0.500; the gate is 0.600. This is the certification PRM800K *can*
support, and it dodges the errors-come-last artifact that makes whole-trace
PRM800K localisation meaningless.

### Stage 3: encode and screen (1 job, ~1h, then minutes)

Encode the Stage 1 steps at layer 35, `--span_only --model_dtype bfloat16`, under
both prompt styles. Then `scripts/screen_representation.py` over the six dense
representations. The screen predicts calib-20 at Spearman 0.934 over 31 evaluated
cells, so this gives a first T1 read in minutes.

Gate to proceed to full scale: the six-representation screen ranking correlates
with the off-policy screen ranking, and the `last_token -> step_mean` contrast has
the same sign. A flipped sign here is not a failure of the arm, it is the
headline, but it changes what Stage 4 has to measure and should be understood
before 20,000 trajectories are generated.

### Stage 4: T1-transfer at full scale

Generate 2,000 PRM800K test problems x 10 samples (~20k trajectories, ~175k steps,
~50 GB span store at 4096 dims; check `$SCRATCH` against its 1 TB quota first),
judge, encode, then score all 19 saved cells x 3 seeds with
`score_cells_on_split.py` and run calib-20 on the on-policy split.

Reporting, fixed in advance:

- Spearman and Kendall across the 19 seed-averaged cells, with a bootstrap
  interval over traces. With 19 cells a point estimate alone is not
  interpretable.
- **A reliability ceiling, which the backbone-transfer result did not have.**
  Split the three seeds within the off-policy arm and compute the same rank
  correlation between halves. A transfer Spearman cannot exceed the measurement's
  own reliability, and reporting 0.7 against a ceiling of 0.75 is a very different
  claim from reporting 0.7 against a ceiling of 0.95.
- The four load-bearing contrasts individually, each with its seed spread, plus a
  sign test across them.
- Every rank correlation recomputed within length-matched strata of `n_tokens`,
  since step length differs between the arms by construction.

### Stage 5: T2, offline from the stored scores

Every downstream task except verifier-guided search is a simulation over scores
already on disk, so one generation run buys all of them and thresholds and
policies can be swept freely.

- **Best-of-N** at N=10. Trajectory score aggregation is itself a choice, so all
  three standard rules are reported: worst step (`1 - max_i P(error_i)`, the usual
  PRM rule), mean step, and last step. Baselines: random pick, self-consistency
  (unweighted majority), and oracle@10 as the ceiling.
- **Weighted majority voting**, votes weighted by trajectory score.
- **Early termination** and **trajectory pruning**: token savings at matched
  accuracy, swept over thresholds.
- **On-policy first-error F1_PB**: the ProcessBench analogue on self-generated
  traces, which is Stage 4's metric, reused here as the x-axis.

T2's headline is the correlation, across the 19 verifiers, between off-policy
F1_PB and downstream gain. State the power honestly: at n=19 the 95% interval on
a Spearman is roughly +/-0.3, so this study can separate a strong relationship
from none and cannot separate 0.5 from 0.8. That limit is inherent to a controlled
grid and is the price of pinning every nuisance variable; it is better stated in
the paper than discovered by a reviewer.

The hypothesis to state before looking: reranking needs only *within-problem*
ordering while first-error localisation needs a *calibrated threshold across*
problems. A representation can be good at one and mediocre at the other. If
`step_mean` wins F1_PB and `last_token` wins best-of-N, F1_PB is the wrong thing
to optimise, and that is the most useful finding available here.

### Stage 6 (conditional): training-data policy

Only if Stage 4 shows the ranking moving. The confound is that on-policy training
yields far fewer labelled steps, and a rank change caused by data quantity would
be read as a policy effect. Control: subsample the off-policy training split to
the on-policy step count with `--train_cap` (which the cell script already
records as `full_train: false`, and which the merge already keeps in a separate
section), and run the matched pair. The reverse cell, trained on-policy and
evaluated on ProcessBench, completes the 2x2 for free once the on-policy training
store exists.

Deferred: verifier-guided step candidate selection. It is the one task where the
verifier changes what gets generated, so it cannot be simulated offline, and it is
worth nothing until T1 and T2 have answers.

---

## Protocol rules inherited without exception

One trainer, one hyperparameter protocol, three seeds, `--hp_from` reused across
seeds. `--rescale zscore`. Full training split unless a cap is deliberately
recorded. Input fingerprints on every split, and the merge refusing to mix
rescaling settings. calib-20 with quantile thresholds drawn from each split's own
calibration traces. F1 is never compared across arms of differing prevalence;
AUROC is the prevalence-invariant comparison and rank is the claim.

## Cluster rules inherited

Weights downloaded on the login node, nothing heavy run there, whole-node
`--gpus-per-node=h100:4` with in-node `CUDA_VISIBLE_DEVICES` sharding, every
background PID's exit status collected, per-process memory budgets divided by
concurrency, bfloat16 forward passes, and `$SCRATCH` checked against its quota
before a 50 GB store is written.
