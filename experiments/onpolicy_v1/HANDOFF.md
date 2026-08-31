# On-policy transfer: context handoff

You are picking up one arm of a project. Everything below is established; nothing
here needs re-deriving. Read the "What you are doing" section first, then the
constraints, then the details.

---

## What you are doing

We have a controlled leaderboard of **step representations** for detecting
incorrect reasoning steps, measured entirely **off-policy**. Your job is to test
whether that leaderboard transfers to **on-policy** use.

Two distinct questions, both open in the literature:

**T1 — does the ranking survive?** If `step_mean` beats `last_token` off-policy,
does it still beat it when the verifier reads the states of the model that
actually wrote the reasoning? The claim is about **rank**, not score. Absolute
numbers cannot be compared across these settings (different text distribution,
different prevalence), and the project rule is: never compare F1 across datasets
with different prevalence. Measure Spearman across cells, plus whether the
specific load-bearing contrasts survive.

**T2 — does step-classification predict downstream usefulness?** Half the field
reports step-classification metrics and half reports test-time-scaling gains, on
different representations, and nobody measures both on the same ones. Our grid
gives verifiers that differ **only** in representation, so a correlation across
them isolates the metric relationship without model-size or training confounds.

T2 is the more valuable of the two. See "Why this is publishable" below.

---

## Scope: which representations

Use the **dense** representations only. The user explicitly excluded the SAE
family from this arm.

| representation | what it keeps | dim |
|---|---|---|
| `last_token` | state at the step's final token | 4096 |
| `step_delta` | final-token state minus the pre-step boundary state | 4096 |
| `step_mean` | mean over every token state in the step | 4096 |
| `step_stats` | concat[mean, max, min, std, last] | 20480 |
| `boundary_stats` | the above with the pre-step boundary prepended | 24576 |
| `step_tokens` | no reduction: the full variable-length sequence | 4096 x T |

Learners: `linear`, `mlp:h1024`, `mlp:h1024x2` for the fixed vectors;
`attn_query` and `transformer:d128/d256/d512` for `step_tokens`. 19 cells.

---

## The off-policy results you are testing against

Qwen3-8B-Base, layer 35, F1_PB @ calib-20, 4-subset mean, 3 seeds.
Full table: `experiments/unified_harness_7b/leaderboard.md`.
Per-cell artifacts: `cot-checker-results/rep_grid_q3/`.

```
step_tokens x transformer d512     0.566        <- best
step_tokens x attn_query           0.558
step_tokens x transformer d128     0.554
step_stats x mlp:h1024             0.540
boundary_stats x mlp:h1024         0.540
step_stats x linear                0.511
boundary_stats x linear            0.509
step_mean x mlp:h1024              0.495
step_mean x linear                 0.469
last_token x mlp:h1024             0.422
last_token x linear                0.419
step_delta x linear                0.395        <- worst
```

The contrasts that carry the argument, and which T1 must check individually:

```
last_token -> step_mean   (identical dim AND parameter count)   +0.050
step_delta -> last_token                                        +0.024
step_mean  -> step_stats  (5x wider input)                      +0.042
fixed pooling -> learned pooling (step_mean lin -> attn_query)  +0.089
```

The leaderboard also survived a **backbone** change at Spearman 0.919
(Qwen2.5-7B -> Qwen3-8B-Base, 3584 -> 4096 dims, different layer). So it is
already known to be robust to a large perturbation. Policy is the next one.

---

## Why this is publishable, and how to position it

**Do not frame this as a new benchmark.** It is an instance of an established
question applied where nobody has applied it.

- **RewardBench 2** (ICLR 2026) asks exactly T2 for *outcome* reward models:
  benchmark scores predict best-of-N well, RLHF poorly, and they attribute the
  difference explicitly to on-policy vs off-policy factors. **FC-RewardBench**
  reports 0.84 correlation between benchmark and downstream accuracy. So the
  methodology is respected and has precedent — it has simply never been asked of
  *process* verifiers.
- The 2025 survey **"Trust but Verify! A Survey on Verification Design for
  Test-time Scaling"** (arXiv:2508.16665) does **not** systematically address
  whether verifier benchmark scores predict TTS outcomes, and does **not**
  address on/off-policy in verifier evaluation. Both gaps are open and citable.
- **ProcessBench's own authors never claim predictive validity.** They frame it
  as measuring error identification for scalable oversight.
- Some papers report both metrics (FoVer reports ProcessBench *and* Best-of-K
  across 12 benchmarks) but do not analyse their relationship. Worth pulling
  FoVer's numbers as free paired data points before generating anything.

**Our instrument is stronger than RewardBench 2's.** They correlate across
published models, confounded by size, training data and architecture. We have 19
verifiers differing only in representation and learner, on identical frozen
activations, one trainer, one protocol, fingerprint-verified identical inputs. A
correlation across those is a within-study correlation with every nuisance
variable pinned.

Suggested framing:

> ProcessBench has become the de facto selection criterion for step-level
> verifiers, and its predictive validity has never been measured. Following
> RewardBench 2's analysis for outcome reward models, we ask whether step-level
> benchmark rank predicts (a) rank under on-policy evaluation and (b) downstream
> test-time-scaling gains, using a controlled grid where verifiers differ only in
> representation.

---

## Generation: settled, with a passing pilot

**Generator must be Qwen3-8B-Base**, the same model whose states we read. If you
generate with an Instruct model you must encode with Instruct, and then the two
arms differ in backbone as well as policy — confounded again. A base model does
this fine; the pilot proves it.

`scripts/generate_onpolicy_steps.py` already does the whole job: sample N
solutions per problem, grade by final-answer match against `ground_truth_answer`
via `src/eval/math_grade.py`, split on blank lines (the PRM800K `\n\n` step
convention), emit per-step items in the schema the encoders consume.

Pilot (job 430576, `slurm/onpolicy_pilot_qwen3_tamia.sh`), 48 problems x 8
samples:

```
trajectories 384   gradeable 384   correct 167 / incorrect 217   steps 3,369
trajectory accuracy   0.435    both classes well represented
steps per solution    median 7, mean 8.8
single-step solutions 0.042    the \n\n convention holds without instruction tuning
step length           median 16 words, mean 23.8
throughput            ~1,700 trajectories/hour on 4 GPUs at 26% memory
```

**One thing to check before scaling**: on-policy steps run ~24 words against
PRM800K's ~38.8 tokens. The model writes shorter steps than GPT-4 did. Step
length was flagged as a minor confound in the probe-anatomy work — measure it in
tokens, not words, and decide whether it needs controlling.

ReProbe's sampling settings, for comparability: nucleus, top-k 50, top-p 0.95,
temperature 1.0.

---

## Labels: the open decision

On-policy steps have no human labels. ReProbe's recipe, which the user endorsed:

- **Judge: DeepSeek-R1**, following Zheng et al. (the ProcessBench authors).
- **Given**: the question, the target LLM's CoT steps and final answer, **and the
  ground-truth answer**, plus supporting evidence. It grades against a known
  answer, not blind.
- **Validated against human labels**: "DeepSeek-R1 achieves 95% acc. on PRM800k"
  and ~90% on a manually annotated 1000-step set. **That is the number to beat**,
  and the validation design is the one to copy — use PRM800K human labels to
  *certify* the judge, not to train on.
- **They also have a self-supervised mode**: the same LLM annotates its own
  generated CoT steps.

**The constraint that decides this**: compute nodes have **no internet**, so an
API judge cannot run in a batch job. Options:

1. **Self-supervised** — Qwen3-8B judges its own steps, entirely on-node. Free,
   no API, and it is a published setting rather than an improvisation.
2. **A larger local judge** from the HF cache (`$STORE/hf_cache` has Qwen2.5-32B
   among others). On-node, stronger than self-supervision.
3. **Judge from the login node** against saved traces via API — but the login
   node reaps heavy processes (it killed two of ours silently), so this must be
   lightweight and chunked.

Whichever you pick, **report judge–human agreement on a held-out slice of
PRM800K** before using the labels. The project has form here: S3 work found Opus
and Haiku agreed only 44% on failure-mode labels, so judge reliability is not
something to assume.

---

## Downstream tasks for T2

From the literature and this project's own survey:

| task | what the verifier does | metric | source |
|---|---|---|---|
| Best-of-N reranking | score N complete solutions, pick one | accuracy@N vs self-consistency | CLUE, OTV, most PRM work |
| Weighted majority voting | weight each solution's vote by score | accuracy vs unweighted | SSAE |
| Trajectory pruning | kill unpromising traces mid-generation | token savings at matched accuracy | STEP (Liang et al. 2026) |
| Early termination | stop one trace when confident | token savings (OTV 90%, Zhang 24%) | OTV, Zhang et al. |
| Step candidate selection | sample k next-steps, pick best, continue | accuracy at matched budget | ReProbe |
| On-policy first-error | the ProcessBench analogue on self-generated traces | F1_PB | — |

**The design point that saves most of the compute**: all of these except
verifier-guided *search* are evaluable **offline from a single generation run**.
Generate N solutions per problem once, score every step with every
representation, and then best-of-N, weighted voting, early termination and
pruning are simulations over the stored scores — no regeneration, and you can
sweep policies and thresholds freely. Only step candidate selection needs
generation in the loop, because there the verifier changes what gets generated.
Defer that one.

ReProbe's evaluation settings for comparability: BoN with N=10 for math, beam
search with B=5 and N=5 candidates, temperature 1.0.

**A hypothesis worth stating in advance**: downstream reranking needs only
*relative* ordering within a problem, while first-error localisation needs a
*calibrated threshold across* problems. A representation could be excellent at
one and mediocre at the other. If `step_mean` wins on F1_PB but `last_token`
wins on best-of-N, that says F1_PB is the wrong thing to optimise — which would
be the most useful finding available here.

---

## Infrastructure you will need

**Stores** (Qwen3-8B-Base, `hidden_states[35]` = `resid_post` of block 34):
```
$SCRATCH/cot_mech/qwen3_8b_v1/repstore/step_spans      157G  513,810 / 5,000 / 2,000
$SCRATCH/cot_mech/qwen3_8b_v1/repstore/pb_step_spans    17G  25,676 steps, 3,400 traces
$SCRATCH/cot_mech/dense_full_7b_v1/data/               frozen PRM800K splits (jsonl)
/scratch/d/dchikhi/cot-checker/processbench_full/      ProcessBench raw jsonl
```

**Encoding on-policy traces**: `scripts/encode_prm800k_token_store.py
--span_only --layer 35 --model_dtype bfloat16`. Span-only writes the pre-step
boundary row plus the step's own tokens straight out of the forward pass, no
full-sequence intermediate (which would be ~1.1 TB and does not fit).
`slurm/encode_prm800k_span_store_qwen3_tamia.sh` is the template.

**Training cells**: `scripts/train_rep_learner_cell.py`, driven by
`slurm/train_rep_grid_7b_tamia.sh` with a `CELLS_FILE`. Cells files live in
`experiments/unified_harness_7b/`.

**The fast screen**: `scripts/screen_representation.py` trains one linear probe
on a subsample and reports ProcessBench step AUROC in under a minute. Calibrated
against 31 evaluated cells: **step AUROC predicts calib-20 at Spearman 0.934**,
against 0.835 for in-domain AUROC. Use it to triage before spending a grid run.

---

## Protocol rules you must not break

These are what make the leaderboard mean anything.

- **One trainer, one hyperparameter protocol, three seeds** for every cell.
  Tuning is searched once per cell and reused across its seeds (`--hp_from`);
  re-searching per seed lets each seed pick the configuration suiting its own
  initialisation and shrinks the very spread the seeds measure.
- **Full training split, no cap by default.** A cap is recorded as
  `full_train: false` and the merge keeps capped cells in a separate section.
- **`--rescale zscore` is the protocol since 2026-08-28.** The layer we read has
  no normalisation after it, so raw numbers swing by ~22 and pin the probe's
  scores to 0 and 1. The merge **refuses to render a table mixing rescaling
  settings**.
- **Input fingerprints.** Every cell records a digest of every split it reads and
  the merge refuses to build a table if any two cells disagree.
- **calib-20 with quantile thresholds.** 20 held-out traces per subset pick the
  first-error threshold, applied to the rest, averaged over 20 splits. Candidates
  are the **score quantiles of each split's own calibration traces**, not a
  uniform probability grid — an overconfident probe piles scores at 0 and 1 where
  a uniform grid has no resolution, which made one seed score 0.248 against 0.498
  for its siblings while its AUROC was normal.

---

## Cluster gotchas that have already cost us time

- **No internet on compute nodes.** Download weights on the login node first.
  Both encode scripts assert a local snapshot exists and print the download
  command rather than hanging.
- **The login node reaps heavy processes silently.** Two verification runs died
  there with empty logs. Anything that loads a multi-GB tensor must be a batch
  job.
- **H100 nodes allocate whole only**: `--gpus-per-node=h100:4`. Shard in-node via
  `CUDA_VISIBLE_DEVICES`, not job arrays.
- **Background cells hide their failures.** Collect every PID's exit status; a
  grid job once reported COMPLETED in 33 seconds having produced nothing.
- **Per-process memory budgets must divide by concurrency.** Four cells each
  deciding 163 GB "fits" in a 300 GB budget OOM'd a node.
- **`$SCRATCH` has a 1 TB quota** and is the tier that fills. `$STORE`
  (`/project/aip-azouaq/$USER`) is ~2 TB shared and was recently at 100%.
- **HF cache path is `models--org--name`** with a double dash.
- **Qwen3 ships bfloat16.** Encoders default to it and refuse to write a step
  whose activations overflow float16 storage.

---

## Suggested first three steps

1. **Scale the generation.** The pilot passed; run it at ~2,000 problems x 8
   samples on PRM800K *test* problems (held out of probe training, and problems
   the off-policy grid was already scored on, so the two arms share problems and
   differ only in who wrote the solution).
2. **Settle the judge and certify it.** Self-supervised is the cheapest and is a
   published setting; report agreement against held-out PRM800K human labels
   before trusting the labels.
3. **Encode and screen before running the grid.** Encode the on-policy traces at
   layer 35, then run `screen_representation.py` over the six representations.
   That gives a T1 answer in minutes and tells you whether the full 19-cell run
   is worth it.

Open question the user has not decided: whether the on-policy arm should be run
at matched **training size** with the off-policy arm. On-policy will yield far
fewer labelled steps, and a ranking change caused by data quantity would be
mistaken for a policy effect. Subsampling the off-policy grid to match is the
clean control.
