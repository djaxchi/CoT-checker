# onpolicy_v1: does the representation leaderboard transfer to on-policy use?

Status: Stage 0 implemented and tested (2026-08-31), Stage 1 ready to submit.
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

### Stage 2: judge bake-off and certification (1 job, ~1-2h)

As described above. Output: agreement table, chosen judge, and the noise ceiling
that every subsequent first-error number is reported against.

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
