# latent_memory_v0: does a fixed-size latent bottleneck have the capacity to replace a CoT?

Status: design + oracle scaffold (2026-07-21). Not yet run on TamIA.

## Hypothesis (the version worth testing)

A completed chain of thought contains a causally sufficient reasoning state that can be
compressed into a small set of `m` latent memory vectors, such that conditioning the
frozen base model on those vectors (in place of the CoT text) preserves the downstream
behavioural effect of conditioning on the full textual trace.

This is NOT "predict the embedding of a textual summary". Summary-embedding regression
only shows the compressed vector resembles a summary's representation; it never shows
the reasoning state is preserved, and a shortcut ("the answer is 42") solves it. We
therefore test behavioural equivalence directly, and we test capacity BEFORE training
any amortised compressor, mirroring the project's standing oracle-first discipline
(S5 steering, S6 boundary oracle, DAS span/branch, transition_operator Stage 0).

## Why oracle-first, and what it settles

Two existing results bracket the question but do not answer it:

- Whole-step-span patching moves the answer belief by ~0.88 of the fork margin, using
  the ACTUAL step residuals (hundreds of positions). (REPORT L1371, das_span.)
- A single boundary vector recovers ~0 of the answer belief through the suffix
  (transition_operator Stage 0, Target B ~0 at every layer).

So the causally relevant information is distributed across many step positions, and one
natural boundary vector does not carry it. Neither result says how few FREE vectors are
needed. The oracle answers exactly that: per trace, freeze the model, drop `m` trainable
latent vectors into the residual stream at layer L in place of the step tokens, and
optimise them to reproduce the full-CoT teacher's answer distribution. Recovery vs `m`
is a capacity curve. It converts "boundary null / span 0.88" into a number: the minimum
bottleneck width at which a frozen model's answer belief is recoverable.

Interpretation of outcomes:
- m=1 already recovers ~0.88 -> the boundary-null result was about the WRONG single
  vector (a natural boundary state), not about capacity; one free vector suffices.
- recovery rises with m and plateaus at m* -> m* quantifies the distributedness of the
  reasoning state. Sets the target width for any learned compressor.
- even large m stays well below the span ceiling -> the layer-L residual bottleneck /
  injection interface is insufficient; revisit layer or injection mechanism before
  building an encoder.

## Reuse map (what we do NOT rebuild)

| Need | Reused asset |
|---|---|
| trainable-state injection at layer L | `das_span.make_span_patch_hook` (overwrites a residual span; upper blocks recompute K/V during prefill; gradient flows to the states) |
| behavioural readout (answer belief) | `causal_graph.ELICITATION_SUFFIX` + `transition_operator.candidate_mean_logprobs` / `gold_margin` / `belief_from_scores` |
| distractor sets | `transition_operator.build_candidates` (type-matched, pre_generated_answer + wrong-branch + corpus + perturbations) |
| SEP-id tokenization | `causal_graph.encode_pieces` / `assemble_ids` (SEP id 198) |
| full-CoT traces (question + golden steps + gt_answer) | `runs/causal_graph/traces_forks.jsonl`, `runs/transition_operator/forks.jsonl` (fields: question, steps, fork_t, gt_answer, pre_generated_answer) |
| model load (Qwen2.5-7B base, bf16, L20 primary) | cg_stage1_tf load pattern |

## v0 experiment (capacity oracle, same-question answer belief)

Per trace with question q, golden steps s_1..s_n, gold answer a*, distractors D:

- Context skeleton (SEP-id): `[q] SEP [P]*m SEP` where `[P]` is a placeholder token; the
  `m` placeholder positions form the latent span `[lo, hi)`. Question stays in text; only
  the CoT is replaced by latent memory.
- Readout: append `ELICITATION_SUFFIX` then score candidates `[a*, D...]`; `gold_margin`
  and `belief_from_scores` give the behavioural target.
- Teacher (ceiling): full-CoT context `[q] SEP [s_1] SEP ... [s_n] SEP` + suffix.
- Floor: no-CoT context `[q] SEP` + suffix.
- Oracle: initialise `z (m,d)` from chunk-mean-pooled step residuals at layer L; optimise
  z with the frozen model to match the teacher belief (KL over the candidate set), read
  margin recovery and belief-KL.

Sweep `m in {1,2,4,8,16,32}`, layer `L in {20}` primary (24/26 diagnostic). Baselines at
each m: fixed chunk-mean-pool and chunk-max-pool latents (no optimisation), last-step
boundary state (m=1). Metrics: recovery `R = (margin_lat - margin_no)/(margin_full - margin_no)`,
belief-KL to teacher, per-trace optimisation curve (undertraining vs no-capacity).

Report as a capacity curve (median recovery vs m, with per-trace spread), against the
0.88 span ceiling and the boundary-null floor.

## What v0 deliberately does NOT do

- No learned amortised compressor yet (that is the next gate, only if capacity exists).
- No follow-up-query distribution yet. v0 target is same-question answer belief, the
  cheapest already-built readout. The decisive design fork (continue-solution vs
  arbitrary-trace-probe vs transfer-to-related-problem) is chosen only after capacity is
  established, because it dictates expensive new data generation.
- No full-vocab suffix KL yet (candidate-set KL first; full-vocab is a stronger v0.1 target).

## Files

- `src/analysis/latent_memory.py` - oracle core (context build, grad-enabled candidate
  scoring under a latent patch, per-trace optimiser, pooling baselines, recovery).
- `tests/analysis/test_latent_memory.py` - pure + tiny-stub-LM tests (grad flow, loss
  decreases, recovery formula, pooling shapes).
- `scripts/latent_memory/lm_oracle.py` - trace loop, sweep, JSONL out (to add before run).
- `slurm/latent_memory_v0.sbatch` - TamIA whole-node job (to add before run).
