# S3 Research Direction — What Do Hidden States Encode About CoT Correctness?

**Source.** Research proposal by Djalil Chikhi, *"What Do Hidden States Encode About Chain-of-Thought
Correctness? A Mechanistic Study of Step-Level Failure Signals"* (Polytechnique Montréal, Mila).
This is the governing mission for S3 and supersedes the S1/S2 framing as our primary objective.

---

## The pivot (read this first)

S1 and S2 asked: *can we build a representation/probe that beats raw dense hidden states on the
ProcessBench first-error score?* Answer across both sprints: no — raw dense holds the strongest
oracle signal, scale raises the ceiling (0.476 @ 32B) but not deployable calibration, and no learned
representation (SAE/SSAE/fork-preference) wins.

**S3 stops optimizing the score and starts explaining it.** The empirical anchor is fixed: hidden-state
probes reach **~41% on ProcessBench** using only internal activations — too weak to claim general
correctness encoding, too strong to dismiss as noise. The scientific move is to ask *what signal is
being picked up*, not to push the number.

ProcessBench is now the **empirical starting point, not the objective.** Its first-error localization
score is a downstream diagnostic, not the main scientific metric.

---

## Central hypothesis

**Correctness is not encoded as a single unified concept.** A binary correct/incorrect label collapses
several mechanistically distinct failure modes. A probe trained on the binary label succeeds when a
benchmark's failures align with the encoded signal, and fails when the required notion of incorrectness
is not internally represented, not linearly accessible, or not present at the extracted layer/token.

So the aggregate ~41% is plausibly a **mixture of failure-mode-specific signals**, not one
correctness direction.

## Main research question

> How is step-level CoT correctness represented in hidden states, and is the observed correctness
> signal better explained as a mixture of failure-mode-specific signals rather than as one unified
> correctness concept?

### Six subquestions
1. Which hidden states contain information correlated with step-level correctness?
2. Which types of reasoning failure are detectable, and which are not?
3. At which layers and token positions do different failure signals appear?
4. Are these signals linear directions, sparse features, or distributed patterns?
5. Do failure signals learned on controlled examples transfer to natural traces and ProcessBench?
6. Do interventions on detected signals change continuation behavior, or only probe predictions?

---

## Failure taxonomy (analytical lens, experimental)

Moves ROSCOE-style textual reasoning-error typologies *inside the model*: instead of asking whether a
text metric detects an error, ask whether hidden states encode different errors differently.

- **Arithmetic error** — wrong numerical computation (e.g. 7 × 8 = 54).
- **Algebraic transformation error** — invalid equation rearrangement / symbolic manipulation.
- **Variable or entity binding error** — confusion between quantities, people, objects, variables, roles.
- **Quantity or unit mismatch** — combining incompatible quantities or semantic types.
- **Unsupported premise** — information absent from the problem or previous steps.
- **Goal drift** — solving a related but wrong subproblem while locally coherent.
- **Constraint violation** — ignoring domain / integer / inequality / boundary conditions.
- **Logical inference error** — invalid inference not reducible to arithmetic or algebra.
- **Post-hoc reasoning** — plausible trace not causally used for the final answer.

The taxonomy is **refined empirically**: merge categories with indistinguishable hidden-state
directions, split categories that hold multiple separable internal signals. Driven by annotation
quality, probe behavior, and representation geometry.

---

## Methodology (6 stages)

1. **Profile the ProcessBench signal.** Annotate a subset of ProcessBench first-error examples by
   failure mode; measure which categories the hidden-state probe gets right → a per-failure error
   profile that explains the non-trivial-but-limited aggregate score.
2. **Build controlled contrasts.** Per failure mode, paired examples = problem + prior context +
   correct step + minimally corrupted step + label (controls topic, length, vocab, position).
   Estimate the mean corrupted−correct hidden-state difference per type.
3. **Extract representations across layers.** Dense hidden states at the **last token of the current
   step, all layers**, given problem + previous steps + current step. Dense first (avoids
   representation-learning artifacts); SAE / step-level SSAE latents added only *after* locating where
   the dense signal appears.
4. **Probe, localize, compare.** Simple linear probe per (failure mode × layer) — testing geometric
   accessibility, not maximizing a big classifier. Output: layer-by-failure F1 heatmap. Baselines:
   step length, number count, operator count, token likelihood, model self-judgment.
5. **Analyze geometry & transfer.** Compare mean contrastive direction vs failure-specific probe
   direction vs ProcessBench probe direction (cosine sim, seed stability, within/between-class
   distance). Transfer across controlled → natural annotated → ProcessBench. Strong evidence =
   predictive in-domain **and** stable **and** transferable.
6. **Test causality.** Probes only show decodability. Activation patching (swap correct↔corrupted
   hidden states at a chosen layer/token) and steering (add/subtract a failure direction). If a
   direction changes continuation behavior (more/fewer mistakes, final-answer likelihood,
   dose-response), it is causal; if it only moves the probe score, it is diagnostic but not causal.

---

## Metrics & outcomes

- **Failure detection:** macro F1.
- **ProcessBench transfer:** official first-error localization score (downstream diagnostic only).
- **Geometry:** direction alignment, within/between-class distance, stability across seeds and model sizes.
- **Causality:** changes in continuation correctness, final-answer likelihood, dose-response under steering.

Three landing zones:
- **Strong** — several failure modes detectable, geometrically stable, transferable, *and* causally manipulable.
- **Medium** — some modes detectable and transferable, but not causally active.
- **Weak-but-useful** — probes mainly exploit dataset-local artifacts or narrow failures (e.g. arithmetic).
  Still explains the limited ProcessBench score and prevents overclaiming.

## Expected contributions
(1) failure-mode decomposition of step-level CoT correctness; (2) analysis of which ProcessBench
errors drive probe success; (3) a bridge from ROSCOE-style textual taxonomies to internal
representations; (4) layer-wise maps of failure-mode detectability; (5) geometric analysis of failure
directions; (6) transfer tests controlled → natural → ProcessBench; (7) causal tests via activation
patching, steering, and sparse-feature ablation.

---

## What carries over from S1/S2
- **Dense, last-token, all-layer extraction** is the validated default (S2 model-size sweep) — keep it.
- **The locked eval pipeline** (freeze representation → fresh probe → ProcessBench) is reused, but
  ProcessBench is now diagnostic.
- **SAEs/SSAEs are deferred**, not dropped — they re-enter at stage 3 only where the dense signal is
  located (answers subquestion 4).

## References
Golovneva et al. 2022 (ROSCOE, 2212.07919); Lanham et al. 2023 (2307.13702);
Lightman et al. 2023 (2305.20050); Wei et al. 2022 (NeurIPS); Zheng et al. 2024 (ProcessBench, 2412.06559).
