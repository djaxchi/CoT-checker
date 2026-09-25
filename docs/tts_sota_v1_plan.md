# tts_sota_v1: our curves against what the field actually deploys

*2026-09-21. Sprint 8. Follows tts_roster_v1 (`docs/tts_roster_v1_plan.md`,
REPORT.md §21).*

## 1. The question

Sprint 7's result was accepted on its own terms: a verifier tie-break beats
self-consistency at a small budget and converges on it by N=10. The objection
raised against it is that the comparison set was ours, not the field's. Sprint 8
answers one question:

> Does the sprint 7 curve survive against the aggregators and the verifiers that
> the test-time-scaling literature actually reports?

Two halves, in the order they cost compute.

## 2. Half one: the aggregators. Done, no GPU

Landed 2026-09-21, REPORT.md §21.8. The field's rules are weighted votes, so
`src/analysis/tts_frontier.py` grew `weighted_vote` (Lightman et al. 2023),
`filtered_vote` (DeepConf arXiv:2508.15260) and `softmax_vote`, which puts a
plain count and a rerank at the two ends of one dial. All of it rebuilt from the
saved atoms; nothing regenerated.

Outcome: no rule in the family beats the tie-break anywhere with an interval
clear of zero, and accuracy falls monotonically in how much authority the
verifier is given.

Caveat, and the reason half two exists: both scorers in that comparison are
free. They read states or logprobs the sampler already computed. "Weighting
never pays" may be a statement about weak weights.

## 3. Half two: the verifier. Qwen2.5-Math-PRM-7B

The model is already on the cluster (REPORT.md §21 Next Step). What is missing
is a scoring adapter.

### 3.1 What the adapter must produce

Exactly the schema the existing scores already use, so the frontier builder
needs no change at all: it globs `scores/{stem}__*.shard*.jsonl` and parses the
cell name out of the filename.

```
{"traj_uid": ..., "problem_id": ..., "dataset": ..., "split": ...,
 "correct": ..., "n_steps": K, "cell": "prm_qwen25_math_7b",
 "scores": [s_1 ... s_K]}
```

**Orientation is the trap.** Every consumer in this codebase treats `scores` as
suspicion, lower is better, and `tts_build_frontier.py` hardcodes
`quality_from(vals, higher_is_better=False)`. Qwen2.5-Math-PRM-7B emits
P(step correct). The adapter therefore writes `1 - P(correct)` and says so in
its docstring. Get this backwards and every curve is a mirror of the truth
while looking perfectly ordinary, which is the §20.16 failure mode.

The reward the weighted vote then uses is `1 - suspicion = P(correct)`, which is
the quantity Lightman et al. weight by. That falls out for free.

### 3.2 The segmentation gate, checked before any number is read

The PRM must score **the same steps the probe scored**. Our traces are few-shot
base-model output segmented on blank lines (`src/onpolicy/spans.py`); the PRM
was trained with its own chat template and a `<extra_0>` step separator. If the
adapter re-segments, the two scorers are reading different objects and the
head-to-head is confounded.

Gate: for every trajectory, assert `len(prm_scores) == len(probe_scores)`. Abort
the job if the mismatch rate exceeds 0%. Report the rate in the log even when it
is zero, so the check is visible rather than assumed.

Second gate, on a 20-trace sample: the PRM's mean P(correct) on traces graded
correct must exceed its mean on traces graded incorrect. A PRM wired backwards
or fed a malformed template will fail this in seconds and cost nothing.

### 3.3 Cost

18,188 saved trajectories, roughly 4.5M tokens of prefill through a 7B model.
This is a scoring pass, not generation: no sampling, one forward per trace,
batched. One whole H100 node, sharded four ways by `CUDA_VISIBLE_DEVICES` the
way `slurm/tts_score_cells_tamia.sh` already does. Estimate 1 hour; request 2.

### 3.4 Pricing the PRM onto the token axis

Sprint 7's headline axis is generated tokens, and on that axis a 7B PRM is not
free while our probe is. Report both readings and let them disagree:

- **at matched N**, the question is whether the PRM ranks better than the probe;
- **at matched compute**, the PRM's scoring pass is added to the budget, and the
  question is whether that spend beats drawing more samples instead.

The second is the comparison a practitioner makes and the one sprint 7's axis
was built for. Do not collapse them into one curve.

## 4. Also in scope, cheap

**The second probe cell.** `last_token x linear` was roster item 1 in
`docs/tts_roster_v1_plan.md` §4 and never ran. One pass of
`scripts/onpolicy/score_traces_with_cell.py`, same node, same job. It answers
whether "the verifier" is one architecture's quirk.

## 5. Out of scope, and why

**Sequential search** (beam search or lookahead over a PRM, Snell et al. 2024).
It needs regeneration under a different sampler, so it is a different experiment
on a different axis, not another line on this figure. It is the obvious sprint 9.

**Further PRMs** (Math-Shepherd, Skywork). Add only if Qwen2.5-Math-PRM-7B lands
and the result turns on which PRM was chosen.

## 6. What a null looks like

If the PRM's curves land on top of the probe's at matched N, the claim becomes
"a probe on frozen hidden states matches an off-the-shelf 7B PRM at a fraction
of the cost", which is the stronger result and should be reported as such rather
than buried.

If the PRM beats the probe at matched N *and* at matched compute, sprint 7's
recommendation is wrong and the report says so in §21.8's own table.

If weighting pays off for the PRM where it did not for the probe, §21.8's
structural claim is downgraded from "weighting never helps" to "weighting helps
in proportion to verifier quality", which is a finding, not a retraction.
