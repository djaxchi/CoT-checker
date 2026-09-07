# ReProbe / UHead label semantics, recovered from the paper

Source: Ni et al., arXiv:2511.06209 (`https://arxiv.org/html/2511.06209v1`), read
2026-09-05. Everything below is quoted or paraphrased from the paper; where the
paper is silent the gap is marked and no convention is invented to fill it.

## What the judge is shown

> the question, the target LLM's CoT steps and final answer, and the ground-truth
> answer

The judge is **DeepSeek-R1**, and it grades **against a known answer**. Three
differences from the judge this project has been running:

| | ours (onpolicy_v1) | ReProbe |
|---|---|---|
| gold answer shown | no | **yes** |
| outcome told | yes ("reaches an INCORRECT final answer") | not stated |
| asked for | the index of the first wrong step | **the set of steps that contain errors** |

The first two were forced by the certification set: ProcessBench carries
`final_answer_correct` and no gold answer, so a judge certified on it could not
be shown one. The on-policy traces do carry a gold answer, so the ReProbe arm can
follow the paper.

## What counts as an error

> examine each step in the student's solution to determine whether it is both
> logically correct and relevant

> if any step contains an error that would prevent the student from reaching the
> correct solution, identify and report those specific steps

Two things follow, and the second is the one that would have been easy to get
wrong.

1. The criterion is **logical correctness and relevance**, not just arithmetic. A
   redundant step is explicitly in scope ("unnecessary/redundant reasoning").
2. The judge reports **the specific steps that contain errors**, as a set. The
   paper does **not** say that every step after the first error becomes negative.
   So this arm labels only the steps the judge names, and does not propagate.
   That is a real fork: propagation is common in the PRM literature and it
   changes the positive/negative balance substantially.

Labels are binary per step, 1 for correct and 0 for incorrect.

## Step segmentation

> Steps are written on a single line only: NO line breaks, bullet points, or
> substeps within a step.

The paper's generator is prompted to emit one step per line. Ours emits steps
separated by blank lines, the PRM800K convention, because the whole off-policy
grid is built on it and changing it would break comparability with every
existing cell. **Recorded as a deviation**, not silently reconciled.

## Trajectory aggregation for best-of-N

> Q_offline(r^(j)) = min_{1<=t<=T^(j)} (1 - U(r_t^(j) | r_{<t}^(j), x))

The chain score is the **minimum over steps of the step's probability of being
correct**, i.e. the worst step decides. With `U` the probe's probability that a
step is wrong, ranking by `min_t (1 - U_t)` descending is identical to ranking by
`max_t U_t` ascending, which is exactly the `worst_step` rule already implemented
and already reported as primary. No change needed, and the agreement is worth
recording: our primary aggregation was ReProbe's canonical one by coincidence,
and the `mean_step` and `last_step` columns stay as sensitivity checks.

## Architecture

> attention weights to the 1-3 preceding tokens and the logits of the top-K
> candidate generations

> a stack of L Transformer blocks ... mean-pooled across the reasoning step ...
> two-layer classification head with dropout and a GeLU activation

> less than 10M parameters (a 9.8M-parameter UHead)

The paper's headline feature set is **attention weights plus top-K logits**, not
raw hidden states. This project's `step_tokens x transformer` cell reads
last-layer hidden states of every token in the step. It is the same detector
shape over a different feature set, so calling it a ReProbe reproduction without
qualification would be wrong. Two named variants from here on:

- **ReProbe-style (hidden states)**: what we can build from the existing store.
- **ReProbe (published features)**: attention + logits, not currently extracted.

L, the hidden width, the head count, dropout, optimiser, learning rate, batch
size and epochs are deferred to the paper's appendix D.2, which is truncated in
the HTML version. **Unresolved**; the reproduction uses this project's frozen
protocol and records that as a deviation rather than guessing the paper's values.

## Training data scale

> 10.8K problems from the PRM800K training dataset, 3 reasoning chains per
> problem, ~32K data samples

Ours, as measured on the completed annotation run rather than as planned:
991 problems, 5,686 trajectories annotated, of which 5,483 parsed and were kept,
51,711 steps. Of those, 4,657 traces / 43,837 steps over 842 problems went to
training and 826 traces over 149 disjoint problems to validation. Comparable in
steps, an order of magnitude fewer problems. Recorded.

(An earlier draft of this section carried the planning estimate of 5,946
trajectories and 33,447 steps, which was the 991 x 6 target and a projected step
count. The numbers above are the audited ones from
`cot-checker-results/reprobe_v1/label_audit_full.json`.)

## Deviations, in one place

1. Judge is **GPT-OSS-120B run locally**, not DeepSeek-R1. The dataset is named
   "ReProbe-style GPT-OSS-120B on-policy labels" and never described as an exact
   reproduction.
2. Steps segmented on blank lines, not one-per-line.
3. Features are last-layer hidden states, not attention + logits.
4. Optimiser hyperparameters follow this project's protocol; the paper's are not
   recoverable from the public HTML.
5. 991 problems against 10.8K, and 5,483 kept trajectories against ~32K samples.

---

## Running GPT-OSS-120B offline on TamIA: what does and does not work

Recorded because two of these look like solutions and are not.

**MXFP4 needs `triton_kernels` from the Triton repo, and it is not obtainable.**
It is absent from the Alliance offline wheelhouse. The `kernels` library would
fetch kernels from the Hub at run time, which a compute node cannot do. And the
package named `triton_kernels` on PyPI is **a different project** from Kernelize
AI containing `add_vectors` and `rotary_embedding`; installing it would shadow
the import name and fail somewhere less obvious. The real one is a subdirectory
of the Triton git repo and would have to be installed from source on the login
node.

**Without those kernels the checkpoint dequantises to bf16**, from 61 GiB to
about 234 GiB. That fits across four H100s but not under a `device_map="auto"`
that fills the first GPU before spilling.

**Two failures at the same line ruled out capacity.** Jobs 443012 (auto map,
GPU 0 at 79.4/81.5 GiB) and 443041 (explicit 68 GiB cap) both died in
`transformers.core_model_loading._materialize_copy` at `tensor.to(device)` with
`CUDA_ERROR_ILLEGAL_ADDRESS` out of `cuMemcpyHtoDAsync`. A capacity problem would
not survive being given more room.

**vLLM is available after all.** The wheelhouse lists it as `cp38`, which reads
like a Python 3.8 build and TamIA has no 3.8 module, but the wheel is actually
`cp38-abi3` and installs on 3.12. The earlier `pip download` failure was its
`opencv` dependency, not vLLM. So a vLLM environment is a genuine fallback and
would use the native MXFP4 path, avoiding the dequantisation entirely.

---

## What GPT-OSS actually did with the no-propagation rule

Measured on 2,674 parsed annotations, over the 1,393 failing traces where it
found at least one fault:

```
faulty set is the whole suffix from the first error : 0.569
faulty set is a contiguous run, not to the end      : 0.146
faulty set has gaps (clean steps inside the span)   : 0.284
mean fraction of a trace marked faulty              : 0.543
```

**The judge arrives at propagation on its own in 57% of traces**, even though the
prompt never asks for it and the parser and encoder were built to preserve a
sparse set. That is not a defect and not a bug in the protocol: a step that
carries a wrong value forward genuinely is incorrect, and a careful annotator
would mark it. The 28.4% with gaps, median two clean steps sitting inside the
faulty span, is the evidence that it is reading steps individually rather than
applying a rule.

What it changes is what the labels *mean*, and that has to travel with them:

- they sit much closer to the first-error convention than the paper's protocol
  implies, so "we did not propagate" describes the prompt, not the data;
- the positive class is 54% of steps within a failing trace, against 30% over the
  pool as a whole;
- localisation signal is correspondingly weak, since a probe can score well by
  learning "this trace has gone wrong by now" rather than "this step is the
  wrong one".

The third point is the one to carry into the Phase 9 gate. If the on-policy
probe improves on step-level metrics but not on within-problem ranking or
best-of-N, this is the first explanation to test, ahead of anything about
architecture or training.

### A threshold that was recalibrated after seeing data

The audit's original check counted traces where most steps were faulty against a
0.25 bound, and the real labels came in at 0.422. The bound was chosen before the
trace-length distribution was known and it fires on short traces for arithmetic
reasons: two faulty of three steps is "most steps", and the rate falls from 0.66
at three steps to 0.33 at fifteen-plus purely from that.

It is replaced by a bound on the **length-normalised** faulty fraction at 0.75,
which catches a judge marking essentially everything without encoding a prior
about how many steps ought to be wrong, plus the propagation shares above as a
reported metric rather than a gate. Changing a threshold after seeing the data it
judges is a real methodological risk, so both the old bound and the reason for
the change are recorded here rather than quietly edited away, and propagation is
reported prominently instead of being hidden by a passing check.


## The guided-decoding verification gate, and why its tolerance was changed

Before generating anything with the checker in the loop, the runner re-scores
stored steps through the live scoring path and compares against the offline
scores that cell already wrote. A span reconstructed even slightly differently
would produce plausible and wrong generation numbers, so this blocks the run.

It failed four times at essentially one value (0.01095, 0.01095, 0.010947,
0.010947) against a 0.002 tolerance. Two hypotheses were tested and both were
wrong, and in each case the giveaway was that the maximum did not move at all:

1. **Tokenisation.** `encode_processbench_token_store.py` tokenises the prefix
   with special tokens and the step without, then concatenates the id lists,
   while scoring tokenised the joined string. A real mismatch, fixed, max
   unchanged.
2. **Precision.** The store is `np.float16`, so the head was fitted on
   float16-rounded activations while scoring kept float32. Also real, also
   fixed, max also unchanged.

The distribution then ruled out the whole class of systematic explanations:
median 0.000898 against a max of 0.010947. A wrong span, layer or rescaling
shifts every step and would move the median with it.

What remained was measured rather than assumed. Scoring the same step alone and
again inside a batch, with identical inputs and identical weights, moves the
result by median 0.000487, p95 0.005123, max 0.006439. Widening that probe from
three sequences to eight changed the numbers not at all, which locates the
discontinuity at batch-of-one versus batched: the two take different kernel
paths in bf16. The encoder read steps in batches of eight; the verification path
reads them singly.

So a max tolerance of 0.002 sat below the noise floor of the hardware and could
not be met by any implementation. The gate now uses two thresholds instead:

- **median <= 0.002**, which is the real check, since a systematic error shifts
  every step and shows up here;
- **max <= 0.02**, comfortably above the measured floor, to catch gross breakage
  without gating on numerics.

Recorded because a threshold loosened after seeing the data it governs is
exactly the move that should attract suspicion. The evidence for the change is
the floor measurement above, which was taken before the threshold was chosen.


## Efficiency claims, verified from the PDF

The cost argument for hidden-state verification is the part of ReProbe that
transfers most directly, so the numbers are quoted here rather than paraphrased.

> Despite using 750-810x fewer parameters than [PRMs]

> lightweight, containing fewer than 10M parameters

> ReProbe in current implementation achieves a 2.6x-25x speedup over
> state-of-the-art PRMs

Their runtime benchmark (Table 13, 500 MATH samples, batch size 1) excludes LLM
generation and step extraction; for ReProbe it counts feature extraction plus the
classifier forward pass, and for PRMs the full model inference. Training cost is
given as 4 GH200 GPU-hours over 32K samples.

Ours for comparison: the step_tokens transformer cell is 2,759,681 parameters,
roughly a third of their 9.8M UHead and about 2,500x smaller than a 7B PRM, and
it reads a forward pass the generator is already running. Training used 43,837
labelled steps.

Two cost claims follow and they must not be merged, because they point opposite
ways:

1. **Verifier cost.** A hidden-state head is a rounding error against a second
   7B forward pass. This is the claim that survives without qualification, and it
   holds whether or not the verifier is useful.
2. **Generation cost.** Guided decoding branches N ways per step, so it samples
   roughly N times the tokens. A cheap verifier does not make the search cheap.
   The honest comparison is accuracy at a matched token budget against
   self-consistency, which is why scripts/analysis/online_bon_report.py reports
   accuracy per thousand generation tokens with discarded branches counted in
   the denominator.

A third claim is worth testing and is not theirs: because a verifier can identify
a doomed prefix before the solution is finished, it could reduce tokens rather
than only reallocate them. Our labels place first errors 0.38 of the way through
on average, so the headroom is real, and an early-abandonment arm would measure
it directly.
