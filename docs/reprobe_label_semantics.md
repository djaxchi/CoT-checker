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

Ours: 991 problems, 6 chains each, 5,946 trajectories, 33,447 steps. Comparable
in steps, an order of magnitude fewer problems. Recorded.

## Deviations, in one place

1. Judge is **GPT-OSS-120B run locally**, not DeepSeek-R1. The dataset is named
   "ReProbe-style GPT-OSS-120B on-policy labels" and never described as an exact
   reproduction.
2. Steps segmented on blank lines, not one-per-line.
3. Features are last-layer hidden states, not attention + logits.
4. Optimiser hyperparameters follow this project's protocol; the paper's are not
   recoverable from the public HTML.
5. 991 problems against 10.8K.
