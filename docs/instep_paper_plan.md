# InStep paper plan (draft, 2026-11-02)

InStep = Inline Non-intrusive Step Token for Error Prediction. Internal code: RTH-ICS
(`scripts/read_token_inline.py --deep --chain --start_marker --start_deep`).

## Thesis

A frozen policy already encodes whether the step it just wrote is wrong. InStep reads
that signal with one trained read token per step, placed inside the policy's own trace,
that the text never attends to. It beats the best external probe (ReProbe) on ReProbe's
own benchmark with 26x fewer trainable parameters, and closes most of the distance to a
fine-tuned 7B PRM without a second model.

## Benchmarks, in paper order

1. **ReProbe step benchmark (main).** 8 sets (ID: MATH, GSM8K, ProofNet; OOD: Trips,
   Meetings, Calendar, StrategyQA, ScienceQA). On-policy traces written by the verified
   backbone, DeepSeek-R1 labels on every step. Primary metric PR-AUC of "incorrect"
   (the field's metric, threshold-free); secondary AUROC; F1 at the threshold picked on
   ReProbe's validation split, against the always-positive F1 (project F1 rule).
2. **ProcessBench (second).** Off-policy traces, human first-error labels, math only.
   Main text: threshold-free first-error AUROC / PR-AUC (labelled first error against
   the earlier steps and the steps of correct traces) and the label-free first-error
   F1 once that protocol is settled. Appendix: calib-20.

## InStep definition (to freeze)

Frozen backbone; one soft read token after each step's separator, at the next text
position, initialised from " correct", 36 (or n_layers) per-layer offsets at the read
positions, linear head on the final state -> P(incorrect). Step-start marker (init
" step", own offsets) seen only by its step's read token. Chained reads: read k sees
reads 0..k-1. Text never attends to any read token or marker. ~0.31M parameters on
Qwen3-8B. Training: BCE (1 = incorrect), AdamW, OneCycle (5% warm-up, peak lr 3e-3),
clip 1.0, one epoch, 3 seeds, deployed seed picked on validation. Open: a prespecified
rule for runs that diverge at the end of warm-up (restart on a validation drop, or a
lower peak lr), fixed before any new run.

Use, always one pass on the policy's own trace: **online** = during generation, the read
runs over the live KV cache (8.8 ms per MATH-500 trajectory, +0.27 ms when riding a
decode step); **offline** = after generation, one text pass rebuilds the cache, then the
read (35.3 ms).

Training sources: ReProbe's training traces (on-policy, DeepSeek-R1 or self labels) for
the main benchmark; PRM800K human labels (off-policy text laid out in the chat layout)
for ProcessBench and the zero-shot transfer arm.

## Comparison set

| group | method | params added at inference |
|---|---|---|
| free signals | token confidence (mean/min logprob, entropy per step) | 0 |
| | prompted self-verification, zero-shot P(No) | 0 |
| readout of existing states | passive linear: separator / last step state, logistic, layer picked on val | ~4k |
| external probe | ReProbe Attn+Logit (stored published scores) | |
| | ReProbe hidden states: released, matched retrain, future-masked retrain | 8.1M |
| ours | InStep (+ ablations below) | 0.31M |
| fine-tuned PRM (second model) | weak PRM Qwen2.5-Math-7B-PRM800K (same PRM800K labels as InStep) | 7B |
| | strong PRM Qwen2.5-Math-PRM-7B; other PRMs stored in ReProbe's files | 7B+ |

Related work only: ScratchProbe (closest precedent, appears as the "no offsets" ablation),
one sentence on prompt/representation tuning (they change the text's computation, InStep
does not), STEP, Zhang et al., VerifySteer, LiLaVe, Clue, OTV.

## InStep ablations (main text, run on the ReProbe benchmark and ProcessBench)

The ladder, each rung adding one ingredient: zero-shot P(No) -> passive linear ->
embedding only (ScratchProbe design) -> + per-layer offsets -> + chain -> + marker (= InStep).
Plus: label source (human PRM800K, DeepSeek, self). The per-step template read (old RTH)
is dropped from the paper. Appendix: init word, number of tokens, data size, readout.

## Backbones

Qwen3-8B (main), Phi-4 (other family), Qwen3-32B (scale); optional Qwen3-1.7B (ReProbe
Table 4) for a scale curve. All on the ReProbe benchmark; ProcessBench on Qwen3-8B
(others optional).

## Figures and tables

1. Fig 1: method (inline reads, mask, chain, marker, offsets, head).
2. Fig 2: headline Pareto, quality (ReProbe overall PR-AUC; ProcessBench first-error
   AUROC) against added cost (trainable params; ms per trajectory). Every comparison row.
3. Table 1: ReProbe benchmark, Qwen3-8B, 8 sets + ID/OOD/overall, all rows, paired deltas.
4. Fig 3: backbones, overall PR-AUC per backbone for passive / ReProbe / InStep / best PRM.
5. Fig 4: the ladder with CIs (ReProbe overall PR-AUC; ProcessBench).
6. Table 2: ProcessBench per subset, threshold-free + label-free F1; published PRM column.
7. Table 3: cost (params, extra memory, ms per trajectory b1/b10, ms per decode step).
8. Fig 5: where the signal forms, per-layer separability of the read token against the
   last step token (inline states).
9. Fig 6 (compact): use. Abstention risk-coverage, beam search vs random, best-of-N vs
   majority (honest null).
10. Appendix: calib-20, confidence-wrong limit, released ReProbe attends to future steps,
    ReProbe Table 5 reproducibility notes, seeds and the lr rule.

## Gaps, ReProbe-first priority (superseded by the V1 run list below; kept for history)

| # | run | feeds | cost |
|---|---|---|---|
| G1 | PRM rows on Qwen3-8B's 8 sets from the scores stored in ReProbe's managers | Table 1, Fig 2 | CPU |
| G2 | passive linear (layer on val) + token confidence + zero-shot P(No) on the 8 sets, 3 backbones | Table 1, Fig 2-4 | one forward per set |
| G3 | ladder on ReProbe data: InStep without offsets, without chain, without marker, 3 seeds each | Fig 4 | ~12 min per run |
| G4 | threshold-free ProcessBench metrics for every arm already scored | Table 2 | CPU |
| G5 | weak PRM on ProcessBench (and its step scores where needed) | Table 2, Fig 2 | one scoring pass |
| G6 | divergence rule fixed, reseed where it applies | all | small |
| G7 | label-free ProcessBench F1 settled | Table 2 | analysis |
| G8 | inline per-layer separability store | Fig 5 | store + CPU |
| G9 | optional: Qwen3-1.7B; ProcessBench on Phi-4 / Qwen3-32B | Fig 3 | medium |
| G10 | literature refresh | related work | none |

## V1 run list (2026-11-03, revised twice)

Rule (user): on ReProbe's task and test sets, the numbers printed in the ReProbe paper
(arXiv 2511.06209 v5; Table 1 Qwen3-8B, Table 5 Qwen3-32B, Table 9 Phi-4) are cited as given
for every method it reports. We run what the paper does not have, and we fill the gaps where
the paper is incomplete or does not match its own released files.

### Two tiers of comparators

* Tier A (full metrics: PR-AUC, AUROC, F1 at the validation threshold, oracle F1, always-positive
  F1, paired trace bootstrap): InStep, ReProbe (released + matched retrain), strong PRM
  (Qwen2.5-Math-PRM-7B), weak PRM (Qwen2.5-Math-7B-PRM800K), passive linear, MaxProb (best
  token-confidence row in Tables 1 and 9), P(True) (Qwen3-8B). These need step scores on test
  AND validation, which the paper does not give.
* Tier B (PR-AUC as printed, no interval): every other row of Tables 1, 5, 9.

### What the paper lacks, and how V1 fills it

| gap in the paper | V1 fill |
|---|---|
| PR-AUC only; no AUROC, no thresholded F1, no seeds or intervals | Tier A step scores on test + val, F1 at the val threshold, paired bootstrap |
| no ProcessBench, no human labels | every Tier A method on ProcessBench |
| Table 5 GSM8K PRM column disagrees with the released scores (.930 vs .552); two column pairs swapped; traces are non-thinking although the text says native thinking | PRMs re-scored by us on the released 32B traces; table prints both numbers with a footnote |
| Table 9 planning columns on traces that are not released | PRMs scored by us on the released Phi-4 planning traces; headline Phi-4 comparison on the five released sets |
| released ReProbe attends to steps after the one it scores (mask bug) | our causal retrain (RPc) reported next to it (done) |
| efficiency claimed, no latency against a PRM | cost table (done on Qwen3-8B) |
| best-of-N (Table 3) graded by an LLM judge, not reproducible programmatically | best-of-N within our harness only (done; re-scored with the new tokens) |

### Jobs

| job | what | backbones |
|---|---|---|
| E1 | score the sweep's InStep tokens (DS x3, PRM800K x3) on the 8 sets, ProcessBench, validation, best-of-N pools; abstention re-scored | Qwen3-8B |
| P1 | strong + weak PRM on all 8 test sets + validation (one protocol); gate: our test scores reproduce the stored ones where stored | all 3 |
| U1 | MaxProb (+ MaxEntropy, Perplexity) on test + validation; gate: our test scores reproduce the stored ones | all 3 |
| T1 | P(True) on test + validation | Qwen3-8B |
| B1 | passive linear (logistic, layer picked on validation) | all 3 |
| PB | ProcessBench: weak PRM, MaxProb, P(True), passive | Qwen3-8B |
| L1 | ladder on ReProbe DS data, 3 seeds: embedding-only token, + offsets, + chain, InStep without offsets; embedding-only on PRM800K | Qwen3-8B |
| S1 | per-layer InStep and passive states on validation (Fig 5) | Qwen3-8B |
| C1-C3 | CPU: paper rows (Tier B), threshold-free ProcessBench metrics, bootstraps, tables, plots | |

Reporting: PR-AUC cells for paper methods print the paper's number; where our re-score
differs by more than 0.01 the table shows ours in brackets with a footnote. F1 is reported
per set next to that set's always-positive F1 (set prevalences run from 8.6% to 59.1%, so
F1 is never compared across sets). Order: E1 first (no new code), then P1/U1/T1/B1 (one
scoring script, reproduction gate first), then PB, L1, S1, C1-C3.
