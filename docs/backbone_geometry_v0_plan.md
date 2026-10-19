# backbone_geometry_v0: PRM against its parent, native format, and the layer axis

Frozen 2026-10-19, before any result was read.

## Questions

§21.15 found closed-form LDA on Qwen2.5-Math-PRM-7B states matching the best
trained probe (calib-20 0.667) and beating Qwen3-8B Instruct (0.575), with the
class axis far more visible (best single PC 0.79 against 0.67). That comparison
mixes three things: process supervision, a math-specialised base, and a
different model family and size. It also left a 0.04 gap to the PRM's own head
(0.701), measured under a template the head never reads.

* **Q1, process supervision.** Qwen2.5-Math-PRM-7B is initialised from
  Qwen2.5-Math-7B-Instruct. Same architecture, same tokenizer, same template,
  only the PRM training differs. Does the parent already carry the geometry?
* **Q2, format.** Under the PRM's native template, read at the `<extra_0>`
  separator its head reads, does LDA on the final-layer state reach the head?
* **Q3, layer.** Every Qwen3-8B Instruct cell reads `hidden_states[35]`. Is a
  mid layer better, as L20 was on Qwen2.5-7B (§15)?

## Design

`scripts/encode_pooled_multilayer.py` writes pooled vectors (last token, step
mean, boundary; plus the separator state and the head's P(correct) under
`prm_native`) at several `hidden_states` indices from one forward pass.
`scripts/analysis/backbone_geometry.py` fits LDA (shrinkage on val), the raw
class-mean axis and the best single PC on a fixed 60,000-row subsample of
`prm800k_probe_train_full` (seed 0), and reports PRM800K test_2k AUROC and
ProcessBench known-step AUROC and calib-20 F1_PB.

| config | model | style | hidden_states |
|---|---|---|---|
| prm_ver | Qwen2.5-Math-PRM-7B | verifier | 14, 20, 24, 27, 28 |
| parent_ver | Qwen2.5-Math-7B-Instruct | verifier | 14, 20, 24, 27, 28 |
| prm_native | Qwen2.5-Math-PRM-7B | prm_native | 14, 20, 24, 27, 28 |
| q3_ver | Qwen3-8B | verifier | 12, 18, 24, 30, 35, 36 |

Index 28 (36 for Qwen3) is the post-norm final state, the head's input.

## Gates

* prm_ver, layer 27, step mean, LDA test AUROC within 0.01 of §21.15 (0.9055)
  despite the 60k subsample. Otherwise the subsample is too small; refit larger
  before reading anything.
* prm_native head: ProcessBench calib-20 within 0.01 of 0.701.

## Prespecified readings (step-mean readout, LDA, calib-20)

* Q1: PRM minus parent at layer 27 >= +0.03: process supervision reorganises
  the representation (GO). Within +-0.01: the geometry predates PRM training.
* Q2: native separator state at layer 28 within 0.01 of the head: the §21.15
  gap was format and position, and a linear readout recovers the head (GO).
* Q3: best Qwen3-8B layer minus layer 35 >= +0.02: re-read the verifier at that
  layer (GO).

## Cost

One H100:4 node, one config per GPU, pooled vectors on node-local disk only;
geometry on the same node's CPUs. Outputs: `geometry.json` and logs in
`$SCRATCH/cot_mech/backbone_geometry_v0/`.
