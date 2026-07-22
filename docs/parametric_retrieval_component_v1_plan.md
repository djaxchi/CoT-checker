# parametric_retrieval_component_v1: depth x component decomposition of the same-fact rescue

Follow-up to `parametric_retrieval_access_v1` (REPORT section 17). Experiment C
showed that transplanting a same-fact successful donor's whole residual state at
the final prompt token, at one of three hand-picked layers {24, 26, 28},
causally rescues ~45% of failures and is fact-specific. Two questions it left
open:

1. **Depth.** Only three layers were swept. Where along the 28-layer stack does
   the fact-specific rescue actually live?
2. **Component.** The residual state is the sum of every layer's attention and
   MLP contributions. Which sub-block carries the rescuing content: the
   attention output, or the MLP output?

Both are answered in one extraction + one experiment, reusing all access_v1
artifacts (no re-generation, no new prompts).

## Layer / component axis

Decoder layer L in 0 .. num_layers-1. At layer L we can patch three quantities
at the final prompt token, using the same-fact donor:

| mode | quantity patched | source of donor vector | edit object |
|------|------------------|------------------------|-------------|
| full | resid_post of layer L | HSStore (hs_idx L+1) | ResidualEdit |
| attn | self_attn output of layer L | ComponentStore attn_L | ComponentEdit |
| mlp  | mlp output of layer L | ComponentStore mlp_L | ComponentEdit |

Qwen2DecoderLayer adds `self_attn(ln1(h))` then `mlp(ln2(h))` into the residual,
so hooking those submodules edits exactly the additive contribution of that
component.

**Caveat, stated up front:** patching one component at layer L still perturbs
everything downstream, so attn + mlp do NOT sum to full. This is component
patching for localization, not a clean additive decomposition. If a component
dominates, a path-patching follow-up would be the airtight version.

## Pipeline (scripts/parametric_retrieval/, prefix prgc_)

| stage | script | where | outputs |
|-------|--------|-------|---------|
| 3b | `prgc_extract_components.py` | TamIA 01 | component_states_v1/{attn,mlp}_L{K}.safetensors + comp_meta.parquet |
| baseline | `prgc_component.py --phase baseline` | TamIA 02 | expE/baseline.parquet (unedited scores per prompt) |
| val | `prgc_component.py --phase val` | TamIA 02 | expE/val.parquet (matched+noop, all mode x layer cells) |
| select | `prgc_component.py --select` | TamIA 02 | expE/depth_curve.csv, expE/selection.json |
| test | `prgc_component.py --phase test` | TamIA 02 | expE/test.parquet (control battery at selected layers) |
| analyze | `prgc_component.py --analyze` | TamIA 02 | expE/results.csv |
| capture | `prgc_component.py --phase capture` | TamIA 02 | expE/capture.parquet (readout vectors for the figure) |
| figure | `prgc_viz.py` | local | results/parametric_retrieval_component_v1/prgc_v1_results.png |

Component states (~8 GB) redirect to `$SCRATCH/prga_access_v1/component_states_v1`
via symlink. Sharding: baseline/capture by instance; val/test by (mode, layer)
cell so forward batches stay homogeneous. alpha fixed at 1.0. Layer/selection on
val only; test facts scored once.

## Design decisions

- **Two-stage to keep val cheap.** The val phase runs only matched + noop across
  all 84 (mode x layer) cells to get the three depth curves. The full six-way
  control battery (noop / mismatched_type / mismatched_rand / random_noise /
  reverse) runs only at each mode's val-selected best layer, in test.
- **Controls identical to Experiment C** so the numbers are directly comparable
  (reuses `assign_patch_donors`, `budget_pairs`, `fact_bootstrap_ci`).
- **Capture for the spatial figure.** At each mode's selected layer we record the
  final-layer residual readout of every captured test pair under: fail (no
  edit), success donor (no edit), matched patch, mismatched patch, random patch.
  The figure projects these onto an LDA "retrieval axis" fit on fail-vs-success
  to show the failed states migrating into the retrieved region only under the
  matched patch.

## Figure (the deliverable)

`prgc_v1_results.png`, three panels:
- A. matched d_margin vs decoder layer for full / attn / mlp (depth + component).
- B. representation space: fail cloud, success cloud, and where the same failed
  prompts land once patched (matched vs mismatched vs random), with a centroid
  arrow.
- C. control battery at each component's best layer.
