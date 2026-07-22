# parametric_retrieval_minimal_v1: what minimal subset of the edit flips the answer

Follow-up to `parametric_retrieval_component_v1` (expE), which localized the
same-fact rescue to late MLP layers (full-residual patch @ decoder layer 24
flips 46.5%; MLP-output patch @ layer 27 flips 25%; attention ~9%). This asks
the mechanistic question: given the edit `Delta = state_donor - state_recip` at
the final prompt token, **what minimal part of it, injected, still flips the
answer**, and is it a direction or specific activations / neurons?

Reuses all access_v1 + expE artifacts. One new extraction (MLP intermediate
activations) + one experiment. Outputs under `expF/`.

## Three levels of dissection

**coord (residual space, full layer 24).** Inject only the top-k coordinates of
Delta by |Delta| (vs random-k), and only Delta's projection onto the top-r
shared subspace of train-fact deltas (vs r). Distinguishes sparse specific
coordinates from a distributed / low-rank shared direction.

**neuron (MLP space, layer 27).** The MLP output is linear in the intermediate
activation g: `mlp_out = W_down @ g`, so swapping neuron i from recipient to
donor adds exactly `(g_donor_i - g_recip_i) * W_down[:, i]` to the residual.
Inject only the top-k neurons ranked by
- gradient attribution `dlogP(gold)/dg_i * dg_i` (via one backward to the
  layer-27 resid_post, then `grad . W_down[:, i]`),
- magnitude `|dg_i| * ||W_down[:, i]||`,
- random.
Answers which MLP neurons carry the flip and how few.

**greedy (minimal set).** On pairs the full edit flips, greedily forward-select
neurons from the top-attribution pool until the answer flips: the true minimal
set size, and (via recurrence) whether the same neurons recur across facts.

## Flip criterion

Cheap, no generation: the gold answer's first token becomes **rank 1 among the
32-candidate set** at the decision token, given the recipient failed at baseline
(baseline gold_rank > 1). Calibrated against the k=all reference (which equals
the expE full-MLP patch).

## Pipeline (scripts/parametric_retrieval/, prefix prgm_)

| stage | script | where | outputs |
|-------|--------|-------|---------|
| 3c | `prgm_extract_neurons.py` | TamIA 01 | neuron_states_v1/g_L{K}.safetensors + neuron_meta.parquet |
| run | `prgm_minimal.py --phase run` | TamIA 02 | expF/run.parquet, expF/greedy.parquet (sharded by pair) |
| analyze | `prgm_minimal.py --analyze` | TamIA 02 | expF/curves.csv, neuron_recurrence.csv, greedy_summary.json |
| figure | `prgm_viz.py` | local | results/parametric_retrieval_minimal_v1/prgm_v1_results.png |

New shared code in `prga_common.py`: `ResidualEdit` gains a `mode="add"`
(additive masked injection) and a `NeuronStore` (intermediate-activation
lookup). g states (~0.8 GB, one layer) redirect to `$SCRATCH`.

## Figure

`prgm_v1_results.png`, three panels:
- A. residual recovery vs #coordinates (top-|Δ| vs random vs shared-subspace rank).
- B. residual recovery vs #MLP neurons (attribution vs magnitude vs random), with the median greedy minimal-set size marked.
- C. neuron recurrence across facts (fraction of pairs each neuron is in the top set) — concentrated = shared mechanism, flat = fact-specific memories.

## Caveats

- Flip = gold rank-1 among candidates, not full greedy generation (cheaper,
  calibrated to the k=all reference).
- Greedy forward-selection is over a bounded top-attribution pool
  (GREEDY_POOL=32, max 12 steps): an upper bound on the true minimal set, not
  an exhaustive search.
- Neuron patching adds each neuron's contribution to the layer-27 resid_post;
  because down_proj is linear this is exact for the MLP-output component, but
  still propagates downstream (does not isolate the neuron from later layers).
