# S1 small-scale DenseLinear, ALL LAYERS COMBINED

Repeats the small-scale S1 DenseLinear experiment (`probe_train_40k` / `val_1k` ->
ProcessBench PB-F1) but combines **every layer** instead of using only the last layer.
This is the pipeline behind `runs/s1_model_size_dense/qwen2_5_7b` (oracle macro F1
**0.413**, val-selected **0.237**); we change only the representation fed to the probe.

## What is identical vs changed

Identical (reused verbatim): the frozen PRM800K split, the DenseLinear probe trainer
(`s1ms_train_dense_probe.py`), the PRM800K-val threshold selection, the PB-F1 evaluator
(`evaluate_saved_probe_on_processbench.py`), and the aggregator (`s1ms_aggregate_model.py`).

Changed:
- encode every transformer layer in one forward pass (`encode_prm800k_multilayer.py`,
  `encode_processbench_multilayer.py`) instead of the last layer only;
- `assemble_multilayer_concat.py` concatenates the per-layer vectors column-wise and
  z-scores each dimension, writing the same `{stem}_h.npy` contract. The z-score stats are
  fit once on `probe_train_40k` and reused for `val_1k` and every ProcessBench subset.

The one deviation from "byte-identical pipeline" is the z-scoring: the SGD probe trains on
raw hidden states, and concatenating layers whose norms differ by ~100x would otherwise
swamp it (standardization is what made the layer-concat win the offline AUROC bake-off).
To test sensitivity, re-run the assemble steps with `--no_standardize`.

## Run

```bash
sbatch slurm/s1_model_size_multilayer/run_multilayer_dense_7b.sh
# layer count override (default 28): NLAYERS=28 sbatch ...
```

Outputs to `runs/s1_model_size_dense/qwen2_5_7b_multilayer/`. The job prints the all-layer
vs last-layer oracle/val macro F1 delta at the end and writes `per_subset_metrics.json` in
the same format as the baseline for a direct diff.

## Caveat on expectations

The layer-combination win we measured was **+0.05 AUROC** on PRM800K, which is step-level
*detection*. PB-F1 measures first-error *localization*, which REPORT.md:422 identified as
the bottleneck. So expect some gain but not necessarily the full +0.05 to carry into PB-F1.
