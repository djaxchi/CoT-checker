# S3 Stage-2: ablate the probe readout over LAYERS (token positions deferred)

> Status: planned, not started. Parked while exploring another idea first.

## Context

Stage-1 concluded that, at the **last layer / last token of each step**, the 7B
ProcessBench signal has no discrete failure-mode taxonomy, no first-error clusters,
and first-error steps are not globally distinguishable from correct steps; the only
robust axis is step locality. Every one of those conclusions is conditioned on a
single readout choice. Before accepting "the representation lacks failure-mode
structure," we must ablate that choice. This stage does the **layer** half: a
mid-layer or earlier-depth residual stream may carry error/failure structure that
the final layer has already collapsed into a next-token prediction.

Decision (confirmed with user): **staged**. Do the layer sweep first because the
data already exists; use its result to target the (expensive, GPU) token-position
extraction later. Token-position aggregations will be designed in a follow-up.

## What already exists (reuse, do not rebuild)

- **Per-layer encodings on TAMIA, not fetched.** `encode_processbench_multilayer.py`
  already saved 10 depths (fracs 0.1-1.0 -> layer idx **3,6,8,11,14,17,20,22,25,28**)
  for all 4 subsets at `runs/.../qwen2_5_7b/pb_multilayer/<subset>/`:
  `pb_step_L{idx}_h.npy` (n_steps, 3584) float16, `pb_step_meta.jsonl`
  (id, step_idx, label, n_steps), `pb_multilayer_manifest.json` (frac->layer map).
  Same last-token-of-step position as the deployed probe, so this is a clean
  layer-only ablation. ~1.8G total.
- **Fetch helper** `scripts/fetch_s1ms_7b.sh` already has the opt-in:
  `WITH_PB_MULTILAYER=1`.
- **Loader** `src/data/processbench_probe_data.py` joins per-step tables by
  `(trace_id, step_idx)`; `compute_scores`, `is_first_error` logic, `_read_jsonl`,
  `_load_pb_text` are reusable.
- **Stage-1 analysis metrics** to reuse verbatim per layer:
  `classifiability()` (full-dim balanced 5-fold logreg) from
  `scripts/analysis/s3_failure_mode_scatter.py`; the subset-classifiability +
  decorrelation pattern from `s3_first_error_decorrelated.py`.
- **Existing per-layer F1_PB** from S1: `scripts/s1ms_layer_sweep.py` (commit
  568222e). Overlay its numbers if present; do not recompute F1_PB.
- The **200 verified tags** `results/s3_first_error/failure_labels_final.jsonl`
  (join by `sample_id = {trace_id}#s{step_idx}`).

## Plan

### 1. Fetch the per-layer states (user runs; ProcessBench-only, ~1.8G)
```
WITH_PB_MULTILAYER=1 scripts/fetch_s1ms_7b.sh
```
Lands `pb_multilayer/<subset>/...` under the existing run dir. `results/` and the
run dir are gitignored, so nothing here gets committed.

### 2. Extend the loader (TDD)
Add to `src/data/processbench_probe_data.py`:
- `pb_multilayer_layers(run_dir) -> dict[float,int]` reading the manifest.
- `load_multilayer(run_dir, layer_idx, subsets=SUBSETS, *, with_detection=True)
  -> ProbeStepData`: read `pb_multilayer/<subset>/pb_step_L{idx}_h.npy` +
  `pb_step_meta.jsonl`; derive `subset, trace_id, step_idx, n_steps,
  gold_first_error, is_first_error, hidden`. Join `pred_first_error` from the
  eval-shard `predictions.jsonl` by `id` so detection labels match the deployed
  probe. `score` left as NaN (probe is last-layer only; analyses use detection
  labels + geometry, not per-layer score). **Join by `(trace_id, step_idx)`** since
  multilayer rows never skip and may not align by index with the eval shards.
- New test `tests/data/test_processbench_probe_data.py::test_load_multilayer` with
  a tiny synthetic `pb_multilayer` dir (2 layers, 1-2 traces): asserts shape,
  `is_first_error`, and key-based alignment. No model/HF.

### 3. New analysis: `scripts/analysis/s3_layer_sweep.py`
For each layer in the manifest, compute four depth-resolved metrics (the Stage-1
findings, re-asked per layer) on the full 3,584 dims (no PCA), then plot vs depth:
- **M1 Failure-mode decodability** (the 200 tagged): `classifiability()` balanced
  5-fold logreg accuracy vs majority floor. "Does any layer encode the taxonomy?"
- **M2 First-error separability** (all steps): balanced logreg first-error vs
  correct, 5-fold ROC-AUC (+ balanced acc). "Does any layer separate errors?"
- **M3 Subset-classifiability** + **M4 step-position R²** (ridge on hidden ->
  step_frac, CV R²): the confounds. "Is step locality / topic dominant at every
  depth, or only late?"
Run `--space raw` and `--space decorrelated` (per the ablation-design memory).
Outputs (gitignored): `results/s3_first_error/layer_sweep_metrics.csv` and
`layer_sweep.png` + `.html` (4 curves over depth, with the S1 F1_PB overlay if
available). Cost guard: full-dim logreg over ~25k x 10 layers; use lbfgs +
`class_weight=balanced`, optionally cap the correct-step count for M2/M3.

### 4. Interpret + record (after results)
- If a mid/early layer lifts M1 or M2 meaningfully above the last layer -> Stage-1's
  null was a readout artifact; that layer becomes the target for the token-position
  extraction and for re-running `s3_project_all.py` / `s3_cluster_failures.py`.
- If every layer tracks M3/M4 and M1/M2 stay near floor -> the null is robust to
  depth, strengthening "incorrectness is diffuse."
- Append a "Stage-2 (layers)" block to `sprint.md`; update memory
  `project_s3_failure_labeling.md`. Commit `[eval]` (no co-author) when asked.

### Deferred (next stage): token positions
New GPU extraction on TAMIA storing within-step aggregations (mean/max/first/last
+ possibly delimiter token), at the layer(s) flagged by step 4. Designed once layer
results are in; not part of this plan.

## Verification
- `python -m pytest tests/data/test_processbench_probe_data.py -q` (new test green;
  existing 3 still pass).
- After fetch: `python -c "from src.data.processbench_probe_data import
  pb_multilayer_layers, load_multilayer; print(pb_multilayer_layers('runs/s1_model_size_dense/qwen2_5_7b'));
  d=load_multilayer('runs/s1_model_size_dense/qwen2_5_7b', 28); print(d.hidden.shape, d.is_first_error.sum())"`
  — layer 28 (last) should reproduce Stage-1 counts (~25.7k steps, 2221 first-errors)
  and its geometry should match the eval-shard last layer as a sanity check.
- `python scripts/analysis/s3_layer_sweep.py` -> inspect `layer_sweep.png`: read off
  whether M1/M2 peak away from the last layer.
