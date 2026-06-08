# S1 DenseLinear Model-Size Ablation

Scales the Sprint 1 **DenseLinear** baseline across Qwen2.5 backbones
(1.5B → 3B → 7B → 14B → 32B, plus optional 0.5B anchor) while holding the S1
protocol fixed. The only variable is the backbone; everything else (data,
split, probe, metric, thresholding) is the Sprint 1 pipeline.

Reference row to reproduce first (1.5B smoke gate):

| threshold | macro F1_PB |
|---|---|
| PRM800K val-selected | ~0.1855 |
| per-subset oracle (0.005 grid) | ~0.3773 |

If 1.5B does not reproduce this within tolerance, the sweep stops itself.

---

## Protocol (identical to Sprint 1)

| Component | Choice |
|---|---|
| Method | DenseLinear: linear probe on the final-layer hidden state at the **last token of the current step** |
| Representation training | none (raw hidden state) |
| Probe supervision | PRM800K, **40K train / 1K val** (the frozen S1 split, reused verbatim) |
| Evaluation | ProcessBench `gsm8k`, `math`, `olympiadbench`, `omnimath` |
| Metric | official ProcessBench first-error **F1_PB**; macro = unweighted mean of the 4 subsets |
| Score convention | `s = sigmoid(probe_logit) = P(step is first-error)`; trace pred = first step with `s > t`, else `-1` |
| Thresholds | val-selected (PRM800K val balanced-accuracy grid 0.1..1.0) and per-subset oracle (0.005 grid, 199 points) |
| Probe dim | **inferred** from `model.config.hidden_size` / array shape (never hardcoded) |
| dtype | float16 inference + storage (matches S1; reproduction-safe) |
| Seed | 42 |

### No-truncation invariant (hard constraint)

Every step is encoded conditioned on `question + all previous reasoning steps +
current step`. There is **no `max_seq_len=2048`**: encoders run with
`--max_seq_len -1`, which resolves **per model** to that model's
`max_position_embeddings` (131072 for 1.5B / 7B / 14B / 32B; **32768 for 3B**),
and **fail loudly** on any sequence that would exceed it (PB encoder uses
`--fail_on_overlength`; PRM encoder fails by construction), logging the example
id, subset, step index, and token length. Before any GPU work,
`s1ms_audit_token_lengths.py` tokenizes every PRM800K and ProcessBench example,
reports max/mean/p95/p99 per dataset, and asserts `num_truncated_examples == 0`.

**Qwen2.5-3B is the strictest context check** (32768 vs 131072). Its Stage A
audit is the binding constraint; if any example fits the others but not 3B, the
3B model size fails loudly and stops (the others are unaffected). PRM800K and
ProcessBench steps are far under both windows, so this is a guard, not an
expected condition.

Note: all Qwen2.5 sizes share **one identical tokenizer/vocab**, so every
backbone encodes byte-identical token sequences - the no-truncation guarantee
holds uniformly and model size is the sole variable.

---

## Execution model

Sequential **across** model sizes, parallel **within** a model size, via SLURM
`afterok` dependencies. No monolithic job - a failure never loses prior progress.

Per model, a 4-stage DAG:

```
Stage A  encode PRM800K 40K+1K on a whole 4-GPU node: 4 workers (one per GPU),
         each its deterministic shard (global_index %% 4 == g) of both splits
Stage B  merge shards + dump model_config + train probe + select threshold  (afterok: A, CPU)
Stage C  evaluate ProcessBench on a whole 4-GPU node: one subset per GPU      (afterok: B)
            GPU0=gsm8k GPU1=math GPU2=olympiadbench GPU3=omnimath
Stage D  aggregate val/oracle macro + append to leaderboard                  (afterok: C, CPU)
```

TamIA allocates h100 GPUs by whole node, so Stages A and C each request a full
4-GPU node (`--gpus-per-node=h100:4`) and fan out 4 background workers
internally. Stages B and D are CPU-only (a linear probe and a JSON rollup do not
need a GPU node).

The next model's Stage A is chained `afterok` on the current model's Stage D, so
14B/32B never start until the smaller sizes have produced leaderboard rows. The
1.5B Stage D is a **gate**: it exits non-zero if the macro F1_PB is outside
tolerance, which cancels everything downstream.

### Failure isolation & OOM

- `afterok` means a failed stage stops that model and everything after it, but
  completed smaller-model outputs are untouched (never deleted, never rerun).
- Encoding stages auto-retry on CUDA OOM, **halving the batch size** (floor 1)
  for that model/stage only - never truncating, never skipping examples. The
  failing model/stage/shard/batch/exception is logged.
- If 3B succeeds but 7B fails: keep 3B, debug 7B; the chain stops at 7B.
- If 14B/32B OOM at batch 1: the stage fails loudly (logged) rather than
  producing partial encodings.

---

## How to run (on TamIA)

```bash
cd $HOME/CoT-checker
# REQUIRED: point at the frozen S1 split dir (must contain the two JSONLs)
export PRM_SPLIT_DIR=/path/to/s1/split/dir   # prm800k_probe_train_40k.jsonl, prm800k_val_1k.jsonl
export PB_DIR=/scratch/d/dchikhi/cot-checker/processbench   # ProcessBench raw subsets
# HF cache contract has sane defaults in models.env; override HF_CACHE_ROOT if needed.

# Phase 1 - smoke test 1.5B and gate:
slurm/s1_model_size/launch_smoke_1p5b.sh        # prints the gate (Stage D) job id

# Phase 2 - after the gate job SUCCEEDS, submit the rest chained on it:
slurm/s1_model_size/launch_rest.sh <gate_jobid>

# Or submit the whole gated chain in one shot:
slurm/s1_model_size/launch_sweep.sh
```

All knobs (model ids, batch sizes, paths, gate tolerance, account, walltimes)
live in `slurm/s1_model_size/models.env` and are env-overridable.

Default starting batch sizes (auto-halve on OOM): 1.5B=32, 3B=32, 7B=16,
14B=4, 32B=1.

---

## Output structure

```
runs/s1_model_size_dense/
  <tag>/                              # qwen2_5_1_5b, qwen2_5_3b, ...
    prm800k_encode_shards/shard_0{0..3}/   # per-shard PRM h/y/meta + manifest
    merged/                                # merged probe_train_40k_* , val_1k_*
    processbench_eval_shards/<subset>/     # pb_step_h.npy, metrics.json, step_scores.jsonl, predictions.jsonl
    logs/                                  # per stage/shard logs + length_audit_shard*.json
    model_config.json
    length_audit.json
    linear_probe.pt
    threshold.json
    train_metrics.json
    val_threshold_metrics.json
    processbench_val_threshold.json
    processbench_oracle_threshold.json
    per_subset_metrics.json
  leaderboard_model_size.csv
  leaderboard_model_size_val_threshold.md
  leaderboard_model_size_oracle_threshold.md
  README.md
```

Leaderboard columns: model, params_label, hidden_size, num_hidden_layers,
num_attention_heads, num_key_value_heads, train_examples, val_examples,
val_selected_threshold, macro_f1_val_threshold, macro_f1_oracle, and per-subset
val/oracle F1_PB, plus encode/train/eval walltimes.

---

## Components

Reused S1 scripts (parameterized by backbone; minimal diffs):

- `scripts/encode_prm800k_hidden_states.py` - added deterministic example
  sharding (`--shard_idx/--num_shards`), `--max_seq_len -1` (model context),
  length-audit + model metadata in the manifest. Fails hard on overlength.
- `scripts/encode_processbench_hidden_states.py` - added `--fail_on_overlength`,
  `--max_seq_len -1`, length-audit + model metadata.
- `scripts/evaluate_saved_probe_on_processbench.py` - unchanged S1 evaluator
  (val + 0.005 oracle), called once per subset.

New ablation glue:

- `scripts/s1ms_audit_token_lengths.py` - pre-encode no-truncation audit.
- `scripts/merge_prm800k_encoded_shards.py` - reassemble sharded PRM encodings.
- `scripts/s1ms_dump_model_config.py` - `model_config.json` from HF config.
- `scripts/s1ms_train_dense_probe.py` - Stage B; **imports** the exact S1 probe +
  threshold functions from `train_easy_probe_method.py` (no reimplementation),
  decoupled from PB eval to fit the DAG.
- `scripts/s1ms_aggregate_model.py` - Stage D macro rollup + 1.5B gate.
- `scripts/s1ms_merge_leaderboard.py` - leaderboard CSV/MD.

SLURM (`slurm/s1_model_size/`): `models.env`, `_common.sh`, `stageA..D_*.sh`,
`submit_model_dag.sh`, `launch_smoke_1p5b.sh`, `launch_rest.sh`, `launch_sweep.sh`.

---

## Caveats

- **Split provenance is the reproduction lever.** `PRM_SPLIT_DIR` must hold the
  exact 40K/1K JSONLs that produced the deadline S1 row. The 1.5B gate validates
  this; if it fails, recheck the split before touching the larger models.
- **Gate tolerance** defaults to `S1MS_GATE_TOL=0.01` on both macros. Small
  drift from GPU/transformers-version float16 nondeterminism is possible; loosen
  if a genuine reproduction trips the gate.
- **float16 for all sizes** (Qwen2.5 ship bf16). This matches S1 for the 1.5B
  reproduction; switch to a bf16 path only if 32B fp16 range becomes an issue,
  and re-validate the 1.5B gate if you do.
- The older `scripts/model_size_ablation_*.py` are a **different, unrelated**
  ablation (Math-Shepherd, left-truncation, sklearn, per-step macro-F1) and are
  not used here.
