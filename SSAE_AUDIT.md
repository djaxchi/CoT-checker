# SSAE Phase-1 Implementation Audit

This document tracks the audit items required by the SSAE task specification
(section 19) before launching production training.

## Files added

- `src/ssae/__init__.py`
- `src/ssae/model_qwen_ssae.py` — `QwenSSAE` (phase-1) + `Autoencoder`
- `src/ssae/dataset.py` — `SSAEJsonlDataset`, `SSAECollator`, `tokenize_row`, length audit
- `scripts/train_ssae_official.py` — DDP-ready phase-1 trainer
- `scripts/extract_ssae_latents.py` — latent extraction from JSONL
- `scripts/train_eval_ssae_probe.py` — linear probe + threshold + ProcessBench eval
- `scripts/run_ssae_method.py` — orchestrator (launches torchrun for step 1)
- `scripts/merge_ssae_leaderboard.py` — CSV + MD leaderboard
- `scripts/run_ssae_smoke.sh` — single-GPU smoke launcher
- `slurm/train_ssae_methods_tamia.sh` — production SLURM (4× H100, DDP)

## Audit checklist

| # | Item | Status | Evidence |
|---|---|---|---|
| 1 | Official-style phase-1 SSAE implemented | ✅ | `model_qwen_ssae.py::QwenSSAE` mirrors `papers/SSAE/model_qwen.py::MyModel(phase=1)`; `Autoencoder` is verbatim copy of `papers/SSAE/sentenceSAE.py::Autoencoder`. |
| 2 | Official repo inspected first | ✅ | Read `sentenceSAE.py`, `model_qwen.py`, `train.py`, `configs/train.yaml`, `dataloader.py` before coding (see chat transcript). |
| 3 | Same selected JSONL examples as easy benchmark | ✅ | `run_ssae_method.py` hard-codes the five required JSONLs under `$SCRATCH/cot_mech/prestudy_v1/data/`; no resampling. |
| 4 | No dense `.npy` cache used for SSAE training | ✅ | `train_ssae_official.py` and `extract_ssae_latents.py` ingest JSONL only via `SSAEJsonlDataset`; no `np.load(*_h.npy)` anywhere in SSAE code. |
| 5 | Token-level CE (not MSE over hidden vectors) | ✅ | `compute_loss` in `train_ssae_official.py` runs `F.cross_entropy(logits_for_labels.reshape(-1,V), input_ids.reshape(-1), reduction='none') * loss_mask`. No MSE on hidden vectors. |
| 6 | Labels ignored for ssae_positive / ssae_mixed | ✅ | `contrastive = (method == 'ssae_contrastive')`; `compute_loss` only reads `batch['labels']` when `contrastive=True`. |
| 7 | Labels used only via aux BCE for ssae_contrastive | ✅ | Aux head exists only when `contrastive=True`; loss adds `bce_weight * BCEWithLogits(aux_logit, label)` only on that branch. |
| 8 | ssae_contrastive aux head NOT used for final score | ✅ | `train_eval_ssae_probe.py` instantiates a fresh `LinearProbe(in_dim=n_latents)` and discards the aux head. The aux head is not loaded or referenced anywhere in the probe/eval pipeline. |
| 9 | Final classifier is a fresh linear probe for all 3 methods | ✅ | Same `train_eval_ssae_probe.py` flow for all three methods. |
| 10 | ProcessBench not used for training or threshold selection | ✅ | Training data comes from `prm800k_*` JSONLs only; threshold grid is searched on `prm800k_val_1k` `val_scores.npy`. ProcessBench JSONL is only loaded at evaluation time. |
| 11 | Threshold grid is 0.1..1.0 (10 points) | ✅ | Imported verbatim from `train_easy_probe_method.py::THRESHOLD_GRID = [round(0.1*i,1) for i in range(1,11)]`. |
| 12 | Official ProcessBench F1_PB metric | ✅ | Reuses `evaluate_processbench` from `train_easy_probe_method.py` (same first-error grouping, same harmonic-mean F1_PB). |
| 13 | DDP across all 4 H100 GPUs for production | ✅ | `slurm/train_ssae_methods_tamia.sh` calls `run_ssae_method.py --nproc_per_node 4` which runs `torchrun --standalone --nproc_per_node=4 train_ssae_official.py`. The trainer wraps the model in `DDP(..., find_unused_parameters=True)` (required because frozen `hints_encoder` has no grad path in phase 1). |
| 14 | Smoke mode clearly separated | ✅ | `--smoke` flag limits train/val rows; `--nproc_per_node 1` in `run_ssae_smoke.sh`. Production wave does not pass `--smoke`. |
| 15 | No internet download | ✅ | All `from_pretrained` calls accept `--local_files_only`; SLURM exports `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`. |
| 16 | No truncation | ✅ | `tokenize_row` uses `add_special_tokens=False` only (no `truncation` argument). Any row exceeding `max_seq_len` raises `DatasetLengthError` with the offending uid. `--length_audit_only` mode in `train_ssae_official.py` produces a JSON report without training. |
| 17 | All outputs under `$SCRATCH` | ✅ | SLURM script defines `RUN_ROOT="$SCRATCH/cot_mech/prestudy_v1"`; all `--out_dir` paths sit under `$RUN_ROOT/runs/`. Only code lives under `$HOME/Code/CoT-checker`. |
| 18 | Evaluation reuses easy-probe code | ✅ | `train_eval_ssae_probe.py` imports `train_linear_probe`, `probe_scores`, `select_threshold`, `evaluate_processbench`, `LinearProbe`, `THRESHOLD_GRID`, `seed_everything` from `scripts/train_easy_probe_method.py`. |

## Memory-risk notes

- **Three Qwen2.5-1.5B copies per rank** (encoder + decoder + hints_encoder).
  hints_encoder is frozen and not used in phase-1 forward but is kept to mirror
  the official architecture, per user decision #1. If OOM occurs, halt and
  report — do not silently remove hints_encoder.
- **First fallback** (per user decision #1): drop per-GPU `batch_size` from 16
  to 8 and raise `grad_accum_steps` from 8 to 16. This is exposed via the
  `BATCH_SIZE` and `GRAD_ACCUM` SLURM env vars in `train_ssae_methods_tamia.sh`.

## Deliberate deviations from the official Miaow-Lab/SSAE code

These are intentional and required by the task spec; none simplify the model.

1. **Tokenization** uses `<|step_sep|>` and our PRM800K JSONL schema
   (`problem` / `prefix` / `candidate_step`) instead of the official
   `dataloader.py::ProblemAnswerDataset` (which used `question` / `answer`
   with line-split heuristics). Approved by user decision #4.
2. **Phase 2 and Phase 3** of the official `MyModel` (mean_mlp, var_mlp,
   `sample_*`, `generate_*`) are not implemented — out of scope.
3. **Auxiliary BCE head** is added on top of the official architecture for
   ssae_contrastive only. It is never used at scoring time.
4. **DWA controller** from the official `train.py` is omitted; we use a
   fixed `l1_weight=1e-4` as specified in §10 of the task spec.
5. **`find_unused_parameters=True`** in DDP (official uses False). Required
   because hints_encoder is frozen and has no gradient path in phase-1
   forward, so DDP would otherwise error out on the unused params.

## Audit fixes applied (post-audit revision)

| Issue | Fix | File(s) |
|---|---|---|
| **B1** ProcessBench JSONL not producible | Added a builder that mirrors the same per-step expansion used by `encode_processbench_hidden_states.py` and verifies row-by-row alignment of `(id, step_idx, label, n_steps)` against `pb_gsm8k_step_meta.jsonl`. Refuses to write a misaligned file. | `scripts/build_processbench_gsm8k_jsonl.py` (new) |
| **H1** 10% attention masking missing | Implemented `train_attn_mask_ratio` (default 0.1, mirrors `papers/SSAE/train.py:474-480`). Applied to attention_mask between `sep_pos` and `val_len` during training only; disabled at val and at latent-extraction time. Recorded in metrics JSON. | `scripts/train_ssae_official.py::compute_loss`, `scripts/run_ssae_method.py` |
| **H2** `--length_audit_only` could hang ranks 1..N-1 under DDP | Audit mode now refuses to launch if `RANK` env var is set, with a clear message telling the user to drop torchrun. | `scripts/train_ssae_official.py` main entry |
| **H3** Silent partial checkpoint load | `extract_ssae_latents.py` now restricts `missing`/`unexpected` to exactly `{aux_head.weight, aux_head.bias}` and raises `RuntimeError` on anything else. | `scripts/extract_ssae_latents.py` |
| **M1** No git commit hash captured | `git rev-parse HEAD` recorded into both `config.yaml` and `ssae_train_metrics.json` (best-effort; falls back to `"unknown"`). | `scripts/train_ssae_official.py` |
| **M8** Wheelhouse uncertainty for transformers/tokenizers | Added a pre-flight script that builds an offline virtualenv and confirms `pip install --no-index transformers tokenizers` resolves. SLURM script header now points to it. | `scripts/check_ssae_deps.sh` (new), `slurm/train_ssae_methods_tamia.sh` |

## Pre-flight steps (run on TamIA, in order)

```
# 0a. Verify the offline wheelhouse can install transformers/tokenizers.
bash scripts/check_ssae_deps.sh

# 0b. Materialize processbench_gsm8k.jsonl from the same raw source used to
#     build pb_gsm8k_step_h.npy (replace RAW_PB_GSM8K with your real path).
python scripts/build_processbench_gsm8k_jsonl.py \
  --raw_file "$RAW_PB_GSM8K" \
  --pb_meta "$SCRATCH/cot_mech/prestudy_v1/cache/qwen2_5_1_5b_processbench/pb_gsm8k_step_meta.jsonl" \
  --out_jsonl "$SCRATCH/cot_mech/prestudy_v1/data/processbench_gsm8k.jsonl"

# 0c. Verify all five JSONLs exist
for f in prm800k_pos_base_20k prm800k_mixed_train_40k prm800k_probe_train_40k prm800k_val_1k processbench_gsm8k ; do
  test -s "$SCRATCH/cot_mech/prestudy_v1/data/${f}.jsonl" || echo "MISSING: $f.jsonl"
done

# 1. Length audit (no model load, just tokenization stats)
python scripts/train_ssae_official.py --length_audit_only \
  --method ssae_mixed \
  --train_jsonl "$SCRATCH/cot_mech/prestudy_v1/data/prm800k_mixed_train_40k.jsonl" \
  --val_jsonl "$SCRATCH/cot_mech/prestudy_v1/data/prm800k_val_1k.jsonl" \
  --out_dir "$SCRATCH/cot_mech/prestudy_v1/runs/length_audit" \
  --model_name_or_path Qwen/Qwen2.5-1.5B --local_files_only

# 2. Smoke (single GPU)
bash scripts/run_ssae_smoke.sh

# 3. Production (sbatch)
sbatch slurm/train_ssae_methods_tamia.sh
```
