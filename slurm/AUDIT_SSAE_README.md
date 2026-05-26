# SSAE audit jobs

Two independent Slurm jobs that audit two distinct hypotheses for why our
SSAE results on full ProcessBench were poor:

| # | Hypothesis | Job |
|---|---|---|
| 1 | Our training recipe is the bug; the SSAE method itself works | `slurm/audit_ssae_original_paper_ckpt_tamia.sh` — evaluate the original Miaow-Lab/SSAE paper checkpoint (Qwen2.5-0.5B) end-to-end. |
| 2 | Our SSAE never converged because of undertraining; longer training + DWA should recover signal | `slurm/audit_ssae_mixed_dwa_long_tamia.sh` — re-train `ssae_mixed` on Qwen2.5-1.5B with `LR=1e-4`, `MAX_ITERS=3000`, and the paper's DWA controller restored. |

Each job is one self-contained 4×H100 allocation. The two jobs are
independent and can be submitted in parallel.

## What's added or changed in the repo

| Path | Status | Purpose |
|---|---|---|
| `scripts/audit_load_paper_ssae_checkpoint.py` | **new** | Converts a downloaded Miaow-Lab/SSAE checkpoint into a `QwenSSAE` state_dict (strips phase-2/3 modules, validates vocab size, writes `ssae_model.pt` + `load_manifest.json`). |
| `scripts/train_ssae_official.py` | **patched** | Adds `DWAController` and CLI flags `--use_dwa --l1_target --dwa_update_interval --dwa_min_w --dwa_max_w --dwa_alpha`. Inner loop now feeds the per-micro-step sparsity loss to the controller and uses its current weight for the next batch. Per-iter log line and `train_metrics_history` carry the live `dwa_l1_weight` and rolling `dwa_l1_avg`. |
| `scripts/run_ssae_method.py` | **patched** | Forwards `--use_dwa --l1_target --dwa_update_interval` to the trainer. Registers `ssae_mixed_dwa_lr1e-4_iter3000`. |
| `scripts/extract_ssae_latents.py` | **patched** | Adds `--shard_idx --num_shards`. When `num_shards > 1`, slices `SSAEJsonlDataset.rows` by `global_step_index % num_shards`, writes shard outputs under `out_dir/shards/shard_NN/`, and runs a fail-fast finite check on every produced array. |
| `scripts/build_full_processbench_eval_jobs.py` | **patched** | `METHOD_SPECS` registers the two audit method names. |
| `slurm/audit_ssae_original_paper_ckpt_tamia.sh` | **new** | Job 1. |
| `slurm/audit_ssae_mixed_dwa_long_tamia.sh` | **new** | Job 2. |

The merge step reuses the existing `scripts/merge_processbench_encoded_shards.py`
(generic over `--array_name` / `--meta_name` and sorts by `global_step_index`).

## Prerequisites for Job 1 (compute nodes have NO internet)

TamIA compute nodes cannot reach `huggingface.co` once a job is allocated.
**Both** of the following must be downloaded on a login node before
submitting Job 1, and **both** must land in the HF cache directory that
the slurm script exports as `$HF_HOME` (default `/scratch/d/dchikhi/hf_cache`).

### Prerequisite 0: the Qwen2.5-0.5B base model

Your existing runs use Qwen2.5-1.5B, which is already cached. The paper
checkpoint is tied to Qwen2.5-**0.5B**, which probably is **not** in your
cache yet. `QwenSSAE.__init__` calls `AutoModel.from_pretrained(..., local_files_only=True)`
three times (encoder / hints_encoder / decoder) plus
`AutoTokenizer.from_pretrained(..., local_files_only=True)`; all four
fail offline if the snapshot is missing.

Predownload it on a login node:

```bash
export HF_HOME=/scratch/d/dchikhi/hf_cache    # same cache the job uses
huggingface-cli download Qwen/Qwen2.5-0.5B

# Verify the snapshot is complete (the slurm script also checks this
# before allocating GPU work):
ls -ld $HF_HOME/hub/models--Qwen--Qwen2.5-0.5B/snapshots/*/config.json
ls -ld $HF_HOME/hub/models--Qwen--Qwen2.5-0.5B/snapshots/*/tokenizer.json
```

Both `ls` commands must print a real path. If either fails, the snapshot
is incomplete and you should rerun the download.

Job 2 reuses Qwen2.5-1.5B and does not need this step.

### Prerequisite 1: the Miaow-Lab/SSAE paper checkpoint

Pick one of the phase-1 checkpoints from
<https://huggingface.co/Miaow-Lab/SSAE-Checkpoints>.

### Step 1: download on a login node

`huggingface-cli download` writes into a **directory** (the file you pick
plus any metadata files), not directly to a `.pt` path. So:

```bash
# On a login node with internet access:
PAPER_CKPT_DIR=$SCRATCH/cot_mech/prestudy_v1/paper_ckpts/miaow_ssae_phase1
mkdir -p "$PAPER_CKPT_DIR"

# 1a. List what's in the repo to pick one phase-1 checkpoint:
huggingface-cli repo files Miaow-Lab/SSAE-Checkpoints | grep -i phase

# 1b. Download exactly the phase-1 file you chose (replace <FILE> with the
#     real file name from the listing above):
huggingface-cli download Miaow-Lab/SSAE-Checkpoints <FILE> \
  --local-dir "$PAPER_CKPT_DIR"
```

### Step 2: verify the actual on-disk path

Before submitting, list everything that landed under the download dir:

```bash
find $SCRATCH/cot_mech/prestudy_v1/paper_ckpts -maxdepth 3 -type f | sort
```

You'll typically see something like:

```
.../paper_ckpts/miaow_ssae_phase1/.gitattributes
.../paper_ckpts/miaow_ssae_phase1/<actual-checkpoint>.pt
```

Note the **actual** checkpoint file. `$PAPER_CKPT` may be either:

- the exact `.pt` / `.bin` / `.safetensors` file path, **or**
- a directory that contains **exactly one** such file (the loader's
  `resolve_ckpt_file` helper will pick it up automatically and refuse if
  there are multiple).

So both of the following forms are valid:

```bash
# Form A: point at the file directly (recommended; unambiguous)
PAPER_CKPT=$SCRATCH/cot_mech/prestudy_v1/paper_ckpts/miaow_ssae_phase1/<actual-file>.pt

# Form B: point at a clean directory with exactly one checkpoint file
PAPER_CKPT=$SCRATCH/cot_mech/prestudy_v1/paper_ckpts/miaow_ssae_phase1
```

**Do not submit Job 1 until `find` confirms exactly what file is at the path
you'll export.** The previous version of this README contained a misleading
example that put `.pt` directly on the `--local-dir` path; that does not
match what `huggingface-cli download` actually produces.

## Submitting both jobs in parallel

```bash
cd ~/CoT-checker
export HF_HOME=/scratch/d/dchikhi/hf_cache

# 0a. Confirm Qwen2.5-0.5B is in the offline HF cache (Job 1 prerequisite):
ls -ld "$HF_HOME"/hub/models--Qwen--Qwen2.5-0.5B/snapshots/*/config.json
ls -ld "$HF_HOME"/hub/models--Qwen--Qwen2.5-0.5B/snapshots/*/tokenizer.json
# Both must resolve. If not, download on a login node:
#   huggingface-cli download Qwen/Qwen2.5-0.5B

# 0b. Confirm the paper checkpoint path BEFORE Job 1 submission:
find $SCRATCH/cot_mech/prestudy_v1/paper_ckpts -maxdepth 3 -type f | sort

# 1. Job 1: original paper checkpoint audit (Qwen2.5-0.5B).
#    Replace the placeholder with the exact path printed by `find` above.
PAPER_CKPT=$SCRATCH/cot_mech/prestudy_v1/paper_ckpts/miaow_ssae_phase1/<actual-file>.pt \
  sbatch slurm/audit_ssae_original_paper_ckpt_tamia.sh

# 2. Job 2: corrected DWA long-run audit (Qwen2.5-1.5B).
#    Reuses the cached 1.5B model; no extra download needed.
sbatch slurm/audit_ssae_mixed_dwa_long_tamia.sh

# 3. Track both:
squeue -u $USER
```

Optional knobs (override at submission time):

```bash
# Job 1
ORACLE_STEP=0.005 FORCE=1 BATCH_SIZE=16 \
  PAPER_CKPT=/path/to/ckpt \
  sbatch slurm/audit_ssae_original_paper_ckpt_tamia.sh

# Job 2
MAX_ITERS=3000 LEARNING_RATE=1e-4 L1_TARGET=3.0 DWA_UPDATE_INTERVAL=100 \
  BATCH_SIZE=4 GRAD_ACCUM=32 ORACLE_STEP=0.005 FORCE=1 \
  sbatch slurm/audit_ssae_mixed_dwa_long_tamia.sh
```

## Latent file isolation (Job 1)

The probe is trained and calibrated **only on PRM800K**; ProcessBench is
strictly evaluation-only. The audit pipeline enforces this with separate
per-split files on disk:

| File written under `runs/ssae_original_paper_ckpt_qwen0p5b/latents/` | Role |
|---|---|
| `probe_train_40k_z.npy` + `probe_train_40k_y.npy` + `probe_train_40k_meta.jsonl` | PRM800K **train** — probe fitting only |
| `val_1k_z.npy` + `val_1k_y.npy` + `val_1k_meta.jsonl` | PRM800K **val_1k** — threshold selection only |
| `pb_gsm8k_step_z.npy` + `pb_gsm8k_step_meta.jsonl` | ProcessBench-GSM8K **eval only**, used for the in-script single-subset diagnostic F1 |

Then under `runs/ssae_original_paper_ckpt_qwen0p5b/latents_full_pb/`:

| Path | Role |
|---|---|
| `gsm8k/pb_step_z.npy` + `pb_step_meta.jsonl` | PB-GSM8K subset, eval only |
| `math/pb_step_z.npy` + `pb_step_meta.jsonl` | PB-Math subset, eval only |
| `olympiadbench/pb_step_z.npy` + `pb_step_meta.jsonl` | PB-OlympiadBench subset, eval only |
| `omnimath/pb_step_z.npy` + `pb_step_meta.jsonl` | PB-OmniMath subset, eval only |
| `combined/pb_step_z.npy` + `pb_step_meta.jsonl` | Namespaced pool of the 4 subsets, eval only |

`scripts/train_eval_ssae_probe.py:73-74` loads **only** `probe_train_40k_*`
for training and **only** `val_1k_*` for threshold selection
(`select_threshold(val_scores, y_val)` on line 116). The `pb_gsm8k_step_*`
files are read into a separate variable (`pb_z`, `pb_meta`) on lines 83-89
and used only on line 131 to compute the single-subset diagnostic, after
both the probe and the threshold have already been fixed from PRM800K.
The full PB latents (the four `latents_full_pb/*` subsets plus the
namespaced `combined`) are scored later by
`evaluate_existing_probes_full_processbench_worker.py` using that same
fixed probe + fixed threshold. No PB row ever lands in
`probe_train_40k_*` or `val_1k_*`.

The shard merger
(`scripts/merge_processbench_encoded_shards.py`) operates **per split**
(invoked once per `--array_name`), so the 4-way fan-out never mixes split
contents either.

## Output locations

| Job | Run dir | Leaderboards |
|---|---|---|
| 1 | `$SCRATCH/cot_mech/prestudy_v1/runs/ssae_original_paper_ckpt_qwen0p5b/` | `$SCRATCH/cot_mech/prestudy_v1/runs/full_processbench_eval_audit_original_paper_ckpt_qwen0p5b/leaderboard_full_pb_*.md` |
| 2 | `$SCRATCH/cot_mech/prestudy_v1/runs/ssae_mixed_dwa_lr1e-4_iter3000/` | `$SCRATCH/cot_mech/prestudy_v1/runs/full_processbench_eval_ssae_mixed_dwa_lr1e-4_iter3000/leaderboard_full_pb_*.md` |

The audit jobs **do not touch** the original
`$SCRATCH/cot_mech/prestudy_v1/runs/full_processbench_eval/` leaderboards.

Job 1 PB latents live under
`$SCRATCH/cot_mech/prestudy_v1/cache/qwen2_5_0_5b_processbench_full_ssae_original_paper_ckpt`
(not literally, but conceptually — they sit at
`$RUNS_DIR/ssae_original_paper_ckpt_qwen0p5b/latents_full_pb/` because that
is what the existing build-jobs script expects; a symlink under the cache
root is created so the documented path resolves).

Job 2 PB latents live under
`$SCRATCH/cot_mech/prestudy_v1/runs/ssae_mixed_dwa_lr1e-4_iter3000/latents_full_pb/`
with a symlink at
`$SCRATCH/cot_mech/prestudy_v1/cache/qwen2_5_1_5b_processbench_full_ssae_mixed_dwa_lr1e-4_iter3000/latents`.

## Post-run: print the comparison rows

After both jobs finish, run this on a login node (or interactive shell):

```bash
RUN_ROOT=$SCRATCH/cot_mech/prestudy_v1
MAIN=$RUN_ROOT/runs/full_processbench_eval
A1=$RUN_ROOT/runs/full_processbench_eval_audit_original_paper_ckpt_qwen0p5b
A2=$RUN_ROOT/runs/full_processbench_eval_ssae_mixed_dwa_lr1e-4_iter3000

# Rows we care about (audited methods + reference rows from the main run).
ROW_REGEX='dense_linear|sae_mixed|sae_contrastive|ssae_positive|ssae_mixed|ssae_contrastive|ssae_original_paper_ckpt_qwen0p5b|ssae_mixed_dwa_lr1e-4_iter3000'

echo "===== val-selected leaderboard (deployable) ====="
echo "--- main run ---"
grep -E "^\| (method|---|$ROW_REGEX)" "$MAIN/leaderboard_full_pb_val_threshold.md" 2>/dev/null
echo "--- audit job 1 (paper ckpt, Qwen0.5B) ---"
grep -E "^\| (method|---|$ROW_REGEX)" "$A1/leaderboard_full_pb_val_threshold.md" 2>/dev/null
echo "--- audit job 2 (DWA long-run, Qwen1.5B) ---"
grep -E "^\| (method|---|$ROW_REGEX)" "$A2/leaderboard_full_pb_val_threshold.md" 2>/dev/null

echo
echo "===== oracle leaderboard (diagnostic ceiling) ====="
echo "--- main run ---"
grep -E "^\| (method|---|$ROW_REGEX)" "$MAIN/leaderboard_full_pb_oracle_threshold.md" 2>/dev/null
echo "--- audit job 1 ---"
grep -E "^\| (method|---|$ROW_REGEX)" "$A1/leaderboard_full_pb_oracle_threshold.md" 2>/dev/null
echo "--- audit job 2 ---"
grep -E "^\| (method|---|$ROW_REGEX)" "$A2/leaderboard_full_pb_oracle_threshold.md" 2>/dev/null

echo
echo "===== method averages (macro across 4 subsets) ====="
echo "--- main run ---"
grep -E "^\| (method|---|$ROW_REGEX)" "$MAIN/leaderboard_full_pb_method_averages.md" 2>/dev/null
echo "--- audit job 1 ---"
grep -E "^\| (method|---|$ROW_REGEX)" "$A1/leaderboard_full_pb_method_averages.md" 2>/dev/null
echo "--- audit job 2 ---"
grep -E "^\| (method|---|$ROW_REGEX)" "$A2/leaderboard_full_pb_method_averages.md" 2>/dev/null
```

DWA convergence sanity check (Job 2 only):

```bash
RUN_DIR=$SCRATCH/cot_mech/prestudy_v1/runs/ssae_mixed_dwa_lr1e-4_iter3000
# Per-iter mean L1 vs target=3.0 and the live DWA-controlled weight:
python - <<'PY'
import json, os
from pathlib import Path
run = Path(os.environ["RUN_DIR"])
hist = json.loads((run / "ssae_train_metrics.json").read_text())["train"]
print(f"{'iter':>5} {'loss_nll':>10} {'loss_spa':>10} {'dwa_l1_w':>12} {'dwa_l1_avg':>12}")
for r in hist[::max(1, len(hist)//30)]:
    print(f"{r['iter']:>5} {r['loss_nll']:>10.4f} {r['loss_spa']:>10.4f} "
          f"{r['dwa_l1_weight']:>12.4e} {str(r.get('dwa_l1_avg', '')):>12}")
PY
```

## Decision logic

Read both audit leaderboards together. The two jobs are designed to isolate
the source of the previous failure.

| Observation | Interpretation |
|---|---|
| **Job 1 oracle macro F1_PB > ~0.30** | The original paper SSAE method works on PB. Our previous SSAE runs failed because of **our training recipe** (undertraining and missing DWA), not the method. |
| **Job 1 oracle macro F1_PB ≈ random (~0.17)** | Either the SSAE bottleneck doesn't transfer to ProcessBench, or the 0.5B-vs-1.5B mismatch is doing the harm. Note this is **not conclusive** because the paper checkpoint uses Qwen2.5-0.5B (hidden 896), so the dense_linear comparison row should be a 0.5B dense probe, which we have not trained. |
| **Job 2 oracle macro F1_PB jumps to ≥ 0.30** | The previous SSAE failure was mostly undertraining. Longer training + DWA recovers first-error signal in the latent. |
| **Job 2 oracle macro F1_PB stays ~0.16-0.18** | Longer training + DWA did not recover ProcessBench first-error signal. The SSAE bottleneck does not encode first-error semantics for our PRM800K + ProcessBench setup. |
| **Job 2 oracle improves but val_selected F1_PB remains poor** | The signal exists in the latent, but PRM800K val cannot calibrate a deployable threshold (same gap we already saw for `sae_mixed`). |
| **Job 2 val_selected also improves** | Corrected SSAE training gives a deployable improvement; the result is publishable as "SSAE with DWA + sufficient iters approaches dense_linear". |

For Job 2 specifically, also inspect the DWA convergence log: if `dwa_l1_avg`
stays far from `L1_TARGET=3.0` even after 3 000 iters, the latent code never
converged on a useful sparsity regime and the F1_PB result should be read
with that caveat (the controller may need a different target, alpha, or
update interval).

## Caveats

- **Apples-to-apples disclaimer for Job 1:** the paper checkpoint is
  Qwen2.5-0.5B (hidden 896). The `dense_linear` baseline we compare against
  is Qwen2.5-1.5B (hidden 1536). A fair comparison would require a 0.5B
  dense probe baseline, which we do not have and is out of scope for this
  audit. Treat Job 1 numbers as evidence about whether the **method**
  works, not whether **our 1.5B SSAE pipeline** beats the dense baseline.
- **Memory:** Job 2 trains 3 Qwen2.5-1.5B copies (encoder/hints_encoder/
  decoder). The existing slurm script's `BATCH_SIZE=4 × GRAD_ACCUM=32` with
  `--gradient_checkpointing` is the tested setting; if you raise
  `BATCH_SIZE`, halve `GRAD_ACCUM` to keep the effective global batch at
  128.
- **Wall time:** Job 2's 3 000 iter run at the documented batch size has
  not been timed end-to-end here. The `--time=24:00:00` allocation is a
  generous upper bound; tune after the first run.
- **No nested `sbatch`:** both audit slurm scripts run the full pipeline
  inline (no `sbatch`-from-`sbatch`). Each is one allocation.
- **No overwrite:** all audit outputs land under unique directories
  (`*_audit_*` / `*_ssae_mixed_dwa_*`). The original leaderboards in
  `runs/full_processbench_eval/` are not touched.
