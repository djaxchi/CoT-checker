#!/bin/bash
# Audit Job 1: evaluate the Miaow-Lab/SSAE-Checkpoints (Qwen2.5-0.5B) pretrained
# SSAE checkpoint on full ProcessBench with the same probe + threshold protocol
# the rest of the leaderboard uses.
#
# Pipeline (all inside one 4xH100 allocation):
#   1. Load + convert the paper checkpoint into our QwenSSAE state_dict
#      (audit_load_paper_ssae_checkpoint.py).
#   2. Encode PRM800K probe_train_40k + val_1k + pb_gsm8k via SSAE, sharded
#      across 4 GPUs (extract_ssae_latents.py --num_shards 4), then merge
#      with merge_processbench_encoded_shards.py.
#   3. Train the linear probe and select a deployable threshold on val_1k
#      (train_eval_ssae_probe.py) -- writes linear_probe.pt + threshold.json.
#   4. Encode full ProcessBench (4 subsets) via SSAE, 4-way sharded by
#      subset (extract_ssae_pb_all.py --num_shards 4 --shard_subdir).
#   5. Evaluate the probe on full PB using the existing 4-worker pipeline
#      (build_full_processbench_eval_jobs.py +
#       evaluate_existing_probes_full_processbench_worker.py +
#       merge_full_processbench_eval_results.py).
#
# CAVEAT: the paper checkpoint is Qwen2.5-0.5B (hidden 896), NOT directly
# apples-to-apples with the 1.5B dense_linear baseline. Output paths
# encode that with the suffix _qwen0p5b.
#
# Submission env (the paper checkpoint must be predownloaded; the compute
# node has no internet). PAPER_CKPT may be either an exact file path
# (.pt / .bin / .safetensors) OR a directory containing exactly one such
# file (the loader's resolve_ckpt_file picks it up). Run
#   find $SCRATCH/cot_mech/prestudy_v1/paper_ckpts -maxdepth 3 -type f
# before submitting to confirm the actual path.
#   PAPER_CKPT=/abs/path/to/<file>.pt \
#   sbatch slurm/audit_ssae_original_paper_ckpt_tamia.sh
#
# Optional env overrides:
#   METHOD_NAME=ssae_original_paper_ckpt_qwen0p5b   # already the default
#   ORACLE_STEP=0.005
#   FORCE=1

#SBATCH --job-name=audit_ssae_orig_paper
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail
module load StdEnv/2023 python/3.12

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH/cot_mech/prestudy_v1}"
RUNS_DIR="$RUN_ROOT/runs"
DATA_DIR="$RUN_ROOT/data"
PB_DIR="${PB_DIR:-/scratch/d/dchikhi/cot-checker/processbench_full}"
HF_CACHE="${HF_CACHE:-/scratch/d/dchikhi/hf_cache}"
LOG_DIR="$RUN_ROOT/logs"
METHOD_NAME="${METHOD_NAME:-ssae_original_paper_ckpt_qwen0p5b}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-0.5B}"
SPARSITY_FACTOR="${SPARSITY_FACTOR:-1}"
NUM_SHARDS=4
ORACLE_STEP="${ORACLE_STEP:-0.005}"
BATCH_SIZE="${BATCH_SIZE:-16}"

if [[ -z "${PAPER_CKPT:-}" ]]; then
  echo "[FATAL] PAPER_CKPT env var must point to the predownloaded "\
"Miaow-Lab/SSAE-Checkpoints checkpoint (compute nodes are offline)." >&2
  exit 2
fi

# Pre-flight: the compute node has NO internet. Qwen2.5-0.5B must already
# be in $HF_HOME / $TRANSFORMERS_CACHE, or the from_pretrained calls below
# will fail with "We couldn't connect to https://huggingface.co". Check
# before allocating GPU work. The HF snapshot layout is:
#   $HF_HOME/hub/models--<org>--<name>/snapshots/<sha>/<files>
_HF_HUB_DIR="$HF_HOME/hub/models--Qwen--Qwen2.5-0.5B"
_REQUIRED_FILES=("config.json" "tokenizer.json")
if [[ ! -d "$_HF_HUB_DIR" ]]; then
  echo "[FATAL] Qwen/Qwen2.5-0.5B is not in the offline HF cache." >&2
  echo "Expected: $_HF_HUB_DIR (does not exist)." >&2
  echo "Predownload it on a login node BEFORE submitting Job 1, e.g.:" >&2
  echo "  HF_HOME=$HF_HOME huggingface-cli download Qwen/Qwen2.5-0.5B" >&2
  echo "Then re-run:  ls -ld $_HF_HUB_DIR" >&2
  exit 2
fi
for _F in "${_REQUIRED_FILES[@]}"; do
  if ! find "$_HF_HUB_DIR/snapshots" -maxdepth 2 -name "$_F" -print -quit | grep -q .; then
    echo "[FATAL] $_F not found under $_HF_HUB_DIR/snapshots/*. The Qwen2.5-0.5B "\
"snapshot in the offline cache looks incomplete." >&2
    echo "Re-download on a login node: HF_HOME=$HF_HOME huggingface-cli download Qwen/Qwen2.5-0.5B" >&2
    exit 2
  fi
done
echo "[preflight] Qwen/Qwen2.5-0.5B present at $_HF_HUB_DIR"
# Accept either an exact file path or a directory containing exactly one
# checkpoint file (matching audit_load_paper_ssae_checkpoint::resolve_ckpt_file).
if [[ -f "$PAPER_CKPT" ]]; then
  echo "[PAPER_CKPT] file: $PAPER_CKPT ($(stat -c %s "$PAPER_CKPT" 2>/dev/null || echo '?') bytes)"
elif [[ -d "$PAPER_CKPT" ]]; then
  _CKPT_FILES=$(find "$PAPER_CKPT" -maxdepth 1 -type f \
                  \( -name '*.pt' -o -name '*.bin' -o -name '*.safetensors' \))
  _N=$(echo "$_CKPT_FILES" | grep -c '.' || true)
  if [[ "$_N" -ne 1 ]]; then
    echo "[FATAL] PAPER_CKPT directory $PAPER_CKPT must contain EXACTLY ONE "\
".pt/.bin/.safetensors file; found $_N:" >&2
    echo "$_CKPT_FILES" >&2
    echo "Run: find \"$PAPER_CKPT\" -maxdepth 3 -type f | sort   and pass the exact file path." >&2
    exit 2
  fi
  echo "[PAPER_CKPT] dir resolved to single file: $_CKPT_FILES"
else
  echo "[FATAL] PAPER_CKPT path does not exist: $PAPER_CKPT" >&2
  echo "Run: find \$SCRATCH/cot_mech/prestudy_v1/paper_ckpts -maxdepth 3 -type f | sort" >&2
  exit 2
fi

RUN_DIR="$RUNS_DIR/$METHOD_NAME"
LATENTS_DIR="$RUN_DIR/latents"
PB_LAT_ROOT="$RUN_DIR/latents_full_pb"
# Job-1 specific caches (kept entirely separate from the 1.5B caches)
EVAL_OUT_DIR="$RUNS_DIR/full_processbench_eval_audit_original_paper_ckpt_qwen0p5b"

mkdir -p "$RUN_DIR" "$LATENTS_DIR" "$PB_LAT_ROOT" "$EVAL_OUT_DIR" "$LOG_DIR"

export HF_HOME="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_CACHE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$PROJECT_ROOT"
GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
JID="${SLURM_JOB_ID:-$$}"
LOG_FILE="$LOG_DIR/audit_ssae_orig_paper-${JID}.log"

cat <<BANNER | tee -a "$LOG_FILE"
================================================================
job          : ${SLURM_JOB_NAME:-audit_ssae_orig_paper}
job_id       : ${JID}
hostname     : $(hostname)
date         : $(date -Iseconds)
git_commit   : $GIT_COMMIT
method_name  : $METHOD_NAME
model        : $MODEL_NAME_OR_PATH   (Qwen2.5-0.5B, hidden=896)
paper_ckpt   : $PAPER_CKPT
run_dir      : $RUN_DIR
pb_dir       : $PB_DIR
eval_out_dir : $EVAL_OUT_DIR
oracle_step  : $ORACLE_STEP
log_file     : $LOG_FILE
note         : Qwen2.5-0.5B latents are NOT directly comparable to the
               Qwen2.5-1.5B dense_linear baseline (different backbone size).
================================================================
BANNER

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch numpy tqdm transformers

# ---------- Stage 1: convert the paper checkpoint --------------------------
echo "[$(date -Iseconds)] stage 1: convert paper ckpt -> QwenSSAE state_dict" \
  | tee -a "$LOG_FILE"
python scripts/audit_load_paper_ssae_checkpoint.py \
  --raw_ckpt "$PAPER_CKPT" \
  --out_dir "$RUN_DIR" \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --sparsity_factor "$SPARSITY_FACTOR" \
  --local_files_only \
  ${FORCE:+--force} 2>&1 | tee -a "$LOG_FILE"

CKPT="$RUN_DIR/ssae_model.pt"
if [[ ! -f "$CKPT" ]]; then
  echo "[FATAL] missing $CKPT after conversion" | tee -a "$LOG_FILE"
  exit 3
fi

# ---------- Stage 2: encode PRM800K + pb_gsm8k splits (4-way shard) ---------
echo "[$(date -Iseconds)] stage 2: encode PRM800K + pb_gsm8k (4-way shard)" \
  | tee -a "$LOG_FILE"
PROBE_TRAIN_JSONL="$DATA_DIR/prm800k_probe_train_40k.jsonl"
VAL_JSONL="$DATA_DIR/prm800k_val_1k.jsonl"
PB_GSM8K_JSONL="$DATA_DIR/processbench_gsm8k.jsonl"
for J in "$PROBE_TRAIN_JSONL" "$VAL_JSONL" "$PB_GSM8K_JSONL"; do
  if [[ ! -s "$J" ]]; then
    echo "[FATAL] missing JSONL: $J" | tee -a "$LOG_FILE"; exit 4
  fi
done

WORKER_PIDS=()
WORKER_LOGS=()
for ((SHARD=0; SHARD<NUM_SHARDS; SHARD++)); do
  WLOG="$LOG_DIR/audit_ssae_orig_paper-${JID}-prm800k-shard${SHARD}.log"
  WORKER_LOGS+=("$WLOG")
  (
    echo "[worker] split=prm800k shard=$SHARD gpu=$SHARD pid=$$ start=$(date -Iseconds)"
    CUDA_VISIBLE_DEVICES="$SHARD" python scripts/extract_ssae_latents.py \
      --ckpt "$CKPT" \
      --model_name_or_path "$MODEL_NAME_OR_PATH" \
      --local_files_only \
      --sparsity_factor "$SPARSITY_FACTOR" \
      --max_seq_len 2048 \
      --batch_size "$BATCH_SIZE" \
      --out_dir "$LATENTS_DIR" \
      --probe_train_jsonl "$PROBE_TRAIN_JSONL" \
      --val_jsonl "$VAL_JSONL" \
      --pb_jsonl "$PB_GSM8K_JSONL" \
      --num_shards "$NUM_SHARDS" \
      --shard_idx "$SHARD"
    echo "[worker] split=prm800k shard=$SHARD rc=$? end=$(date -Iseconds)"
  ) >"$WLOG" 2>&1 &
  WORKER_PIDS+=("$!")
  echo "[worker-launch] split=prm800k shard=$SHARD pid=${WORKER_PIDS[-1]} log=$WLOG" \
    | tee -a "$LOG_FILE"
done
FAIL=0
for PID in "${WORKER_PIDS[@]}"; do
  if ! wait "$PID"; then FAIL=1; fi
done
for WLOG in "${WORKER_LOGS[@]}"; do
  echo "----- BEGIN $WLOG -----" >> "$LOG_FILE"
  cat "$WLOG" >> "$LOG_FILE" || true
  echo "----- END   $WLOG -----" >> "$LOG_FILE"
done
if [[ "$FAIL" -ne 0 ]]; then
  echo "[FATAL] PRM800K shard worker failed" | tee -a "$LOG_FILE"; exit 5
fi

# Merge the 4 PRM800K shards into single .npy / _y.npy / _meta.jsonl files.
for NAME in probe_train_40k val_1k pb_gsm8k_step; do
  echo "[merge] $NAME: combining $NUM_SHARDS shards" | tee -a "$LOG_FILE"
  python scripts/merge_processbench_encoded_shards.py \
    --shard_root "$LATENTS_DIR/shards" \
    --out_h "$LATENTS_DIR/${NAME}_z.npy" \
    --out_meta "$LATENTS_DIR/${NAME}_meta.jsonl" \
    --array_name "${NAME}_z.npy" \
    --meta_name "${NAME}_meta.jsonl" \
    ${FORCE:+--force} 2>&1 | tee -a "$LOG_FILE"
done
# Merge the per-shard label arrays for probe_train_40k and val_1k by hand
# (the merger only concatenates one array; we mirror it for the _y.npy files).
python - <<PY 2>&1 | tee -a "$LOG_FILE"
import json, sys
from pathlib import Path
import numpy as np
shards = sorted((Path("${LATENTS_DIR}") / "shards").iterdir())
for name in ("probe_train_40k", "val_1k"):
    ys = []
    metas = []
    for sd in shards:
        y_path = sd / f"{name}_y.npy"
        if not y_path.exists():
            sys.exit(f"missing {y_path}")
        ys.append(np.load(y_path))
        meta_path = sd / f"{name}_meta.jsonl"
        for line in meta_path.read_text().splitlines():
            line = line.strip()
            if line:
                metas.append(json.loads(line))
    y_concat = np.concatenate(ys)
    # Reorder by global_step_index to match the merged z order.
    order = np.argsort([m["global_step_index"] for m in metas])
    y_sorted = y_concat[order]
    out = Path("${LATENTS_DIR}") / f"{name}_y.npy"
    np.save(out, y_sorted)
    print(f"[label-merge] {name}: {y_sorted.shape} -> {out}")
PY

# Quick row-count + finite checks before continuing.
python - <<PY 2>&1 | tee -a "$LOG_FILE"
from pathlib import Path
import numpy as np
lat = Path("${LATENTS_DIR}")
for name in ("probe_train_40k", "val_1k", "pb_gsm8k_step"):
    z = np.load(lat / f"{name}_z.npy")
    n_meta = sum(1 for _ in (lat / f"{name}_meta.jsonl").open())
    ok = z.shape[0] == n_meta and bool(np.all(np.isfinite(z)))
    print(f"[check] {name}: z.shape={z.shape} meta_rows={n_meta} finite={bool(np.all(np.isfinite(z)))} ok={ok}")
PY

# ---------- Stage 3: train linear probe + select threshold ------------------
echo "[$(date -Iseconds)] stage 3: train probe + select threshold on val_1k" \
  | tee -a "$LOG_FILE"
CUDA_VISIBLE_DEVICES=0 python scripts/train_eval_ssae_probe.py \
  --method "$METHOD_NAME" \
  --latents_dir "$LATENTS_DIR" \
  --out_dir "$RUN_DIR" \
  --seed 42 \
  --epochs_probe 50 \
  --batch_size 512 \
  --lr_probe 1e-3 2>&1 | tee -a "$LOG_FILE"

for f in linear_probe.pt threshold.json; do
  if [[ ! -s "$RUN_DIR/$f" ]]; then
    echo "[FATAL] missing $RUN_DIR/$f after probe step" | tee -a "$LOG_FILE"; exit 6
  fi
done

# ---------- Stage 4: encode full PB (4 GPUs, 1 subset per GPU) --------------
echo "[$(date -Iseconds)] stage 4: encode full PB with paper ckpt" | tee -a "$LOG_FILE"

# Round-robin subsets across GPUs (4 subsets, 4 GPUs). extract_ssae_pb_all.py
# uses subset-internal sharding; for this single-checkpoint audit we run one
# subset per GPU with --num_shards 1 (no in-subset sharding) for simplicity.
SUBSETS=(gsm8k math olympiadbench omnimath)
WORKER_PIDS=()
WORKER_LOGS=()
for i in "${!SUBSETS[@]}"; do
  SUB="${SUBSETS[$i]}"
  GPU="$i"
  SRC="$PB_DIR/processbench_${SUB}.jsonl"
  if [[ ! -s "$SRC" ]]; then
    echo "[FATAL] missing $SRC" | tee -a "$LOG_FILE"; exit 7
  fi
  WLOG="$LOG_DIR/audit_ssae_orig_paper-${JID}-pb-${SUB}.log"
  WORKER_LOGS+=("$WLOG")
  (
    echo "[worker] subset=$SUB gpu=$GPU pid=$$ start=$(date -Iseconds)"
    CUDA_VISIBLE_DEVICES="$GPU" python scripts/extract_ssae_pb_all.py \
      --ckpt "$CKPT" \
      --model_name_or_path "$MODEL_NAME_OR_PATH" \
      --local_files_only \
      --sparsity_factor "$SPARSITY_FACTOR" \
      --max_seq_len 2048 \
      --batch_size "$BATCH_SIZE" \
      --num_workers 2 \
      --out_root "$PB_LAT_ROOT" \
      --pb_files "${SUB}:${SRC}" \
      --no_combined_view \
      ${FORCE:+--force}
    echo "[worker] subset=$SUB rc=$? end=$(date -Iseconds)"
  ) >"$WLOG" 2>&1 &
  WORKER_PIDS+=("$!")
  echo "[worker-launch] subset=$SUB gpu=$GPU pid=${WORKER_PIDS[-1]} log=$WLOG" \
    | tee -a "$LOG_FILE"
done
FAIL=0
for PID in "${WORKER_PIDS[@]}"; do
  if ! wait "$PID"; then FAIL=1; fi
done
for WLOG in "${WORKER_LOGS[@]}"; do
  echo "----- BEGIN $WLOG -----" >> "$LOG_FILE"
  cat "$WLOG" >> "$LOG_FILE" || true
  echo "----- END   $WLOG -----" >> "$LOG_FILE"
done
if [[ "$FAIL" -ne 0 ]]; then
  echo "[FATAL] full PB subset encoder failed" | tee -a "$LOG_FILE"; exit 8
fi

# Build the per-method combined view (subset concat with "<sub>::<id>"
# namespacing, identical to extract_processbench_full_ssae_tamia.sh).
echo "[$(date -Iseconds)] stage 4b: build per-method combined view" \
  | tee -a "$LOG_FILE"
export METHOD_OUT_ROOT="$PB_LAT_ROOT"
python - <<'PY' 2>&1 | tee -a "$LOG_FILE"
import json, os, sys
from pathlib import Path
import numpy as np
root = Path(os.environ["METHOD_OUT_ROOT"])
combined = root / "combined"
combined.mkdir(parents=True, exist_ok=True)
zs, metas = [], []
sub_dirs = sorted([d for d in root.iterdir()
                   if d.is_dir() and d.name != "combined"
                   and (d / "pb_step_z.npy").exists()])
expected = {"gsm8k": 2082, "math": 6505, "olympiadbench": 8819, "omnimath": 8291}
for d in sub_dirs:
    z = np.load(d / "pb_step_z.npy"); zs.append(z)
    rows = [json.loads(l) for l in (d / "pb_step_meta.jsonl").read_text().splitlines() if l.strip()]
    if z.shape[0] != len(rows):
        sys.exit(f"[combined] {d.name}: rows mismatch {z.shape[0]} vs {len(rows)}")
    if d.name in expected and len(rows) != expected[d.name]:
        sys.exit(f"[combined] {d.name}: expected {expected[d.name]} rows, got {len(rows)}")
    for r in rows:
        r["pb_subset"] = r.get("pb_subset", d.name)
        r["id"] = f"{d.name}::{r['id']}"
    metas.extend(rows)
big = np.concatenate(zs, axis=0) if zs else np.zeros((0,))
np.save(combined / "pb_step_z.npy", big)
with (combined / "pb_step_meta.jsonl").open("w") as f:
    for r in metas:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
(combined / "encoding_manifest_pb.json").write_text(json.dumps({
    "n_subsets": len(sub_dirs),
    "subsets": [d.name for d in sub_dirs],
    "n_rows": int(big.shape[0]),
    "id_namespacing": "<subset>::<id>",
    "expected_total": sum(expected.values()),
}, indent=2))
print(f"[combined] {big.shape[0]} rows across {len(sub_dirs)} subsets "
      f"(expected {sum(expected.values())})")
PY

# ---------- Stage 5: full-PB evaluation -------------------------------------
echo "[$(date -Iseconds)] stage 5: full-PB evaluation (4 parallel workers)" \
  | tee -a "$LOG_FILE"
JOBS_DIR="$EVAL_OUT_DIR/_jobs"
mkdir -p "$JOBS_DIR"
SHARD_PREFIX="$JOBS_DIR/eval_jobs"

python scripts/build_full_processbench_eval_jobs.py \
  --runs_root "$RUNS_DIR" \
  --dense_pb_cache_root "$RUN_ROOT/cache/qwen2_5_1_5b_processbench_full" \
  --out_dir "$EVAL_OUT_DIR" \
  --methods "$METHOD_NAME" \
  --out_jobs_jsonl "$JOBS_DIR/eval_jobs.jsonl" \
  --num_workers "$NUM_SHARDS" \
  --shard_prefix "$SHARD_PREFIX" \
  --include_combined 2>&1 | tee -a "$LOG_FILE"

TOTAL_JOBS=$(wc -l < "$JOBS_DIR/eval_jobs.jsonl" | awk '{print $1}')
echo "[plan] total_jobs=$TOTAL_JOBS workers=$NUM_SHARDS" | tee -a "$LOG_FILE"
if [[ "$TOTAL_JOBS" -eq 0 ]]; then
  echo "[FATAL] no eval jobs built" | tee -a "$LOG_FILE"; exit 9
fi

WORKER_PIDS=()
WORKER_LOGS=()
for ((W=0; W<NUM_SHARDS; W++)); do
  GPU=$W
  JOBS_JSON="${SHARD_PREFIX}_worker_${W}.json"
  WLOG="$LOG_DIR/audit_ssae_orig_paper-${JID}-eval-worker${W}.log"
  WORKER_LOGS+=("$WLOG")
  (
    echo "[worker] worker_id=$W gpu=$GPU pid=$$ start=$(date -Iseconds)"
    CUDA_VISIBLE_DEVICES="$GPU" python scripts/evaluate_existing_probes_full_processbench_worker.py \
      --jobs_json "$JOBS_JSON" \
      --worker_id "$W" \
      --device cuda \
      --oracle_threshold_step "$ORACLE_STEP" \
      ${FORCE:+--force}
    echo "[worker] worker_id=$W rc=$? end=$(date -Iseconds)"
  ) >"$WLOG" 2>&1 &
  WORKER_PIDS+=("$!")
  echo "[worker-launch] eval worker=$W gpu=$GPU pid=${WORKER_PIDS[-1]} log=$WLOG" \
    | tee -a "$LOG_FILE"
done
FAIL=0
for PID in "${WORKER_PIDS[@]}"; do
  if ! wait "$PID"; then FAIL=1; fi
done
for WLOG in "${WORKER_LOGS[@]}"; do
  echo "----- BEGIN $WLOG -----" >> "$LOG_FILE"
  cat "$WLOG" >> "$LOG_FILE" || true
  echo "----- END   $WLOG -----" >> "$LOG_FILE"
done
if [[ "$FAIL" -ne 0 ]]; then
  echo "[FATAL] at least one eval worker failed" | tee -a "$LOG_FILE"; exit 10
fi

python scripts/merge_full_processbench_eval_results.py \
  --out_dir "$EVAL_OUT_DIR" ${FORCE:+--force} 2>&1 | tee -a "$LOG_FILE"

echo "[$(date -Iseconds)] audit_ssae_orig_paper DONE" | tee -a "$LOG_FILE"
echo "Leaderboards: $EVAL_OUT_DIR/leaderboard_full_pb_*.md" | tee -a "$LOG_FILE"
