#!/bin/bash
# Audit Job 2: corrected ssae_mixed training on Qwen2.5-1.5B with the DWA
# controller restored and a real iteration budget (3 000 iters at LR 1e-4).
# Trains one method end-to-end, then re-uses the existing 4-GPU full-PB
# encoder + evaluator to produce a leaderboard row comparable to the rest
# of the table.
#
# Pipeline:
#   1. Train ssae_mixed_dwa_lr1e-4_iter3000 via the existing
#      run_ssae_method.py orchestrator (--use_dwa --l1_target 3.0
#      --learning_rate 1e-4 --max_iters 3000). DDP across all 4 H100s, since
#      train_ssae_official.py already supports torchrun.
#   2. Per-method run_ssae_method.py also (a) extracts SSAE latents for
#      probe_train_40k + val_1k + pb_gsm8k, and (b) trains the fresh linear
#      probe + selects a deployable threshold on val_1k.
#   3. Extract full PB latents for the new checkpoint, 4-way sharded by
#      subset (extract_ssae_pb_all.py with subset-level fan-out).
#   4. Run the existing full-PB evaluator pipeline restricted to this
#      single method.
#
# Submission:
#   sbatch slurm/audit_ssae_mixed_dwa_long_tamia.sh
#
# Optional env overrides:
#   MAX_ITERS=3000  LEARNING_RATE=1e-4  BATCH_SIZE=4  GRAD_ACCUM=32
#   L1_TARGET=3.0   DWA_UPDATE_INTERVAL=100
#   ORACLE_STEP=0.005  FORCE=1

#SBATCH --job-name=audit_ssae_mixed_dwa_long
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=24:00:00
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

METHOD_NAME="ssae_mixed_dwa_lr1e-4_iter3000"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-1.5B}"
MAX_ITERS="${MAX_ITERS:-3000}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-32}"
L1_WEIGHT="${L1_WEIGHT:-1e-4}"
L1_TARGET="${L1_TARGET:-3.0}"
DWA_UPDATE_INTERVAL="${DWA_UPDATE_INTERVAL:-100}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
WARMUP_ITERS="${WARMUP_ITERS:-50}"
MIN_LR="${MIN_LR:-1e-5}"
ATTN_IMPL="${ATTN_IMPL:-eager}"
LATENT_NORM_EPS="${LATENT_NORM_EPS:-1e-2}"
CE_CHUNK_SIZE="${CE_CHUNK_SIZE:-2048}"
ORACLE_STEP="${ORACLE_STEP:-0.005}"

RUN_DIR="$RUNS_DIR/$METHOD_NAME"
PB_LAT_ROOT="$RUN_DIR/latents_full_pb"
EVAL_OUT_DIR="$RUNS_DIR/full_processbench_eval_${METHOD_NAME}"
PB_CACHE_AUDIT="$RUN_ROOT/cache/qwen2_5_1_5b_processbench_full_ssae_mixed_dwa_lr1e-4_iter3000"

mkdir -p "$RUN_DIR" "$PB_LAT_ROOT" "$EVAL_OUT_DIR" "$LOG_DIR" "$PB_CACHE_AUDIT"

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
LOG_FILE="$LOG_DIR/audit_ssae_mixed_dwa_long-${JID}.log"

cat <<BANNER | tee -a "$LOG_FILE"
================================================================
job          : ${SLURM_JOB_NAME:-audit_ssae_mixed_dwa_long}
job_id       : ${JID}
hostname     : $(hostname)
date         : $(date -Iseconds)
git_commit   : $GIT_COMMIT
method_name  : $METHOD_NAME
model        : $MODEL_NAME_OR_PATH   (Qwen2.5-1.5B, hidden=1536)
max_iters    : $MAX_ITERS
lr           : $LEARNING_RATE  (warmup=$WARMUP_ITERS, min_lr=$MIN_LR)
batch        : per-GPU=$BATCH_SIZE x grad_accum=$GRAD_ACCUM => effective ~$((BATCH_SIZE*GRAD_ACCUM*4))
DWA          : target=$L1_TARGET interval=$DWA_UPDATE_INTERVAL init_l1_weight=$L1_WEIGHT
run_dir      : $RUN_DIR
pb_lat_root  : $PB_LAT_ROOT
eval_out_dir : $EVAL_OUT_DIR
oracle_step  : $ORACLE_STEP
log_file     : $LOG_FILE
================================================================
BANNER

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch numpy tqdm transformers pyyaml

# ---------- Stage 1: train + extract probe latents + train probe ------------
# run_ssae_method.py orchestrates training (torchrun, 4 GPUs DDP), latent
# extraction for probe/val/pb_gsm8k, and probe training + threshold pick.
# It writes ssae_model.pt, latents/, linear_probe.pt, threshold.json under
# $RUN_DIR.
echo "[$(date -Iseconds)] stage 1: train SSAE + extract probe latents + train probe" \
  | tee -a "$LOG_FILE"
python scripts/run_ssae_method.py \
  --method "$METHOD_NAME" \
  --data_dir "$DATA_DIR" \
  --out_dir "$RUN_DIR" \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --local_files_only \
  --phase 1 \
  --sparsity_factor 1 \
  --l1_weight "$L1_WEIGHT" \
  --use_dwa \
  --l1_target "$L1_TARGET" \
  --dwa_update_interval "$DWA_UPDATE_INTERVAL" \
  --max_seq_len "$MAX_SEQ_LEN" \
  --batch_size "$BATCH_SIZE" \
  --grad_accum_steps "$GRAD_ACCUM" \
  --learning_rate "$LEARNING_RATE" \
  --min_lr "$MIN_LR" \
  --warmup_iters "$WARMUP_ITERS" \
  --max_iters "$MAX_ITERS" \
  --nproc_per_node 4 \
  --ce_chunk_size "$CE_CHUNK_SIZE" \
  --max_grad_norm 1.0 \
  --attn_implementation "$ATTN_IMPL" \
  --latent_norm_eps "$LATENT_NORM_EPS" \
  --gradient_checkpointing \
  --extract_batch_size 8 \
  --epochs_probe 50 \
  --probe_batch_size 512 \
  --seed 42 2>&1 | tee -a "$LOG_FILE"

CKPT="$RUN_DIR/ssae_model.pt"
for f in ssae_model.pt linear_probe.pt threshold.json; do
  if [[ ! -s "$RUN_DIR/$f" ]]; then
    echo "[FATAL] missing $RUN_DIR/$f after stage 1" | tee -a "$LOG_FILE"; exit 3
  fi
done

# ---------- Stage 2: extract full PB latents for the new checkpoint ---------
# 4 parallel workers, one subset per GPU.
echo "[$(date -Iseconds)] stage 2: full PB encoding (4 GPUs, 1 subset / GPU)" \
  | tee -a "$LOG_FILE"
SUBSETS=(gsm8k math olympiadbench omnimath)
WORKER_PIDS=()
WORKER_LOGS=()
for i in "${!SUBSETS[@]}"; do
  SUB="${SUBSETS[$i]}"
  GPU="$i"
  SRC="$PB_DIR/processbench_${SUB}.jsonl"
  if [[ ! -s "$SRC" ]]; then
    echo "[FATAL] missing $SRC" | tee -a "$LOG_FILE"; exit 4
  fi
  WLOG="$LOG_DIR/audit_ssae_mixed_dwa_long-${JID}-pb-${SUB}.log"
  WORKER_LOGS+=("$WLOG")
  (
    echo "[worker] subset=$SUB gpu=$GPU pid=$$ start=$(date -Iseconds)"
    CUDA_VISIBLE_DEVICES="$GPU" python scripts/extract_ssae_pb_all.py \
      --ckpt "$CKPT" \
      --model_name_or_path "$MODEL_NAME_OR_PATH" \
      --local_files_only \
      --sparsity_factor 1 \
      --max_seq_len "$MAX_SEQ_LEN" \
      --batch_size 8 \
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
  echo "[FATAL] full PB subset encoder failed" | tee -a "$LOG_FILE"; exit 5
fi

# Build the per-method combined view.
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

# Mirror the latents to the audit-only cache root so the user can locate
# the encoded latents using the documented path convention even though the
# evaluator reads them from $RUN_DIR/latents_full_pb/.
ln -sfn "$PB_LAT_ROOT" "$PB_CACHE_AUDIT/latents"
echo "[link] $PB_CACHE_AUDIT/latents -> $PB_LAT_ROOT" | tee -a "$LOG_FILE"

# ---------- Stage 3: full-PB evaluation -------------------------------------
echo "[$(date -Iseconds)] stage 3: full-PB evaluation (4 parallel workers)" \
  | tee -a "$LOG_FILE"
NUM_SHARDS=4
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
  echo "[FATAL] no eval jobs built" | tee -a "$LOG_FILE"; exit 6
fi

WORKER_PIDS=()
WORKER_LOGS=()
for ((W=0; W<NUM_SHARDS; W++)); do
  GPU=$W
  JOBS_JSON="${SHARD_PREFIX}_worker_${W}.json"
  WLOG="$LOG_DIR/audit_ssae_mixed_dwa_long-${JID}-eval-worker${W}.log"
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
  echo "[FATAL] at least one eval worker failed" | tee -a "$LOG_FILE"; exit 7
fi

python scripts/merge_full_processbench_eval_results.py \
  --out_dir "$EVAL_OUT_DIR" ${FORCE:+--force} 2>&1 | tee -a "$LOG_FILE"

echo "[$(date -Iseconds)] audit_ssae_mixed_dwa_long DONE" | tee -a "$LOG_FILE"
echo "Leaderboards: $EVAL_OUT_DIR/leaderboard_full_pb_*.md" | tee -a "$LOG_FILE"
