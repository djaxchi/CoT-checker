#!/bin/bash
# For each existing SSAE run, extract ProcessBench latents on every subset
# using ALL 4 H100 GPUs via per-subset 4-way sharding.
#
# Per (method, subset):
#   <run>/latents_full_pb/<subset>/shards/shard_NN/pb_step_z.npy   (one/GPU)
#   <run>/latents_full_pb/<subset>/pb_step_z.npy                   (merged)
#   <run>/latents_full_pb/<subset>/pb_step_meta.jsonl
# Plus per-run combined/ view (subset concat with "<sub>::<id>" namespacing).
#
# Methods default:
#   ssae_positive  ssae_mixed  ssae_contrastive  ssae_contrastive_auxlr1e-3_full
#
# Override knobs:
#   PB_DIR=...                # processbench_<subset>.jsonl root
#   FORCE=1                   # allow overwriting latents_full_pb
#   METHODS="ssae_positive ssae_mixed ..."
#   SPARSITY_FACTOR=1
#   NUM_SHARDS=4              # default 4 (one worker per GPU)
#   BATCH_SIZE=8              # per-GPU batch size

#SBATCH --job-name=ssae_pb_extract_full
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail
module load StdEnv/2023 python/3.12

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH/cot_mech/prestudy_v1}"
RUNS_DIR="$RUN_ROOT/runs"
PB_DIR_DEFAULT="/scratch/d/dchikhi/cot-checker/processbench"
PB_DIR="${PB_DIR:-$PB_DIR_DEFAULT}"
LOG_DIR="$RUN_ROOT/logs"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-1.5B}"
HF_CACHE="${HF_CACHE:-$SCRATCH/hf_cache}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_SHARDS="${NUM_SHARDS:-4}"
SPARSITY_FACTOR="${SPARSITY_FACTOR:-1}"
METHODS="${METHODS:-ssae_positive ssae_mixed ssae_contrastive ssae_contrastive_auxlr1e-3_full}"

mkdir -p "$LOG_DIR" "$HF_CACHE"
export HF_HOME="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_CACHE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd "$PROJECT_ROOT"
GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
JID="${SLURM_JOB_ID:-$$}"
LOG_FILE="$LOG_DIR/ssae_pb_extract_full-${JID}.log"

cat <<BANNER | tee -a "$LOG_FILE"
================================================================
job          : ${SLURM_JOB_NAME:-ssae_pb_extract_full}
job_id       : ${JID}
hostname     : $(hostname)
date         : $(date -Iseconds)
git_commit   : $GIT_COMMIT
runs_dir     : $RUNS_DIR
pb_root      : $PB_DIR
methods      : $METHODS
num_shards   : $NUM_SHARDS  (one worker per GPU)
batch_size   : $BATCH_SIZE  (per worker)
log_file     : $LOG_FILE
monitor      :
  nvidia-smi
  squeue -j $JID
  grep -nE "worker|shard|merge|extract_ssae|GPU|ERROR|Traceback" $LOG_FILE | tail -200
  ls -1 $RUNS_DIR/*/latents_full_pb/*/pb_step_z.npy
================================================================
BANNER

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch transformers numpy tqdm

# Discover PB subsets once
python scripts/list_processbench_subsets.py \
    --pb_root "$PB_DIR" \
    --out_manifest "$RUN_ROOT/processbench_full_manifest.json" \
    --quiet 2>&1 | tee -a "$LOG_FILE"

mapfile -t PB_PAIRS < <(python - <<'PY' "$RUN_ROOT/processbench_full_manifest.json"
import json, sys
m = json.load(open(sys.argv[1]))
for s in m["subsets"]:
    print(f"{s['subset']}:{s['path']}")
PY
)
if [[ ${#PB_PAIRS[@]} -eq 0 ]]; then
  echo "[FATAL] No PB subsets discovered under $PB_DIR" | tee -a "$LOG_FILE"
  exit 2
fi
echo "[plan] subsets: ${PB_PAIRS[*]}" | tee -a "$LOG_FILE"

FORCE_FLAG=""
if [[ "${FORCE:-0}" == "1" ]]; then FORCE_FLAG="--force"; fi

for METHOD in $METHODS; do
  RUN_DIR="$RUNS_DIR/$METHOD"
  CKPT="$RUN_DIR/ssae_model.pt"
  export METHOD_OUT_ROOT="$RUN_DIR/latents_full_pb"
  if [[ ! -f "$CKPT" ]]; then
    echo "[SKIP] $METHOD: no ssae_model.pt at $CKPT" | tee -a "$LOG_FILE"
    continue
  fi
  CONTRASTIVE_FLAG=""
  case "$METHOD" in
    ssae_contrastive*|ssae_*contrastive*)
      CONTRASTIVE_FLAG="--contrastive_ckpt"
      ;;
  esac

  mkdir -p "$METHOD_OUT_ROOT"

  for PAIR in "${PB_PAIRS[@]}"; do
    NAME="${PAIR%%:*}"
    SRC="${PAIR#*:}"
    SUB_ROOT="$METHOD_OUT_ROOT/$NAME"
    SHARDS_DIR="$SUB_ROOT/shards"
    FINAL_Z="$SUB_ROOT/pb_step_z.npy"
    FINAL_META="$SUB_ROOT/pb_step_meta.jsonl"

    if [[ -f "$FINAL_Z" && "${FORCE:-0}" != "1" ]]; then
      echo "[SKIP] $METHOD/$NAME: $FINAL_Z exists (set FORCE=1)" \
        | tee -a "$LOG_FILE"
      continue
    fi
    mkdir -p "$SHARDS_DIR"

    echo "[$(date -Iseconds)] [$METHOD/$NAME] launching $NUM_SHARDS shard workers" \
      | tee -a "$LOG_FILE"
    WORKER_PIDS=()
    WORKER_LOGS=()
    for ((SHARD=0; SHARD<NUM_SHARDS; SHARD++)); do
      GPU=$SHARD
      WLOG="$LOG_DIR/ssae_extract-${JID}-${METHOD}-${NAME}-shard${SHARD}.log"
      WORKER_LOGS+=("$WLOG")
      (
        echo "[worker] method=$METHOD subset=$NAME shard=$SHARD/$NUM_SHARDS "\
"gpu=$GPU pid=$$ start=$(date -Iseconds)"
        CUDA_VISIBLE_DEVICES="$GPU" python scripts/extract_ssae_pb_all.py \
          --ckpt "$CKPT" \
          --model_name_or_path "$MODEL_NAME_OR_PATH" \
          --local_files_only \
          --sparsity_factor "$SPARSITY_FACTOR" \
          --max_seq_len 2048 \
          --batch_size "$BATCH_SIZE" \
          --num_workers 2 \
          --out_root "$METHOD_OUT_ROOT" \
          --pb_files "${NAME}:${SRC}" \
          --shard_idx "$SHARD" \
          --num_shards "$NUM_SHARDS" \
          --shard_subdir \
          $CONTRASTIVE_FLAG \
          $FORCE_FLAG
        echo "[worker] method=$METHOD subset=$NAME shard=$SHARD rc=$? end=$(date -Iseconds)"
      ) >"$WLOG" 2>&1 &
      WORKER_PIDS+=("$!")
      echo "[worker-launch] method=$METHOD subset=$NAME shard=$SHARD/$NUM_SHARDS "\
"gpu=$GPU pid=${WORKER_PIDS[-1]} log=$WLOG" | tee -a "$LOG_FILE"
    done

    FAIL=0
    for PID in "${WORKER_PIDS[@]}"; do
      if ! wait "$PID"; then
        echo "[worker-fail] pid=$PID exited non-zero" | tee -a "$LOG_FILE"
        FAIL=1
      fi
    done
    for WLOG in "${WORKER_LOGS[@]}"; do
      echo "----- BEGIN $WLOG -----" >> "$LOG_FILE"
      cat "$WLOG" >> "$LOG_FILE" || true
      echo "----- END   $WLOG -----" >> "$LOG_FILE"
    done
    if [[ "$FAIL" -ne 0 ]]; then
      echo "[FATAL] $METHOD/$NAME: at least one shard worker failed" \
        | tee -a "$LOG_FILE"
      exit 3
    fi

    echo "[merge] $METHOD/$NAME: combining $NUM_SHARDS shards -> $FINAL_Z" \
      | tee -a "$LOG_FILE"
    python scripts/merge_processbench_encoded_shards.py \
      --shard_root "$SHARDS_DIR" \
      --out_h "$FINAL_Z" \
      --out_meta "$FINAL_META" \
      --array_name pb_step_z.npy \
      --meta_name pb_step_meta.jsonl \
      $FORCE_FLAG 2>&1 | tee -a "$LOG_FILE"
  done

  # Per-method combined view
  echo "[$(date -Iseconds)] [$METHOD] building combined view" \
    | tee -a "$LOG_FILE"
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
for d in sub_dirs:
    z = np.load(d / "pb_step_z.npy"); zs.append(z)
    rows = [json.loads(l) for l in (d / "pb_step_meta.jsonl").read_text().splitlines() if l.strip()]
    if z.shape[0] != len(rows):
        sys.exit(f"[combined] {d.name}: rows mismatch {z.shape[0]} vs {len(rows)}")
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
}, indent=2))
print(f"[combined] {big.shape[0]} rows across {len(sub_dirs)} subsets")
PY

done

echo "[$(date -Iseconds)] ssae_pb_extract_full done" | tee -a "$LOG_FILE"
