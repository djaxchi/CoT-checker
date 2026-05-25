#!/bin/bash
# Encode every available ProcessBench subset into dense Qwen2.5-1.5B hidden
# states using ALL 4 H100 GPUs (per-subset 4-way sharding).
#
# Layout per subset:
#   <out_root>/<subset>/shards/shard_00/pb_step_h.npy   (GPU 0)
#   <out_root>/<subset>/shards/shard_01/pb_step_h.npy   (GPU 1)
#   <out_root>/<subset>/shards/shard_02/pb_step_h.npy   (GPU 2)
#   <out_root>/<subset>/shards/shard_03/pb_step_h.npy   (GPU 3)
#   <out_root>/<subset>/pb_step_h.npy                   (merged, sorted by
#                                                        global_step_index)
#   <out_root>/<subset>/pb_step_meta.jsonl              (merged)
#   <out_root>/<subset>/encoding_manifest_pb.json
#   <out_root>/combined/pb_step_h.npy + meta            (concat of subsets,
#                                                        ids prefixed "<sub>::")
#
# Each subset is encoded by launching 4 parallel Python workers, one per
# GPU, pinned with CUDA_VISIBLE_DEVICES=k. Workers write to disjoint shard
# dirs; deterministic sharding by (global_step_index % 4 == shard_idx).
# After workers complete, we merge with scripts/merge_processbench_encoded_shards.py.
#
# Override knobs:
#   PB_DIR=...           # processbench_<subset>.jsonl root
#   FORCE=1              # allow re-encoding existing per-subset outputs
#   BATCH_SIZE=...       # per-GPU batch size (default 16)
#   NUM_SHARDS=...       # default 4

#SBATCH --job-name=pb_encode_full_prestudy
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=05:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail
module load StdEnv/2023 python/3.12

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH/cot_mech/prestudy_v1}"
PB_DIR_DEFAULT="/scratch/d/dchikhi/cot-checker/processbench"
PB_DIR="${PB_DIR:-$PB_DIR_DEFAULT}"
export OUT_ROOT="$RUN_ROOT/cache/qwen2_5_1_5b_processbench_full"
LOG_DIR="$RUN_ROOT/logs"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-1.5B}"
HF_CACHE="${HF_CACHE:-$SCRATCH/hf_cache}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_SHARDS="${NUM_SHARDS:-4}"

mkdir -p "$OUT_ROOT" "$LOG_DIR" "$HF_CACHE"
export HF_HOME="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_CACHE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd "$PROJECT_ROOT"
GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
JID="${SLURM_JOB_ID:-$$}"
LOG_FILE="$LOG_DIR/pb_encode_full_prestudy-${JID}.log"

cat <<BANNER | tee -a "$LOG_FILE"
================================================================
job          : ${SLURM_JOB_NAME:-pb_encode_full_prestudy}
job_id       : ${JID}
hostname     : $(hostname)
date         : $(date -Iseconds)
git_commit   : $GIT_COMMIT
pb_root      : $PB_DIR
out_root     : $OUT_ROOT
num_shards   : $NUM_SHARDS  (one worker per GPU)
batch_size   : $BATCH_SIZE  (per worker)
log_file     : $LOG_FILE
monitor      :
  nvidia-smi
  squeue -j $JID
  grep -nE "worker|encode|shard|merge|GPU|ERROR|Traceback" $LOG_FILE | tail -200
  ls -1 $OUT_ROOT/*/pb_step_h.npy
================================================================
BANNER

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch transformers numpy

# ---- Discover PB subsets via the standalone utility ----------------------
python scripts/list_processbench_subsets.py \
    --pb_root "$PB_DIR" \
    --out_manifest "$RUN_ROOT/processbench_full_manifest.json" \
    --quiet 2>&1 | tee -a "$LOG_FILE"

# Build a name->path map from the manifest
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

echo "[plan] subsets to encode (${#PB_PAIRS[@]}):" | tee -a "$LOG_FILE"
printf '   %s\n' "${PB_PAIRS[@]}" | tee -a "$LOG_FILE"

FORCE_FLAG=""
if [[ "${FORCE:-0}" == "1" ]]; then FORCE_FLAG="--force"; fi

# ---- For each subset: fan out N shard workers, then merge ---------------
for PAIR in "${PB_PAIRS[@]}"; do
  NAME="${PAIR%%:*}"
  SRC="${PAIR#*:}"
  SUB_ROOT="$OUT_ROOT/$NAME"
  SHARDS_DIR="$SUB_ROOT/shards"
  FINAL_H="$SUB_ROOT/pb_step_h.npy"
  FINAL_META="$SUB_ROOT/pb_step_meta.jsonl"

  if [[ -f "$FINAL_H" && "${FORCE:-0}" != "1" ]]; then
    echo "[SKIP] $NAME: $FINAL_H exists (set FORCE=1 to re-encode)" \
      | tee -a "$LOG_FILE"
    continue
  fi

  mkdir -p "$SHARDS_DIR"
  echo "[$(date -Iseconds)] [subset=$NAME] launching $NUM_SHARDS workers" \
    | tee -a "$LOG_FILE"
  WORKER_PIDS=()
  WORKER_LOGS=()
  for ((SHARD=0; SHARD<NUM_SHARDS; SHARD++)); do
    GPU=$SHARD
    SHARD_DIR="$SHARDS_DIR/$(printf 'shard_%02d' "$SHARD")"
    mkdir -p "$SHARD_DIR"
    WLOG="$LOG_DIR/encode-${JID}-${NAME}-shard${SHARD}.log"
    WORKER_LOGS+=("$WLOG")
    (
      echo "[worker] subset=$NAME shard=$SHARD/$NUM_SHARDS gpu=$GPU pid=$$ start=$(date -Iseconds)"
      CUDA_VISIBLE_DEVICES="$GPU" python scripts/encode_processbench_hidden_states.py \
        --raw_file "$SRC" \
        --out_dir "$SHARD_DIR" \
        --model_name_or_path "$MODEL_NAME_OR_PATH" \
        --run_name "prestudy_v1_pb_full__${NAME}__shard${SHARD}" \
        --max_seq_len 2048 \
        --batch_size "$BATCH_SIZE" \
        --model_dtype float16 \
        --save_dtype float16 \
        --subset_name "$NAME" \
        --output_layout generic \
        --shard_idx "$SHARD" \
        --num_shards "$NUM_SHARDS" \
        --local_files_only \
        $FORCE_FLAG
      echo "[worker] subset=$NAME shard=$SHARD/$NUM_SHARDS rc=$? end=$(date -Iseconds)"
    ) >"$WLOG" 2>&1 &
    WORKER_PIDS+=("$!")
    echo "[worker-launch] subset=$NAME shard=$SHARD/$NUM_SHARDS gpu=$GPU "\
"pid=${WORKER_PIDS[-1]} log=$WLOG" | tee -a "$LOG_FILE"
  done

  # Wait for all workers; if any fails, abort the whole job.
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
    echo "[FATAL] $NAME: at least one shard worker failed" | tee -a "$LOG_FILE"
    exit 3
  fi

  echo "[merge] $NAME: combining $NUM_SHARDS shards -> $FINAL_H" \
    | tee -a "$LOG_FILE"
  python scripts/merge_processbench_encoded_shards.py \
    --shard_root "$SHARDS_DIR" \
    --out_h "$FINAL_H" \
    --out_meta "$FINAL_META" \
    --array_name pb_step_h.npy \
    --meta_name pb_step_meta.jsonl \
    $FORCE_FLAG 2>&1 | tee -a "$LOG_FILE"
done

# ---- Combined view across subsets ----------------------------------------
echo "[$(date -Iseconds)] building combined view across subsets" \
  | tee -a "$LOG_FILE"
python - <<'PY' 2>&1 | tee -a "$LOG_FILE"
import json, os, sys
from pathlib import Path
import numpy as np
out_root = Path(os.environ["OUT_ROOT"])
combined = out_root / "combined"
combined.mkdir(parents=True, exist_ok=True)
hs, metas = [], []
subset_dirs = sorted([d for d in out_root.iterdir()
                      if d.is_dir() and d.name != "combined"
                      and (d / "pb_step_h.npy").exists()])
for d in subset_dirs:
    h = np.load(d / "pb_step_h.npy"); hs.append(h)
    rows = [json.loads(l) for l in (d / "pb_step_meta.jsonl").read_text().splitlines() if l.strip()]
    if h.shape[0] != len(rows):
        sys.exit(f"[combined] {d.name}: rows mismatch {h.shape[0]} vs {len(rows)}")
    for r in rows:
        r["pb_subset"] = r.get("pb_subset", d.name)
        r["id"] = f"{d.name}::{r['id']}"
    metas.extend(rows)
big = np.concatenate(hs, axis=0) if hs else np.zeros((0,))
np.save(combined / "pb_step_h.npy", big)
with (combined / "pb_step_meta.jsonl").open("w") as f:
    for r in metas:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
(combined / "encoding_manifest_pb.json").write_text(json.dumps({
    "n_subsets": len(subset_dirs),
    "subsets": [d.name for d in subset_dirs],
    "n_rows": int(big.shape[0]),
    "id_namespacing": "<subset>::<id>",
}, indent=2))
print(f"[combined] {big.shape[0]} rows across {len(subset_dirs)} subsets -> {combined / 'pb_step_h.npy'}")
PY

echo "[$(date -Iseconds)] pb_encode_full_prestudy done" | tee -a "$LOG_FILE"
