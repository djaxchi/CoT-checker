#!/bin/bash
# Encode every ProcessBench subset into dense Qwen2.5-7B BASE hidden states,
# 4-way in-node sharding per subset, then merge + build the combined view.
# 7B variant of encode_processbench_full_dense_tamia.sh for the unified harness.
#
# Output layout (dense-full contract consumed by train_easy_probe_method.py):
#   <OUT_ROOT>/<subset>/pb_step_h.npy + pb_step_meta.jsonl
#   <OUT_ROOT>/combined/pb_step_h.npy + meta   (ids prefixed "<subset>::")

#SBATCH --job-name=pb_encode_7b
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=05:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH/cot_mech/dense_full_7b_v1}"
PB_DIR="${PB_DIR:-/scratch/d/dchikhi/cot-checker/processbench}"
export OUT_ROOT="$RUN_ROOT/cache/qwen2_5_7b_processbench"
LOG_DIR="$RUN_ROOT/logs"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-7B}"
HF_CACHE="${HF_CACHE:-/project/aip-azouaq/$USER/hf_cache}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_SHARDS="${NUM_SHARDS:-4}"

mkdir -p "$OUT_ROOT" "$LOG_DIR"
export HF_HOME="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_CACHE"
export HF_DATASETS_CACHE="$HF_CACHE/datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd "$PROJECT_ROOT"
GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
JID="${SLURM_JOB_ID:-$$}"
LOG_FILE="$LOG_DIR/pb_encode_7b-${JID}.log"

cat <<BANNER | tee -a "$LOG_FILE"
================================================================
job          : ${SLURM_JOB_NAME:-pb_encode_7b}   job_id: ${JID}
git_commit   : $GIT_COMMIT
model        : $MODEL_NAME_OR_PATH  (HF_HOME=$HF_CACHE)
pb_root      : $PB_DIR
out_root     : $OUT_ROOT
num_shards   : $NUM_SHARDS   batch: $BATCH_SIZE
log_file     : $LOG_FILE
================================================================
BANNER

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch transformers numpy

python scripts/list_processbench_subsets.py \
    --pb_root "$PB_DIR" \
    --out_manifest "$RUN_ROOT/processbench_full_manifest_7b.json" \
    --quiet 2>&1 | tee -a "$LOG_FILE"

mapfile -t PB_PAIRS < <(python - <<'PY' "$RUN_ROOT/processbench_full_manifest_7b.json"
import json, sys
m = json.load(open(sys.argv[1]))
for s in m["subsets"]:
    print(f"{s['subset']}:{s['path']}")
PY
)
if [[ ${#PB_PAIRS[@]} -eq 0 ]]; then
  echo "[FATAL] No PB subsets discovered under $PB_DIR" | tee -a "$LOG_FILE"; exit 2
fi
echo "[plan] subsets (${#PB_PAIRS[@]}):" | tee -a "$LOG_FILE"
printf '   %s\n' "${PB_PAIRS[@]}" | tee -a "$LOG_FILE"

FORCE_FLAG=""
if [[ "${FORCE:-0}" == "1" ]]; then FORCE_FLAG="--force"; fi

for PAIR in "${PB_PAIRS[@]}"; do
  NAME="${PAIR%%:*}"
  SRC="${PAIR#*:}"
  SUB_ROOT="$OUT_ROOT/$NAME"
  SHARDS_DIR="$SUB_ROOT/shards"
  FINAL_H="$SUB_ROOT/pb_step_h.npy"
  FINAL_META="$SUB_ROOT/pb_step_meta.jsonl"

  if [[ -f "$FINAL_H" && "${FORCE:-0}" != "1" ]]; then
    echo "[SKIP] $NAME: $FINAL_H exists (FORCE=1 to re-encode)" | tee -a "$LOG_FILE"
    continue
  fi
  mkdir -p "$SHARDS_DIR"
  echo "[$(date -Iseconds)] [subset=$NAME] launching $NUM_SHARDS workers" | tee -a "$LOG_FILE"
  WORKER_PIDS=(); WORKER_LOGS=()
  for ((SHARD=0; SHARD<NUM_SHARDS; SHARD++)); do
    GPU=$SHARD
    SHARD_DIR="$SHARDS_DIR/$(printf 'shard_%02d' "$SHARD")"
    mkdir -p "$SHARD_DIR"
    WLOG="$LOG_DIR/pb_encode-${JID}-${NAME}-shard${SHARD}.log"
    WORKER_LOGS+=("$WLOG")
    (
      CUDA_VISIBLE_DEVICES="$GPU" python scripts/encode_processbench_hidden_states.py \
        --raw_file "$SRC" \
        --out_dir "$SHARD_DIR" \
        --model_name_or_path "$MODEL_NAME_OR_PATH" \
        --run_name "dense_full_7b_v1_pb__${NAME}__shard${SHARD}" \
        --max_seq_len 2048 \
        --batch_size "$BATCH_SIZE" \
        --model_dtype float16 --save_dtype float16 \
        --subset_name "$NAME" \
        --output_layout generic \
        --shard_idx "$SHARD" --num_shards "$NUM_SHARDS" \
        --local_files_only \
        $FORCE_FLAG
      echo "[worker] subset=$NAME shard=$SHARD rc=$?"
    ) >"$WLOG" 2>&1 &
    WORKER_PIDS+=("$!")
  done
  FAIL=0
  for PID in "${WORKER_PIDS[@]}"; do wait "$PID" || FAIL=1; done
  for WLOG in "${WORKER_LOGS[@]}"; do
    echo "----- $WLOG -----" >> "$LOG_FILE"; cat "$WLOG" >> "$LOG_FILE" || true
  done
  if [[ "$FAIL" -ne 0 ]]; then
    echo "[FATAL] $NAME: a shard worker failed" | tee -a "$LOG_FILE"; exit 3
  fi
  echo "[merge] $NAME -> $FINAL_H" | tee -a "$LOG_FILE"
  python scripts/merge_processbench_encoded_shards.py \
    --shard_root "$SHARDS_DIR" \
    --out_h "$FINAL_H" --out_meta "$FINAL_META" \
    --array_name pb_step_h.npy --meta_name pb_step_meta.jsonl \
    $FORCE_FLAG 2>&1 | tee -a "$LOG_FILE"
done

echo "[$(date -Iseconds)] building combined view" | tee -a "$LOG_FILE"
python - <<'PY' 2>&1 | tee -a "$LOG_FILE"
import json, os, sys
from pathlib import Path
import numpy as np
out_root = Path(os.environ["OUT_ROOT"])
combined = out_root / "combined"; combined.mkdir(parents=True, exist_ok=True)
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
        r["pb_subset"] = r.get("pb_subset", d.name); r["id"] = f"{d.name}::{r['id']}"
    metas.extend(rows)
big = np.concatenate(hs, axis=0) if hs else np.zeros((0,))
np.save(combined / "pb_step_h.npy", big)
with (combined / "pb_step_meta.jsonl").open("w") as f:
    for r in metas: f.write(json.dumps(r, ensure_ascii=False) + "\n")
(combined / "encoding_manifest_pb.json").write_text(json.dumps({
    "n_subsets": len(subset_dirs), "subsets": [d.name for d in subset_dirs],
    "n_rows": int(big.shape[0]), "id_namespacing": "<subset>::<id>"}, indent=2))
print(f"[combined] {big.shape[0]} rows across {len(subset_dirs)} subsets")
PY
echo "[$(date -Iseconds)] pb_encode_7b done" | tee -a "$LOG_FILE"
