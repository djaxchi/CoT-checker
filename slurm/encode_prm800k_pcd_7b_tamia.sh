#!/bin/bash
#SBATCH --job-name=pcd_encode_7b
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out

# Stage 2+3 of the pcd pipeline. Encode the future-delta (pcd) vectors for the
# PRM800K splits from the generated-continuation jsonl (data_pcd/*_next.jsonl),
# 4-GPU sharded then merged into the dense harness cache; then derive the
# ProcessBench pcd from the full-solution store (offline). Feeds
# train_harness_7b_tamia.sh with REP_TAG=pcd.

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH/cot_mech/dense_full_7b_v1}"
GEN_DIR="$RUN_ROOT/data_pcd"
CACHE_DIR="$RUN_ROOT/cache/qwen2_5_7b_pcd"
PB_FULL="$RUN_ROOT/repstore/pb_full_solution"
PB_OUT="$RUN_ROOT/cache/qwen2_5_7b_pcd_pb"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-7B}"
HF_CACHE="${HF_CACHE:-/project/aip-azouaq/$USER/hf_cache}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
NUM_SHARDS="${NUM_SHARDS:-4}"

mkdir -p "$CACHE_DIR" "$PB_OUT"
export HF_HOME="$HF_CACHE"; export TRANSFORMERS_CACHE="$HF_CACHE"
export HF_HUB_OFFLINE=1; export TRANSFORMERS_OFFLINE=1; export TOKENIZERS_PARALLELISM=false
cd "$PROJECT_ROOT"

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch transformers numpy

# cache_stem : generated_jsonl  (stem names must match the harness TRAIN/VAL/TEST)
declare -A STEMS=(
  [probe_train_full]="$GEN_DIR/prm800k_probe_train_full_next.jsonl"
  [val_5k]="$GEN_DIR/prm800k_val_5k_next.jsonl"
  [test_2k]="$GEN_DIR/prm800k_test_2k_next.jsonl"
)

for stem in "${!STEMS[@]}"; do
  in_jsonl="${STEMS[$stem]}"
  [[ -f "$in_jsonl" ]] || { echo "[FATAL] missing $in_jsonl"; exit 2; }
  echo "[encode] $stem <- $in_jsonl"
  pids=()
  for i in $(seq 0 $((NUM_SHARDS-1))); do
    CUDA_VISIBLE_DEVICES=$i python scripts/encode_prm800k_pcd.py \
      --in_jsonl "$in_jsonl" --out_dir "$CACHE_DIR" --stem "$stem" \
      --model_name_or_path "$MODEL_NAME_OR_PATH" --local_files_only \
      --layer -1 --max_seq_len "$MAX_SEQ_LEN" --batch_size "$BATCH_SIZE" \
      --shard_idx "$i" --num_shards "$NUM_SHARDS" &
    pids+=($!)
  done
  fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
  [[ "$fail" == "0" ]] || { echo "[FATAL] a shard failed for $stem"; exit 1; }
  python - "$CACHE_DIR" "$stem" "$NUM_SHARDS" <<'PY'
import sys, numpy as np, pathlib
cache, stem, ns = sys.argv[1], sys.argv[2], int(sys.argv[3])
cd = pathlib.Path(cache)
H = np.concatenate([np.load(cd / f"{stem}_h.shard{i}.npy") for i in range(ns)], 0)
Y = np.concatenate([np.load(cd / f"{stem}_y.shard{i}.npy") for i in range(ns)], 0)
np.save(cd / f"{stem}_h.npy", H); np.save(cd / f"{stem}_y.npy", Y)
for i in range(ns):
    (cd / f"{stem}_h.shard{i}.npy").unlink(); (cd / f"{stem}_y.shard{i}.npy").unlink()
print(f"[merge] {stem}: H={H.shape} Y={Y.shape}", flush=True)
PY
done

echo "[derive-pb] pcd from full-solution store"
python scripts/derive_pb_pcd_from_full_store.py \
  --store_root "$PB_FULL" --out_dir "$PB_OUT"

echo "[verify]"; ls -la "$CACHE_DIR"/*_h.npy; ls "$PB_OUT"/*/pb_step_h.npy
echo "[$(date)] pcd_encode_7b done"
