#!/bin/bash
#SBATCH --job-name=s1ms_probe_L20
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=00:45:00
#SBATCH --output=%x-%j.out

# Train the SAME 7B DenseLinear probe as the deployed L28 one, but on L20/last.
# The only change vs the L28 pipeline is the encode layer: reuse the exact frozen
# Sprint-1 PRM800K splits (probe_train_40k, val_1k), the exact trainer
# (s1ms_train_dense_probe.py), and the exact held-out eval; just feed L20/last
# features via encode_prm800k_hidden_states.py --layer 20.
#
# TamIA allocates h100 by node, so we take all 4 GPUs, shard the 41k-step encode
# across them into shard_0N/ dirs, merge, then train + eval on one GPU.
#
#   -> $RUNS_ROOT/qwen2_5_7b_L20/linear_probe.pt (+ threshold.json, train_metrics.json)
#   -> results/prm800k_heldout_eval/7B_L20.json  (appended to heldout_eval.csv)
#
# Usage:
#   sbatch slurm/s1ms_dense_probe_L20_tamia.sh
#   FORCE=1 sbatch slurm/s1ms_dense_probe_L20_tamia.sh     # re-encode + overwrite

set -uo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
cd "$PROJECT_ROOT"
source slurm/s1_model_size/models.env   # S1MS_MODEL_ID, S1MS_BATCH, RUNS_ROOT, PRM_SPLIT_DIR, ...

TAG="${TAG:-qwen2_5_7b}"
LAYER="${LAYER:-20}"
MODEL_ID="${S1MS_MODEL_ID[$TAG]}"
BATCH="${BATCH:-${S1MS_BATCH[$TAG]}}"
NUM_SHARDS="${NUM_SHARDS:-4}"

CACHE="${CACHE:-$RUNS_ROOT/${TAG}_L${LAYER}/cache}"
RUN_DIR="${RUN_DIR:-$RUNS_ROOT/${TAG}_L${LAYER}}"
HELDOUT_ENC_DIR="${HELDOUT_ENC_DIR:-$RUNS_ROOT/$TAG/prm_multitoken}"
EVAL_OUT="${EVAL_OUT:-$PROJECT_ROOT/results/prm800k_heldout_eval}"

export HF_HOME="${HF_HOME:-${HF_CACHE_ROOT:-$SCRATCH/hf_cache}}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
mkdir -p "$CACHE/shards" "$RUN_DIR" "$EVAL_OUT"

cat <<BANNER
================================================================
job    : ${SLURM_JOB_NAME:-s1ms_probe_L20}  id ${SLURM_JOB_ID:-N/A}
host   : $(hostname)  date $(date -Iseconds)  git $(git rev-parse --short HEAD 2>/dev/null || echo ?)
model  : $MODEL_ID   layer L$LAYER   batch $BATCH   shards $NUM_SHARDS
splits : $PRM_SPLIT_DIR/{$PRM_TRAIN_JSONL:probe_train_40k, $PRM_VAL_JSONL:val_1k}
run_dir: $RUN_DIR
eval   : $HELDOUT_ENC_DIR (L$LAYER,last) -> $EVAL_OUT/7B_L${LAYER}.json
================================================================
BANNER

for j in "$PRM_TRAIN_JSONL" "$PRM_VAL_JSONL"; do
  [[ -f "$PRM_SPLIT_DIR/$j" ]] || { echo "[FATAL] missing $PRM_SPLIT_DIR/$j"; exit 1; }
done

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch transformers numpy scikit-learn

EXTRA=()
[[ "${FORCE:-0}" == "1" ]] && EXTRA+=(--force)

# ---- 1. Encode L$LAYER/last for both splits, sharded one process per GPU ----
pids=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  sdir="$CACHE/shards/shard_0$i"; mkdir -p "$sdir"
  CUDA_VISIBLE_DEVICES="$i" python scripts/encode_prm800k_hidden_states.py \
    --data_dir "$PRM_SPLIT_DIR" --out_dir "$sdir" \
    --model_name_or_path "$MODEL_ID" --local_files_only \
    --run_name "L${LAYER}_${TAG}" --max_seq_len -1 --batch_size "$BATCH" \
    --model_dtype float16 --save_dtype float16 --layer "$LAYER" \
    --splits "${PRM_TRAIN_JSONL}:probe_train_40k" "${PRM_VAL_JSONL}:val_1k" \
    --shard_idx "$i" --num_shards "$NUM_SHARDS" "${EXTRA[@]}" \
    > "$CACHE/encode_shard${i}.log" 2>&1 &
  pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
if [[ $fail -ne 0 ]]; then
  echo "[FATAL] an encode shard failed:"; tail -n 40 "$CACHE"/encode_shard*.log; exit 1
fi

# ---- 2. Merge shards per split ----
for stem in probe_train_40k val_1k; do
  python scripts/merge_prm800k_encoded_shards.py \
    --shard_root "$CACHE/shards" --stem "$stem" --out_dir "$CACHE" "${EXTRA[@]}" \
    || { echo "[FATAL] merge $stem failed"; exit 1; }
done

# ---- 3. Train the DenseLinear probe (identical protocol to L28) ----
python scripts/s1ms_train_dense_probe.py \
  --cache_dir "$CACHE" --out_dir "$RUN_DIR" \
  --probe_train_stem probe_train_40k --val_stem val_1k \
  --model_name "$MODEL_ID" || { echo "[FATAL] training failed"; exit 1; }

# ---- 4. Eval on the 6k held-out test at L$LAYER/last (reuses multitoken encode) ----
if [[ -f "$HELDOUT_ENC_DIR/prm800k_heldout_test_h.npy" ]]; then
  python scripts/eval_prm800k_heldout_probe.py \
    --run_dir "$RUN_DIR" --enc_dir "$HELDOUT_ENC_DIR" \
    --stem prm800k_heldout_test --layer "$LAYER" --token last \
    --tag "7B_L${LAYER}" --out_dir "$EVAL_OUT" --csv \
    || echo "[WARN] heldout eval failed (probe still trained)"
else
  echo "[WARN] no $HELDOUT_ENC_DIR/prm800k_heldout_test_h.npy; skipping eval."
  echo "       (re-encode the heldout multitoken, or point HELDOUT_ENC_DIR at it.)"
fi

echo "[$(date -Iseconds)] L$LAYER probe done -> $RUN_DIR/linear_probe.pt"
