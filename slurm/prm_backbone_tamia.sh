#!/bin/bash
#SBATCH --job-name=prm_backbone
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=20:00:00
#SBATCH --output=%x-%j.out

# prm_backbone_v1 on TamIA: encode with Qwen2.5-Math-PRM-7B into NODE-LOCAL
# disk, then train one group of the Instruct leaderboard roster on it.
#
# Why node-local: the PRM stores need ~155 GB and shared scratch had 127 GB free
# (2026-10-02). TamIA nodes have 7.2 TB of $SLURM_TMPDIR and 500 GB of RAM, and
# the encode is ~1h of the node, so each job encodes its own copy. The stores
# are deterministic functions of the same inputs; the cells' input fingerprints
# say whether the copies agree.
#
# Spans are read through the page cache of the local file (PRELOAD_BUDGET_GB=1
# disables the per-process RAM copy), so four concurrent sequence cells share one
# 139 GB copy instead of needing four.
#
# Protocol is the Instruct arm's: same frozen splits, verifier template, span
# store, RESCALE=none, 30 epochs, patience 3, batch 256, seeds 42 43 44 with the
# seed-42 search reused. Loading and layer choice: see
# slurm/prm_backbone_encode_rorqual.sh (AutoModelForCausalLM, 0 missing keys;
# LAYER=27 = block 26 of 28, matching Instruct's block 34 of 36).
#
#   GROUP=vectors | seq_a | seq_b   (cell files in $RUN_ROOT/cellfiles)
#   GROUP=geometry  closed-form geometry study (scripts/analysis/prm_geometry.py)
#                   on these PRM states and on the Instruct stores, no learner

set -euo pipefail
: "${GROUP:?vectors, seq_a, seq_b or geometry}"
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH/cot_mech/prm_backbone_v1}"
DATA_DIR="${DATA_DIR:-$SCRATCH/cot_mech/dense_full_7b_v1/data}"
PB_DIR="${PB_DIR:-/scratch/d/dchikhi/cot-checker/processbench_full}"
HF_CACHE="${HF_CACHE:-/project/aip-azouaq/$USER/hf_cache}"
MODEL="${MODEL:-Qwen/Qwen2.5-Math-PRM-7B}"
LAYER="${LAYER:-27}"
LOCAL="$SLURM_TMPDIR/prm_backbone"
CELLS_FILE="$RUN_ROOT/cellfiles/$GROUP.cells"
[[ "$GROUP" == geometry || -r "$CELLS_FILE" ]] || { echo "[FATAL] no $CELLS_FILE" >&2; exit 2; }
INSTRUCT_ROOT="${INSTRUCT_ROOT:-$SCRATCH/cot_mech/qwen3_8b_instruct_v1/repstore}"
mkdir -p "$LOCAL/repstore" "$RUN_ROOT/cells" "$RUN_ROOT/logs"

module purge
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0
export HF_HOME="$HF_CACHE"
unset TRANSFORMERS_CACHE
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$PROJECT_ROOT"
echo "git $(git rev-parse --short HEAD) group=$GROUP model=$MODEL layer=$LAYER local=$LOCAL"
df -h "$SLURM_TMPDIR" | tail -1

virtualenv --no-download "$SLURM_TMPDIR/encenv"
source "$SLURM_TMPDIR/encenv/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch transformers numpy 2>&1 | tail -1

common=(--model_name_or_path "$MODEL" --local_files_only --span_only --layer "$LAYER"
        --max_seq_len 2048 --batch_size 8 --model_dtype bfloat16 --num_shards 4)
LOG="$RUN_ROOT/logs/encode_${GROUP}-${SLURM_JOB_ID}.log"
pids=()
for i in 0 1 2 3; do
  ( CUDA_VISIBLE_DEVICES=$i python scripts/encode_processbench_token_store.py \
      --raw_specs gsm8k:$PB_DIR/processbench_gsm8k.jsonl math:$PB_DIR/processbench_math.jsonl \
                  olympiadbench:$PB_DIR/processbench_olympiadbench.jsonl \
                  omnimath:$PB_DIR/processbench_omnimath.jsonl \
      --rep_root "$LOCAL/repstore/pb_step_spans" --shard_idx $i "${common[@]}" &&
    CUDA_VISIBLE_DEVICES=$i python scripts/encode_prm800k_token_store.py \
      --data_dir "$DATA_DIR" --rep_root "$LOCAL/repstore/step_spans" \
      --splits prm800k_val_5k.jsonl:val_5k prm800k_test_2k.jsonl:test_2k \
               prm800k_probe_train_full.jsonl:probe_train_full \
      --shard_idx $i "${common[@]}" ) >>"$LOG.shard$i" 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
if [[ "$fail" == 1 ]]; then tail -20 "$LOG".shard* >&2; exit 1; fi
du -sh "$LOCAL"/repstore/* | tee -a "$LOG"
echo "[$(date)] encode done"

if [[ "$GROUP" == geometry ]]; then
  # One backbone per GPU, concurrently; each holds its train split in RAM.
  GEO="$RUN_ROOT/geometry"
  CUDA_VISIBLE_DEVICES=0 python scripts/analysis/prm_geometry.py --out_dir "$GEO" \
    --backbone "prm=$LOCAL/repstore/step_spans,$LOCAL/repstore/pb_step_spans" \
    > "$RUN_ROOT/logs/geometry_prm-${SLURM_JOB_ID}.log" 2>&1 &
  p1=$!
  CUDA_VISIBLE_DEVICES=1 python scripts/analysis/prm_geometry.py --out_dir "$GEO" \
    --backbone "instruct=$INSTRUCT_ROOT/step_spans,$INSTRUCT_ROOT/pb_step_spans" \
    > "$RUN_ROOT/logs/geometry_instruct-${SLURM_JOB_ID}.log" 2>&1 &
  p2=$!
  fail=0
  wait $p1 || fail=1
  wait $p2 || fail=1
  tail -30 "$RUN_ROOT"/logs/geometry_*-"${SLURM_JOB_ID}".log
  [[ "$fail" == 0 ]] || { echo "[FATAL] geometry failed" >&2; exit 1; }
  echo "[$(date)] geometry done"
  exit 0
fi
deactivate

export PROJECT_ROOT RUN_ROOT
export PRM_STORE="$LOCAL/repstore/step_spans" PB_STORE="$LOCAL/repstore/pb_step_spans"
export OUT_ROOT="$RUN_ROOT/cells" VEC_CACHE="$LOCAL/grid_vectors" CELLS_FILE
export RESCALE=none SEEDS="42 43 44" EPOCHS=30 BATCH_SIZE=256 HP_SEARCH_CAP=100000
export N_GPUS=4 PRELOAD_BUDGET_GB=1
module purge
bash "$PROJECT_ROOT/slurm/train_rep_grid_7b_tamia.sh"
