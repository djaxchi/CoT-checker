#!/bin/bash
#SBATCH --job-name=cg_stage1
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out

# cot_causal_graph_v0 Stage 1: (a) Stage-0 trace build if missing (offline HF
# cache), (b) on-policy trajectory generation if missing (GPU 0), (c) teacher-
# forced interventions, arm both, 4 in-node shards, (d) CPU merge + gates G1/G3.
#
# Usage:  sbatch slurm/causal_graph_stage1_tamia.sh

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_DIR="${RUN_DIR:-$PROJECT_ROOT/runs/causal_graph}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-7B}"
DIRECTIONS_DIR="${DIRECTIONS_DIR:-$PROJECT_ROOT/runs/fork_rep_audit/qwen2_5_7b/steering}"
N_TRACES="${N_TRACES:-800}"

source "$PROJECT_ROOT/slurm/s1_model_size/models.env"
export HF_HOME="${HF_HOME:-$HF_CACHE_ROOT}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

cd "$PROJECT_ROOT"
for f in "$DIRECTIONS_DIR/directions_L28.npz" "$DIRECTIONS_DIR/directions_L20.npz"; do
  test -e "$f" || { echo "missing $f (build_steering_directions.py output)"; exit 1; }
done

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch transformers numpy scipy scikit-learn pandas pyarrow huggingface_hub

cat <<BANNER
================================================================
job        : ${SLURM_JOB_NAME:-cg_stage1}  id ${SLURM_JOB_ID:-N/A}
host       : $(hostname)   date $(date -Iseconds)
run_dir    : $RUN_DIR
model      : $MODEL_NAME_OR_PATH
directions : $DIRECTIONS_DIR
================================================================
BANNER

# ---- (a) Stage 0 build (CPU, deterministic seed 42; PRM800K from offline cache)
if [ ! -e "$RUN_DIR/traces_forks.jsonl" ]; then
  python scripts/causal_graph/cg_build_traces.py \
    --run_dir "$RUN_DIR" --n_traces "$N_TRACES"
fi

# ---- (b) on-policy trajectories (GPU 0) --------------------------------------
if [ ! -e "$RUN_DIR/onpolicy_trajectories.jsonl" ]; then
  CUDA_VISIBLE_DEVICES=0 python scripts/generate_onpolicy_steps.py \
    --fork_items "$RUN_DIR/onpolicy_problems.jsonl" \
    --out_dir "$RUN_DIR" --stem onpolicy \
    --model_name_or_path "$MODEL_NAME_OR_PATH" --local_files_only \
    --run_name causal_graph_v0 --n_samples 4 --max_problems 300
fi

# ---- (c) teacher-forced pass, 4 in-node shards -------------------------------
pids=()
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g python scripts/causal_graph/cg_stage1_tf.py \
    --run_dir "$RUN_DIR" --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --directions_npz "$DIRECTIONS_DIR/directions_L28.npz" \
                     "$DIRECTIONS_DIR/directions_L20.npz" \
    --arm both --onpolicy_trajectories "$RUN_DIR/onpolicy_trajectories.jsonl" \
    --shard_id $g --num_shards 4 --local_files_only \
    > "cg_stage1_shard$g-${SLURM_JOB_ID:-manual}.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done

# ---- (d) merge + gates (CPU) -------------------------------------------------
python scripts/causal_graph/cg_stage1_tf.py --run_dir "$RUN_DIR" --merge

echo "[cg_stage1] done $(date -Iseconds)"
