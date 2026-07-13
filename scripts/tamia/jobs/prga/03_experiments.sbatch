#!/bin/bash
#SBATCH --job-name=prga-03-exp
#SBATCH --account=aip-azouaq
#SBATCH --time=02:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --gpus-per-node=h100:4
#SBATCH --output=results/logs/%x-%j.out
#SBATCH --error=results/logs/%x-%j.err

# ---------------------------------------------------------------------------
# parametric_retrieval_access_v1 Experiments A(decoder)+B+C+D, one node.
#
#   step 1 (parallel): GPU0 expA candidate decoder | GPU1 expB probes
#   step 2: expC same-fact patching  val(4 shards) -> select -> test -> analyze
#   step 3: expD access subspace     estimate -> val -> select -> test -> analyze
#
# Requires the completed stage 1-4 run dir (generations, grading, pairs,
# candidates, hidden_states_v1 -> $SCRATCH symlink, logitlens scores).
# Experiment E was gated OFF by data (reasoning-unlocked groups already show
# hits@1 0.47 pre-generation, spec section 16 condition not met).
# ---------------------------------------------------------------------------

set -euo pipefail

PROJECT_DIR="$HOME/CoT-checker"
STORE="/project/aip-azouaq/$USER"
RUN_DIR="${RUN_DIR:-runs/parametric_retrieval_access_v1}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
NUM_SHARDS="${NUM_SHARDS:-4}"

cd "$PROJECT_DIR"
mkdir -p results/logs

module purge
module load StdEnv/2023 gcc arrow/24.0.0 python/3.11 cuda/12.2
source "$HOME/venvs/cot/bin/activate"

if [[ -f slurm/s1_model_size/models.env ]]; then
    source slurm/s1_model_size/models.env
fi
export HF_HOME="${HF_CACHE_ROOT:-$STORE/hf_cache}"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

FORCEFLAG=()
[[ -n "${FORCE:-}" ]] && FORCEFLAG+=(--force)

run_shards () {  # script, phase
    local script=$1 phase=$2
    local pids=() rc=0
    for s in $(seq 0 $((NUM_SHARDS - 1))); do
        CUDA_VISIBLE_DEVICES=$s python "$script" \
            --out_dir "$RUN_DIR" --model_name_or_path "$MODEL" \
            --local_files_only --phase "$phase" \
            --shard_idx "$s" --num_shards "$NUM_SHARDS" "${FORCEFLAG[@]}" \
            > "results/logs/prga-03-$(basename "$script" .py)-${phase}-${SLURM_JOB_ID}_shard${s}.out" 2>&1 &
        pids+=($!)
    done
    for s in $(seq 0 $((NUM_SHARDS - 1))); do
        wait "${pids[$s]}" || { echo "$script $phase shard $s FAILED"; rc=1; }
    done
    return $rc
}

echo "== step 1: expA decoder (GPU0) + expB probes (GPU1) =="
CUDA_VISIBLE_DEVICES=0 python scripts/parametric_retrieval/prga_expA_decoder.py \
    --out_dir "$RUN_DIR" --model_name_or_path "$MODEL" --local_files_only \
    > "results/logs/prga-03-expA-${SLURM_JOB_ID}.out" 2>&1 &
A_PID=$!
CUDA_VISIBLE_DEVICES=1 python scripts/parametric_retrieval/prga_expB_probe.py \
    --out_dir "$RUN_DIR" \
    > "results/logs/prga-03-expB-${SLURM_JOB_ID}.out" 2>&1 &
B_PID=$!
wait $A_PID || { echo "expA FAILED"; exit 1; }
wait $B_PID || { echo "expB FAILED"; exit 1; }
echo "step 1 done"

echo "== step 2: expC patching =="
EXPC=scripts/parametric_retrieval/prga_expC_patch.py
run_shards "$EXPC" val
python "$EXPC" --out_dir "$RUN_DIR" --num_shards "$NUM_SHARDS" --select
run_shards "$EXPC" test
python "$EXPC" --out_dir "$RUN_DIR" --num_shards "$NUM_SHARDS" --analyze
echo "step 2 done"

echo "== step 3: expD access subspace =="
EXPD=scripts/parametric_retrieval/prga_expD_steer.py
CUDA_VISIBLE_DEVICES=0 python "$EXPD" --out_dir "$RUN_DIR" --estimate
run_shards "$EXPD" val
python "$EXPD" --out_dir "$RUN_DIR" --num_shards "$NUM_SHARDS" --select
run_shards "$EXPD" test
python "$EXPD" --out_dir "$RUN_DIR" --num_shards "$NUM_SHARDS" --analyze
echo "step 3 done"
echo "ALL EXPERIMENTS COMPLETE"
