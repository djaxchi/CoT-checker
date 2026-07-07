#!/bin/bash
#SBATCH --job-name=prg-05-roll
#SBATCH --account=aip-azouaq
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --gpus-per-node=h100:4
#SBATCH --output=results/logs/%x-%j.out
#SBATCH --error=results/logs/%x-%j.err

# ---------------------------------------------------------------------------
# parametric_retrieval_geometry_v0 exp 2: within-instance CoT rollout
# divergence. 4 shards in-node (one GPU each), then merge + analysis on CPU.
# Generates k=8 CoT rollouts for every mixed-outcome instance and saves
# per-rollout hidden states at trajectory checkpoints. --time is a generous
# estimate; tighten after a smoke.
#
# Smoke (whole node): sbatch --time=00:30:00 --export=ALL,MAXI=20 \
#     scripts/tamia/jobs/prg/05_rollouts.sbatch
# ---------------------------------------------------------------------------

set -euo pipefail
PROJECT_DIR="$HOME/CoT-checker"
STORE="/project/aip-azouaq/$USER"
RUN_DIR="${RUN_DIR:-runs/parametric_retrieval_geometry_v0}"
LAYERS="${LAYERS:-20 24}"
K="${K:-8}"
NUM_SHARDS=4

cd "$PROJECT_DIR"
mkdir -p results/logs
module purge
module load StdEnv/2023 gcc arrow/24.0.0 python/3.11 cuda/12.2
source "$HOME/venvs/cot/bin/activate"
[[ -f slurm/s1_model_size/models.env ]] && source slurm/s1_model_size/models.env
export HF_HOME="${HF_CACHE_ROOT:-$STORE/hf_cache}"
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

EXTRA=()
[[ -n "${MAXI:-}" ]] && EXTRA+=(--max_instances "$MAXI")

pids=()
for s in $(seq 0 $((NUM_SHARDS - 1))); do
    CUDA_VISIBLE_DEVICES=$s python scripts/parametric_retrieval/prg_rollouts.py \
        --out_dir "$RUN_DIR" \
        --model_name_or_path Qwen/Qwen2.5-7B-Instruct --local_files_only \
        --layers $LAYERS --k "$K" \
        --shard_idx "$s" --num_shards "$NUM_SHARDS" "${EXTRA[@]}" \
        > "results/logs/prg-05-roll-${SLURM_JOB_ID}_shard${s}.out" 2>&1 &
    pids+=($!)
done
rc=0
for s in $(seq 0 $((NUM_SHARDS - 1))); do
    wait "${pids[$s]}" || { echo "shard $s FAILED"; rc=1; }
done
[[ $rc -ne 0 ]] && exit $rc

echo "--- merge ---"
python scripts/parametric_retrieval/prg_rollouts.py --merge \
    --num_shards "$NUM_SHARDS" --out_dir "$RUN_DIR" --layers $LAYERS
echo "--- analysis ---"
python scripts/parametric_retrieval/prg_rollout_analysis.py \
    --out_dir "$RUN_DIR" --hs $LAYERS
