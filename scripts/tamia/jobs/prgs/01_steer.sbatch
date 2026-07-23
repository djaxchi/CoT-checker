#!/bin/bash
#SBATCH --job-name=prgs-01-steer
#SBATCH --account=aip-azouaq
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --gpus-per-node=h100:4
#SBATCH --output=results/logs/%x-%j.out
#SBATCH --error=results/logs/%x-%j.err

# ---------------------------------------------------------------------------
# parametric_retrieval_steer_v1: Golden-Gate-style clamp of the prgm
# flip-neurons on unrelated prompts. Single job:
#   run (4 shards by fact) -> analyze (curves + specificity matrix)
#
# Reuses (all already on the cluster): neuron_states_v1 (max activations),
# expF/run.parquet (fact->neuron map), expE/selection.json (mlp layer),
# metadata.parquet. No new extraction.
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

EXP=scripts/parametric_retrieval/prgs_steer.py
FORCEFLAG=()
[[ -n "${FORCE:-}" ]] && FORCEFLAG+=(--force)

echo "== run: neuron steering (4 shards by fact) =="
pids=()
for s in $(seq 0 $((NUM_SHARDS - 1))); do
    CUDA_VISIBLE_DEVICES=$s python "$EXP" \
        --out_dir "$RUN_DIR" --model_name_or_path "$MODEL" \
        --local_files_only --phase run \
        --shard_idx "$s" --num_shards "$NUM_SHARDS" "${FORCEFLAG[@]}" \
        > "results/logs/prgs-01-run-${SLURM_JOB_ID}_shard${s}.out" 2>&1 &
    pids+=($!)
done
rc=0
for s in $(seq 0 $((NUM_SHARDS - 1))); do
    wait "${pids[$s]}" || { echo "run shard $s FAILED"; rc=1; }
done
[[ $rc -ne 0 ]] && exit $rc

echo "== analyze =="
python "$EXP" --out_dir "$RUN_DIR" --num_shards "$NUM_SHARDS" --analyze
echo "ALL PRGS STEERING COMPLETE"
