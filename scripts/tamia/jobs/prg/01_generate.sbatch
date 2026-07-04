#!/bin/bash
#SBATCH --job-name=prg-01-gen
#SBATCH --account=aip-azouaq
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --gpus-per-node=h100:4
#SBATCH --output=results/logs/%x-%j.out
#SBATCH --error=results/logs/%x-%j.err

# ---------------------------------------------------------------------------
# parametric_retrieval_geometry_v0 stage 1: generation + hidden-state
# extraction. TamIA allocates H100s by whole node, so this is ONE node job
# that runs the 4 WikiProfile shards in parallel, one per GPU
# (CUDA_VISIBLE_DEVICES), and fails if any shard fails.
#
# Requires runs/parametric_retrieval_geometry_v0/metadata.parquet (committed)
# and Qwen/Qwen2.5-7B-Instruct in $HF_CACHE_ROOT (download on a login node).
#
# Smoke (whole node, ~10 min):
#   sbatch --time=00:30:00 --export=ALL,LIMIT=40 \
#       scripts/tamia/jobs/prg/01_generate.sbatch
# --time above is a smoke-based estimate; override on submit if needed.
# ---------------------------------------------------------------------------

set -euo pipefail

PROJECT_DIR="$HOME/CoT-checker"
STORE="/project/aip-azouaq/$USER"
RUN_DIR="${RUN_DIR:-runs/parametric_retrieval_geometry_v0}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
NUM_SHARDS="${NUM_SHARDS:-4}"

cd "$PROJECT_DIR"
mkdir -p results/logs

module purge
module load StdEnv/2023 gcc arrow/24.0.0 python/3.11 cuda/12.2
source "$HOME/venvs/cot/bin/activate"

# Qwen weights live in HF_CACHE_ROOT (models.env), not $SCRATCH/hf_cache.
if [[ -f slurm/s1_model_size/models.env ]]; then
    source slurm/s1_model_size/models.env
fi
export HF_HOME="${HF_CACHE_ROOT:-$STORE/hf_cache}"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

EXTRA=()
[[ -n "${LIMIT:-}" ]] && EXTRA+=(--limit "$LIMIT")
[[ -n "${FORCE:-}" ]] && EXTRA+=(--force)  # overwrite smoke shard outputs

echo "======================================================="
echo "  PRG stage 01: generate + extract, $NUM_SHARDS shards on one node"
echo "  run dir : $RUN_DIR"
echo "  model   : $MODEL"
echo "  limit   : ${LIMIT:-none}"
echo "======================================================="

pids=()
for s in $(seq 0 $((NUM_SHARDS - 1))); do
    CUDA_VISIBLE_DEVICES=$s python scripts/parametric_retrieval/prg_generate.py \
        --out_dir "$RUN_DIR" \
        --model_name_or_path "$MODEL" \
        --local_files_only \
        --model_dtype bfloat16 \
        --shard_idx "$s" \
        --num_shards "$NUM_SHARDS" \
        "${EXTRA[@]}" \
        > "results/logs/prg-01-gen-${SLURM_JOB_ID}_shard${s}.out" 2>&1 &
    pids+=($!)
done

rc=0
for s in $(seq 0 $((NUM_SHARDS - 1))); do
    if ! wait "${pids[$s]}"; then
        echo "shard $s FAILED (see results/logs/prg-01-gen-${SLURM_JOB_ID}_shard${s}.out)"
        rc=1
    else
        echo "shard $s done"
    fi
done
exit $rc
