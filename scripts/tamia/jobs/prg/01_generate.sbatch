#!/bin/bash
#SBATCH --job-name=prg-01-gen
#SBATCH --account=aip-azouaq
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gpus=h100:1
#SBATCH --array=0-3
#SBATCH --output=results/logs/%x-%A_%a.out
#SBATCH --error=results/logs/%x-%A_%a.err

# ---------------------------------------------------------------------------
# parametric_retrieval_geometry_v0 stage 1: generation + hidden-state
# extraction, one WikiProfile shard per array task (facts stay whole within a
# shard). Requires runs/parametric_retrieval_geometry_v0/metadata.parquet
# (materialized locally by prg_sample_facts.py and synced) and
# Qwen/Qwen2.5-7B-Instruct in the HF cache (download on a login node first:
#   HF_HOME=$HF_CACHE_ROOT huggingface-cli download Qwen/Qwen2.5-7B-Instruct )
#
# --time is a smoke-based estimate (~1k instances/shard, ~2.6k generations +
# ~2k teacher-forced forwards); override on submit if the smoke run says
# otherwise:  sbatch --time=... scripts/tamia/jobs/prg/01_generate.sbatch
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

echo "======================================================="
echo "  PRG stage 01: generate + extract, shard $SLURM_ARRAY_TASK_ID/$NUM_SHARDS"
echo "  run dir : $RUN_DIR"
echo "  model   : $MODEL"
echo "======================================================="

python scripts/parametric_retrieval/prg_generate.py \
    --out_dir "$RUN_DIR" \
    --model_name_or_path "$MODEL" \
    --local_files_only \
    --model_dtype bfloat16 \
    --shard_idx "$SLURM_ARRAY_TASK_ID" \
    --num_shards "$NUM_SHARDS"
