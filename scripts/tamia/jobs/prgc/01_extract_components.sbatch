#!/bin/bash
#SBATCH --job-name=prgc-01-components
#SBATCH --account=aip-azouaq
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --gpus-per-node=h100:4
#SBATCH --output=results/logs/%x-%j.out
#SBATCH --error=results/logs/%x-%j.err

# ---------------------------------------------------------------------------
# parametric_retrieval_component_v1 stage 3b: per-layer attention- and
# MLP-contribution extraction at the final prompt token (4 fact-sharded
# workers -> merge). Reuses the existing access_v1 run dir: needs
# extraction_set.json, metadata.parquet, grading.jsonl (all already present).
#
# Component states (~8 GB) are redirected to $SCRATCH via symlink; $HOME must
# never hold them.
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

EXTRA=()
[[ -n "${LIMIT:-}" ]] && EXTRA+=(--limit "$LIMIT")
[[ -n "${FORCE:-}" ]] && EXTRA+=(--force)

# ---- redirect bulky component states to $SCRATCH --------------------------
tgt="$SCRATCH/prga_access_v1/component_states_v1"
mkdir -p "$tgt"
if [[ -e "$RUN_DIR/component_states_v1" && ! -L "$RUN_DIR/component_states_v1" ]]; then
    echo "ERROR: $RUN_DIR/component_states_v1 exists and is not a symlink" >&2
    exit 1
fi
ln -sfn "$tgt" "$RUN_DIR/component_states_v1"

EXT=scripts/parametric_retrieval/prgc_extract_components.py
pids=()
for s in $(seq 0 $((NUM_SHARDS - 1))); do
    CUDA_VISIBLE_DEVICES=$s python "$EXT" \
        --out_dir "$RUN_DIR" --model_name_or_path "$MODEL" \
        --local_files_only --model_dtype bfloat16 \
        --shard_idx "$s" --num_shards "$NUM_SHARDS" "${EXTRA[@]}" \
        > "results/logs/prgc-01-components-${SLURM_JOB_ID}_shard${s}.out" 2>&1 &
    pids+=($!)
done
rc=0
for s in $(seq 0 $((NUM_SHARDS - 1))); do
    if ! wait "${pids[$s]}"; then
        echo "shard $s FAILED (see results/logs/prgc-01-components-${SLURM_JOB_ID}_shard${s}.out)"
        rc=1
    else
        echo "shard $s done"
    fi
done
[[ $rc -ne 0 ]] && exit $rc

python "$EXT" --merge --num_shards "$NUM_SHARDS" --out_dir "$RUN_DIR"
echo "COMPONENT EXTRACTION COMPLETE"
