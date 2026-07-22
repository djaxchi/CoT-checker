#!/bin/bash
#SBATCH --job-name=prgm-01-neurons
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
# parametric_retrieval_minimal_v1 stage 3c: MLP intermediate-activation (g)
# extraction at the final prompt token, at the MLP-selected layer (from
# expE/selection.json), 4 fact-sharded workers -> merge. Reuses the access_v1
# run dir. g states (~0.8 GB) redirect to $SCRATCH.
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

tgt="$SCRATCH/prga_access_v1/neuron_states_v1"
mkdir -p "$tgt"
if [[ -e "$RUN_DIR/neuron_states_v1" && ! -L "$RUN_DIR/neuron_states_v1" ]]; then
    echo "ERROR: $RUN_DIR/neuron_states_v1 exists and is not a symlink" >&2
    exit 1
fi
ln -sfn "$tgt" "$RUN_DIR/neuron_states_v1"

EXT=scripts/parametric_retrieval/prgm_extract_neurons.py
pids=()
for s in $(seq 0 $((NUM_SHARDS - 1))); do
    CUDA_VISIBLE_DEVICES=$s python "$EXT" \
        --out_dir "$RUN_DIR" --model_name_or_path "$MODEL" \
        --local_files_only --model_dtype bfloat16 \
        --shard_idx "$s" --num_shards "$NUM_SHARDS" "${EXTRA[@]}" \
        > "results/logs/prgm-01-neurons-${SLURM_JOB_ID}_shard${s}.out" 2>&1 &
    pids+=($!)
done
rc=0
for s in $(seq 0 $((NUM_SHARDS - 1))); do
    if ! wait "${pids[$s]}"; then
        echo "shard $s FAILED (see results/logs/prgm-01-neurons-${SLURM_JOB_ID}_shard${s}.out)"
        rc=1
    else
        echo "shard $s done"
    fi
done
[[ $rc -ne 0 ]] && exit $rc

python "$EXT" --merge --num_shards "$NUM_SHARDS" --out_dir "$RUN_DIR"
echo "NEURON EXTRACTION COMPLETE"
