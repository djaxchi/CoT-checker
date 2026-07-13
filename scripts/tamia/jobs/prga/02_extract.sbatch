#!/bin/bash
#SBATCH --job-name=prga-02-extract
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
# parametric_retrieval_access_v1 stages 2+3+4, chained after 01 via
# --dependency=afterok: grading + pairing + candidate sets (prga_pairs.py,
# CPU, runs in-job so no laptop round trip), then all-layer multi-position
# residual extraction (4 fact-sharded workers), merge, and the logit-lens
# answer-content readout (Experiment A method A) on GPU 0.
#
# Requires in the run dir: metadata.parquet + generations.jsonl (from 01),
# and data/wikiprofile/wikiprofile.csv synced (compute nodes are offline).
#
# Hidden states (tens of GB) are redirected to $SCRATCH via symlink; $HOME
# must never hold them (~50 GB quota).
#
# Submit chained:
#   sbatch --dependency=afterok:<JOB01> scripts/tamia/jobs/prga/02_extract.sbatch
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

echo "======================================================="
echo "  PRGA stage 02: pairs -> extract all layers ($NUM_SHARDS shards)"
echo "                 -> merge -> logit lens"
echo "  run dir : $RUN_DIR"
echo "======================================================="

# ---- stage 2: grading + pairs + candidates + extraction set (CPU) ---------
# Deterministic from generations.jsonl; safe to rerun on requeue.
python scripts/parametric_retrieval/prga_pairs.py \
    --out_dir "$RUN_DIR" \
    --csv_cache data/wikiprofile/wikiprofile.csv

# ---- redirect bulky hidden states to $SCRATCH ------------------------------
for sub in hidden_states_v1 logitlens; do
    tgt="$SCRATCH/prga_access_v1/$sub"
    mkdir -p "$tgt"
    if [[ -e "$RUN_DIR/$sub" && ! -L "$RUN_DIR/$sub" ]]; then
        echo "ERROR: $RUN_DIR/$sub exists and is not a symlink" >&2
        exit 1
    fi
    ln -sfn "$tgt" "$RUN_DIR/$sub"
done

pids=()
for s in $(seq 0 $((NUM_SHARDS - 1))); do
    CUDA_VISIBLE_DEVICES=$s python scripts/parametric_retrieval/prga_extract.py \
        --out_dir "$RUN_DIR" \
        --model_name_or_path "$MODEL" \
        --local_files_only \
        --model_dtype bfloat16 \
        --shard_idx "$s" \
        --num_shards "$NUM_SHARDS" \
        "${EXTRA[@]}" \
        > "results/logs/prga-02-extract-${SLURM_JOB_ID}_shard${s}.out" 2>&1 &
    pids+=($!)
done

rc=0
for s in $(seq 0 $((NUM_SHARDS - 1))); do
    if ! wait "${pids[$s]}"; then
        echo "shard $s FAILED (see results/logs/prga-02-extract-${SLURM_JOB_ID}_shard${s}.out)"
        rc=1
    else
        echo "shard $s done"
    fi
done
[[ $rc -ne 0 ]] && exit $rc

python scripts/parametric_retrieval/prga_extract.py --merge \
    --num_shards "$NUM_SHARDS" --out_dir "$RUN_DIR"

CUDA_VISIBLE_DEVICES=0 python scripts/parametric_retrieval/prga_logitlens.py \
    --out_dir "$RUN_DIR" \
    --model_name_or_path "$MODEL" \
    --local_files_only \
    ${FORCE:+--force}
