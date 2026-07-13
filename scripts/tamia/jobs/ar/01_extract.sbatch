#!/bin/bash
#SBATCH --job-name=ar-01-extract
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
# attention_routing_v0: attention-feature extraction for the 4000 matched
# correct/incorrect PRM800K forks (runs/contrib_cluster/forks.jsonl) on
# Qwen2.5-7B base, teacher-forced, output_attentions=eager, batch size 1.
#
# 4 fork-sharded workers pinned to one GPU each via CUDA_VISIBLE_DEVICES
# (whole-node H100 allocation; never job arrays), then merge, then the
# paired analysis (CPU) so one rsync brings back tables + plots.
#
# Requires synced from laptop (sync.sh covers code only):
#   runs/contrib_cluster/forks.jsonl
# Requires in $HF_CACHE_ROOT (login node download, compute nodes offline):
#   Qwen/Qwen2.5-7B
#
# Peak GPU memory: full eager attention for all layers is materialized per
# pass; at the 4096-token cap this is ~26 GB on top of ~15 GB weights.
# Outputs (~400 MB total) stay in the run dir under the project.
#
# Submit:  sbatch scripts/tamia/jobs/ar/01_extract.sbatch
# ---------------------------------------------------------------------------

set -euo pipefail

PROJECT_DIR="$HOME/CoT-checker"
STORE="/project/aip-azouaq/$USER"
RUN_DIR="${RUN_DIR:-runs/attention_routing/forks_attn}"
ANALYSIS_DIR="${ANALYSIS_DIR:-runs/attention_routing/analysis}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B}"
NUM_SHARDS=4

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
export OMP_NUM_THREADS=8  # 4 numpy-heavy workers share the node's CPUs

[[ -f runs/contrib_cluster/forks.jsonl ]] || {
    echo "ERROR: runs/contrib_cluster/forks.jsonl not synced" >&2; exit 1; }

EXTRA=()
[[ -n "${LIMIT:-}" ]] && EXTRA+=(--limit "$LIMIT")
[[ -n "${FORCE:-}" ]] && EXTRA+=(--force)

echo "======================================================="
echo "  AR stage 01: backend parity -> fork attention extraction"
echo "               ($NUM_SHARDS shards) -> merge -> paired analysis"
echo "  run dir : $RUN_DIR   model : $MODEL"
echo "======================================================="

# ---- stage 0: eager vs sdpa parity on 100 forks (GPU 0, ~minutes) ----------
# Records whether the eager forward matches the S4 sdpa forward; gates
# interpretation, not the run.
CUDA_VISIBLE_DEVICES=0 python scripts/attention_routing/ar_backend_parity.py \
    --forks runs/contrib_cluster/forks.jsonl \
    --out_dir runs/attention_routing/parity \
    --model_name_or_path "$MODEL" \
    --local_files_only --limit 100

pids=()
for s in $(seq 0 $((NUM_SHARDS - 1))); do
    CUDA_VISIBLE_DEVICES=$s python scripts/attention_routing/ar_extract_forks.py \
        --forks runs/contrib_cluster/forks.jsonl \
        --out_dir "$RUN_DIR" \
        --model_name_or_path "$MODEL" \
        --local_files_only \
        --shard_idx "$s" --num_shards "$NUM_SHARDS" \
        "${EXTRA[@]}" &
    pids+=($!)
done
for pid in "${pids[@]}"; do
    wait "$pid"
done

python scripts/attention_routing/ar_extract_forks.py \
    --out_dir "$RUN_DIR" --num_shards "$NUM_SHARDS" --merge

python scripts/analysis/ar_fork_analysis.py \
    --run_dir "$RUN_DIR" --out_dir "$ANALYSIS_DIR"

python scripts/analysis/ar_inspect_heatmaps.py --run_dir "$RUN_DIR"

echo "DONE. rsync back: $RUN_DIR (metadata/features/manifests/inspect_plots)"
echo "                  $ANALYSIS_DIR (tables/plots/summary.md)"
