#!/bin/bash
#SBATCH --job-name=das_p1b
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out

# DAS branch-subspace, Phase 1b (boundary-patch ORACLE gate). Free-generation
# rollouts under correct / wrong / oracle_L{12,20,26} contexts; the oracle injects
# the correct branch's boundary residual state into the wrong prompt during prefill.
# Reports per-layer recovery = (oracle - wrong) / (correct - wrong). A positive,
# significant oracle is the green light for the Phase-2 subspace search.
#
# Usage:  sbatch slurm/das_branch_phase1b_oracle_tamia.sh
#         LAYERS=8,16,24 PILOT_TRACES=300 sbatch ...

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_DIR="${RUN_DIR:-$PROJECT_ROOT/runs/causal_graph}"
OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/runs/das_branch}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-7B}"
LAYERS="${LAYERS:-12,20,26}"
K_ROLLOUTS="${K_ROLLOUTS:-8}"
PILOT_TRACES="${PILOT_TRACES:-200}"

source "$PROJECT_ROOT/slurm/s1_model_size/models.env"
export HF_HOME="${HF_HOME:-$HF_CACHE_ROOT}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

cd "$PROJECT_ROOT"
test -e "$RUN_DIR/traces_forks.jsonl" || { echo "missing traces_forks.jsonl"; exit 1; }

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch transformers numpy scipy scikit-learn pandas pyarrow

cat <<BANNER
================================================================
job        : ${SLURM_JOB_NAME:-das_p1b}  id ${SLURM_JOB_ID:-N/A}
host       : $(hostname)   date $(date -Iseconds)
layers     : $LAYERS   k_rollouts $K_ROLLOUTS   pilot_traces $PILOT_TRACES
out_dir    : $OUT_DIR
================================================================
BANNER

pids=()
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g python scripts/das_branch/das_oracle.py \
    --run_dir "$RUN_DIR" --out_dir "$OUT_DIR" \
    --model_name_or_path "$MODEL_NAME_OR_PATH" --layers "$LAYERS" \
    --k_rollouts "$K_ROLLOUTS" --pilot_traces "$PILOT_TRACES" \
    --local_files_only --shard_id $g --num_shards 4 \
    > "das_p1b_oracle_shard$g-${SLURM_JOB_ID:-manual}.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done

python scripts/das_branch/das_oracle.py --out_dir "$OUT_DIR" --merge

echo "[das_p1b] done $(date -Iseconds); gates -> $OUT_DIR/gates_oracle.json"
