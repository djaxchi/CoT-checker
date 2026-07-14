#!/bin/bash
#SBATCH --job-name=to_stage0
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

# transition_operator_v0 Stage 0 gates on Qwen2.5-7B
# (docs/transition_operator_v0_plan.md section 8): suffix selection, directional
# sanity, tokenization sanity, boundary-sufficiency oracle at L20/L24/L26.
# Single process on GPU 0; the gates are sequential (2 and 5 need gate 1's suffix).
# Requires runs/transition_operator/{forks,golden}.jsonl, built on the LOGIN node
# (internet) by scripts/transition_operator/to_build_forks.py.
#
# Usage:
#   sbatch slurm/transition_operator_stage0_tamia.sh
#   N_ORACLE=50 sbatch --time=00:20:00 slurm/transition_operator_stage0_tamia.sh  # smoke

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_DIR="${RUN_DIR:-$PROJECT_ROOT/runs/transition_operator}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-7B}"
LAYERS="${LAYERS:-20 24 26}"
N_GATE1="${N_GATE1:-300}"
N_GATE2="${N_GATE2:-500}"
N_ORACLE="${N_ORACLE:-500}"
N_ORACLE_B="${N_ORACLE_B:-200}"

source "$PROJECT_ROOT/slurm/s1_model_size/models.env"
export HF_HOME="${HF_HOME:-$HF_CACHE_ROOT}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

cd "$PROJECT_ROOT"
test -f "$RUN_DIR/forks.jsonl" || { echo "missing $RUN_DIR/forks.jsonl (build on login node)"; exit 1; }
test -f "$RUN_DIR/golden.jsonl" || { echo "missing $RUN_DIR/golden.jsonl (build on login node)"; exit 1; }

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch transformers numpy scipy

cat <<BANNER
================================================================
job        : ${SLURM_JOB_NAME:-to_stage0}  id ${SLURM_JOB_ID:-N/A}
host       : $(hostname)   date $(date -Iseconds)
git_commit : $(git rev-parse HEAD 2>/dev/null || echo unknown)
model      : $MODEL_NAME_OR_PATH   patch layers: $LAYERS
forks      : $RUN_DIR/forks.jsonl ($(wc -l < "$RUN_DIR/forks.jsonl") forks)
golden     : $RUN_DIR/golden.jsonl ($(wc -l < "$RUN_DIR/golden.jsonl") trajectories)
gates      : n1=$N_GATE1 n2=$N_GATE2 n5=$N_ORACLE n5b=$N_ORACLE_B
================================================================
BANNER

CUDA_VISIBLE_DEVICES=0 python scripts/transition_operator/to_stage0.py \
  --run_dir "$RUN_DIR" \
  --model_name_or_path "$MODEL_NAME_OR_PATH" --local_files_only \
  --layers $LAYERS \
  --n_gate1 "$N_GATE1" --n_gate2 "$N_GATE2" \
  --n_oracle "$N_ORACLE" --n_oracle_b "$N_ORACLE_B" \
  --device cuda

echo "[done] $(date -Iseconds) -> $RUN_DIR/stage0"
