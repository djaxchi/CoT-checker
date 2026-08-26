#!/bin/bash
#SBATCH --job-name=sae_gate
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:20:00
#SBATCH --output=%x-%j.out

# Go/no-go for the SAE arm: does any Qwen-Scope SAE layer reconstruct the states
# we actually stored? Cheap (a few matmuls over a sample), but it decides whether
# the encode currently running produced an SAE-compatible store, so it is worth
# knowing before that job finishes rather than after.
#
# CPU-only and short. Do NOT run this on the login node; it loads a 2.1 GB SAE
# and gets reaped there (twice, so far).

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
SPLIT_DIR="${SPLIT_DIR:-$SCRATCH/cot_mech/qwen3_8b_v1/repstore/pb_step_spans/gsm8k}"
HF_CACHE="${HF_CACHE:-/project/aip-azouaq/$USER/hf_cache}"
REPO_ID="${REPO_ID:-Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50}"
LAYERS="${LAYERS:-35}"
N_ITEMS="${N_ITEMS:-400}"
OUT="${OUT:-$SCRATCH/cot_mech/qwen3_8b_v1/sae_pairing_gate.json}"

export HF_HOME="$HF_CACHE" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

cd "$PROJECT_ROOT"
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch numpy

python scripts/public_sae/qwen_scope_pairing_gate.py \
  --split_dir "$SPLIT_DIR" --hf_cache "$HF_CACHE" --repo_id "$REPO_ID" \
  --layers $LAYERS --n_items "$N_ITEMS" --out "$OUT"
echo "[$(date)] sae_gate done"
