#!/bin/bash
#SBATCH --job-name=reprobe_7b
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=05:00:00
#SBATCH --output=%x-%j.out

# ReProbe (Ni et al. 2025) under our protocol: a small (<10M-param) transformer
# over ALL of a step's last-layer tokens, then masked mean-pool + linear head.
# Same token stores as attn_pool (no re-encode). Reports in-domain PRM800K test +
# ProcessBench per subset (+ pb_step_scores for offline calib-20).

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH/cot_mech/dense_full_7b_v1}"
PRM_STORE="$RUN_ROOT/repstore/tokens_last_layer"
PB_STORE="$RUN_ROOT/repstore/pb_tokens_last_layer"
OUT_DIR="$RUN_ROOT/runs/reprobe"
LOG_DIR="$RUN_ROOT/logs"
TRAIN_CAP="${TRAIN_CAP:-150000}"
T_MAX="${T_MAX:-128}"
EPOCHS="${EPOCHS:-15}"
D_MODEL="${D_MODEL:-256}"
NLAYERS="${NLAYERS:-2}"
NHEAD="${NHEAD:-4}"

mkdir -p "$OUT_DIR" "$LOG_DIR"
cd "$PROJECT_ROOT"
LOG_FILE="$LOG_DIR/reprobe_7b-${SLURM_JOB_ID:-$$}.log"

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch numpy pyyaml

python scripts/train_reprobe_probe.py \
  --prm_store "$PRM_STORE" --pb_store "$PB_STORE" \
  --out_dir "$OUT_DIR" \
  --train_cap "$TRAIN_CAP" --t_max "$T_MAX" --epochs "$EPOCHS" \
  --d_model "$D_MODEL" --nlayers "$NLAYERS" --nhead "$NHEAD" \
  --threshold_grid 0.01 2>&1 | tee -a "$LOG_FILE"

echo "[$(date)] reprobe_7b done"
