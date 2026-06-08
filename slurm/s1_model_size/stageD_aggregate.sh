#!/bin/bash
#SBATCH --job-name=s1ms_D_agg
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#
# Stage D: aggregate this model's per-subset metrics into the val/oracle macro
# rows and append to the global leaderboard. CPU only.
#
# For the 1.5B smoke model (GATE=1) this stage ENFORCES the Sprint 1
# reproduction: it exits non-zero if the macro F1_PB is outside tolerance, which
# (via afterok) stops the rest of the sweep.
#
# Required env: TAG ; optional: GATE=1
set -euo pipefail

# sbatch spools this script; use the launcher-exported S1MS_DIR (real repo path).
HERE="${S1MS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
# shellcheck disable=SC1091
source "$HERE/models.env"
# shellcheck disable=SC1091
source "$HERE/_common.sh"

: "${TAG:?TAG must be exported by the launcher}"
MODEL_DIR="$RUNS_ROOT/$TAG"
LOG_DIR="$MODEL_DIR/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/stageD_aggregate.log"
export S1MS_STAGE="D_aggregate"

s1ms_env_setup
echo "[D] TAG=$TAG aggregate (GATE=${GATE:-0})" | tee "$LOG"
s1ms_venv

GATE_ARGS=()
if [[ "${GATE:-0}" == "1" ]]; then
  GATE_ARGS=( --expect_val_macro "$S1MS_EXPECT_VAL_MACRO"
              --expect_oracle_macro "$S1MS_EXPECT_ORACLE_MACRO"
              --tol "$S1MS_GATE_TOL" )
fi

python scripts/s1ms_aggregate_model.py \
  --eval_dir "$MODEL_DIR/processbench_eval_shards" \
  --out_dir "$MODEL_DIR" \
  --subsets "${S1MS_SUBSETS[@]}" \
  "${GATE_ARGS[@]}" 2>&1 | tee -a "$LOG"

# Refresh the global leaderboard with every model completed so far.
python scripts/s1ms_merge_leaderboard.py --runs_root "$RUNS_ROOT" 2>&1 | tee -a "$LOG"

echo "[D] done -> $MODEL_DIR/per_subset_metrics.json + leaderboard refreshed" | tee -a "$LOG"
