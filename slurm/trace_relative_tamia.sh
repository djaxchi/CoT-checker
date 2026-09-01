#!/bin/bash
#SBATCH --job-name=tracerel
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out

# Represent a step by where it sits in its own trace.
#
# R1, trace-relative coordinates. The pre-step boundary state alone scores 0.7412
# in domain and 0.5035 on ProcessBench, so most of what an in-domain probe reads
# is which problem this is and where in the solution we are, and none of it
# survives the domain change. The task is relative -- which step goes wrong FIRST
# in this solution -- and every representation so far is absolute. GeoReason
# (arXiv:2605.13772) reaches the same normalisation independently.
#
# R2, between-step dynamics. `diffs` measured motion between TOKENS inside a step.
# Nothing has ever measured motion between STEPS. `contribution` used the raw
# velocity vector and lost, but never its length relative to the trace, its angle
# to the previous step's motion, or the second difference.
#
# `pos` is the control: relative position and trace length and nothing else.
# First errors sit later on average, so position is a shortcut of exactly the
# kind step length turned out to be, and a trace-relative gain can only be read
# against it.

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH/cot_mech/qwen3_8b_v1}"
OUT_DIR="${OUT_DIR:-$RUN_ROOT/trace_relative}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
mkdir -p "$OUT_DIR"
cd "$PROJECT_ROOT"

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch numpy

python scripts/trace_relative_reps.py \
  --prm_store "$RUN_ROOT/repstore/step_spans" \
  --pb_store  "$RUN_ROOT/repstore/pb_step_spans" \
  --out_dir "$OUT_DIR" --n_traces "${N_TRACES:-9000}" \
  --n_pb_traces "${N_PB_TRACES:-1200}"

# Ridge, so the verdict is not about a training budget. The winner is recomputed
# on these same rows inside the deriver, so every row here is one sample.
python scripts/ridge_screen.py --npz "$OUT_DIR"/*.npz \
  --bootstrap 1000 --ref winner --out "$RUN_ROOT/trace_relative.json"
echo "[$(date)] tracerel done"
