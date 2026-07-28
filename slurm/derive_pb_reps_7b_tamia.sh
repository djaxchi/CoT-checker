#!/bin/bash
#SBATCH --job-name=derive_pb_reps_7b
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out

# Derive dense_last and delta ProcessBench caches (all subsets) from the PB token
# store, offline. Emits the pb_step contract per subset for the harness.

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH/cot_mech/dense_full_7b_v1}"
STORE_ROOT="$RUN_ROOT/repstore/pb_tokens_last_layer"
DENSE_PB="$RUN_ROOT/cache/qwen2_5_7b_densepb"
DELTA_PB="$RUN_ROOT/cache/qwen2_5_7b_delta_pb"
SUBSETS="${SUBSETS:-gsm8k math olympiadbench omnimath}"

cd "$PROJECT_ROOT"
# subsets present in the store
PRESENT=""
for s in $SUBSETS; do [[ -d "$STORE_ROOT/$s" ]] && PRESENT="$PRESENT $s"; done
echo "[derive_pb] subsets:$PRESENT"

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index numpy

python scripts/derive_delta_from_token_store.py --store_root "$STORE_ROOT" \
  --splits $PRESENT --out_dir "$DENSE_PB" --mode pb --readout last
python scripts/derive_delta_from_token_store.py --store_root "$STORE_ROOT" \
  --splits $PRESENT --out_dir "$DELTA_PB" --mode pb --readout delta

echo "[verify] dense_pb:"; ls "$DENSE_PB"/*/pb_step_h.npy 2>/dev/null
echo "[verify] delta_pb:"; ls "$DELTA_PB"/*/pb_step_h.npy 2>/dev/null
echo "[$(date)] derive_pb_reps done"
