#!/bin/bash
#SBATCH --job-name=derive_rep_7b
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=05:00:00
#SBATCH --output=%x-%j.out

# Derive a vector representation (READOUT) from the PRM800K and ProcessBench
# token stores into the dense-cache contract, offline (no GPU). Pooled readouts
# (mean/max/multistat/boundary_stats) do a per-item span reduction over the ~1TB
# store, so this is I/O-heavy on the train split; last/delta are vectorized gathers.
#   READOUT: delta | last | mean | max | multistat | boundary_stats
#   TAG:     cache suffix (default = READOUT)

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH/cot_mech/dense_full_7b_v1}"
READOUT="${READOUT:-multistat}"
TAG="${TAG:-$READOUT}"
STORE_PRM="$RUN_ROOT/repstore/tokens_last_layer"
STORE_PB="$RUN_ROOT/repstore/pb_tokens_last_layer"
# Derived caches default to $SCRATCH next to the store, but data_setup.md's policy
# is "master token store in $SCRATCH (regenerable), derived reps in $STORE", and
# $SCRATCH is the tier that runs out (the 984G token store dominates its quota).
# Set CACHE_ROOT to follow the policy: CACHE_ROOT=$STORE/cot_mech/dense_full_7b_v1/cache
CACHE_ROOT="${CACHE_ROOT:-$RUN_ROOT/cache}"
OUT_PRM="$CACHE_ROOT/qwen2_5_7b_${TAG}"
OUT_PB="$CACHE_ROOT/qwen2_5_7b_${TAG}_pb"
PRM_SPLITS="${PRM_SPLITS:-probe_train_full val_5k test_2k}"
PB_SUBSETS="${PB_SUBSETS:-gsm8k math olympiadbench omnimath}"

cd "$PROJECT_ROOT"
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index numpy

mkdir -p "$OUT_PRM" "$OUT_PB"
echo "[derive_rep] READOUT=$READOUT TAG=$TAG CACHE_ROOT=$CACHE_ROOT"
df -h "$CACHE_ROOT" | tail -1
python scripts/derive_delta_from_token_store.py --store_root "$STORE_PRM" \
  --splits $PRM_SPLITS --out_dir "$OUT_PRM" --mode prm --readout "$READOUT"

PBP=""
for s in $PB_SUBSETS; do [[ -d "$STORE_PB/$s" ]] && PBP="$PBP $s"; done
python scripts/derive_delta_from_token_store.py --store_root "$STORE_PB" \
  --splits $PBP --out_dir "$OUT_PB" --mode pb --readout "$READOUT"

echo "[verify] PRM:"; ls -la "$OUT_PRM"/*.npy 2>/dev/null | head
echo "[verify] PB:";  ls "$OUT_PB"/*/pb_step_h.npy 2>/dev/null
echo "[$(date)] derive_rep ($TAG) done"
