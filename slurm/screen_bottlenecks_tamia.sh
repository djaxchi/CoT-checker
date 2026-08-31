#!/bin/bash
#SBATCH --job-name=screen_bneck
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

# Screen every bottleneck objective against the base representation it compresses.
#
# The question: can a bottleneck be made to KEEP the correctness signal rather
# than the variance? The sparse dictionary did the opposite, costing up to 0.195
# F1, because step correctness is a ~0.01%-variance margin and a
# reconstruction objective has no reason to spend its budget there.
#
# Ranked by ProcessBench step AUROC, which predicted the full metric at Spearman
# 0.934 across 31 evaluated cells. signal_share is reported alongside: it needs no
# training and is the quantity these objectives are meant to raise, so it says WHY
# a bottleneck screens as it does.
#
# `none` is the base representation uncompressed and must always be in the table.
# A bottleneck that does not beat the thing it compresses is not interesting.

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH/cot_mech/qwen3_8b_v1}"
VEC_CACHE="${VEC_CACHE:-$RUN_ROOT/cache/grid_vectors}"
OUT_DIR="${OUT_DIR:-$RUN_ROOT/bottlenecks}"
BASE_REP="${BASE_REP:-step_mean}"
D_CODE="${D_CODE:-256}"
N_TRAIN="${N_TRAIN:-150000}"
EPOCHS="${EPOCHS:-6}"

mkdir -p "$OUT_DIR"
cd "$PROJECT_ROOT"
export TOKENIZERS_PARALLELISM=false

cat <<BANNER
================================================================
job       : ${SLURM_JOB_NAME:-screen_bneck}  id: ${SLURM_JOB_ID:-N/A}
base rep  : $BASE_REP   code width: $D_CODE   train rows: $N_TRAIN
cache     : $VEC_CACHE
out       : $OUT_DIR
================================================================
BANNER

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch numpy

# name:objective:beta -- the beta sweep is the trade-off curve, not one point
RUNS=(
  "base:none:0"
  "recon:recon:0"
  "recon_white:recon_white:0"
  "mixed_b1:mixed:1"
  "mixed_b10:mixed:10"
  "mixed_b100:mixed:100"
  "ib:ib:10"
  "ib_b100:ib:100"
)

pids=(); tags=(); gpu=0
for r in "${RUNS[@]}"; do
  IFS=: read -r name obj beta <<< "$r"
  echo "[launch] gpu$((gpu % 4)) $name ($obj, beta=$beta)"
  CUDA_VISIBLE_DEVICES=$((gpu % 4)) python scripts/train_bottleneck_rep.py \
    --vec_cache "$VEC_CACHE" --base_rep "$BASE_REP" \
    --objective "$obj" --beta "$beta" --d_code "$D_CODE" \
    --n_train "$N_TRAIN" --epochs "$EPOCHS" \
    --out "$OUT_DIR/$name.npz" > "$OUT_DIR/$name.log" 2>&1 &
  pids+=($!); tags+=("$name"); gpu=$((gpu+1))
  if (( gpu % 4 == 0 )); then
    for i in "${!pids[@]}"; do
      wait "${pids[$i]}" && echo "[ok] ${tags[$i]}" || { echo "[FAIL] ${tags[$i]}"; tail -12 "$OUT_DIR/${tags[$i]}.log"; }
    done
    pids=(); tags=()
  fi
done
for i in "${!pids[@]}"; do
  wait "${pids[$i]}" && echo "[ok] ${tags[$i]}" || { echo "[FAIL] ${tags[$i]}"; tail -12 "$OUT_DIR/${tags[$i]}.log"; }
done

echo
echo "=== signal share, per objective ==="
grep -h "signal share" "$OUT_DIR"/*.log || true
echo
echo "=== screen ==="
python scripts/screen_representation.py --npz "$OUT_DIR"/*.npz \
  --out "$OUT_DIR/screen.json"
echo "[$(date)] screen_bneck done"
