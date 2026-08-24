#!/bin/bash
#SBATCH --job-name=rep_grid_7b
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out

# Run cells of the representation x learner grid. One cell = one (rep, learner,
# seed) trained by scripts/train_rep_learner_cell.py under the fixed protocol:
# the FULL PRM800K train split (no cap), the same lr x wd search, the same
# trainer and early stopping, the same evaluation.
#
# TamIA allocates H100 nodes whole, so the four GPUs are used by running four
# cells concurrently via CUDA_VISIBLE_DEVICES rather than by a job array.
#
# CELLS: newline- or semicolon-separated "rep learner" pairs, or a file path via
# CELLS_FILE (one "rep learner" per line, # comments allowed).
#   e.g. CELLS="last_token linear;step_mean linear;step_stats mlp:h1024"
# SEEDS: seeds to run for every cell (default "42 43 44"); the grid reports
# mean +- std across them, because a single seed cannot rank cells that differ
# by a couple of F1 points.

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_ROOT="${RUN_ROOT:-$STORE/cot_mech/dense_full_7b_v1}"
PRM_STORE="${PRM_STORE:-$RUN_ROOT/repstore/step_spans}"
PB_STORE="${PB_STORE:-$RUN_ROOT/repstore/pb_step_spans}"
VEC_CACHE="${VEC_CACHE:-$RUN_ROOT/cache/grid_vectors}"
OUT_ROOT="${OUT_ROOT:-$RUN_ROOT/runs/rep_grid_v1}"
SEEDS="${SEEDS:-42 43 44}"
TRAIN_STEM="${TRAIN_STEM:-probe_train_full}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-256}"
HP_SEARCH_CAP="${HP_SEARCH_CAP:-100000}"
N_GPUS="${N_GPUS:-4}"
CELLS_FILE="${CELLS_FILE:-}"
CELLS="${CELLS:-}"

cd "$PROJECT_ROOT"
GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
mkdir -p "$OUT_ROOT" "$VEC_CACHE"

# ---- assemble the cell list ------------------------------------------------
declare -a PAIRS=()
if [[ -n "$CELLS_FILE" ]]; then
  while IFS= read -r line; do
    line="${line%%#*}"; line="$(echo "$line" | xargs || true)"
    [[ -n "$line" ]] && PAIRS+=("$line")
  done < "$CELLS_FILE"
else
  [[ -n "$CELLS" ]] || { echo "[FATAL] set CELLS or CELLS_FILE" >&2; exit 2; }
  IFS=';' read -r -a raw <<< "$CELLS"
  for c in "${raw[@]}"; do
    c="$(echo "$c" | xargs || true)"
    [[ -n "$c" ]] && PAIRS+=("$c")
  done
fi

cat <<BANNER
================================================================
job         : ${SLURM_JOB_NAME:-rep_grid_7b}  job_id: ${SLURM_JOB_ID:-N/A}
git_commit  : $GIT_COMMIT
prm_store   : $PRM_STORE
pb_store    : $PB_STORE
out_root    : $OUT_ROOT
cells       : ${#PAIRS[@]}  seeds: $SEEDS  gpus: $N_GPUS
train_stem  : $TRAIN_STEM (no cap: the full split)
================================================================
BANNER
printf '  cell: %s\n' "${PAIRS[@]}"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch numpy

# ---- vector caches first, once, shared by every learner on that rep --------
# Deriving inside the concurrent cells would have four processes racing to write
# the same .npy; do it serially up front instead.
for pair in "${PAIRS[@]}"; do
  rep="${pair%% *}"
  [[ "$rep" == "step_tokens" ]] && continue
  marker="$VEC_CACHE/${rep}__${TRAIN_STEM}_h.npy"
  [[ -f "$marker" ]] && { echo "[cache] $rep present"; continue; }
  echo "[cache] deriving $rep"
  CUDA_VISIBLE_DEVICES="" python scripts/train_rep_learner_cell.py \
    --rep "$rep" --learner linear \
    --prm_store "$PRM_STORE" --pb_store "$PB_STORE" \
    --out_dir "$SLURM_TMPDIR/cache_warm_$rep" --vec_cache_dir "$VEC_CACHE" \
    --train_stem "$TRAIN_STEM" --train_cap 2000 --hp_search_cap 1000 \
    --lr_grid 1e-3 --wd_grid 0.0 --epochs 1 --patience 1 >/dev/null
done

# ---- run the cells, four at a time ----------------------------------------
i=0
for seed in $SEEDS; do
  for pair in "${PAIRS[@]}"; do
    rep="${pair%% *}"; learner="${pair#* }"
    tag="${rep}__$(echo "$learner" | tr ':,' '__')__seed${seed}"
    out="$OUT_ROOT/$tag"
    if [[ -f "$out/results.json" ]]; then
      echo "[skip] $tag already done"; continue
    fi
    gpu=$(( i % N_GPUS )); i=$(( i + 1 ))
    echo "[launch] gpu$gpu $tag"
    CUDA_VISIBLE_DEVICES="$gpu" python scripts/train_rep_learner_cell.py \
      --rep "$rep" --learner "$learner" \
      --prm_store "$PRM_STORE" --pb_store "$PB_STORE" \
      --out_dir "$out" --vec_cache_dir "$VEC_CACHE" \
      --train_stem "$TRAIN_STEM" --seed "$seed" \
      --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" \
      --hp_search_cap "$HP_SEARCH_CAP" \
      > "$OUT_ROOT/${tag}.log" 2>&1 &
    if (( i % N_GPUS == 0 )); then wait; fi
  done
done
wait

echo "[$(date)] rep_grid done: $(find "$OUT_ROOT" -name results.json | wc -l) cells"
