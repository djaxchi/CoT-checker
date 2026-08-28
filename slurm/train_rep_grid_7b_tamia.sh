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
# CELLS_FILE must live on a SHARED filesystem. Compute nodes have their own
# /tmp, so a file written to /tmp on the login node is not there at runtime;
# the cell lists are kept in experiments/unified_harness_7b/ for that reason.
# SEEDS: seeds to run for every cell (default "42 43 44"); the grid reports
# mean +- std across them, because a single seed cannot rank cells that differ
# by a couple of F1 points.
#
# Runs in two phases so the lr x wd search happens once per cell rather than once
# per seed. Phase 1 runs every cell at the first seed and does the search; phase 2
# runs the remaining seeds with --hp_from pointing at the phase-1 result. Both
# phases keep all four GPUs busy, and the saving is roughly 40% of the sequence
# cells' cost. Selecting per seed would also let each seed pick the config that
# suits its own initialisation, shrinking the very spread the seeds measure.

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
RESCALE="${RESCALE:-none}"
N_GPUS="${N_GPUS:-4}"
# Span preloading is a PER-PROCESS budget, but N_GPUS cells run concurrently on
# one node. Job 429667 died OOM because each of 4 cells independently decided
# 163.4 GB fitted in a 300 GB budget: 4 x 163.4 = 654 GB on a node with less
# than that free. Derive the per-cell budget from the node's actual memory
# divided by the concurrency, so the decision is made with the right denominator.
MEM_KB="$(awk "/MemTotal/ {print \$2}" /proc/meminfo 2>/dev/null || echo 0)"
PRELOAD_BUDGET_GB="${PRELOAD_BUDGET_GB:-$(awk -v m="$MEM_KB" -v n="$N_GPUS" \
  "BEGIN{printf \"%.0f\", (m/1048576)*0.60/n}")}"
CELLS_FILE="${CELLS_FILE:-}"
CELLS="${CELLS:-}"

cd "$PROJECT_ROOT"
GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
mkdir -p "$OUT_ROOT" "$VEC_CACHE"

# ---- assemble the cell list ------------------------------------------------
declare -a PAIRS=()
if [[ -n "$CELLS_FILE" ]]; then
  [[ -r "$CELLS_FILE" ]] || {
    echo "[FATAL] CELLS_FILE not readable from the compute node: $CELLS_FILE" >&2
    echo "        it must be on a shared filesystem, not the login node's /tmp" >&2
    exit 2; }
  while IFS= read -r line; do
    line="${line%%#*}"; line="$(echo "$line" | xargs || true)"
    [[ -n "$line" ]] && PAIRS+=("$line")
  done < "$CELLS_FILE"
  [[ ${#PAIRS[@]} -gt 0 ]] || { echo "[FATAL] no cells read from $CELLS_FILE" >&2; exit 2; }
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
preload cap : ${PRELOAD_BUDGET_GB} GB per cell (node mem / $N_GPUS x 0.60)
rescale     : $RESCALE
train_stem  : $TRAIN_STEM (no cap: the full split)
================================================================
BANNER
printf '  cell: %s\n' "${PAIRS[@]}"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch numpy pyyaml

# ---- preflight: import the runner once, before launching anything ----------
# A missing dependency otherwise shows up as twelve background processes dying
# silently; here it is one clear message before any GPU time is spent.
python -c "import scripts.train_rep_learner_cell" || {
  echo "[FATAL] the cell runner does not import in this environment" >&2; exit 3; }
echo "[preflight] cell runner imports cleanly"

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

# ---- run the cells, four at a time, in two phases ---------------------------
read -r -a SEED_ARR <<< "$SEEDS"
FIRST_SEED="${SEED_ARR[0]}"

cell_tag() { echo "${1}__$(echo "$2" | tr ':,' '__')__seed${3}"; }

# Cells run in the background so four share the node's GPUs, which means a
# crashed cell is invisible unless its exit status is collected. Job 426312
# "COMPLETED" in 33 seconds with every cell dead on a missing import, so PIDs are
# tracked and waited on individually, and the job fails if no cell produced a
# result.
declare -a PIDS=() PID_TAGS=()
FAILED=0

run_cell() {  # rep learner seed gpu [extra args...]
  local rep="$1" learner="$2" seed="$3" gpu="$4"; shift 4
  local tag; tag="$(cell_tag "$rep" "$learner" "$seed")"
  local out="$OUT_ROOT/$tag"
  if [[ -f "$out/results.json" ]]; then echo "[skip] $tag already done"; return; fi
  echo "[launch] gpu$gpu $tag $*"
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/train_rep_learner_cell.py \
    --rep "$rep" --learner "$learner" \
    --prm_store "$PRM_STORE" --pb_store "$PB_STORE" \
    --out_dir "$out" --vec_cache_dir "$VEC_CACHE" \
    --train_stem "$TRAIN_STEM" --seed "$seed" \
    --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" \
    --hp_search_cap "$HP_SEARCH_CAP" --rescale "$RESCALE" \
    --preload_budget_gb "$PRELOAD_BUDGET_GB" "$@" \
    > "$OUT_ROOT/${tag}.log" 2>&1 &
  PIDS+=("$!"); PID_TAGS+=("$tag")
}

wait_batch() {
  local i rc
  for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
      echo "[ok] ${PID_TAGS[$i]}"
    else
      rc=$?
      FAILED=$(( FAILED + 1 ))
      echo "[FAIL rc=$rc] ${PID_TAGS[$i]}"
      tail -5 "$OUT_ROOT/${PID_TAGS[$i]}.log" 2>/dev/null | sed 's/^/    /'
    fi
  done
  PIDS=(); PID_TAGS=()
}

echo "=== phase 1: seed $FIRST_SEED, hyperparameter search ==="
i=0
for pair in "${PAIRS[@]}"; do
  run_cell "${pair%% *}" "${pair#* }" "$FIRST_SEED" "$(( i % N_GPUS ))"
  i=$(( i + 1 ))
  if (( i % N_GPUS == 0 )); then wait_batch; fi
done
wait_batch

echo "=== phase 2: remaining seeds, reusing each cell's selection ==="
i=0
for seed in "${SEED_ARR[@]:1}"; do
  for pair in "${PAIRS[@]}"; do
    rep="${pair%% *}"; learner="${pair#* }"
    hp="$OUT_ROOT/$(cell_tag "$rep" "$learner" "$FIRST_SEED")/results.json"
    if [[ ! -f "$hp" ]]; then
      echo "[warn] no phase-1 result for $rep x $learner, searching in place"
      run_cell "$rep" "$learner" "$seed" "$(( i % N_GPUS ))"
    else
      run_cell "$rep" "$learner" "$seed" "$(( i % N_GPUS ))" --hp_from "$hp"
    fi
    i=$(( i + 1 ))
    if (( i % N_GPUS == 0 )); then wait_batch; fi
  done
done
wait_batch

DONE=$(find "$OUT_ROOT" -name results.json | wc -l)
echo "[$(date)] rep_grid done: $DONE cells written, $FAILED cell(s) failed"
if (( DONE == 0 )); then
  echo "[FATAL] no cell produced a results.json; see the per-cell logs above" >&2
  exit 1
fi
