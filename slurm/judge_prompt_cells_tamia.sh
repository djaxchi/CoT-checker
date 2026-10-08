#!/bin/bash
#SBATCH --job-name=jp_cells
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=01:00:00

# judge_prompt_v1 vector cells on TamIA, up to four at once (one per GPU).
# CELLS: ';'-separated "arm rep learner seed", arm = span | verdict.
#   span     the pre-derived step-span vectors ($RUN/vec/span/<rep>, --prederived)
#   verdict  the judge-token store, rep last_token = the verdict-token state
# A seed other than 42 reuses that cell's seed-42 selection (--hp_from), as in
# the leaderboard protocol, so its seed-42 job must have finished first.
# ZEROSHOT=1 also writes the zero-shot P(No) cell (CPU, a few minutes).

set -euo pipefail
: "${RUN:?}" "${CELLS:?}"
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
module purge
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
cd "$PROJECT_ROOT"
echo "git $(git rev-parse --short HEAD) cells=$CELLS"
virtualenv --no-download "$SLURM_TMPDIR/env" >/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip >/dev/null
pip install --no-index torch numpy pyyaml 2>&1 | tail -1
mkdir -p "$RUN/cells/span" "$RUN/cells/verdict" "$RUN/cells/zeroshot" "$RUN/logs"

if [[ "${ZEROSHOT:-0}" == 1 && ! -f "$RUN/cells/zeroshot/judge_zeroshot/results.json" ]]; then
  CUDA_VISIBLE_DEVICES="" python scripts/analysis/judge_zeroshot_score.py \
    --judge_store "$RUN/judge_token" --pb_judge_store "$RUN/pb_judge_token" \
    --out_dir "$RUN/cells/zeroshot/judge_zeroshot" > "$RUN/logs/zeroshot-${SLURM_JOB_ID}.log" 2>&1 &
  ZPID=$!
fi

tag() { echo "${1}__$(echo "$2" | tr ':,' '__')__seed${3}"; }
IFS=';' read -r -a LINES <<< "$CELLS"
pids=(); tags=(); fail=0; gpu=0
flush() {
  local i
  for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then echo "[ok] ${tags[$i]}"
    else fail=$((fail + 1)); echo "[FAIL] ${tags[$i]}"; tail -5 "$RUN/logs/${tags[$i]}.log" | sed 's/^/    /'; fi
  done
  pids=(); tags=(); gpu=0
}
for line in "${LINES[@]}"; do
  read -r arm rep learner seed <<< "$line"
  [[ -n "${arm:-}" ]] || continue
  t=$(tag "$rep" "$learner" "$seed"); out="$RUN/cells/$arm/$t"
  [[ -f "$out/results.json" ]] && { echo "[skip] $arm/$t done"; continue; }
  if [[ "$arm" == span ]]; then
    store=(--prm_store "$RUN/vec/span/$rep" --pb_store "$RUN/vec/pb_span/$rep" --prederived)
  else
    store=(--prm_store "$RUN/judge_token" --pb_store "$RUN/pb_judge_token")
  fi
  hp=()
  if [[ "$seed" != 42 ]]; then
    h="$RUN/cells/$arm/$(tag "$rep" "$learner" 42)/results.json"
    [[ -f "$h" ]] || { echo "[FATAL] $arm/$t needs $h" >&2; exit 2; }
    hp=(--hp_from "$h")
  fi
  echo "[launch] gpu$gpu $arm/$t"
  CUDA_VISIBLE_DEVICES=$gpu python scripts/train_rep_learner_cell.py \
    --rep "$rep" --learner "$learner" "${store[@]}" --out_dir "$out" \
    --vec_cache_dir "$SLURM_TMPDIR/vc_${arm}_${t}" --train_stem probe_train_full --seed "$seed" \
    --epochs 30 --patience 3 --batch_size 256 --hp_search_cap 100000 --rescale none \
    "${hp[@]}" > "$RUN/logs/${arm}_${t}.log" 2>&1 &
  pids+=($!); tags+=("${arm}_${t}")
  gpu=$((gpu + 1))
  (( gpu == 4 )) && flush
done
flush
if [[ -n "${ZPID:-}" ]]; then wait "$ZPID" || { fail=$((fail + 1)); echo "[FAIL] zeroshot"; }; fi
echo "[$(date)] cells done, $fail failed"
(( fail == 0 ))
