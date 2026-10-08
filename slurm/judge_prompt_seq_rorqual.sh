#!/bin/bash
#SBATCH --job-name=jp_seq_r
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=02:30:00

# judge_prompt_v1 on Rorqual: the step_tokens x transformer:d512 cell (the best
# Instruct cell) at one SEED. Spans are copied to node-local disk first, because
# training straight off Rorqual's Lustre memmaps starves the GPU
# (context_ablation_v3); spans are then read through the local page cache. A
# seed other than 42 reuses seed 42's lr x wd selection.
set -euo pipefail
: "${SEED:?}" "${RUN:?}"
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
LEARNER="${LEARNER:-transformer:d512,l2,f2048,h8}"
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0
export PYTHONUNBUFFERED=1
cd "$PROJECT_ROOT"
echo "git $(git rev-parse --short HEAD) seed=$SEED learner=$LEARNER"
df -h "$SLURM_TMPDIR" | tail -1
virtualenv --no-download "$SLURM_TMPDIR/env" >/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip >/dev/null
pip install --no-index torch numpy pyyaml 2>&1 | tail -1

L="$SLURM_TMPDIR/repstore"; mkdir -p "$L"
t0=$(date +%s)
for s in step_spans pb_step_spans; do cp -r "$RUN/repstore/$s" "$L/"; done
echo "[stage] $(du -sh "$L" | cut -f1) in $(( $(date +%s) - t0 ))s"

tag="step_tokens__$(echo "$LEARNER" | tr ':,' '__')__seed${SEED}"
out="$RUN/cells/span/$tag"; mkdir -p "$RUN/cells/span" "$RUN/logs"
hp=()
if [[ "$SEED" != 42 ]]; then
  h="$RUN/cells/span/step_tokens__$(echo "$LEARNER" | tr ':,' '__')__seed42/results.json"
  [[ -f "$h" ]] || { echo "[FATAL] needs $h" >&2; exit 2; }
  hp=(--hp_from "$h")
fi
python scripts/train_rep_learner_cell.py --rep step_tokens --learner "$LEARNER" \
  --prm_store "$L/step_spans" --pb_store "$L/pb_step_spans" --out_dir "$out" \
  --train_stem probe_train_full --seed "$SEED" --epochs 30 --patience 3 \
  --batch_size 256 --hp_search_cap 100000 --rescale none --preload_budget_gb 1 \
  "${hp[@]}"
echo "[$(date)] $tag done"
