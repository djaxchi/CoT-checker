#!/bin/bash
#SBATCH --job-name=rp1_read
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=02:00:00

# risk_probe_v1 readouts (docs/risk_probe_v1_plan.md, sections 2, 4 to 7):
#   1. freeze the label set: summarize_values.py on the complete label shards
#      (5,874 traces) unless step_values.jsonl already exists; record its sha256
#   2. fit_readouts.py --stage real on one GPU (arms A and B, E2 folds and E1)
#   3. fit_readouts.py --stage null, N_PERM permutations split over the 4 GPUs
#   4. analyze.py: gates, Q1 to Q3, A against B, the prespecified reading
# Required env: SNAP, RUN (the encode/fit job's OUT_ROOT), FIT (output dir), ANALYSIS.
set -euo pipefail
: "${SNAP:?}" "${RUN:?}" "${FIT:?}" "${ANALYSIS:?}"
N_PERM="${N_PERM:-200}"
C=/project/aip-azouaq/$USER/cot_mech
SVL=$C/step_value_labels_v1
R=$C/risk_probe_v1
SET=$SVL/split/trace_set_label.json
SV="${STEP_VALUES:-$SVL/label/summary/step_values.jsonl}"   # STEP_VALUES: a partial label set, smoke only
mkdir -p "$FIT" "$ANALYSIS"

module purge
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
cd "$SNAP"
grep -E "snapshot|git_head" SNAPSHOT_INFO
virtualenv --no-download "$SLURM_TMPDIR/env" >/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip >/dev/null
pip install --no-index torch numpy scipy scikit-learn 2>&1 | tail -1

if [[ -z "${STEP_VALUES:-}" && ! -s "$SV" ]]; then
  n=$(cat "$SVL"/label/values.shard*.jsonl | wc -l)
  [[ "$n" == 5874 ]] || { echo "[FATAL] label set has $n of 5874 traces" >&2; exit 1; }
  python scripts/step_value_labels/summarize_values.py --values_dir "$SVL/label" --out_dir "$SVL/label/summary" \
    --trace_set "$SET" > "$FIT/summarize_values.log"
fi
sha256sum "$SV" | tee "$FIT/step_values.sha256"

ARGS=(--rows "$RUN/features/rows_L35.npz" --lab_views "$R/views/lab_views.jsonl" --step_values "$SV"
      --trace_set "$SET" --out_dir "$FIT")
[[ -s "$FIT/hparams.json" ]] || CUDA_VISIBLE_DEVICES=0 python scripts/risk_probe_v1/fit_readouts.py "${ARGS[@]}" \
  --stage real 2>&1 | tee "$FIT/fit_real.log"
per=$(( (N_PERM + 3) / 4 )); pids=()
for g in 0 1 2 3; do
  s=$((g * per)); e=$(( (g + 1) * per < N_PERM ? (g + 1) * per : N_PERM ))
  out=$(printf "%s/null_%03d_%03d.npz" "$FIT" "$s" "$e")
  [[ -s "$out" || $s -ge $e ]] && continue
  CUDA_VISIBLE_DEVICES=$g python scripts/risk_probe_v1/fit_readouts.py "${ARGS[@]}" --stage null \
    --perm_start $s --perm_end $e > "$FIT/fit_null_$g.log" 2>&1 &
  pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
[[ $fail == 0 ]] || { tail -5 "$FIT"/fit_null_*.log >&2; exit 1; }

python scripts/risk_probe_v1/analyze.py --fit_dir "$FIT" --run_dir "$RUN" --v0_run "$C/risk_probe_v0/prod" \
  --views "$R/views" --step_values "$SV" --trace_set "$SET" \
  --fork_factors "$C/factorized_step_labels_v1/full/summary/fork_factors.jsonl" \
                 "$C/factorized_step_labels_v1/pilot/summary_xb/fork_factors.jsonl" \
  --track1_dirs "$C/factorized_step_labels_v1/full/track1" "$C/factorized_step_labels_v1/pilot/track1" \
  --out_dir "$ANALYSIS" > "$ANALYSIS/analyze.log" 2>&1 || { tail -30 "$ANALYSIS/analyze.log" >&2; exit 1; }
tail -60 "$ANALYSIS/summary.md"
echo "[$(date)] readouts done"
