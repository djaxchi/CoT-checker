#!/bin/bash
#SBATCH --job-name=prm800k_fulltest_eval
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out

# Eval-only re-run: the all-sizes encode job already wrote each size's full-test
# encoding to <RUNS_ROOT>/<tag>/merged/<stem>_{h,y,meta}; only the eval step failed
# (it needed src/ on PYTHONPATH). This re-evals every size's dense probe against
# those saved encodings and aggregates -- NO GPU, NO re-encoding.
#
# Usage:
#   STEM=prm800k_test_full sbatch slurm/eval_prm800k_fulltest_allsizes_tamia.sh
set -uo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
cd "$PROJECT_ROOT"
source slurm/s1_model_size/models.env   # S1MS_*, RUNS_ROOT

STEM="${STEM:-prm800k_test_full}"
EVAL_OUT="${EVAL_OUT:-$PROJECT_ROOT/results/prm800k_test_full_eval}"
TAGS="${TAGS:-qwen2_5_1_5b qwen2_5_3b qwen2_5_7b qwen2_5_14b qwen2_5_32b}"

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
mkdir -p "$EVAL_OUT"

cat <<BANNER
================================================================
job      : ${SLURM_JOB_NAME:-fulltest_eval}  id ${SLURM_JOB_ID:-N/A}
host     : $(hostname)   date $(date -Iseconds)
git      : $(git rev-parse --short HEAD 2>/dev/null || echo unknown)
stem     : $STEM   tags: $TAGS
eval_out : $EVAL_OUT
================================================================
BANNER

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch numpy scikit-learn

for tag in $TAGS; do
  run_dir="$RUNS_ROOT/$tag"
  enc_dir="$run_dir/merged"
  label="${S1MS_PARAMS_LABEL[$tag]:-$tag}"
  if [[ ! -f "$enc_dir/${STEM}_h.npy" ]]; then
    echo "[WARN] $tag: no $enc_dir/${STEM}_h.npy; skipping"; continue
  fi
  if [[ ! -f "$run_dir/linear_probe.pt" ]]; then
    echo "[WARN] $tag: no $run_dir/linear_probe.pt; skipping"; continue
  fi
  echo "[eval] $tag ($label)"
  python scripts/eval_prm800k_heldout_probe.py \
    --run_dir "$run_dir" --enc_dir "$enc_dir" --stem "$STEM" \
    --tag "$label" --out_dir "$EVAL_OUT" \
    || echo "[ERR] $tag eval FAILED"
done

python scripts/aggregate_heldout_eval.py --in_dir "$EVAL_OUT" \
  || echo "[WARN] aggregation failed (per-size JSONs still in $EVAL_OUT)"
echo "[$(date -Iseconds)] full-test eval complete -> $EVAL_OUT"
