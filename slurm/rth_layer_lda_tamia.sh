#!/bin/bash
#SBATCH --job-name=rth_m1
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=01:15:00
#SBATCH --output=%x-%j.out
# read_token_v1 mechanistic study M1 (docs/read_token_v1_plan.md): per-layer LDA on the
# mechanistic bundle for the Qwen3-8B last step token, the read token and the PRM's
# last step token, one representation per GPU (scripts/analysis/rth_layer_lda.py).
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
MECH="${MECH:-/project/aip-azouaq/$USER/cot_mech/read_token_v1/mech}"
OUT="${OUT:-$SCRATCH/cot_mech/read_token_v1/mech_study}"
mkdir -p "$OUT/logs"
cd "$PROJECT_ROOT"
module purge
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0
export OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 MKL_NUM_THREADS=12
virtualenv --no-download "$SLURM_TMPDIR/env" >/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip >/dev/null
pip install --no-index torch numpy scipy scikit-learn pyyaml 2>&1 | tail -1
pids=(); g=0
for rep in last_step read prm; do
  CUDA_VISIBLE_DEVICES=$g python scripts/analysis/rth_layer_lda.py --bundle "$MECH/bundle" --rep $rep \
    --out "$OUT/m1_$rep.json" > "$OUT/logs/m1_$rep-$SLURM_JOB_ID.log" 2>&1 &
  pids+=($!); g=$((g + 1))
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
tail -n 2 "$OUT"/logs/m1_*-"$SLURM_JOB_ID".log
exit $fail
