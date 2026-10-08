#!/bin/bash
#SBATCH --job-name=cav3_fit
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

# context_ablation_v3 on Rorqual: one fit (line LINE of FITS_FILE) on the shared
# scratch store of its context. Resumable: a re-run continues from last.pt.
set -euo pipefail
: "${SNAP:?}" "${RUN:?}" "${CTX:?}" "${LINE:?}"
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 TORCH_THREADS=4
virtualenv --no-download "$SLURM_TMPDIR/env" >/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index torch numpy 2>&1 | tail -1
cd "$SNAP"
line=$(grep -v '^\s*#' "experiments/context_ablation_v3/$CTX.fits" | grep -v '^\s*$' | sed -n "${LINE}p")
read -r name args <<< "$line"
echo "[fit] $name"
python scripts/train_contextual_token_probe.py --store "$RUN/stores/$CTX" --out_dir "$RUN/prod/fits/$name" $args
