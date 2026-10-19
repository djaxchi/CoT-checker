#!/bin/bash
#SBATCH --job-name=das_probe_dir
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=00:50:00
#SBATCH --output=%x-%j.out
# das_probe_direction_v0 (docs/das_probe_direction_v0_plan.md): fixed-subspace
# interchange on the §18.1 caches, layers 12 and 20 on two GPUs.
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
module purge
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0
export HF_HOME=/project/aip-azouaq/$USER/hf_cache
unset TRANSFORMERS_CACHE
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$PROJECT_ROOT"
echo "git $(git rev-parse --short HEAD)"
virtualenv --no-download "$SLURM_TMPDIR/env" >/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip >/dev/null
pip install --no-index torch transformers numpy scipy pandas 2>&1 | tail -1
pids=()
for pair in "0 12" "1 20"; do
  set -- $pair
  CUDA_VISIBLE_DEVICES=$1 python scripts/das_branch/das_probe_direction.py --layer $2 \
    --local_files_only --report runs/das_train/probe_direction_L$2.json > runs/das_train/probe_direction_L$2.log 2>&1 &
  pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
tail -n 30 runs/das_train/probe_direction_L*.log
exit $fail
