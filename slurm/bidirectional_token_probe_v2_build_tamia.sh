#!/bin/bash
#SBATCH --job-name=btp2_build
#SBATCH --account=aip-azouaq
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out

# Build the frozen v2 manifest from the pinned ReProbe parquets (CPU only).
set -euo pipefail
: "${SNAP:?}" "${R:?}"
module purge
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0
virtualenv --no-download "$SLURM_TMPDIR/env" >/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index torch transformers numpy pandas 2>&1 | tail -1
export HF_HOME=/project/aip-azouaq/$USER/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$SNAP"
python scripts/build_reprobe_trajectory_dataset.py --ds_parquet "$R/raw/ds.parquet" \
  --self_parquet "$R/raw/self.parquet" \
  --v1_manifest /project/aip-azouaq/$USER/cot_mech/bidirectional_token_probe_v1/manifest_v1 \
  --out_dir "$R/manifest_v2" --tokenizer Qwen/Qwen3-8B --local_files_only
