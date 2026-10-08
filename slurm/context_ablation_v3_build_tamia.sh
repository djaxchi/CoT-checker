#!/bin/bash
#SBATCH --job-name=cav3_build
#SBATCH --account=aip-azouaq
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=%x-%j.out

# Build the four context-view manifests (CPU, tokenizer only).
set -euo pipefail
: "${SNAP:?}" "${R:?}"
module purge
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0
virtualenv --no-download "$SLURM_TMPDIR/env" >/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index torch transformers numpy 2>&1 | tail -1
export HF_HOME=/project/aip-azouaq/$USER/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$SNAP"
P=/project/aip-azouaq/$USER/cot_mech
python scripts/build_context_views.py --v1_manifest $P/bidirectional_token_probe_v1/manifest_v1 \
  --v2_manifest $P/bidirectional_token_probe_v2/manifest_v2 --out_root "$R/views" --local_files_only
