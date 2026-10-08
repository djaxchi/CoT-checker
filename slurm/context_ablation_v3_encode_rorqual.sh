#!/bin/bash
#SBATCH --job-name=cav3_enc
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00

# context_ablation_v3 on Rorqual (per-GPU allocation): encode SHARDS of one
# context's view manifest into a shared scratch store. Same encoder, model
# revision and layer as TamIA; shards are disjoint, completion markers are atomic.
set -euo pipefail
: "${SNAP:?}" "${RUN:?}" "${CTX:?}" "${SHARDS:?space-free list like 0:1:2:3}"
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
unset TRANSFORMERS_CACHE
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 TORCH_THREADS=4
virtualenv --no-download "$SLURM_TMPDIR/env" >/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index torch transformers numpy 2>&1 | tail -1
cd "$SNAP"
python scripts/encode_trajectory_token_store.py --manifest_dir "$RUN/views/$CTX" \
  --rep_root "$RUN/stores/$CTX" --model_name_or_path Qwen/Qwen3-8B --local_files_only \
  --layer 35 --shards ${SHARDS//:/ } --batch_tokens 32768 --local_tmp "$SLURM_TMPDIR/stage"
