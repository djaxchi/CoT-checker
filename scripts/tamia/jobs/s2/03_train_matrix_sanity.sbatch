#!/bin/bash
#SBATCH --job-name=s2-03-sanity
#SBATCH --account=aip-azouaq
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gpus-per-node=h100:4
#SBATCH --output=results/logs/%x-%j.out
#SBATCH --error=results/logs/%x-%j.err

# ---------------------------------------------------------------------------
# Sprint 2, Stage 03 (SANITY): tiny subset, 1 epoch, GSM8K-only ProcessBench.
# Exercises all 4 cells across all 4 GPUs to catch wiring/path errors fast
# before committing the full run. Fails if any worker exits non-zero.
#
# Submit:  sbatch scripts/tamia/jobs/s2/03_train_matrix_sanity.sbatch
# ---------------------------------------------------------------------------

set -euo pipefail

PROJECT_DIR="$HOME/CoT-checker"
STORE="/project/aip-azouaq/$USER"
SCRATCH_BASE="$SCRATCH/cot_mech"

export PROJECT_DIR
export CACHE_DIR="$SCRATCH_BASE/prestudy_v1/encoded"     # probe_train_40k / val_1k
export ENC_DIR="$SCRATCH_BASE/s2_forks/encoded"
export FORKS_DIR="$SCRATCH_BASE/s2_forks/data"
export PB_BASE="$SCRATCH_BASE/processbench_full"
export OUT_BASE="$PROJECT_DIR/runs/s2/sanity"

# Sanity knobs: small subset + 1 epoch + one PB subset + coarse threshold grid.
export MAX_PAIRS=512
export EPOCHS_SAE=1
export EPOCHS_PROBE=1
export BATCH=256
export THRESHOLD_GRID=0.05
export PB_SUBSETS="gsm8k"

# Smoke all paths: recon controls + the 4 objective cells at obj_weight=1.
JOBS=(ae_recon sae_recon ae_rank:1 sae_rank:1 ae_triplet:1 sae_triplet:1)

cd "$PROJECT_DIR"
mkdir -p results/logs

module purge
module load StdEnv/2023 gcc arrow/24.0.0 python/3.11 cuda/12.2
source "$HOME/venvs/cot/bin/activate"

export HF_HOME="$STORE/hf_cache"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

source scripts/tamia/jobs/s2/_matrix_common.sh
