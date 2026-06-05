#!/bin/bash
#SBATCH --job-name=s2-04-full
#SBATCH --account=aip-azouaq
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gpus-per-node=h100:4
#SBATCH --output=results/logs/%x-%j.out
#SBATCH --error=results/logs/%x-%j.err

# ---------------------------------------------------------------------------
# Sprint 2, Stage 04 (FULL): the 2x2 representation-shaping matrix.
# One node, 4 H100s, one method pinned per GPU, run concurrently:
#   GPU0 -> ae_rank   GPU1 -> sae_rank   GPU2 -> ae_triplet   GPU3 -> sae_triplet
# Full fork dataset, full epochs, all 4 ProcessBench subsets, fine threshold
# grid. Fails if any worker exits non-zero.
#
# Prereqs: stages 00 (build), 02 (encode) complete; CACHE_DIR holds the S1
# PRM800K probe_train_40k / val_1k; PB_BASE holds the 4-subset PB caches.
#
# Submit:  sbatch scripts/tamia/jobs/s2/04_train_matrix_full.sbatch
# ---------------------------------------------------------------------------

set -euo pipefail

PROJECT_DIR="$HOME/CoT-checker"
STORE="/project/aip-azouaq/$USER"
SCRATCH_BASE="$SCRATCH/cot_mech"

export PROJECT_DIR
: "${CACHE_DIR:=$SCRATCH_BASE/prestudy_v1/cache/qwen2_5_1_5b}"                  # probe_train_40k / val_1k
: "${ENC_DIR:=$SCRATCH_BASE/s2_forks/encoded}"
: "${FORKS_DIR:=$SCRATCH_BASE/s2_forks/data}"
: "${PB_BASE:=$SCRATCH_BASE/prestudy_v1/cache/qwen2_5_1_5b_processbench_full}"  # per-subset pb_step_*
export CACHE_DIR ENC_DIR FORKS_DIR PB_BASE
export OUT_BASE="$PROJECT_DIR/runs/s2/full"

# Full knobs (match S1 SAE training defaults).
export EPOCHS_SAE=20
export EPOCHS_PROBE=50
export BATCH=512
export THRESHOLD_GRID=0.005
export PB_SUBSETS="gsm8k math olympiadbench omnimath"
export RANK_KIND=logistic
export TRIPLET_METRIC=l2
export TRIPLET_MARGIN=1.0

# Matched comparison at obj_weight=1: recon-only controls + the 4 objective
# cells. ae_recon/sae_recon differ from ae_rank/sae_rank etc. ONLY by the
# objective term, so any F1 delta is attributable to the objective.
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
