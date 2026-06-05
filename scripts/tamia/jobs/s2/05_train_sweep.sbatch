#!/bin/bash
#SBATCH --job-name=s2-05-sweep
#SBATCH --account=aip-azouaq
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gpus-per-node=h100:4
#SBATCH --output=results/logs/%x-%j.out
#SBATCH --error=results/logs/%x-%j.err

# ---------------------------------------------------------------------------
# Sprint 2, Stage 05: obj_weight sweep + recon-only controls.
# One node, 4 H100s; 14 jobs distributed round-robin across the GPUs, run
# concurrently, fail if any job exits non-zero.
#
#   controls : ae_recon, sae_recon                       (objective off)
#   sweep    : {ae,sae}_{rank,triplet} x {1, 10, 100}    (obj_weight)
#
# Each run logs final_objective_loss, final_pair_accuracy,
# final_margin_satisfaction (train_metrics.json) + per-epoch curves
# (representation_history.json), so the objective's effect is measurable and
# attributable against the recon-only controls.
#
# Submit:  sbatch scripts/tamia/jobs/s2/05_train_sweep.sbatch
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
export OUT_BASE="$PROJECT_DIR/runs/s2/sweep"

export EPOCHS_SAE=20
export EPOCHS_PROBE=50
export BATCH=512
export THRESHOLD_GRID=0.005
export PB_SUBSETS="gsm8k math olympiadbench omnimath"
export RANK_KIND=logistic
export TRIPLET_METRIC=l2
export TRIPLET_MARGIN=1.0

# 2 controls + 4 objective methods x {1,10,100} = 14 jobs.
JOBS=(
    ae_recon
    sae_recon
    ae_rank:1     ae_rank:10     ae_rank:100
    sae_rank:1    sae_rank:10    sae_rank:100
    ae_triplet:1  ae_triplet:10  ae_triplet:100
    sae_triplet:1 sae_triplet:10 sae_triplet:100
)

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
