#!/bin/bash
#SBATCH --job-name=das_p1a
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out

# DAS branch-subspace, Phase 1a (premise gate). Free-generation rollouts on the
# FORKS arm only (reuses cot_causal_graph_v0 stage 2). Decides whether branch
# identity matters behaviorally at all: does free-gen from the WRONG candidate
# step reach the correct answer less often than from the CORRECT (golden) step,
# and is that gap content-specific (vs swap_pos / swap_xprob controls)?
#   base       = golden correct step   (the "source"/correct branch)
#   swap_wrong = fork's rating -1 step  (the "base"/wrong branch)
# Gate fg_wrong_effect: mean(swap_wrong - base) < 0, Wilcoxon less.
# Gate G3: |wrong-base| AUC vs |control-base| > 0.55.
# No stage-1 dependency (fork contexts only). Merge reuses cg_stage2_fg --merge,
# which tolerates a forks-only run (onpolicy pivot is skipped when absent).
#
# Usage:  sbatch slurm/das_branch_phase1a_premise_tamia.sh
#         K_ROLLOUTS=16 sbatch ...

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_DIR="${RUN_DIR:-$PROJECT_ROOT/runs/causal_graph}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-7B}"
K_ROLLOUTS="${K_ROLLOUTS:-8}"

source "$PROJECT_ROOT/slurm/s1_model_size/models.env"
export HF_HOME="${HF_HOME:-$HF_CACHE_ROOT}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

cd "$PROJECT_ROOT"
test -e "$RUN_DIR/traces_forks.jsonl" || { echo "missing traces_forks.jsonl"; exit 1; }

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch transformers numpy scipy scikit-learn pandas pyarrow

cat <<BANNER
================================================================
job        : ${SLURM_JOB_NAME:-das_p1a}  id ${SLURM_JOB_ID:-N/A}
host       : $(hostname)   date $(date -Iseconds)
run_dir    : $RUN_DIR   k_rollouts $K_ROLLOUTS   model $MODEL_NAME_OR_PATH
================================================================
BANNER

# ---- full forks arm, 4 in-node shards ----------------------------------------
pids=()
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g python scripts/causal_graph/cg_stage2_fg.py \
    --run_dir "$RUN_DIR" --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --arm forks --k_rollouts "$K_ROLLOUTS" --local_files_only \
    --shard_id $g --num_shards 4 \
    > "das_p1a_forks_shard$g-${SLURM_JOB_ID:-manual}.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done

# ---- merge + premise/G3 gates (CPU) ------------------------------------------
python scripts/causal_graph/cg_stage2_fg.py --run_dir "$RUN_DIR" --merge

echo "[das_p1a] done $(date -Iseconds); gates -> $RUN_DIR/stage2/gates_stage2.json"
