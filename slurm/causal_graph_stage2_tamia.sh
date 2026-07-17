#!/bin/bash
#SBATCH --job-name=cg_stage2
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out

# cot_causal_graph_v0 Stage 2: free-generation rollouts. Pilot block (50 arm-F
# traces on GPU 0) prints gate G2 first, then the full 4-shard run over both
# arms, CPU merge with gates, stage-3 assembly and the explorer build.
# Requires stage 1 outputs (node_features.parquet drives the on-policy
# intervention-site policy).
#
# Usage:  sbatch slurm/causal_graph_stage2_tamia.sh
#         K_ROLLOUTS=16 sbatch ...   # if the pilot fails G2

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_DIR="${RUN_DIR:-$PROJECT_ROOT/runs/causal_graph}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-7B}"
K_ROLLOUTS="${K_ROLLOUTS:-8}"
MAX_ONPOLICY="${MAX_ONPOLICY:-200}"

source "$PROJECT_ROOT/slurm/s1_model_size/models.env"
export HF_HOME="${HF_HOME:-$HF_CACHE_ROOT}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

cd "$PROJECT_ROOT"
for f in "$RUN_DIR/traces_forks.jsonl" "$RUN_DIR/onpolicy_trajectories.jsonl" \
         "$RUN_DIR/stage1/node_features.parquet"; do
  test -e "$f" || { echo "missing $f (run stage 1 first)"; exit 1; }
done

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch transformers numpy scipy scikit-learn pandas pyarrow

cat <<BANNER
================================================================
job        : ${SLURM_JOB_NAME:-cg_stage2}  id ${SLURM_JOB_ID:-N/A}
host       : $(hostname)   date $(date -Iseconds)
run_dir    : $RUN_DIR   k_rollouts $K_ROLLOUTS
================================================================
BANNER

# ---- pilot: gate G2 before committing the budget -----------------------------
CUDA_VISIBLE_DEVICES=0 python scripts/causal_graph/cg_stage2_fg.py \
  --run_dir "$RUN_DIR" --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --arm forks --pilot --k_rollouts "$K_ROLLOUTS" --local_files_only \
  --shard_id 0 --num_shards 1
echo "[cg_stage2] pilot done; see stage2/fg_rollouts_shard0_pilot.parquet"

# ---- full run, both arms, 4 in-node shards -----------------------------------
for arm in forks onpolicy; do
  pids=()
  for g in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$g python scripts/causal_graph/cg_stage2_fg.py \
      --run_dir "$RUN_DIR" --model_name_or_path "$MODEL_NAME_OR_PATH" \
      --arm $arm --k_rollouts "$K_ROLLOUTS" \
      --max_onpolicy_traces "$MAX_ONPOLICY" --local_files_only \
      --shard_id $g --num_shards 4 \
      > "cg_stage2_${arm}_shard$g-${SLURM_JOB_ID:-manual}.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
done

# ---- merge + gates, stage 3, explorer (CPU) ----------------------------------
python scripts/causal_graph/cg_stage2_fg.py --run_dir "$RUN_DIR" --merge
python scripts/causal_graph/cg_stage3_assemble.py --run_dir "$RUN_DIR"
python scripts/causal_graph/cg_explorer.py --run_dir "$RUN_DIR"

echo "[cg_stage2] done $(date -Iseconds)"
