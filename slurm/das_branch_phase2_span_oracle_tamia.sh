#!/bin/bash
#SBATCH --job-name=das_span
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --output=%x-%j.out

# DAS branch-subspace, Phase 2 whole-step-span ORACLE (teacher-forced gold margin).
# Injects the correct branch's full candidate-step residual span into the wrong
# branch at L12/20/26. Runs both alignments:
#   lastk : final k tokens, all forks (powered)     -> 4 shards
#   equal : identical-length siblings (~25, clean)  -> 1 shard
# Primary readout is the TF gold-answer margin (S6 answer-belief extension); a
# positive, control-beating recovery is the green light for free-gen + DAS.
#
# Usage:  sbatch slurm/das_branch_phase2_span_oracle_tamia.sh
#         K=16 LAYERS=8,16,24 sbatch ...

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_DIR="${RUN_DIR:-$PROJECT_ROOT/runs/causal_graph}"
OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/runs/das_span}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-7B}"
LAYERS="${LAYERS:-12,20,26}"
K="${K:-8}"

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
job     : ${SLURM_JOB_NAME:-das_span}  id ${SLURM_JOB_ID:-N/A}
host    : $(hostname)   date $(date -Iseconds)
layers  : $LAYERS   k(lastk) $K   out $OUT_DIR
================================================================
BANNER

run_align () {  # $1=align  $2=num_shards
  local align="$1" nshards="$2" pids=()
  for ((g=0; g<nshards; g++)); do
    CUDA_VISIBLE_DEVICES=$g python scripts/das_branch/das_span_oracle.py \
      --run_dir "$RUN_DIR" --out_dir "$OUT_DIR" \
      --model_name_or_path "$MODEL_NAME_OR_PATH" --align "$align" --k "$K" \
      --layers "$LAYERS" --mode tf_margin --local_files_only \
      --shard_id $g --num_shards $nshards \
      > "das_span_${align}_shard$g-${SLURM_JOB_ID:-manual}.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  python scripts/das_branch/das_span_oracle.py --out_dir "$OUT_DIR" \
    --align "$align" --merge
}

run_align lastk 4
run_align equal 1

echo "[das_span] done $(date -Iseconds); gates -> $OUT_DIR/gates_span_{lastk,equal}.json"
