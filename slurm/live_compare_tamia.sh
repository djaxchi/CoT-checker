#!/bin/bash
#SBATCH --job-name=live_compare
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
# submit from a writable dir (snapshots are read-only), e.g. with -o $SCRATCH/.../live/logs/%x-%j.out
# read_token_v1 Stage J (docs/read_token_v1_plan.md): the read token against the PRM while
# Qwen3-8B generates (scripts/analysis/live_compare.py), arms plain / read / read_stop /
# prm / prm_stop at B = 1 and B = 16 on confirmatory MATH-500 problems.
#   smoke  GPU0: 1 + 4 problems, 512-token cap, 2 replayed pool trajectories (fails fast)
#   then   GPU0-3: one shard of the confirmatory problems each (B = 1: 16 problems,
#          B = 16: 64 problems x 2 samples), 10 replayed pool trajectories per shard
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
OUT="${OUT:-$SCRATCH/cot_mech/read_token_v1}"
POOL="$OUT/downstream/pools/instruct_t10"
LV="$OUT/live"
LOGS="$LV/logs"
CKPT="$OUT/tokens/rthics_seed1/read_token.pt"
mkdir -p "$LOGS"
cd "$PROJECT_ROOT"
echo "snapshot ${PROJECT_ROOT}"

module purge
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0
export HF_HOME=/project/aip-azouaq/$USER/hf_cache
unset TRANSFORMERS_CACHE
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 MKL_NUM_THREADS=12
virtualenv --no-download "$SLURM_TMPDIR/env" >/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip >/dev/null
pip install --no-index torch transformers numpy scipy scikit-learn pyyaml sympy safetensors huggingface_hub 2>&1 | tail -1

filt() { grep -vE "Loading weights|UNEXPECTED|MISSING|Notes:|can be ignored|LOAD REPORT|^Key|^-+\+|^ *$" "$1" | tail -${2:-40}; }

run() {  # gpu name extra-args...
  local gpu=$1 name=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu python scripts/analysis/live_compare.py --inline_ckpt "$CKPT" \
    --trajectories "$POOL"/tts_math500.shard*_trajectories.jsonl --pool_scores "$POOL/scores" "$OUT/downstream/scores/instruct_t10" \
    --out "$LV/${name}-$SLURM_JOB_ID" "$@" > "$LOGS/${name}-$SLURM_JOB_ID.log" 2>&1
}

SHARDS=(${SHARDS:-0 1 2 3})  # a subset reruns failed shards, one per GPU
if [[ "${SMOKE:-1}" == 1 ]]; then
  echo "[$(date)] smoke"
  run 0 smoke --batch 1 4 --problems_b1 1 --problems_bn 4 --samples 1 --n_replay 2 --max_new_tokens 512 \
    || { filt "$LOGS/smoke-$SLURM_JOB_ID.log" 60; exit 1; }
  filt "$LOGS/smoke-$SLURM_JOB_ID.log" 40
fi

echo "[$(date)] shards ${SHARDS[*]}"
pids=()
for g in "${!SHARDS[@]}"; do
  i=${SHARDS[$g]}
  run $g shard$i --shard_idx $i --num_shards 4 --n_replay 10 & pids+=($!)
done
fail=0
for g in "${!SHARDS[@]}"; do
  i=${SHARDS[$g]}
  wait "${pids[$g]}" || { echo "[FAIL] shard$i"; filt "$LOGS/shard$i-$SLURM_JOB_ID.log" 40; fail=1; }
done
for i in "${SHARDS[@]}"; do echo "== shard$i"; filt "$LOGS/shard$i-$SLURM_JOB_ID.log" 30; done
echo "[$(date)] done"
exit $fail
