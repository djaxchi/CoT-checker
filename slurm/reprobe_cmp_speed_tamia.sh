#!/bin/bash
#SBATCH --job-name=rpc_speed
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=00:40:00
#SBATCH --output=%x-%j.out
# reprobe_comparison_v2 cost: the inline read token (v1 RT-self) against ReProbe's released
# hidden-state head (self labels) and the PRM, per trajectory and per step, with the Stage H4
# benchmark (scripts/analysis/inline_speed.py --reprobe_release). One GPU, the node otherwise
# idle. smoke (8 trajectories), then 200 trajectories of the T=1.0 MATH-500 pool at batch 1
# and 10, then the same again (run-to-run spread).
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
V1=/project/aip-azouaq/$USER/cot_mech/reprobe_cmp_v1
CKPT="${CKPT:-$V1/runs/tokens/rt_self_seed0/read_token.pt}"
REL="${REL:-$V1/released/self}"
TRAJ="$SCRATCH/cot_mech/read_token_v1/downstream/pools/instruct_t10/tts_math500.shard00_trajectories.jsonl"
SP="${SP:-/project/aip-azouaq/$USER/cot_mech/reprobe_cmp_v2/speed}"
mkdir -p "$SP/logs"
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
pip install --no-index torch transformers numpy scipy scikit-learn pyyaml pandas 2>&1 | tail -1
pip install --no-index triton 2>&1 | tail -1 || echo "no triton wheel"
export CUDA_VISIBLE_DEVICES=0

run() {  # name n batches...
  local name=$1 n=$2; shift 2
  python scripts/analysis/inline_speed.py --inline_ckpt "$CKPT" --reprobe_release "$REL" \
    --trajectories "$TRAJ" --n "$n" --batch "$@" --out "$SP/${name}-$SLURM_JOB_ID.json" \
    > "$SP/logs/${name}-$SLURM_JOB_ID.log" 2>&1 || { tail -30 "$SP/logs/${name}-$SLURM_JOB_ID.log"; exit 1; }
  grep -vE "Loading weights|UNEXPECTED|Notes:|can be ignored|LOAD REPORT|^Key|^-+\+|^ *$" \
    "$SP/logs/${name}-$SLURM_JOB_ID.log" | tail -45
}
run smoke 8 1 2
run rt_self 200 1 10
run rt_self_rep 200 1 10
echo "[$(date)] done"
