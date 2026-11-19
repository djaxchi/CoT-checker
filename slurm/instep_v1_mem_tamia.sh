#!/bin/bash
#SBATCH --job-name=instep_mem
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
# InStep paper, reviewer Q3/W4: peak GPU memory (scripts/instep_v1/mem_bench.py) on the speed
# benchmark's inputs (reprobe_cmp_speed_tamia.sh: the same token, released ReProbe head and
# 200 MATH-500 trajectories), one GPU, the node otherwise idle, one fresh process per mode:
# policy alone, InStep, Qwen2.5-Math-PRM-7B, ReProbe head; 1, 5 and 10 trajectories per call.
# N=8 smoke first (fails fast).
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-review}"
V1=/project/aip-azouaq/$USER/cot_mech/reprobe_cmp_v1
CKPT="${CKPT:-$V1/runs/tokens/rt_self_seed0/read_token.pt}"
REL="${REL:-$V1/released/self}"
TRAJ="$SCRATCH/cot_mech/read_token_v1/downstream/pools/instruct_t10/tts_math500.shard00_trajectories.jsonl"
SP="${SP:-/project/aip-azouaq/$USER/cot_mech/instep_v1/review/mem}"
mkdir -p "$SP/logs"
cd "$PROJECT_ROOT"
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
run() {  # name mode n batches...
  local name=$1 mode=$2 n=$3; shift 3
  python scripts/instep_v1/mem_bench.py --mode "$mode" --inline_ckpt "$CKPT" --reprobe_release "$REL" \
    --trajectories "$TRAJ" --n "$n" --batch "$@" --out "$SP/${name}-$SLURM_JOB_ID.json" \
    > "$SP/logs/${name}-$SLURM_JOB_ID.log" 2>&1 || { tail -30 "$SP/logs/${name}-$SLURM_JOB_ID.log"; exit 1; }
  grep "^\[b\|^\[out\]" "$SP/logs/${name}-$SLURM_JOB_ID.log"
}
# MODES: which modes to run (default all); instep_fused runs once per batch size, each in a fresh
# process, since its captured graphs and static buffers stay allocated
MODES="${MODES:-policy instep instep_fused prm reprobe}"
for m in $MODES; do
  if [[ $m == instep_fused ]]; then run smoke_${m}_b2 $m 8 2; else run smoke_$m $m 8 1 2; fi
done
for m in $MODES; do
  if [[ $m == instep_fused ]]; then for b in 1 5 10; do run ${m}_b$b $m 200 $b; done
  else run $m $m 200 1 5 10; fi
done
echo "[$(date)] done"
