#!/bin/bash
#SBATCH --job-name=inline_speed
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=00:45:00
#SBATCH --output=%x-%j.out
# read_token_v1 Stage H4 (docs/read_token_v1_plan.md): how fast the inline read can score a
# trajectory against the PRM (scripts/analysis/inline_speed.py). One GPU at a time, the rest
# of the node idle, so no run shares the host with another:
#   smoke  8 trajectories, batch 1 and 2 (fails fast)
#   then   RTH-ICS (H3's pick), RTH-IC, and RTH-ICS again (run-to-run spread), 200 trajectories
#          of the T=1.0 MATH-500 pool, batch 1 and 10
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
OUT="${OUT:-$SCRATCH/cot_mech/read_token_v1}"
DS="$OUT/downstream"
SP="$OUT/speed"
LOGS="$SP/logs"
TRAJ="$DS/pools/instruct_t10/tts_math500.shard00_trajectories.jsonl"
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
pip install --no-index torch transformers numpy scipy scikit-learn pyyaml 2>&1 | tail -1
pip install --no-index triton 2>&1 | tail -1 || echo "no triton wheel"
python -c "import torch, triton; print(\"torch\", torch.__version__, \"triton\", triton.__version__)" || true
export CUDA_VISIBLE_DEVICES=0

run() {  # name ckpt n batches...
  local name=$1 ck=$2 n=$3; shift 3
  python scripts/analysis/inline_speed.py --inline_ckpt "$OUT/tokens/$ck/read_token.pt" \
    --trajectories "$TRAJ" --n "$n" --batch "$@" --out "$SP/${name}-$SLURM_JOB_ID.json" \
    > "$LOGS/${name}-$SLURM_JOB_ID.log" 2>&1 || { tail -30 "$LOGS/${name}-$SLURM_JOB_ID.log"; exit 1; }
  grep -vE "Loading weights|UNEXPECTED|Notes:|can be ignored|LOAD REPORT|^Key|^-+\+|^ *$" \
    "$LOGS/${name}-$SLURM_JOB_ID.log" | tail -40
}
run smoke rthics_seed1 8 1 2
run rthics_seed1 rthics_seed1 200 1 10
run rthic_seed0 rthic_seed0 200 1 10
run rthics_seed1_rep rthics_seed1 200 1 10
echo "[$(date)] done"
