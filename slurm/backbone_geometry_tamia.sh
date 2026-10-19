#!/bin/bash
#SBATCH --job-name=bb_geom
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out

# backbone_geometry_v0 (docs/backbone_geometry_v0_plan.md). One config per GPU,
# pooled vectors to node-local disk, then closed-form geometry on the node's CPUs.
# Only geometry.json and logs leave the node.
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
OUT="${OUT:-$SCRATCH/cot_mech/backbone_geometry_v0}"
DATA="${DATA:-$SCRATCH/cot_mech/dense_full_7b_v1/data}"
PB="${PB:-/scratch/d/dchikhi/cot-checker/processbench_full}"
N_TRAIN="${N_TRAIN:-60000}"
SMOKE="${SMOKE:-0}"
LOCAL="$SLURM_TMPDIR/bbg"
mkdir -p "$OUT/logs" "$LOCAL"

module purge
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0
export HF_HOME=/project/aip-azouaq/$USER/hf_cache
unset TRANSFORMERS_CACHE
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$PROJECT_ROOT"
echo "git $(git rev-parse --short HEAD) smoke=$SMOKE n_train=$N_TRAIN"
virtualenv --no-download "$SLURM_TMPDIR/env" >/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip >/dev/null
pip install --no-index torch transformers numpy scipy scikit-learn 2>&1 | tail -1

splits=(--prm800k "train:$DATA/prm800k_probe_train_full.jsonl:$N_TRAIN" "val:$DATA/prm800k_val_5k.jsonl"
        "test:$DATA/prm800k_test_2k.jsonl"
        --pb "gsm8k:$PB/processbench_gsm8k.jsonl" "math:$PB/processbench_math.jsonl"
             "olympiadbench:$PB/processbench_olympiadbench.jsonl" "omnimath:$PB/processbench_omnimath.jsonl")
q25=(14 20 24 27 28)
q3=(12 18 24 30 35 36)
configs=("prm_ver|Qwen/Qwen2.5-Math-PRM-7B|verifier|${q25[*]}"
         "parent_ver|Qwen/Qwen2.5-Math-7B-Instruct|verifier|${q25[*]}"
         "prm_native|Qwen/Qwen2.5-Math-PRM-7B|prm_native|${q25[*]}"
         "q3_ver|Qwen/Qwen3-8B|verifier|${q3[*]}")
pids=()
for i in 0 1 2 3; do
  IFS='|' read -r name model style layers <<<"${configs[$i]}"
  # shellcheck disable=SC2086
  ( CUDA_VISIBLE_DEVICES=$i python scripts/encode_pooled_multilayer.py --model "$model" \
      --style "$style" --layers $layers "${splits[@]}" --out "$LOCAL/$name" \
      --max_seq_len 4096 --smoke "$SMOKE" ) > "$OUT/logs/encode_${name}-${SLURM_JOB_ID}.log" 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
for i in 0 1 2 3; do IFS='|' read -r name _ <<<"${configs[$i]}"; tail -2 "$OUT/logs/encode_${name}-${SLURM_JOB_ID}.log"; done
du -sh "$LOCAL"/*
export OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 MKL_NUM_THREADS=12
gpids=()
for i in 0 1 2 3; do
  IFS='|' read -r name _ <<<"${configs[$i]}"
  [[ -e "$LOCAL/$name/pb_omnimath.shard0.npz" ]] || { echo "[skip] $name has no output"; continue; }
  python scripts/analysis/backbone_geometry.py --configs "$name:$LOCAL/$name" \
    --out "$OUT/geometry_$name" > "$OUT/logs/geometry_${name}-${SLURM_JOB_ID}.log" 2>&1 &
  gpids+=($!)
done
for p in "${gpids[@]}"; do wait "$p" || fail=1; done
tail -n 3 "$OUT"/logs/geometry_*-"${SLURM_JOB_ID}".log
exit $fail
