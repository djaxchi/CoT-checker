#!/bin/bash
#SBATCH --job-name=instep_lr
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=01:45:00
#SBATCH --output=%x-%j.out
# InStep lr x warm-up sweep on Qwen3-8B (docs/instep_lr_sweep_plan.md). Every config of the
# grid trains in ONE process per seed, on the same batches, sharing each batch's text pass
# (read_token_inline.py --sweep); each config is then scored on the source's VALIDATION split
# only (--val, one shared text pass). No test set is read.
#   GPU g = seed g (0-2): the ReProbe DeepSeek-labelled split (ReProbe layout), then 100k
#   PRM800K human-labelled items (chat layout); GPU3 idle. Then the frozen selection rule.
# SMOKE=N: GPU0 the ReProbe source, GPU1 PRM800K, seed 0, the first N training and N validation items
# (pipeline and per-step timing for --time). Submit the smoke with --time=00:25:00.
# --time: set from the smoke (text and read seconds are logged every 50 steps).
set -euo pipefail
SMOKE="${SMOKE:-0}"
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RP="${RP:-/project/aip-azouaq/$USER/cot_mech/reprobe_cmp_v1/data/rt}"
PRMD="${PRMD:-$SCRATCH/cot_mech/dense_full_7b_v1/data}"
OUT="${OUT:-/project/aip-azouaq/$USER/cot_mech/instep_lr_sweep}$([[ $SMOKE == 0 ]] || echo "_smoke$SMOKE")"
GRID="${GRID:-5e-4:0.05,5e-4:0.2,1e-3:0.05,1e-3:0.2,1.5e-3:0.05,1.5e-3:0.2,2e-3:0.05,2e-3:0.2,3e-3:0.05,3e-3:0.2}"
Q3=Qwen/Qwen3-8B
LOGS="$OUT/logs"
mkdir -p "$LOGS"
module purge
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0
export HF_HOME=/project/aip-azouaq/$USER/hf_cache
unset TRANSFORMERS_CACHE
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 MKL_NUM_THREADS=12
cd "$PROJECT_ROOT"
echo "git $(git rev-parse --short HEAD 2>/dev/null || echo none) smoke=$SMOKE out=$OUT grid=$GRID"
virtualenv --no-download "$SLURM_TMPDIR/env" >/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip >/dev/null
pip install --no-index torch transformers numpy scipy scikit-learn 2>&1 | tail -1
COMMON=(--model $Q3 --init_text " correct" --deep --head linear --chain --start_marker --start_deep
        --soft_attn no_cudnn --sweep "$GRID")

# sweep <gpu> <seed> <source>: one process trains the whole grid, then scores validation
sweep() {
  local g=$1 s=$2 src=$3 sm=() args=()
  [[ $SMOKE == 0 ]] || sm=(--smoke "$SMOKE")
  if [[ $src == ds ]]; then
    args=(--layout reprobe --train "$RP/ds_train.items.jsonl" --n_train 0 --val "$RP/ds_val.items.jsonl")
  else
    args=(--layout chat --train "$PRMD/prm800k_probe_train_full.jsonl" --n_train 100000
          --val "$PRMD/prm800k_val_5k.jsonl")
  fi
  CUDA_VISIBLE_DEVICES=$g python scripts/read_token_inline.py train "${COMMON[@]}" "${args[@]}" \
    --seed "$s" --out "$OUT/${src}_seed$s" "${sm[@]}" > "$LOGS/${src}_seed$s-$SLURM_JOB_ID.log" 2>&1
}

pids=()
if [[ $SMOKE != 0 ]]; then
  sweep 0 0 ds & pids+=($!)
  sweep 1 0 prm & pids+=($!)
else
  for g in 0 1 2; do ( sweep $g $g ds; sweep $g $g prm ) & pids+=($!); done
fi
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
grep -h "^\[train inline\] step\|^\[val\]" "$LOGS"/*-"$SLURM_JOB_ID".log | tail -n 40 || true
python scripts/analysis/instep_lr_select.py \
  --source "reprobe_ds:ap:$OUT/ds_seed*/sweep.json" \
  --source "prm800k:auroc:$OUT/prm_seed*/sweep.json" --out "$OUT/selection" || fail=1
exit $fail
