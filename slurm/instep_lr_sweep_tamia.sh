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
#   Seeds 0-2 on two sources: `ds` = the ReProbe DeepSeek-labelled split (ReProbe layout),
#   `prm` = 100k PRM800K human-labelled items (chat layout). Smoke 662416 (per step, 10
#   configs): ds 1.25 s (text 0.28, reads 0.085 per config), prm 2.25 s (text 0.46, reads
#   0.16 per config), i.e. ds ~41 min and prm ~70 min per seed. Balanced over 4 GPUs:
#     GPU0 prm seed 0, then ds seed 2 configs 1-5     GPU1 prm seed 1, then ds seed 2 configs 6-10
#     GPU2 prm seed 2                                 GPU3 ds seed 0, then ds seed 1
#   (configs are independent, so splitting one seed's grid changes nothing but the text
#   passes shared). Then the frozen selection rule. Submit with --time=02:00:00.
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
        --soft_attn no_cudnn)
IFS=, read -r -a CFG <<< "$GRID"
GRID_A=$(IFS=,; echo "${CFG[*]:0:5}"); GRID_B=$(IFS=,; echo "${CFG[*]:5}")

# sweep <gpu> <seed> <source> [grid] [out suffix]: one process trains the grid, then scores validation
sweep() {
  local g=$1 s=$2 src=$3 grid=${4:-$GRID} suf=${5:-} sm=() args=()
  [[ $SMOKE == 0 ]] || sm=(--smoke "$SMOKE")
  if [[ $src == ds ]]; then
    args=(--layout reprobe --train "$RP/ds_train.items.jsonl" --n_train 0 --val "$RP/ds_val.items.jsonl")
  else
    args=(--layout chat --train "$PRMD/prm800k_probe_train_full.jsonl" --n_train 100000
          --val "$PRMD/prm800k_val_5k.jsonl")
  fi
  CUDA_VISIBLE_DEVICES=$g python scripts/read_token_inline.py train "${COMMON[@]}" "${args[@]}" \
    --sweep "$grid" --seed "$s" --out "$OUT/${src}_seed$s$suf" "${sm[@]}" \
    > "$LOGS/${src}_seed$s$suf-$SLURM_JOB_ID.log" 2>&1
}

pids=()
if [[ $SMOKE != 0 ]]; then
  sweep 0 0 ds & pids+=($!)
  sweep 1 0 prm & pids+=($!)
else
  ( f=0; sweep 0 0 prm || f=1; sweep 0 2 ds "$GRID_A" _a || f=1; exit $f ) & pids+=($!)
  ( f=0; sweep 1 1 prm || f=1; sweep 1 2 ds "$GRID_B" _b || f=1; exit $f ) & pids+=($!)
  sweep 2 2 prm & pids+=($!)
  ( f=0; sweep 3 0 ds || f=1; sweep 3 1 ds || f=1; exit $f ) & pids+=($!)
fi
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
grep -h "^\[train inline\] step\|^\[val\]" "$LOGS"/*-"$SLURM_JOB_ID".log | tail -n 40 || true
python scripts/analysis/instep_lr_select.py \
  --source "reprobe_ds:ap:$OUT/ds_seed*/sweep.json" \
  --source "prm800k:auroc:$OUT/prm_seed*/sweep.json" --out "$OUT/selection" || fail=1
exit $fail
