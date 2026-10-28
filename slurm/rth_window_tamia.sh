#!/bin/bash
#SBATCH --job-name=rth_window
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=01:40:00
#SBATCH --output=%x-%j.out
# read_token_v1 mechanistic study M6: RTH retrained with offsets in some layers only
# (scripts/analysis/rth_mech_window_train.py), RTH's recipe otherwise (dq_correct:
# " correct", m = 1, deep, linear head, 100k steps, seed 0), one arm per GPU, two
# complementary pairs:
#   GPU0 layers 0-17      GPU1 layers 18-35
#   GPU2 layers 9-17,27-35 (the two windows M2 found heaviest)   GPU3 layers 0-8,18-26
# Each arm is scored like every read_token_v1 run (geometry, ProcessBench scores in the
# leaderboard format), then M3's attention dump runs on all arms with RTH and BL1.
# Readings, fixed before the run (RTH 3-seed calib-20 0.659, BL1 from workstream A):
#   a half within 0.01 of RTH (>= 0.649) is enough; only 0-17 enough = the gain comes
#   from early layers (where M3 sees the attention shift onto the step); only 18-35 =
#   late readout; both = offsets are interchangeable across depth.
# SEED=1 (or 2): the same arms at another seed, names suffixed _seed<SEED>, no attention dump.
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
OUT="${OUT:-$SCRATCH/cot_mech/read_token_v1}"
MS="${MS:-$OUT/mech_study}"
MECH="${MECH:-/project/aip-azouaq/$USER/cot_mech/read_token_v1/mech}"
DATA="${DATA:-$SCRATCH/cot_mech/dense_full_7b_v1/data}"
PB="${PB:-/scratch/d/dchikhi/cot-checker/processbench_full}"
Q3=Qwen/Qwen3-8B
LOCAL="$SLURM_TMPDIR/m6"
LOGS="$MS/logs"
J="$SLURM_JOB_ID"
mkdir -p "$LOGS" "$LOCAL" "$MS/m6"
cd "$PROJECT_ROOT"
echo "git $(git rev-parse --short HEAD 2>/dev/null || echo snapshot)"
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
TRAIN="$DATA/prm800k_probe_train_full.jsonl"
ENC=(--prm800k "train:$TRAIN:30000" "val:$DATA/prm800k_val_5k.jsonl" "test:$DATA/prm800k_test_2k.jsonl"
     --pb "gsm8k:$PB/processbench_gsm8k.jsonl" "math:$PB/processbench_math.jsonl"
          "olympiadbench:$PB/processbench_olympiadbench.jsonl" "omnimath:$PB/processbench_omnimath.jsonl")
LIKE="$OUT/stage_f/cells_rth/rth_dq_correct"
SEED="${SEED:-0}"
SUF=""; [[ "$SEED" != 0 ]] && SUF="_seed$SEED"

arm() {
  local gpu=$1 layers=$2 name="m6_layers${2//,/_}$SUF"
  (
    set -e
    export CUDA_VISIBLE_DEVICES=$gpu
    python scripts/analysis/rth_mech_window_train.py --layers "$layers" -- --model $Q3 --m 1 --init text \
      --init_text " correct" --deep --head linear --n_train 100000 --seed "$SEED" --soft_attn no_cudnn \
      --train "$TRAIN" --out "$LOCAL/$name"
    python scripts/read_token_probe.py encode --model $Q3 --out "$LOCAL/$name" "${ENC[@]}"
    python scripts/analysis/backbone_geometry.py --configs "$name:$LOCAL/$name" --reps sep \
      --out "$OUT/geometry_$name"
    mkdir -p "$OUT/tokens/$name" && cp "$LOCAL/$name/read_token.pt" "$OUT/tokens/$name/"
    python scripts/analysis/read_token_pb_scores.py --enc "$LOCAL/$name" --out "$OUT/pb_scores/$name" \
      --like "$LIKE"
  ) > "$LOGS/$name-$J.log" 2>&1
}

pids=()
arm 0 0-17 & pids+=($!)
arm 1 18-35 & pids+=($!)
arm 2 9-17,27-35 & pids+=($!)
arm 3 0-8,18-26 & pids+=($!)
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
for f in "$LOGS"/m6_*-"$J".log; do echo "== $f"; tail -n 3 "$f"; done

if [[ "$SEED" != 0 ]]; then echo "[$(date)] done, fail=$fail"; exit $fail; fi
toks=("rth=$OUT/tokens/dq_correct/read_token.pt" "bl1=$OUT/tokens/bl1_correct_seed0/read_token.pt")
for n in m6_layers0-17 m6_layers18-35 m6_layers9-17_27-35 m6_layers0-8_18-26; do
  toks+=("$n=$OUT/tokens/$n/read_token.pt")
done
CUDA_VISIBLE_DEVICES=0 python scripts/analysis/rth_mech_attention.py --model $Q3 --bundle "$MECH/bundle" \
  --split test_2k --source "prm800k:$DATA/prm800k_test_2k.jsonl" --tokens "${toks[@]}" --out "$MS/m6" \
  > "$LOGS/m6_attention-$J.log" 2>&1 || fail=1
tail -n 8 "$LOGS/m6_attention-$J.log"
echo "[$(date)] done, fail=$fail"
exit $fail
