#!/bin/bash
#SBATCH --job-name=read_token_v1
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out
# read_token_v1 (docs/read_token_v1_plan.md), after read_token_v0. STAGE picks the node:
#   B  GPU0 B1 frozen slot, then B6 re-encodes; GPU1-2 B2 seeds 1, 2; GPU3 B3 400k steps
#   C  GPU0-2 parent Qwen2.5-Math-7B-Instruct seeds 0-2; GPU3 B7 (mean of all m read tokens)
#   Cparent  GPU0-2 parent seeds 0-2 only, eager soft pass (rerun of C's parent arm)
#   D  GPU0-2 deep read token seeds 0-2 (HEAD=linear|mlp), eager soft pass
# Every stage first re-encodes PRM800K val with v0's four tokens and picks C* and
# m_all on val AUROC (scripts/analysis/read_token_select.py). B and C stop if the
# v0 gate sends the experiment to Stage D. Only results, tokens and logs leave the node.
set -euo pipefail
STAGE="${STAGE:?set STAGE=B, C, Cparent or D}"
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
V0="${V0:-$SCRATCH/cot_mech/read_token_v0}"
OUT="${OUT:-$SCRATCH/cot_mech/read_token_v1}"
DATA="${DATA:-$SCRATCH/cot_mech/dense_full_7b_v1/data}"
PB="${PB:-/scratch/d/dchikhi/cot-checker/processbench_full}"
HEAD="${HEAD:-linear}"
Q3=Qwen/Qwen3-8B
PARENT=Qwen/Qwen2.5-Math-7B-Instruct
LOCAL="$SLURM_TMPDIR/rt1"
mkdir -p "$OUT/logs" "$LOCAL"
module purge
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0
export HF_HOME=/project/aip-azouaq/$USER/hf_cache
unset TRANSFORMERS_CACHE
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$PROJECT_ROOT"
echo "git $(git rev-parse --short HEAD) stage=$STAGE head=$HEAD"
virtualenv --no-download "$SLURM_TMPDIR/env" >/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip >/dev/null
pip install --no-index torch transformers numpy scipy scikit-learn 2>&1 | tail -1
TRAIN="$DATA/prm800k_probe_train_full.jsonl"
ENC=(--prm800k "train:$TRAIN:30000" "val:$DATA/prm800k_val_5k.jsonl" "test:$DATA/prm800k_test_2k.jsonl"
     --pb "gsm8k:$PB/processbench_gsm8k.jsonl" "math:$PB/processbench_math.jsonl"
          "olympiadbench:$PB/processbench_olympiadbench.jsonl" "omnimath:$PB/processbench_omnimath.jsonl")
LOGS="$OUT/logs"
export OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 MKL_NUM_THREADS=12

# --- selection: v0's four tokens on PRM800K val ------------------------------------
v0=(m1 m4 m16 m4text)
pids=()
for i in 0 1 2 3; do
  n=${v0[$i]}
  [[ -s "$V0/tokens/$n/read_token.pt" ]] || { echo "missing $V0/tokens/$n/read_token.pt"; exit 1; }
  CUDA_VISIBLE_DEVICES=$i python scripts/read_token_probe.py encode --model $Q3 \
    --ckpt "$V0/tokens/$n/read_token.pt" --out "$LOCAL/val_$n" \
    --prm800k "val:$DATA/prm800k_val_5k.jsonl" > "$LOGS/select_val_${n}-${SLURM_JOB_ID}.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
SEL="$OUT/selection_${STAGE}-${SLURM_JOB_ID}.json"
line=$(python scripts/analysis/read_token_select.py --v0 "$V0" --out "$SEL" \
  --val m1:"$LOCAL/val_m1" m4:"$LOCAL/val_m4" m16:"$LOCAL/val_m16" m4text:"$LOCAL/val_m4text" | tail -1)
echo "[select] $line"
eval "$line"
if [[ "$STAGE" != D && "$GATE_STAGE" == D ]]; then
  echo "[gate] v0 best ProcessBench calib-20 <= 0.568: the plan sends this to Stage D; stage $STAGE does not run"
  exit 0
fi

# --- one run: train, encode, geometry, keep the token ---------------------------------
# run <gpu> <name> <model> <m> <init> <extra train args...>
run() {
  local gpu=$1 name=$2 model=$3 m=$4 init=$5; shift 5
  (
    set -e
    export CUDA_VISIBLE_DEVICES=$gpu
    python scripts/read_token_probe.py train --model "$model" --m "$m" --init "$init" \
      --train "$TRAIN" --out "$LOCAL/$name" "$@"
    python scripts/read_token_probe.py encode --model "$model" --out "$LOCAL/$name" "${ENC[@]}"
    python scripts/analysis/backbone_geometry.py --configs "$name:$LOCAL/$name" --reps sep \
      --out "$OUT/geometry_$name"
    mkdir -p "$OUT/tokens/$name" && cp "$LOCAL/$name/read_token.pt" "$OUT/tokens/$name/"
  ) > "$LOGS/${name}-${SLURM_JOB_ID}.log" 2>&1
}

# B6: re-encode v0 tokens keeping all m states, then LDA (geometry) and trained readouts.
b6() {
  local gpu=$1
  (
    set -e
    export CUDA_VISIBLE_DEVICES=$gpu
    local names=("$CSTAR") cfgs=()
    [[ "$MALL" != "$CSTAR" ]] && names+=("$MALL")
    for n in "${names[@]}"; do
      python scripts/read_token_probe.py encode --model $Q3 --ckpt "$V0/tokens/$n/read_token.pt" \
        --keep_all --out "$LOCAL/b6_$n" "${ENC[@]}"
      cfgs+=("b6_$n:$LOCAL/b6_$n")
    done
    python scripts/analysis/backbone_geometry.py --configs "${cfgs[@]}" --reps sep sep_mean \
      --out "$OUT/geometry_b6"
    python scripts/analysis/read_token_readouts.py --configs "${cfgs[@]}" --out "$OUT/readouts_b6"
  ) > "$LOGS/b6-${SLURM_JOB_ID}.log" 2>&1
}

pids=()
case "$STAGE" in
  B)
    ( run 0 b1_frozen $Q3 "$CSTAR_M" "$CSTAR_INIT" --freeze_soft --n_train 100000 --seed 0; b6 0 ) & pids+=($!)
    run 1 b2_seed1 $Q3 "$CSTAR_M" "$CSTAR_INIT" --n_train 100000 --seed 1 & pids+=($!)
    run 2 b2_seed2 $Q3 "$CSTAR_M" "$CSTAR_INIT" --n_train 100000 --seed 2 & pids+=($!)
    run 3 b3_400k $Q3 "$CSTAR_M" "$CSTAR_INIT" --n_train 400000 --seed 0 & pids+=($!)
    ;;
  C)
    for s in 0 1 2; do
      run $s "c_parent_seed$s" $PARENT "$CSTAR_M" "$CSTAR_INIT" --n_train 100000 --seed $s & pids+=($!)
    done
    run 3 b7_mean $Q3 "$MALL_M" "$MALL_INIT" --read mean --n_train 100000 --seed 0 & pids+=($!)
    ;;
  Cparent)
    # Stage C rerun without B7: the cuDNN SDPA backward gave NaN gradients on the
    # parent (job 629664), so the soft pass runs with eager attention.
    for s in 0 1 2; do
      run $s "c_parent_seed$s" $PARENT "$CSTAR_M" "$CSTAR_INIT" --n_train 100000 --seed $s \
        --soft_attn eager & pids+=($!)
    done
    ;;
  D)
    for s in 0 1 2; do
      run $s "d_deep_${HEAD}_seed$s" $Q3 "$CSTAR_M" "$CSTAR_INIT" --deep --head "$HEAD" \
        --n_train 100000 --seed $s --soft_attn eager & pids+=($!)
    done
    ;;
  *) echo "unknown STAGE=$STAGE"; exit 1 ;;
esac
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
for f in "$LOGS"/*-"${SLURM_JOB_ID}".log; do echo "== $f"; tail -n 3 "$f"; done
exit $fail
