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
#   Cparent  GPU0-2 parent seeds 0-2 only, soft pass without cuDNN SDPA (rerun of C's parent arm)
#   D  GPU0-2 deep read token seeds 0-2 (HEAD=linear|mlp), soft pass without cuDNN SDPA
#   DQ GPU0-3 Stage D2, deep read token start-point ablation, seed 0, m = start text length
#   BLA GPU0-2 BL1 embedding only (" correct", m = 1, no offsets) seeds 0-2; GPU3 BL1-SP
#      (ScratchProbe-style: 8 scratch tokens + the read token, m = 9 from vocabulary
#      rows, MLP head, no offsets) seed 0
#   BLB GPU0-1 BL1-SP seeds 1, 2; GPU2 BL2 (offsets only, embedding frozen at
#      " correct") seeds 0 then 2; GPU3 BL2 seed 1, then the ProcessBench scores of
#      the RTH tokens (dq_correct, rth_correct_seed1, rth_correct_seed2)
#   Every run also writes its ProcessBench head scores in the leaderboard format to
#   $OUT/pb_scores/<name>, in the trace order of the RTH cell, for the paired bootstrap.
#   BLC GPU0-2 BL3 (LoReFT rank 4 at the read position, " correct" embedding, linear
#      head, no offsets) seeds 0-2 at peak lr 1e-3; GPU3 BL3 seed 0 at 3e-3 (lr check)
#   BLD the same for BL4 (LoRA rank 8 on q_proj, v_proj, read position only)
#   M2  the RTH token dq_correct re-scored with its offsets switched off by layer window:
#      0-8, 9-17, 18-26, 27-35 (GPU0-3), then all of them (GPU0)
#   RTH GPU0-1 RTH seeds 1, 2: dq_correct (seed 0) again at 100k steps (submit with
#      --time=01:15:00)
#   DFLR GPU0 DF seed 2 again with peak lr 1e-3 (the 3e-3 run diverged at the end of
#      warm-up, job 635351); the other GPUs idle
#   DF GPU0-2 the D2 pick (" correct", m = 1, deep, linear head) on all 513,810 train
#      steps, seeds 0-2 (about 1.6 h of training each; submit with --time=02:45:00)
# Every stage but DQ and DF first re-encodes PRM800K val with v0's four tokens and picks C* and
# m_all on val AUROC (scripts/analysis/read_token_select.py). B and C stop if the
# v0 gate sends the experiment to Stage D. Only results, tokens and logs leave the node.
set -euo pipefail
STAGE="${STAGE:?set STAGE=B, C, Cparent, D, DQ, DF, DFLR, RTH, BLA, BLB, BLC, BLD or M2}"
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
PBENC=(--pb "gsm8k:$PB/processbench_gsm8k.jsonl" "math:$PB/processbench_math.jsonl"
            "olympiadbench:$PB/processbench_olympiadbench.jsonl" "omnimath:$PB/processbench_omnimath.jsonl")
LIKE="$OUT/stage_f/cells_rth/rth_dq_correct"
LOGS="$OUT/logs"
export OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 MKL_NUM_THREADS=12

# --- selection: v0's four tokens on PRM800K val ------------------------------------
if [[ "$STAGE" != DQ && "$STAGE" != DF && "$STAGE" != DFLR && "$STAGE" != RTH && "$STAGE" != BL* && "$STAGE" != M2 ]]; then
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
    python scripts/analysis/read_token_pb_scores.py --enc "$LOCAL/$name" --out "$OUT/pb_scores/$name" \
      --like "$LIKE"
  ) > "$LOGS/${name}-${SLURM_JOB_ID}.log" 2>&1
}

# ProcessBench scores of an existing token, in the leaderboard format
pb_scores() {
  local gpu=$1 name=$2
  (
    set -e
    export CUDA_VISIBLE_DEVICES=$gpu
    python scripts/read_token_probe.py encode --model $Q3 --ckpt "$OUT/tokens/$name/read_token.pt" \
      --out "$LOCAL/pbs_$name" "${PBENC[@]}"
    python scripts/analysis/read_token_pb_scores.py --enc "$LOCAL/pbs_$name" --out "$OUT/pb_scores/$name" \
      --like "$LIKE"
  ) > "$LOGS/pb_scores_${name}-${SLURM_JOB_ID}.log" 2>&1
}

# M2: an existing token re-encoded with some offsets set to zero, scored like a run
ablate() {
  local gpu=$1 src=$2 window=$3 name="m2_${2}_zero${3}"
  (
    set -e
    export CUDA_VISIBLE_DEVICES=$gpu
    python scripts/read_token_probe.py encode --model $Q3 --ckpt "$OUT/tokens/$src/read_token.pt" \
      --zero_deep "$window" --out "$LOCAL/$name" "${ENC[@]}"
    python scripts/analysis/backbone_geometry.py --configs "$name:$LOCAL/$name" --reps sep \
      --out "$OUT/geometry_$name"
    python scripts/analysis/read_token_pb_scores.py --enc "$LOCAL/$name" --out "$OUT/pb_scores/$name" \
      --like "$LIKE"
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
    # parent (job 629664), so the soft pass excludes that backend (eager drifts 7-8% in bf16).
    for s in 0 1 2; do
      run $s "c_parent_seed$s" $PARENT "$CSTAR_M" "$CSTAR_INIT" --n_train 100000 --seed $s \
        --soft_attn no_cudnn & pids+=($!)
    done
    ;;
  D)
    for s in 0 1 2; do
      run $s "d_deep_${HEAD}_seed$s" $Q3 "$CSTAR_M" "$CSTAR_INIT" --deep --head "$HEAD" \
        --n_train 100000 --seed $s --soft_attn no_cudnn & pids+=($!)
    done
    ;;
  DQ)
    # Stage D2: same as Stage D seed 0, only the start text (and so m) changes.
    # Qwen3 tokens: " Is step correct?" 4, " correct?" 2, "?" 1, " correct" 1.
    dq=(--deep --head "$HEAD" --n_train 100000 --seed 0 --soft_attn no_cudnn)
    run 0 dq_is_step_correct $Q3 4 text --init_text " Is step correct?" "${dq[@]}" & pids+=($!)
    run 1 dq_correct_q $Q3 2 text --init_text " correct?" "${dq[@]}" & pids+=($!)
    run 2 dq_q $Q3 1 text --init_text "?" "${dq[@]}" & pids+=($!)
    run 3 dq_correct $Q3 1 text --init_text " correct" "${dq[@]}" & pids+=($!)
    ;;
  DF)
    # The D2 pick at full scale: every train step once (n_train >= the split keeps all
    # 513,810, in a seed-shuffled batch order), everything else as dq_correct.
    for s in 0 1 2; do
      run $s df_correct_full_seed$s $Q3 1 text --init_text " correct" --deep --head "$HEAD" \
        --n_train 513810 --seed $s --soft_attn no_cudnn & pids+=($!)
    done
    ;;
  BLA)
    for s in 0 1 2; do
      run $s bl1_correct_seed$s $Q3 1 text --init_text " correct" --head linear \
        --n_train 100000 --seed $s --soft_attn no_cudnn & pids+=($!)
    done
    run 3 bl1sp_seed0 $Q3 9 vocab --head mlp --n_train 100000 --seed 0 --soft_attn no_cudnn & pids+=($!)
    ;;
  BLB)
    bl2=(1 text --init_text " correct" --deep --freeze_soft --head linear --n_train 100000 --soft_attn no_cudnn)
    for s in 1 2; do
      run $((s - 1)) bl1sp_seed$s $Q3 9 vocab --head mlp --n_train 100000 --seed $s --soft_attn no_cudnn &
      pids+=($!)
    done
    ( run 2 bl2_offsets_seed0 $Q3 "${bl2[@]}" --seed 0 && run 2 bl2_offsets_seed2 $Q3 "${bl2[@]}" --seed 2 ) &
    pids+=($!)
    ( run 3 bl2_offsets_seed1 $Q3 "${bl2[@]}" --seed 1 &&
      for n in dq_correct rth_correct_seed1 rth_correct_seed2; do pb_scores 3 $n; done ) & pids+=($!)
    ;;
  BLC|BLD)
    if [[ "$STAGE" == BLC ]]; then iv=(--interv loreft --rank 4); tag=bl3_loreft; else
      iv=(--interv lora --rank 8); tag=bl4_lora; fi
    for s in 0 1 2; do
      run $s ${tag}_seed$s $Q3 1 text --init_text " correct" --head linear "${iv[@]}" \
        --n_train 100000 --seed $s --lr 1e-3 --soft_attn no_cudnn & pids+=($!)
    done
    run 3 ${tag}_seed0_lr3e-3 $Q3 1 text --init_text " correct" --head linear "${iv[@]}" \
      --n_train 100000 --seed 0 --lr 3e-3 --soft_attn no_cudnn & pids+=($!)
    ;;
  M2)
    ( ablate 0 dq_correct 0-8 && ablate 0 dq_correct 0-35 ) & pids+=($!)
    ablate 1 dq_correct 9-17 & pids+=($!)
    ablate 2 dq_correct 18-26 & pids+=($!)
    ablate 3 dq_correct 27-35 & pids+=($!)
    ;;
  RTH)
    # RTH = the read token classified by its own head; dq_correct is seed 0.
    for s in 1 2; do
      run $((s - 1)) rth_correct_seed$s $Q3 1 text --init_text " correct" --deep --head "$HEAD" \
        --n_train 100000 --seed $s --soft_attn no_cudnn & pids+=($!)
    done
    ;;
  DFLR)
    run 0 df_correct_full_seed2_lr1e-3 $Q3 1 text --init_text " correct" --deep --head "$HEAD" \
      --n_train 513810 --seed 2 --lr 1e-3 --soft_attn no_cudnn & pids+=($!)
    ;;
  *) echo "unknown STAGE=$STAGE"; exit 1 ;;
esac
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
for f in "$LOGS"/*-"${SLURM_JOB_ID}".log; do echo "== $f"; tail -n 3 "$f"; done
exit $fail
