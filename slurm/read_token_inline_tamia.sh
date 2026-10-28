#!/bin/bash
#SBATCH --job-name=rt_inline
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=01:45:00
#SBATCH --output=%x-%j.out
# read_token_v1 Stage H (Plan B, docs/read_token_v1_plan.md): the RTH read token inside the
# trace, one pass per trajectory in Qwen3-8B's chat layout (scripts/read_token_inline.py).
#   GPU0-2  H2: RTH-I seeds 0-2, the RTH recipe (" correct", m = 1, deep offsets, linear head,
#           100k PRM800K items of the same seed, lr 3e-3) trained inline; encode; geometry;
#           LDA on the read token and the passive text states; ProcessBench scores
#   GPU3    H1: the RTH tokens (dq_correct = seed 0, rth_correct_seed1, rth_correct_seed2),
#           unchanged, read inline (the layout gap before any retraining)
# then the paired trace bootstrap of RTH-I against RTH, the PRM-state cells and H1.
# MODE=marker (Stage H-S, after H2 missed its bar): a start marker per step, seen only by
# that step's read token, standing in for the template's "Current step:" header
#   GPU0-2  RTH-IS seeds 0-2: the marker is one trained embedding (" step" at the start)
#   GPU3    RTH-ISD seeds 0, 1, 2 in turn: the marker also has its own per-layer offsets
# then the bootstrap of each against RTH-I and RTH. Submit with --time=02:00:00.
# MODE=chain (Stage H-C): a read token after every step, each seeing the earlier ones
#   GPU0-2  RTH-IC seeds 0-2: chained read tokens, no marker
#   GPU3    RTH-ICS seeds 0, 1, 2 in turn: chained, plus the start marker with its own offsets
# then the bootstrap against RTH-I, RTH-ISD and RTH. Submit with --time=02:00:00.
set -euo pipefail
MODE="${MODE:-plain}"
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
OUT="${OUT:-$SCRATCH/cot_mech/read_token_v1}"
DATA="${DATA:-$SCRATCH/cot_mech/dense_full_7b_v1/data}"
PB="${PB:-/scratch/d/dchikhi/cot-checker/processbench_full}"
PRM_CELLS=/scratch/d/dchikhi/cot_mech/prm_backbone_v1/cells
Q3=Qwen/Qwen3-8B
LOCAL="$SLURM_TMPDIR/rti"
H="$OUT/inline"
LOGS="$OUT/logs"
mkdir -p "$LOGS" "$LOCAL" "$H"
module purge
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0
export HF_HOME=/project/aip-azouaq/$USER/hf_cache
unset TRANSFORMERS_CACHE
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 MKL_NUM_THREADS=12
cd "$PROJECT_ROOT"
echo "git $(git rev-parse --short HEAD 2>/dev/null || echo none) snapshot ${PROJECT_ROOT}"
virtualenv --no-download "$SLURM_TMPDIR/env" >/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip >/dev/null
pip install --no-index torch transformers numpy scipy scikit-learn 2>&1 | tail -1
TRAIN="$DATA/prm800k_probe_train_full.jsonl"
ENC=(--prm800k "train:$TRAIN:30000" "val:$DATA/prm800k_val_5k.jsonl" "test:$DATA/prm800k_test_2k.jsonl"
     --pb "gsm8k:$PB/processbench_gsm8k.jsonl" "math:$PB/processbench_math.jsonl"
          "olympiadbench:$PB/processbench_olympiadbench.jsonl" "omnimath:$PB/processbench_omnimath.jsonl")
LIKE="$OUT/stage_f/cells_rth/rth_dq_correct"

# score <name>: everything after an encode in $LOCAL/<name>
score() {
  local name=$1
  python scripts/analysis/backbone_geometry.py --configs "$name:$LOCAL/$name" --reps sep \
    --out "$H/geometry_$name"
  python scripts/analysis/inline_passive_lda.py --enc "$LOCAL/$name" --out "$H/lda_$name.json"
  python scripts/analysis/read_token_pb_scores.py --enc "$LOCAL/$name" --out "$OUT/pb_scores/$name" \
    --like "$LIKE"
}

# train <gpu> <seed> <name> <extra train args...>
train() {
  local g=$1 s=$2 name=$3; shift 3
  (
    set -e
    export CUDA_VISIBLE_DEVICES=$g
    python scripts/read_token_inline.py train --model $Q3 --init_text " correct" --deep --head linear \
      --train "$TRAIN" --n_train 100000 --seed "$s" --lr 3e-3 --soft_attn no_cudnn --out "$LOCAL/$name" "$@"
    mkdir -p "$OUT/tokens/$name" && cp "$LOCAL/$name/read_token.pt" "$OUT/tokens/$name/"
    python scripts/read_token_inline.py encode --model $Q3 --out "$LOCAL/$name" "${ENC[@]}"
    score "$name"
  ) > "$LOGS/${name}-${SLURM_JOB_ID}.log" 2>&1
}

transfer() {
  local src=$1 name=h1_$1
  (
    set -e
    export CUDA_VISIBLE_DEVICES=3
    python scripts/read_token_inline.py encode --model $Q3 --ckpt "$OUT/tokens/$src/read_token.pt" \
      --out "$LOCAL/$name" "${ENC[@]}"
    score "$name"
  ) > "$LOGS/${name}-${SLURM_JOB_ID}.log" 2>&1
}

pids=()
if [[ "$MODE" == chain ]]; then
  for s in 0 1 2; do train $s $s rthic_seed$s --chain & pids+=($!); done
  ( for s in 0 1 2; do train 3 $s rthics_seed$s --chain --start_marker --start_deep || true; done ) & pids+=($!)
elif [[ "$MODE" == marker ]]; then
  for s in 0 1 2; do train $s $s rthis_seed$s --start_marker & pids+=($!); done
  ( for s in 0 1 2; do train 3 $s rthisd_seed$s --start_marker --start_deep || true; done ) & pids+=($!)
else
  for s in 0 1 2; do train $s $s rthi_seed$s & pids+=($!); done
  ( for t in dq_correct rth_correct_seed1 rth_correct_seed2; do transfer $t || true; done ) & pids+=($!)
fi
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
tail -n 3 "$LOGS"/rthi*_seed*-"$SLURM_JOB_ID".log "$LOGS"/h1_*-"$SLURM_JOB_ID".log 2>/dev/null || true

# --- paired trace bootstrap (calib-20 F1_PB), only over score dirs that exist --------------
S="$OUT/pb_scores"
dirs() { local out=() d; for d in "$@"; do [[ -s "$d/pb_step_scores_math.jsonl" ]] && out+=("$d"); done
         (IFS=,; echo "${out[*]}"); }
RTHI=$(dirs "$S/rthi_seed0" "$S/rthi_seed1" "$S/rthi_seed2")
RTH=$(dirs "$S/dq_correct" "$S/rth_correct_seed1" "$S/rth_correct_seed2")
H1=$(dirs "$S/h1_dq_correct" "$S/h1_rth_correct_seed1" "$S/h1_rth_correct_seed2")
PRMC=$(dirs "$PRM_CELLS"/boundary_stats__mlp_h1024x2__seed4{2,3,4})
echo "bootstrap sides: RTH-I [$RTHI] RTH [$RTH] H1 [$H1] PRM [$PRMC]"
pairs=()
if [[ "$MODE" == marker ]]; then
  RTHIS=$(dirs "$S/rthis_seed0" "$S/rthis_seed1" "$S/rthis_seed2")
  RTHISD=$(dirs "$S/rthisd_seed0" "$S/rthisd_seed1" "$S/rthisd_seed2")
  echo "marker sides: RTH-IS [$RTHIS] RTH-ISD [$RTHISD]"
  [[ -n "$RTHIS" && -n "$RTHI" ]] && pairs+=("RTH-IS - RTH-I=$RTHIS:$RTHI")
  [[ -n "$RTHIS" && -n "$RTH" ]] && pairs+=("RTH-IS - RTH=$RTHIS:$RTH")
  [[ -n "$RTHISD" && -n "$RTHI" ]] && pairs+=("RTH-ISD - RTH-I=$RTHISD:$RTHI")
  [[ -n "$RTHISD" && -n "$RTH" ]] && pairs+=("RTH-ISD - RTH=$RTHISD:$RTH")
  [[ -n "$RTHISD" && -n "$RTHIS" ]] && pairs+=("RTH-ISD - RTH-IS=$RTHISD:$RTHIS")
  H1=""; RTHI_PAIRS=0
elif [[ "$MODE" == chain ]]; then
  RTHIC=$(dirs "$S/rthic_seed0" "$S/rthic_seed1" "$S/rthic_seed2")
  RTHICS=$(dirs "$S/rthics_seed0" "$S/rthics_seed1" "$S/rthics_seed2")
  RTHISD=$(dirs "$S/rthisd_seed0" "$S/rthisd_seed1" "$S/rthisd_seed2")
  echo "chain sides: RTH-IC [$RTHIC] RTH-ICS [$RTHICS] RTH-ISD [$RTHISD]"
  [[ -n "$RTHIC" && -n "$RTHI" ]] && pairs+=("RTH-IC - RTH-I=$RTHIC:$RTHI")
  [[ -n "$RTHIC" && -n "$RTHISD" ]] && pairs+=("RTH-IC - RTH-ISD=$RTHIC:$RTHISD")
  [[ -n "$RTHIC" && -n "$RTH" ]] && pairs+=("RTH-IC - RTH=$RTHIC:$RTH")
  [[ -n "$RTHICS" && -n "$RTHIC" ]] && pairs+=("RTH-ICS - RTH-IC=$RTHICS:$RTHIC")
  [[ -n "$RTHICS" && -n "$RTH" ]] && pairs+=("RTH-ICS - RTH=$RTHICS:$RTH")
  H1=""; RTHI_PAIRS=0
else
  RTHI_PAIRS=1
fi
(( RTHI_PAIRS )) && [[ -n "$RTHI" && -n "$RTH" ]] && pairs+=("RTH-I - RTH=$RTHI:$RTH")
(( RTHI_PAIRS )) && [[ -n "$RTHI" && -n "$PRMC" ]] && pairs+=("RTH-I - PRM boundary_stats mlp h1024x2=$RTHI:$PRMC")
[[ -n "$H1" && -n "$RTH" ]] && pairs+=("H1 (RTH read inline) - RTH=$H1:$RTH")
(( RTHI_PAIRS )) && [[ -n "$RTHI" && -n "$H1" ]] && pairs+=("RTH-I - H1=$RTHI:$H1")
if (( ${#pairs[@]} )); then
  python scripts/analysis/stage_f_bootstrap.py --pairs "${pairs[@]}" --out "$H/bootstrap_${MODE}.json" \
    > "$LOGS/inline_bootstrap-$SLURM_JOB_ID.log" 2>&1 || fail=1
  tail -n 12 "$LOGS/inline_bootstrap-$SLURM_JOB_ID.log"
fi
exit $fail
