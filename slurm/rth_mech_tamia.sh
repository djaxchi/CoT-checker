#!/bin/bash
#SBATCH --job-name=rth_mech
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --output=%x-%j.out
# read_token_v1 mechanistic study M2 (offsets), M3, M4, M5 (docs/read_token_v1_plan.md,
# Stage G), on the bundle built with the RTH token dq_correct (seed 0). M2's re-encodes
# are stage M2 of slurm/read_token_v1_tamia.sh.
#   phase 1, in parallel
#     GPU0  M3 attention from the read position, RTH vs BL1 (bl1_correct_seed0, skipped
#           if missing) vs the untrained " correct" token, PRM800K test_2k
#     GPU1  M4 encode: the policy's confidence on train sample + test
#     GPU2  M4 encode: val + ProcessBench
#     GPU3  M5 the PRM's own head regressed from every layer (PB)
#     CPU   M2 offset norms and their vocabulary reading
#   phase 2  M4 analyze (projected heads), then the MATH-500 mixed-pair questions
#            (instruct_t10 pool) scored with dq_correct, w_proj and w_decor (GPU0-2,
#            prefix sharing), then the pair accuracies
# PARTS (default "m3 m4 m5 m2 analyze pairs") picks what runs, e.g. PARTS="m5 pairs" to
# redo M5 and the pair test only. POOL_CONF: the pool's *_conf.jsonl directory (the
# copies under $POOL are symlinks to the laptop).
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
OUT="${OUT:-$SCRATCH/cot_mech/read_token_v1}"
MS="${MS:-$OUT/mech_study}"
MECH="${MECH:-/project/aip-azouaq/$USER/cot_mech/read_token_v1/mech}"
DATA="${DATA:-$SCRATCH/cot_mech/dense_full_7b_v1/data}"
PB="${PB:-/scratch/d/dchikhi/cot-checker/processbench_full}"
PRM_HEAD="${PRM_HEAD:-/scratch/d/dchikhi/cot_mech/prm_backbone_v1/prm_head_scores}"
POOL="${POOL:-$OUT/downstream/pools/instruct_t10}"
TOKEN="${TOKEN:-dq_correct}"
BL1="${BL1:-bl1_correct_seed0}"
Q3=Qwen/Qwen3-8B
LOGS="$MS/logs"
J="${SLURM_JOB_ID}"
mkdir -p "$LOGS" "$MS/m4_conf" "$MS/m4_tokens" "$MS/m4_scores"
cd "$PROJECT_ROOT"
echo "git $(git rev-parse --short HEAD 2>/dev/null || echo snapshot) token=$TOKEN"
CKPT="$OUT/tokens/$TOKEN/read_token.pt"
[[ -s "$CKPT" ]] || { echo "missing $CKPT"; exit 1; }

module purge
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0
export HF_HOME=/project/aip-azouaq/$USER/hf_cache
unset TRANSFORMERS_CACHE
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
virtualenv --no-download "$SLURM_TMPDIR/env" >/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip >/dev/null
pip install --no-index torch transformers numpy scipy scikit-learn pyyaml safetensors 2>&1 | tail -1
TRAIN="$DATA/prm800k_probe_train_full.jsonl"
PARTS="${PARTS:-m3 m4 m5 m2 analyze pairs}"
POOL_CONF="${POOL_CONF:-$MS/pool_conf}"
has() { [[ " $PARTS " == *" $1 "* ]]; }

# --- phase 1 ---------------------------------------------------------------------------
pids=()
has m3 && { CUDA_VISIBLE_DEVICES=0 python scripts/analysis/rth_mech_attention.py --model $Q3 --bundle "$MECH/bundle" \
  --split test_2k --source "prm800k:$DATA/prm800k_test_2k.jsonl" \
  --tokens "rth=$CKPT" "bl1=$OUT/tokens/$BL1/read_token.pt" --plain --out "$MS" \
  > "$LOGS/m3-$J.log" 2>&1 & pids+=($!); }
has m4 && { CUDA_VISIBLE_DEVICES=1 python scripts/analysis/rth_mech_confidence.py encode --model $Q3 --bundle "$MECH/bundle" \
  --splits "probe_train_full=prm800k:$TRAIN" "test_2k=prm800k:$DATA/prm800k_test_2k.jsonl" \
  --out "$MS/m4_conf" > "$LOGS/m4_encode_a-$J.log" 2>&1 & pids+=($!)
CUDA_VISIBLE_DEVICES=2 python scripts/analysis/rth_mech_confidence.py encode --model $Q3 --bundle "$MECH/bundle" \
  --splits "val_5k=prm800k:$DATA/prm800k_val_5k.jsonl" \
  $(for s in gsm8k math olympiadbench omnimath; do echo "$s=pb:$s:$PB/processbench_$s.jsonl"; done) \
  --out "$MS/m4_conf" > "$LOGS/m4_encode_b-$J.log" 2>&1 & pids+=($!); }
has m5 && { CUDA_VISIBLE_DEVICES=3 python scripts/analysis/rth_mech_prm_head.py --bundle "$MECH/bundle" \
  --prm_scores "$PRM_HEAD" --out "$MS/m5_prm_head.json" > "$LOGS/m5-$J.log" 2>&1 & pids+=($!); }
has m2 && { CUDA_VISIBLE_DEVICES="" python scripts/analysis/rth_mech_offsets.py --model $Q3 --ckpt "$CKPT" \
  --bundle "$MECH/bundle" --split test_2k --out "$MS/m2_offsets.json" > "$LOGS/m2_offsets-$J.log" 2>&1 & pids+=($!); }
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
for f in "$LOGS"/m*-"$J".log; do [[ -e "$f" ]] && { echo "== $f"; tail -n 3 "$f"; }; done

# --- phase 2: M4 analyze, then the confidence-wrong pairs ------------------------------
if ! has analyze && ! has pairs; then
  :
elif [[ -s "$MS/m4_conf/conf_omnimath.npz" && -s "$MS/m4_conf/conf_probe_train_full.npz" ]]; then
  has analyze && { CUDA_VISIBLE_DEVICES=1 python scripts/analysis/rth_mech_confidence.py analyze --bundle "$MECH/bundle" \
    --conf "$MS/m4_conf" --ckpt "$CKPT" --ckpt_out "$MS/m4_tokens" --out "$MS/m4_confidence.json" \
    > "$LOGS/m4_analyze-$J.log" 2>&1 || fail=1
  tail -n 4 "$LOGS/m4_analyze-$J.log"; }
  TRAJ="$POOL/tts_math500.shard*_trajectories.jsonl"
  CONF="$POOL_CONF/tts_math500.shard*_conf.jsonl"
  has pairs && python scripts/analysis/rth_mech_confidence.py pairs --trajectories "$TRAJ" --conf "$CONF" \
    --cells "prm=$POOL/scores/tts_math500__prm_qwen25_math_7b.shard*.jsonl" \
    --questions_out "$MS/m4_pair_questions.json" --n_boot 100 > "$LOGS/m4_questions-$J.log" 2>&1 || fail=1
  if ! has pairs; then
    :
  elif [[ -s "$MS/m4_tokens/${TOKEN}_w_proj/read_token.pt" && -s "$MS/m4_pair_questions.json" ]]; then
    spids=(); g=0
    for c in "$TOKEN:$CKPT" "${TOKEN}_w_proj:$MS/m4_tokens/${TOKEN}_w_proj/read_token.pt" \
             "${TOKEN}_w_decor:$MS/m4_tokens/${TOKEN}_w_decor/read_token.pt"; do
      name=${c%%:*}; ck=${c#*:}
      CUDA_VISIBLE_DEVICES=$g python scripts/onpolicy/score_traces_with_read_token.py --model $Q3 --ckpt "$ck" \
        --cell_name "m4_$name" --trajectories $TRAJ --fork_ids "$MS/m4_pair_questions.json" --branched \
        --verify_against "$POOL"/scores/tts_math500__prm_qwen25_math_7b.shard*.jsonl \
        --out "$MS/m4_scores/tts_math500__m4_$name.shard0.jsonl" > "$LOGS/m4_score_$name-$J.log" 2>&1 &
      spids+=($!); g=$((g + 1))
    done
    for p in "${spids[@]}"; do wait "$p" || fail=1; done
    python scripts/analysis/rth_mech_confidence.py pairs --trajectories "$TRAJ" --conf "$CONF" \
      --cells "$TOKEN=$MS/m4_scores/tts_math500__m4_$TOKEN.shard0.jsonl" \
              "w_proj=$MS/m4_scores/tts_math500__m4_${TOKEN}_w_proj.shard0.jsonl" \
              "w_decor=$MS/m4_scores/tts_math500__m4_${TOKEN}_w_decor.shard0.jsonl" \
              "prm=$POOL/scores/tts_math500__prm_qwen25_math_7b.shard*.jsonl" \
      --out "$MS/m4_pairs.json" > "$LOGS/m4_pairs-$J.log" 2>&1 || fail=1
    tail -n 4 "$LOGS/m4_pairs-$J.log"
  else
    echo "M4: no projected heads or no pair questions; pair test skipped"; fail=1
  fi
else
  echo "M4: confidence encode incomplete; analyze skipped"; fail=1
fi
echo "[$(date)] done, fail=$fail"
exit $fail
