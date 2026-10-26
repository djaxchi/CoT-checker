#!/bin/bash
#SBATCH --job-name=rth_downstream
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
# read_token_v1 Stage E with RTH (docs/read_token_v1_plan.md, "Stage E: RTH downstream"):
#   pick   the RTH seed with the best PRM800K val AUROC (dq_correct = seed 0,
#          rth_correct_seed1, rth_correct_seed2); ProcessBench is not read
#   D1/D2  score every step of the saved Instruct TTS pools of §21.13 (T=1.0 and T=0.7,
#          GSM8K test and MATH-500, ten samples) with RTH, segmentation checked
#          against the PRM's score files; selection rules and the confidence-wrong
#          pairs are computed afterwards from these score files
#   E1     wall time per trajectory and per step: RTH, a passive Qwen3-8B pass, the PRM
# The pools are copied to $DS/pools beforehand (trajectories, confidence, PRM scores).
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
OUT="${OUT:-$SCRATCH/cot_mech/read_token_v1}"
DS="${DS:-$OUT/downstream}"
DATA="${DATA:-$SCRATCH/cot_mech/dense_full_7b_v1/data}"
CANDS=(dq_correct rth_correct_seed1 rth_correct_seed2)
POOLS=(instruct_t10 instruct_t07)
STEMS=(tts_gsm8k tts_math500)
Q3=Qwen/Qwen3-8B
LOGS="$DS/logs"
mkdir -p "$LOGS" "$DS/scores"
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

# --- pick the RTH seed on PRM800K val AUROC ----------------------------------------
LOCAL="$SLURM_TMPDIR/rth"
mkdir -p "$LOCAL"
vals=(); pids=(); i=0
for n in "${CANDS[@]}"; do
  [[ -s "$OUT/tokens/$n/read_token.pt" ]] || { echo "missing $OUT/tokens/$n/read_token.pt"; exit 1; }
  CUDA_VISIBLE_DEVICES=$i python scripts/read_token_probe.py encode --model $Q3 \
    --ckpt "$OUT/tokens/$n/read_token.pt" --out "$LOCAL/val_$n" \
    --prm800k "val:$DATA/prm800k_val_5k.jsonl" > "$LOGS/pick_val_$n-$SLURM_JOB_ID.log" 2>&1 &
  pids+=($!); vals+=("$n:1:$LOCAL/val_$n"); i=$((i + 1))
done
for p in "${pids[@]}"; do wait "$p"; done
line=$(python scripts/analysis/read_token_pick.py --val "${vals[@]}" --out "$DS/pick-$SLURM_JOB_ID.json" | tail -1)
echo "[pick] $line"
eval "$line"
CKPT="$OUT/tokens/$FTOKEN/read_token.pt"
CELL="rth_$FTOKEN"

# --- D1/D2: score the pools, 4 GPUs, every pool x dataset sharded 4 ways --------------
pids=()
for g in 0 1 2 3; do
  (
    set -e
    export CUDA_VISIBLE_DEVICES=$g
    for pool in "${POOLS[@]}"; do
      for stem in "${STEMS[@]}"; do
        python scripts/onpolicy/score_traces_with_read_token.py --model $Q3 --ckpt "$CKPT" \
          --cell_name "$CELL" --trajectories "$DS/pools/$pool/${stem}".shard*_trajectories.jsonl \
          --verify_against "$DS/pools/$pool/scores/${stem}__prm_qwen25_math_7b".shard*.jsonl \
          --out "$DS/scores/$pool/${stem}__${CELL}.shard$g.jsonl" --shard_idx $g --num_shards 4
      done
    done
  ) > "$LOGS/score_gpu$g-$SLURM_JOB_ID.log" 2>&1 &
  pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
tail -n 2 "$LOGS"/score_gpu*-"$SLURM_JOB_ID".log
(( fail == 0 )) || exit 1

# --- E1: cost on one GPU, the node otherwise idle -------------------------------------
CUDA_VISIBLE_DEVICES=0 python scripts/analysis/rth_cost.py --model $Q3 --ckpt "$CKPT" \
  --trajectories "$DS/pools/instruct_t10/tts_math500.shard00_trajectories.jsonl" --n 200 \
  --out "$DS/e1_cost-$SLURM_JOB_ID.json" > "$LOGS/e1-$SLURM_JOB_ID.log" 2>&1 || fail=1
tail -n 3 "$LOGS/e1-$SLURM_JOB_ID.log"
echo "[$(date)] done, fail=$fail"
exit $fail
