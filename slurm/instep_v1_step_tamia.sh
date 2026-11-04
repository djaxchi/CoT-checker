#!/bin/bash
#SBATCH --job-name=instep_v1_step
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
# InStep paper V1: the STEP baseline (Liang et al. 2026; scripts/instep_v1/step_probe.py), a
# 2-layer MLP on the last-layer state at each step's last token, trained on the FULL training
# split of every backbone (the earlier linear baseline used 40k steps on Phi-4 and Qwen3-32B).
#   1. last-layer text states (passive_final) of the full training split of Phi-4 (184,734
#      steps, Qwen3-8B labels; GPUs 0-1) and Qwen3-32B (44,110 steps, GPT-OSS labels; GPUs 2-3),
#      2 trace-disjoint parts each, into $ST/<backbone>_fulltrain/. Text states do not depend on
#      the read token passed to encode (text never attends to it). Qwen3-8B reuses
#      $ST/qwen3_8b_fulltrain (STAGE=fullpassive of instep_v1_states_tamia.sh, 167,355 steps).
#   2. step_probe.py per backbone (one GPU each, seeds 0-2): scores to $OUT/scores/<bb>/step/
#      seed<s>/, summary to $OUT/scores/<bb>/step/summary.json.
# SMOKE=N: first N training steps per part, outputs under *_smokeN roots.
set -euo pipefail
SMOKE="${SMOKE:-0}"
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
P="/project/aip-azouaq/$USER/cot_mech"
SUF="$([[ $SMOKE == 0 ]] || echo "_smoke$SMOKE")"
OUT="${OUT:-$P/instep_v1}$SUF"
ST0="$SCRATCH/cot_mech/instep_v1/states"   # existing val/test states (all backbones)
FT="$SCRATCH/cot_mech/instep_v1$SUF/states" # full-train states written here
LOGS="$OUT/logs"
mkdir -p "$LOGS" "$FT"
module purge
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0
export HF_HOME=$SCRATCH/hf_cache
unset TRANSFORMERS_CACHE
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 MKL_NUM_THREADS=12
cd "$PROJECT_ROOT"
echo "git $(git rev-parse --short HEAD 2>/dev/null || echo none) smoke=$SMOKE out=$OUT ft=$FT"
virtualenv --no-download "$SLURM_TMPDIR/env" >/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip >/dev/null
pip install --no-index torch transformers numpy scipy scikit-learn 2>&1 | tail -1

declare -A R MODEL LAYOUT LAB TOK
R[qwen3_8b]=$P/reprobe_cmp_v1;           LAB[qwen3_8b]=ds
R[phi4]=$P/reprobe_cmp_v2/phi4;          LAB[phi4]=qwen3
R[qwen3_32b]=$P/reprobe_cmp_v2/qwen3_32b; LAB[qwen3_32b]=oss
MODEL[phi4]=microsoft/phi-4;   LAYOUT[phi4]=reprobe_phi4; TOK[phi4]=${R[phi4]}/runs/tokens/rt_qwen3_seed0/read_token.pt
MODEL[qwen3_32b]=Qwen/Qwen3-32B; LAYOUT[qwen3_32b]=reprobe; TOK[qwen3_32b]=${R[qwen3_32b]}/runs/tokens/rt_oss_seed0/read_token.pt

# split <items> <out dir>: 2 trace-disjoint parts (first SMOKE items of each in a smoke)
split() {
  python - "$1" "$2" "$SMOKE" <<'PY'
import json, sys, zlib
src, out, smoke = sys.argv[1], sys.argv[2], int(sys.argv[3])
fs, n = [open(f"{out}/train_p{g}.jsonl", "w") for g in range(2)], [0, 0]
for l in open(src):
    g = zlib.crc32(json.loads(l)["uid"].rsplit("::", 1)[0].encode()) % 2
    if smoke and n[g] >= smoke:
        continue
    fs[g].write(l)
    n[g] += 1
print("parts", n)
PY
}

pids=()
g=0
for bb in phi4 qwen3_32b; do
  [[ -f ${TOK[$bb]} ]] || { echo "missing token ${TOK[$bb]}"; exit 2; }
  mkdir -p "$SLURM_TMPDIR/$bb" "$FT/${bb}_fulltrain"
  split "${R[$bb]}/data/rt/${LAB[$bb]}_train.items.jsonl" "$SLURM_TMPDIR/$bb"
  for part in 0 1; do
    ( CUDA_VISIBLE_DEVICES=$g python scripts/read_token_inline.py encode --model "${MODEL[$bb]}" \
        --layout "${LAYOUT[$bb]}" --ckpt "${TOK[$bb]}" --out "$FT/${bb}_fulltrain" \
        --prm800k "train_p$part:$SLURM_TMPDIR/$bb/train_p$part.jsonl" --passive_blocks "" ) \
      > "$LOGS/stepenc_${bb}_p$part-$SLURM_JOB_ID.log" 2>&1 & pids+=($!)
    g=$((g + 1))
  done
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done; pids=()
[[ $fail == 0 ]] || { grep -h "Error\|error" "$LOGS"/stepenc_*-"$SLURM_JOB_ID".log | tail; exit 1; }

# train <gpu> <backbone> <train states dir> <train names>
train() {
  CUDA_VISIBLE_DEVICES=$1 python scripts/instep_v1/step_probe.py --states "$ST0/$2" --train_states "$3" \
    --train_names "$4" --data "${R[$2]}/data" --val_label "${LAB[$2]}" --out "$OUT/scores/$2/step" \
    > "$LOGS/step_$2-$SLURM_JOB_ID.log" 2>&1
}
T8=train_p0,train_p1,train_p2,train_p3
[[ $SMOKE == 0 ]] || T8=train_p0
train 0 qwen3_8b "$ST0/qwen3_8b_fulltrain" "$T8" & pids+=($!)
train 1 phi4 "$FT/phi4_fulltrain" train_p0,train_p1 & pids+=($!)
train 2 qwen3_32b "$FT/qwen3_32b_fulltrain" train_p0,train_p1 & pids+=($!)
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
grep -h "^\[step\].*\(train\|seed .*best\|done\)\|Error\|error" "$LOGS"/step_*-"$SLURM_JOB_ID".log | tail -n 40 || true
exit $fail
