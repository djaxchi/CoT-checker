#!/bin/bash
#SBATCH --job-name=instep_v1
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=00:45:00
#SBATCH --output=%x-%j.out
# InStep paper V1 runs (docs/instep_paper_plan.md, "V1 run list"). Every stage writes under
# $OUT; step scores use the formats scripts/reprobe_cmp/analyze.py reads (rt: encode npz +
# meta; rp: <set>.jsonl {"uid", "p"}). Large states go to $ST (scratch).
#   STAGE=e1  the paper's Qwen3-8B InStep tokens (lr sweep, 3e-3 / 5% warm-up: DS seeds
#             0, 1, 2 and PRM800K seeds 0, 1, 2) scored on the 8 ReProbe test sets,
#             ProcessBench, ReProbe validation and the best-of-N pools. The DS seed-0 encode
#             also keeps text (passive) and read states at every 4th block on test, val,
#             ProcessBench and a 40k-step draw of the DS training split (B1, S1).
#   STAGE=l1  the ablation ladder on ReProbe's DS data, 3 seeds each, same recipe:
#             emb (read token embedding only, no offsets, no chain, no marker),
#             off (+ per-layer offsets), chain (+ chained reads), nooff (InStep without
#             offsets: chain + marker, no offsets anywhere); plus emb on PRM800K (3 seeds)
#             for the ProcessBench ladder. Each run is scored like e1 (no best-of-N).
# SMOKE=N runs every step on the first N items (pipeline check).
set -euo pipefail
STAGE="${STAGE:?e1 or l1}"
SMOKE="${SMOKE:-0}"
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
V1="/project/aip-azouaq/$USER/cot_mech/reprobe_cmp_v1"
D="$V1/data"
SW="/project/aip-azouaq/$USER/cot_mech/instep_lr_sweep"
PRMD="$SCRATCH/cot_mech/dense_full_7b_v1/data"
SUF="$([[ $SMOKE == 0 ]] || echo "_smoke$SMOKE")"
OUT="${OUT:-/project/aip-azouaq/$USER/cot_mech/instep_v1}$SUF"
ST="${ST:-$SCRATCH/cot_mech/instep_v1}$SUF/states"
LOGS="$OUT/logs"
L="$SLURM_TMPDIR/iv1"
Q3=Qwen/Qwen3-8B
BLOCKS="3,7,11,15,19,23,27,31,35"
mkdir -p "$LOGS" "$L" "$ST"
module purge
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0
export HF_HOME=/project/aip-azouaq/$USER/hf_cache
unset TRANSFORMERS_CACHE
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 MKL_NUM_THREADS=12
cd "$PROJECT_ROOT"
echo "git $(git rev-parse --short HEAD 2>/dev/null || echo none) stage=$STAGE smoke=$SMOKE out=$OUT"
virtualenv --no-download "$SLURM_TMPDIR/env" >/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip >/dev/null
pip install --no-index torch transformers numpy scipy scikit-learn 2>&1 | tail -1

SETS=()
for s in math gsm8k proofnet trips meetings calendar strategyqa scienceqa; do SETS+=("$s:$D/test/$s.items.jsonl"); done
for s in gsm8k math olympiadbench omnimath; do SETS+=("pb_$s:$D/pb/$s.items.jsonl"); done
SETS+=("val:$D/rt/all_val.items.jsonl")
BON=()
for s in math gsm8k strategyqa scienceqa calendar; do BON+=("bon_$s:$V1/bon/inputs/$s.items.jsonl"); done
SM=(); [[ $SMOKE == 0 ]] || SM=(--smoke "$SMOKE")

# encode <gpu> <ckpt> <name> <keep_states 0|1> [extra sets...]: scores to $OUT/rt/<name>
encode() {
  local g=$1 ck=$2 name=$3 keep=$4; shift 4
  local extra=(--passive_blocks "")
  [[ $keep == 1 ]] && extra=(--passive_blocks "$BLOCKS" --read_blocks "$BLOCKS")
  CUDA_VISIBLE_DEVICES=$g python scripts/read_token_inline.py encode --model $Q3 --layout reprobe \
    --ckpt "$ck" --out "$L/$name" --prm800k "${SETS[@]}" "$@" "${extra[@]}" "${SM[@]}" || return 1
  mkdir -p "$OUT/rt/$name"
  python - "$L/$name" "$OUT/rt/$name" <<'PY'
import sys, shutil, numpy as np
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
for z in src.glob("*.shard0.npz"):
    a = np.load(z)
    np.savez(dst / z.name, y=a["y"], kept=a["kept"], head_p_correct=a["head_p_correct"])
    shutil.copy(src / z.name.replace(".npz", ".meta.jsonl"), dst)
PY
  [[ $? == 0 ]] || return 1
  if [[ $keep == 1 ]]; then mkdir -p "$ST/qwen3_8b" && cp "$L/$name"/*.shard0.* "$ST/qwen3_8b/"; fi
  rm -rf "$L/$name"
}

# train <gpu> <src ds|prm> <variant> <seed>, then encode
train() {
  local g=$1 src=$2 var=$3 s=$4 name="${3}_${2}_seed$4" args=() flags=()
  case $var in
    emb)   flags=() ;;
    off)   flags=(--deep) ;;
    chain) flags=(--deep --chain) ;;
    nooff) flags=(--chain --start_marker) ;;
    *) echo "unknown variant $var"; return 2 ;;
  esac
  if [[ $src == ds ]]; then args=(--layout reprobe --train "$D/rt/ds_train.items.jsonl" --n_train 0)
  else args=(--layout chat --train "$PRMD/prm800k_probe_train_full.jsonl" --n_train 100000); fi
  (
    CUDA_VISIBLE_DEVICES=$g python scripts/read_token_inline.py train --model $Q3 --init_text " correct" \
      --head linear "${flags[@]}" "${args[@]}" --seed "$s" --lr 3e-3 --warmup 0.05 --soft_attn no_cudnn \
      --out "$L/tok_$name" "${SM[@]}" || exit 1
    mkdir -p "$OUT/tokens/$name" && cp "$L/tok_$name/read_token.pt" "$OUT/tokens/$name/" || exit 1
    encode "$g" "$L/tok_$name/read_token.pt" "$name" 0 || exit 1
  ) > "$LOGS/$name-$SLURM_JOB_ID.log" 2>&1
}

tok() { echo "$SW/$1/lr0.003_wu0.05/read_token.pt"; }
pids=()
if [[ $STAGE == e1 ]]; then
  ( encode 0 "$(tok ds_seed0)" instep_ds_seed0 1 "${BON[@]}" "train40k:$D/rt/ds_train.items.jsonl:40000"
  ) > "$LOGS/e1_g0-$SLURM_JOB_ID.log" 2>&1 & pids+=($!)
  ( f=0; encode 1 "$(tok ds_seed1)" instep_ds_seed1 0 "${BON[@]}" || f=1
    encode 1 "$(tok prm_seed2)" instep_prm_seed2 0 "${BON[@]}" || f=1; exit $f
  ) > "$LOGS/e1_g1-$SLURM_JOB_ID.log" 2>&1 & pids+=($!)
  ( f=0; encode 2 "$(tok ds_seed2_b)" instep_ds_seed2 0 "${BON[@]}" || f=1
    encode 2 "$(tok prm_seed0)" instep_prm_seed0 0 "${BON[@]}" || f=1; exit $f
  ) > "$LOGS/e1_g2-$SLURM_JOB_ID.log" 2>&1 & pids+=($!)
  encode 3 "$(tok prm_seed1)" instep_prm_seed1 0 "${BON[@]}" > "$LOGS/e1_g3-$SLURM_JOB_ID.log" 2>&1 & pids+=($!)
elif [[ $STAGE == l1 ]]; then
  ( f=0; for s in 0 1 2; do train 0 ds emb $s || f=1; done; train 0 prm emb 0 || f=1; exit $f ) & pids+=($!)
  ( f=0; for s in 0 1 2; do train 1 ds off $s || f=1; done; train 1 prm emb 1 || f=1; exit $f ) & pids+=($!)
  ( f=0; for s in 0 1 2; do train 2 ds chain $s || f=1; done; train 2 prm emb 2 || f=1; exit $f ) & pids+=($!)
  ( f=0; for s in 0 1 2; do train 3 ds nooff $s || f=1; done; exit $f ) & pids+=($!)
else
  echo "unknown STAGE=$STAGE"; exit 2
fi
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
grep -h "^\[encode\]\|^\[train inline\] step .*/.* loss" "$LOGS"/*-"$SLURM_JOB_ID".log | tail -n 60 || true
exit $fail
