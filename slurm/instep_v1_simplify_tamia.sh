#!/bin/bash
#SBATCH --job-name=instep_v1_simp
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --output=%x-%j.out
# InStep V1 follow-up (2026-11-03): (1) does the simplified read token (embedding + per-layer
# offsets + linear head; no chained reads, no start marker = ladder rung `off`) hold up beyond
# Qwen3-8B DS labels? (2) is the Qwen3-32B "instability" a short-training effect? Qwen3-32B
# has 44,110 training steps-to-read (510 optimizer steps per epoch) against 167,355 (1,966)
# on Qwen3-8B, so the stability check (from 25% of training) starts at step 128, still inside
# the initial descent; here Qwen3-32B trains for 2 and 4 epochs (4 epochs = 2,040 steps,
# the Qwen3-8B budget). Recipe everywhere: lr 3e-3, 5% warm-up (OneCycle over all epochs).
# Every run writes sweep.json (validation AP + loss trace) and is scored on the test sets.
#   STAGE=q8    GPU0-2 off on PRM800K seeds 0-2 (ProcessBench zero-shot); GPU3 off on self labels seeds 0-2
#   STAGE=phi4  GPU0-2 off on Phi-4 (Qwen3-8B labels) seeds 0-2; GPU3 Qwen3-32B InStep 2 epochs seed 0
#   STAGE=q32a  GPU0-2 Qwen3-32B InStep 4 epochs seeds 0-2; GPU3 Qwen3-32B InStep 2 epochs seed 1
#   STAGE=q32b  GPU0-2 Qwen3-32B off 4 epochs seeds 0-2;    GPU3 Qwen3-32B InStep 2 epochs seed 2
# Outputs: $OUT/simp/<name>/ (token, sweep.json), scores $OUT/rt_simp/<backbone>/<name>/.
# SMOKE=N: first N items.
set -euo pipefail
STAGE="${STAGE:?q8, phi4, q32a, q32b, redo32 or encode}"
SMOKE="${SMOKE:-0}"
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
CM="/project/aip-azouaq/$USER/cot_mech"
PRMD="$SCRATCH/cot_mech/dense_full_7b_v1/data"
SUF="$([[ $SMOKE == 0 ]] || echo "_smoke$SMOKE")"
OUT="${OUT:-$CM/instep_v1}$SUF"
LOGS="$OUT/logs"
L="$SLURM_TMPDIR/simp"
mkdir -p "$LOGS" "$L" "$OUT/simp"
module purge
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0
unset TRANSFORMERS_CACHE
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 MKL_NUM_THREADS=12
cd "$PROJECT_ROOT"
echo "git $(git rev-parse --short HEAD 2>/dev/null || echo none) stage=$STAGE smoke=$SMOKE out=$OUT"
virtualenv --no-download "$SLURM_TMPDIR/env" >/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip >/dev/null
pip install --no-index torch transformers numpy scipy scikit-learn 2>&1 | tail -1
SM=(); [[ $SMOKE == 0 ]] || SM=(--smoke "$SMOKE")

# run <gpu> <bb 8b|phi4|32b> <variant instep|off> <labels> <seed> <epochs>
run() {
  local g=$1 bb=$2 var=$3 lab=$4 s=$5 ep=$6
  local name="${var}_${lab}_ep${ep}_seed$s" model hf dd layout extra=() flags=() tr va sets=()
  case $bb in
    8b)   model=Qwen/Qwen3-8B; hf=/project/aip-azouaq/$USER/hf_cache; dd=$CM/reprobe_cmp_v1/data; layout=reprobe ;;
    phi4) model=microsoft/phi-4; hf=$SCRATCH/hf_cache; dd=$CM/reprobe_cmp_v2/phi4/data; layout=reprobe_phi4 ;;
    32b)  model=Qwen/Qwen3-32B; hf=$SCRATCH/hf_cache; dd=$CM/reprobe_cmp_v2/qwen3_32b/data; layout=reprobe
          extra=(--micro_tokens 4000) ;;
  esac
  case $var in
    off)    flags=(--deep) ;;
    instep) flags=(--deep --chain --start_marker --start_deep) ;;
  esac
  if [[ $lab == prm ]]; then
    tr=(--layout chat --train "$PRMD/prm800k_probe_train_full.jsonl" --n_train 100000 --val "$PRMD/prm800k_val_5k.jsonl")
  else
    tr=(--layout $layout --train "$dd/rt/${lab}_train.items.jsonl" --n_train 0 --val "$dd/rt/${lab}_val.items.jsonl")
  fi
  for x in math gsm8k proofnet trips meetings calendar strategyqa scienceqa; do sets+=("$x:$dd/test/$x.items.jsonl"); done
  [[ $bb == 8b ]] && for x in gsm8k math olympiadbench omnimath; do sets+=("pb_$x:$dd/pb/$x.items.jsonl"); done
  sets+=("val:$dd/rt/all_val.items.jsonl")
  local tok="$OUT/simp/${bb}_$name" dst="$OUT/rt_simp/$bb/$name" tmp="$L/enc_${bb}_$name"
  (
    HF_HOME=$hf CUDA_VISIBLE_DEVICES=$g python scripts/read_token_inline.py train --model $model --init_text " correct" \
      --head linear "${flags[@]}" "${tr[@]}" "${extra[@]}" --sweep "3e-3:0.05" --epochs "$ep" --seed "$s" \
      --soft_attn no_cudnn --out "$tok" "${SM[@]}" || exit 1
    enc "$g" "$bb" "$name" || exit 1
  ) > "$LOGS/simp_${bb}_$name-$SLURM_JOB_ID.log" 2>&1
}

# enc <gpu> <bb> <name>: score a trained token (a single-config run saves <tok>/read_token.pt)
enc() {
  local g=$1 bb=$2 name=$3 model hf dd layout sets=()
  case $bb in
    8b)   model=Qwen/Qwen3-8B; hf=/project/aip-azouaq/$USER/hf_cache; dd=$CM/reprobe_cmp_v1/data; layout=reprobe ;;
    phi4) model=microsoft/phi-4; hf=$SCRATCH/hf_cache; dd=$CM/reprobe_cmp_v2/phi4/data; layout=reprobe_phi4 ;;
    32b)  model=Qwen/Qwen3-32B; hf=$SCRATCH/hf_cache; dd=$CM/reprobe_cmp_v2/qwen3_32b/data; layout=reprobe ;;
  esac
  for x in math gsm8k proofnet trips meetings calendar strategyqa scienceqa; do sets+=("$x:$dd/test/$x.items.jsonl"); done
  [[ $bb == 8b ]] && for x in gsm8k math olympiadbench omnimath; do sets+=("pb_$x:$dd/pb/$x.items.jsonl"); done
  sets+=("val:$dd/rt/all_val.items.jsonl")
  local tok="$OUT/simp/${bb}_$name" dst="$OUT/rt_simp/$bb/$name" tmp="$L/enc_${bb}_$name" ck
  ck="$tok/read_token.pt"; [[ -f "$ck" ]] || ck="$tok/lr0.003_wu0.05/read_token.pt"
  (
    HF_HOME=$hf CUDA_VISIBLE_DEVICES=$g python scripts/read_token_inline.py encode --model $model --layout $layout \
      --ckpt "$ck" --out "$tmp" --prm800k "${sets[@]}" --passive_blocks "" "${SM[@]}" || exit 1
    mkdir -p "$dst"
    python - "$tmp" "$dst" <<'PY' || exit 1
import sys, shutil, numpy as np
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
for z in src.glob("*.shard0.npz"):
    a = np.load(z)
    np.savez(dst / z.name, y=a["y"], kept=a["kept"], head_p_correct=a["head_p_correct"])
    shutil.copy(src / z.name.replace(".npz", ".meta.jsonl"), dst)
PY
    rm -rf "$tmp"
  )
}

pids=()
case $STAGE in
  q8)
    for s in 0 1 2; do run $s 8b off prm $s 1 & pids+=($!); done
    ( f=0; for s in 0 1 2; do run 3 8b off self $s 1 || f=1; done; exit $f ) & pids+=($!) ;;
  phi4)
    for s in 0 1 2; do run $s phi4 off qwen3 $s 1 & pids+=($!); done
    run 3 32b instep oss 0 2 & pids+=($!) ;;
  q32a)
    for s in 0 1 2; do run $s 32b instep oss $s 4 & pids+=($!); done
    run 3 32b instep oss 1 2 & pids+=($!) ;;
  q32b)
    for s in 0 1 2; do run $s 32b off oss $s 4 & pids+=($!); done
    run 3 32b instep oss 2 2 & pids+=($!) ;;
  redo32)  # 32B off seed 2 hit a CUDA OOM in the startup gate (job 664300)
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    run 0 32b off oss 2 4 & pids+=($!) ;;
  encode)  # score every trained token that has no scores yet (jobs 664297-664300 saved
           # single-config tokens at <tok>/read_token.pt, not <tok>/<tag>/)
    todo=()
    for t in "$OUT"/simp/*; do
      [[ -f $t/read_token.pt || -f $t/lr0.003_wu0.05/read_token.pt ]] || continue
      n=$(basename "$t"); bb=${n%%_*}; name=${n#*_}
      [[ -f "$OUT/rt_simp/$bb/$name/val.shard0.npz" ]] || todo+=("$bb:$name")
    done
    echo "to encode: ${todo[*]}"
    for g in 0 1 2 3; do
      ( f=0; i=0; for x in "${todo[@]}"; do
          if (( i % 4 == g )); then enc $g "${x%%:*}" "${x#*:}" > "$LOGS/enc_${x/:/_}-$SLURM_JOB_ID.log" 2>&1 || f=1; fi
          i=$((i + 1)); done; exit $f ) & pids+=($!)
    done ;;
  *) echo "unknown STAGE=$STAGE"; exit 2 ;;
esac
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
grep -h "^\[val\]\|^\[encode\] val\|step .*/.* loss" "$LOGS"/simp_*-"$SLURM_JOB_ID".log | tail -n 40 || true
exit $fail
