#!/bin/bash
#SBATCH --job-name=instep_v1_allmk
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=00:50:00
#SBATCH --output=%x-%j.out
# InStep variant "all markers": read token k sees the start markers of every step 0..k
# (read_token_inline.py --all_markers), where InStep shows each read only its own step's
# marker. Everything else is the InStep recipe on Qwen3-8B (ReProbe DeepSeek-labelled split,
# reprobe layout, deep offsets, chain, start marker with its own offsets, lr 3e-3, 5% warm-up,
# 1 epoch). GPU g trains seed g (g = 0..2), scores validation, then encodes the eight ReProbe
# test sets, ProcessBench and validation into $OUT/rt/allmk_ds_seed<g>. Then analyze.py
# against InStep ($OUT/rt/instep_ds_seed0-2, same recipe and seeds) with a paired bootstrap.
# SMOKE=N: first N items (pipeline check); submit it with --time=00:20:00.
# --time: one config trains in ~12 min (lr sweep smoke 662416: 0.37 s per step, ~1,970 steps),
# encode ~5 min (pick_g1 log of the rung sweep).
set -euo pipefail
SMOKE="${SMOKE:-0}"
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
CM="/project/aip-azouaq/$USER/cot_mech"
D="$CM/reprobe_cmp_v1/data"
SUF="$([[ $SMOKE == 0 ]] || echo "_smoke$SMOKE")"
OUT="${OUT:-$CM/instep_v1}"
TR="$OUT/allmarkers$SUF"
LOGS="$OUT/logs"
L="$SLURM_TMPDIR/allmk"
mkdir -p "$LOGS" "$L" "$TR"
module purge
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0
export HF_HOME=/project/aip-azouaq/$USER/hf_cache
unset TRANSFORMERS_CACHE
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 MKL_NUM_THREADS=12
cd "$PROJECT_ROOT"
echo "git $(git rev-parse --short HEAD 2>/dev/null || echo none) smoke=$SMOKE out=$TR"
virtualenv --no-download "$SLURM_TMPDIR/env" >/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip >/dev/null
pip install --no-index torch transformers numpy scipy scikit-learn 2>&1 | tail -1
SM=(); [[ $SMOKE == 0 ]] || SM=(--smoke "$SMOKE")

# run <gpu = seed>: train, validation, encode (scores only are kept)
run() {
  local s=$1 dst="$OUT/rt/allmk_ds_seed$1$SUF" tmp="$L/enc_$1" sets=()
  CUDA_VISIBLE_DEVICES=$s python scripts/read_token_inline.py train --model Qwen/Qwen3-8B --layout reprobe \
    --init_text " correct" --head linear --deep --chain --start_marker --start_deep --all_markers \
    --soft_attn no_cudnn --train "$D/rt/ds_train.items.jsonl" --n_train 0 --val "$D/rt/ds_val.items.jsonl" \
    --lr 3e-3 --warmup 0.05 --seed "$s" --out "$TR/seed$s" "${SM[@]}" || return 1
  for t in math gsm8k proofnet trips meetings calendar strategyqa scienceqa; do sets+=("$t:$D/test/$t.items.jsonl"); done
  for t in gsm8k math olympiadbench omnimath; do sets+=("pb_$t:$D/pb/$t.items.jsonl"); done
  sets+=("val:$D/rt/all_val.items.jsonl")
  CUDA_VISIBLE_DEVICES=$s python scripts/read_token_inline.py encode --model Qwen/Qwen3-8B --layout reprobe \
    --ckpt "$TR/seed$s/read_token.pt" --out "$tmp" --prm800k "${sets[@]}" --passive_blocks "" "${SM[@]}" || return 1
  mkdir -p "$dst"
  python - "$tmp" "$dst" <<'PY' || return 1
import sys, shutil, numpy as np
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
for z in src.glob("*.shard0.npz"):
    a = np.load(z)
    np.savez(dst / z.name, y=a["y"], kept=a["kept"], head_p_correct=a["head_p_correct"])
    shutil.copy(src / z.name.replace(".npz", ".meta.jsonl"), dst)
PY
  rm -rf "$tmp"
}

pids=()
for s in 0 1 2; do run $s > "$LOGS/allmk_seed$s$SUF-$SLURM_JOB_ID.log" 2>&1 & pids+=($!); done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
grep -h "^\[gate\]\|^\[val\]\|^\[train inline\] step .*/\|^\[encode\] val" "$LOGS"/allmk_*-"$SLURM_JOB_ID".log | tail -n 30 || true
[[ $fail == 0 && $SMOKE == 0 ]] || exit $fail
A=$(ls -d "$OUT"/rt/allmk_ds_seed? | paste -sd,); B=$(ls -d "$OUT"/rt/instep_ds_seed? | paste -sd,)
python scripts/reprobe_cmp/analyze.py --data "$D" --out "$TR/analysis" \
  --arm "AllMarkers=rt:$A" "InStep=rt:$B" --pairs AllMarkers-InStep
