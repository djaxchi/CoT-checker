#!/bin/bash
#SBATCH --job-name=instep_v1_rung
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=01:45:00
#SBATCH --output=%x-%j.out
# InStep V1 follow-up: per-rung learning rate. Under the frozen stability rule of
# docs/instep_lr_sweep_plan.md (running loss above the constant predictor's from 25% of
# training on), InStep at lr 3e-3 / 5% warm-up is stable on Qwen3-8B (6/6 seeds) and Phi-4,
# but the ablation rungs at the same recipe are not (job 663145: emb_ds 0/3 stable, off 2/3,
# chain 2/3, nooff 0/3, emb_prm 0/3) and neither are the Qwen3-32B InStep seeds (0/3). So
# every such rung, and InStep on Qwen3-32B, gets its own peak lr from the same sweep and the
# same rule (5 lrs x 5% warm-up, 3 seeds, validation only); the picked config's 3 seeds are
# then scored on the test sets (STAGE=pick). The fixed-recipe results stay reported.
#   STAGE=ds    GPU g = rung g (emb, off, chain, nooff) on ReProbe DS data, seeds 0-2 in turn
#   STAGE=prm   GPU0-2 = emb on PRM800K, seeds 0-2
#   STAGE=q32   GPU0-2 = InStep on Qwen3-32B (GPT-OSS labels), seeds 0-2
#   STAGE=pick  selection per source (scripts/analysis/instep_lr_select.py), then encode the
#               picked config's seeds: $OUT/rt/<rung>_<src>_lrpick_seed<s>,
#               $OUT/rt32/instep_lrpick_seed<s>; picks in $OUT/rungsweep/picks.txt
# SMOKE=N: first N items (pipeline check).
set -euo pipefail
STAGE="${STAGE:?ds, prm, q32 or pick}"
SMOKE="${SMOKE:-0}"
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
CM="/project/aip-azouaq/$USER/cot_mech"
V1="$CM/reprobe_cmp_v1"; D="$V1/data"
D32="$CM/reprobe_cmp_v2/qwen3_32b/data"
PRMD="$SCRATCH/cot_mech/dense_full_7b_v1/data"
SUF="$([[ $SMOKE == 0 ]] || echo "_smoke$SMOKE")"
OUT="${OUT:-$CM/instep_v1}$SUF"
SW="$OUT/rungsweep"
LOGS="$OUT/logs"
L="$SLURM_TMPDIR/rung"
GRID="${GRID:-5e-4:0.05,1e-3:0.05,1.5e-3:0.05,2e-3:0.05,3e-3:0.05}"
mkdir -p "$LOGS" "$L" "$SW"
module purge
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0
unset TRANSFORMERS_CACHE
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 MKL_NUM_THREADS=12
cd "$PROJECT_ROOT"
echo "git $(git rev-parse --short HEAD 2>/dev/null || echo none) stage=$STAGE smoke=$SMOKE out=$OUT grid=$GRID"
virtualenv --no-download "$SLURM_TMPDIR/env" >/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip >/dev/null
pip install --no-index torch transformers numpy scipy scikit-learn 2>&1 | tail -1
SM=(); [[ $SMOKE == 0 ]] || SM=(--smoke "$SMOKE")
HF8=/project/aip-azouaq/$USER/hf_cache; HF32=$SCRATCH/hf_cache

flags() {
  case $1 in
    emb)    echo "" ;;
    off)    echo "--deep" ;;
    chain)  echo "--deep --chain" ;;
    nooff)  echo "--chain --start_marker" ;;
    instep) echo "--deep --chain --start_marker --start_deep" ;;
  esac
}

# sweep <gpu> <rung> <src ds|prm|oss> <seed>
sweep() {
  local g=$1 r=$2 src=$3 s=$4 name="${2}_${3}" args=() hf=$HF8 model=Qwen/Qwen3-8B
  case $src in
    ds)  args=(--layout reprobe --train "$D/rt/ds_train.items.jsonl" --n_train 0 --val "$D/rt/ds_val.items.jsonl") ;;
    prm) args=(--layout chat --train "$PRMD/prm800k_probe_train_full.jsonl" --n_train 100000
               --val "$PRMD/prm800k_val_5k.jsonl") ;;
    oss) args=(--layout reprobe --train "$D32/rt/oss_train.items.jsonl" --n_train 0 --val "$D32/rt/oss_val.items.jsonl"
               --micro_tokens 4000); hf=$HF32; model=Qwen/Qwen3-32B ;;
  esac
  HF_HOME=$hf CUDA_VISIBLE_DEVICES=$g python scripts/read_token_inline.py train --model $model --init_text " correct" \
    --head linear $(flags "$r") "${args[@]}" --sweep "$GRID" --seed "$s" --soft_attn no_cudnn \
    --out "$SW/${name}_seed$s" "${SM[@]}" > "$LOGS/rung_${name}_seed$s-$SLURM_JOB_ID.log" 2>&1
}

# encode <gpu> <ckpt> <dst> <backbone 8b|32b>
encode() {
  local g=$1 ck=$2 dst=$3 bb=$4 sets=() hf=$HF8 model=Qwen/Qwen3-8B dd=$D
  if [[ $bb == 32b ]]; then hf=$HF32; model=Qwen/Qwen3-32B; dd=$D32; fi
  for s in math gsm8k proofnet trips meetings calendar strategyqa scienceqa; do sets+=("$s:$dd/test/$s.items.jsonl"); done
  if [[ $bb == 8b ]]; then for s in gsm8k math olympiadbench omnimath; do sets+=("pb_$s:$dd/pb/$s.items.jsonl"); done; fi
  sets+=("val:$dd/rt/all_val.items.jsonl")
  local tmp="$L/enc_$(basename "$dst")"
  HF_HOME=$hf CUDA_VISIBLE_DEVICES=$g python scripts/read_token_inline.py encode --model $model --layout reprobe \
    --ckpt "$ck" --out "$tmp" --prm800k "${sets[@]}" --passive_blocks "" "${SM[@]}" || return 1
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
if [[ $STAGE == ds ]]; then
  g=0
  for r in emb off chain nooff; do
    ( f=0; for s in 0 1 2; do sweep $g $r ds $s || f=1; done; exit $f ) & pids+=($!)
    g=$((g + 1))
  done
elif [[ $STAGE == prm ]]; then
  for s in 0 1 2; do sweep $s emb prm $s & pids+=($!); done
elif [[ $STAGE == q32 ]]; then
  for s in 0 1 2; do sweep $s instep oss $s & pids+=($!); done
elif [[ $STAGE == pick ]]; then
  : > "$SW/picks.txt"
  # pick_one <name> <metric>: the rule's pick; with no stable config, the config with the
  # fewest unstable seeds, then the highest mean (flagged in picks.txt)
  pick_one() {
    python scripts/analysis/instep_lr_select.py --source "$1:$2:$SW/$1_seed*/sweep.json" --out "$SW/selection_$1" > /dev/null || return 1
    python - "$SW/selection_$1/selection.json" "$1" >> "$SW/picks.txt" <<'PY'
import json, sys
j = json.load(open(sys.argv[1])); n = sys.argv[2]; t = j["tables"][n]; p = j["picks"][n]
if p is None:
    p = min(t, key=lambda k: (t[k]["n_unstable"], -(t[k]["mean"] or 0)))
    print(n, p, "NO_STABLE_CONFIG", f"unstable {t[p]['n_unstable']}/3 mean {t[p]['mean']:.4f}")
else:
    print(n, p, "rule", f"mean {t[p]['mean']:.4f}")
PY
  }
  for n in emb_ds off_ds chain_ds nooff_ds; do pick_one $n ap || echo "$n none FAILED" >> "$SW/picks.txt"; done
  pick_one emb_prm auroc || echo "emb_prm none FAILED" >> "$SW/picks.txt"
  pick_one instep_oss ap || echo "instep_oss none FAILED" >> "$SW/picks.txt"
  cat "$SW/picks.txt"
  tag() { awk -v n="$1" '$1 == n {print $2}' "$SW/picks.txt"; }
  # GPU0: emb_ds + off_ds; GPU1: chain_ds + nooff_ds; GPU2: emb_prm; GPU3: Qwen3-32B
  ( f=0; for n in emb_ds off_ds; do t=$(tag $n); [[ $t == none || -z $t ]] && continue
      for s in 0 1 2; do encode 0 "$SW/${n}_seed$s/$t/read_token.pt" "$OUT/rt/${n}_lrpick_seed$s" 8b || f=1; done; done; exit $f
  ) > "$LOGS/pick_g0-$SLURM_JOB_ID.log" 2>&1 & pids+=($!)
  ( f=0; for n in chain_ds nooff_ds; do t=$(tag $n); [[ $t == none || -z $t ]] && continue
      for s in 0 1 2; do encode 1 "$SW/${n}_seed$s/$t/read_token.pt" "$OUT/rt/${n}_lrpick_seed$s" 8b || f=1; done; done; exit $f
  ) > "$LOGS/pick_g1-$SLURM_JOB_ID.log" 2>&1 & pids+=($!)
  ( f=0; t=$(tag emb_prm); if [[ $t != none && -n $t ]]; then
      for s in 0 1 2; do encode 2 "$SW/emb_prm_seed$s/$t/read_token.pt" "$OUT/rt/emb_prm_lrpick_seed$s" 8b || f=1; done; fi; exit $f
  ) > "$LOGS/pick_g2-$SLURM_JOB_ID.log" 2>&1 & pids+=($!)
  ( f=0; t=$(tag instep_oss); if [[ $t != none && -n $t ]]; then
      for s in 0 1 2; do encode 3 "$SW/instep_oss_seed$s/$t/read_token.pt" "$OUT/rt32/instep_lrpick_seed$s" 32b || f=1; done; fi; exit $f
  ) > "$LOGS/pick_g3-$SLURM_JOB_ID.log" 2>&1 & pids+=($!)
else
  echo "unknown STAGE=$STAGE"; exit 2
fi
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
grep -h "^\[val\]\|^\[encode\] val" "$LOGS"/*-"$SLURM_JOB_ID".log | tail -n 40 || true
exit $fail
