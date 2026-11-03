#!/bin/bash
#SBATCH --job-name=instep_v1_tts
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out
# InStep V1, ReProbe Tables 2 (beam search) and 3 (best-of-N) with InStep inside, Qwen3-8B,
# on ReProbe's released pools. The v2 runs covered best-of-N on the five programmatically
# graded sets and beam search on four sets with three arms; this script adds the rest.
#   STAGE=bonenc  best-of-N inputs for ProofNet, Trips, Meetings (build_bon.py), scored by
#                 InStep-DS (seeds 0-2), InStep-self (seeds 0-2) and the released ReProbe
#                 hidden-state probes (DeepSeek, self); scores join the existing arm dirs
#   STAGE=beam    beam search B=5 x N=5 (scripts/reprobe_cmp/beam_search.py) on BEAM_SETS with
#                 BEAM_ARMS (names below), the first BEAM_LIMIT questions of each set split
#                 over the 4 GPUs; rows join $V1/beam (resumable: a (question, arm) already in
#                 any <set>*.jsonl there is skipped)
#   STAGE=judge   LLM judge (scripts/instep_v1/tts_judge.py, Qwen3-32B) on JUDGE_WHAT
#                 (bon,val or beam) into $OUT/tts/judge_<JUDGE_TAG>, then merge
# SMOKE=N: N questions per set (beam), N items per shard (judge), first N items (bonenc).
set -euo pipefail
STAGE="${STAGE:?bonenc, beam or judge}"
SMOKE="${SMOKE:-0}"
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
CM="/project/aip-azouaq/$USER/cot_mech"
V1="$CM/reprobe_cmp_v1"
SW="$CM/instep_lr_sweep"
POOLS="$SCRATCH/cot_mech/reprobe_bon/pools"
SUF="$([[ $SMOKE == 0 ]] || echo "_smoke$SMOKE")"
OUT="${OUT:-$CM/instep_v1}$SUF"
LOGS="$OUT/logs"
L="$SLURM_TMPDIR/tts"
mkdir -p "$LOGS" "$L" "$OUT/tts"
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
pip install --no-index torch transformers numpy scipy scikit-learn pyyaml pandas sympy 2>&1 | tail -1
Q3=Qwen/Qwen3-8B
declare -A ARM=(
  [instep_ds]="rt:$SW/ds_seed0/lr0.003_wu0.05/read_token.pt"
  [rt_self]="rt:$V1/runs/tokens/rt_self_seed0/read_token.pt"
  [rp_ds]="rp:$V1/released/ds"
  [rp_self]="rp:$V1/released/self"
  [prm7b]="prm:Qwen/Qwen2.5-Math-PRM-7B"
  [prm800k]="prm:Qwen/Qwen2.5-Math-7B-PRM800K"
  [random]="random"
)
pids=()

# rt_encode <gpu> <ckpt> <dest arm dir> <sets...>: head scores of a read token, added to an arm dir
rt_encode() {
  local g=$1 ck=$2 dst=$3; shift 3
  local tmp="$L/enc_$(basename "$dst")"
  local sm=(); [[ $SMOKE == 0 ]] || sm=(--smoke "$SMOKE")
  CUDA_VISIBLE_DEVICES=$g python scripts/read_token_inline.py encode --model $Q3 --layout reprobe \
    --ckpt "$ck" --out "$tmp" --prm800k "$@" --passive_blocks "" "${sm[@]}" || return 1
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

if [[ $STAGE == bonenc ]]; then
  NEW="${BON_NEW:-proofnet trips meetings}"
  BI="$V1/bon/inputs"; [[ $SMOKE == 0 ]] || BI="$OUT/bon_inputs"
  python scripts/reprobe_cmp/build_bon.py --pools "$POOLS" --out "$BI" --sets "$(echo $NEW | tr ' ' ',')"
  RT=(); RP=()
  for s in $NEW; do RT+=("bon_$s:$BI/$s.items.jsonl"); RP+=("bon_$s:$BI/$s.traces.jsonl"); done
  RTD="$OUT/rt"; RVD="$V1/runs/rt"; RLD="$V1/runs/released/scores"
  if [[ $SMOKE != 0 ]]; then RVD="$OUT/rt_v1"; RLD="$OUT/released_scores"; fi
  ( set -e
    for lab in ds self; do
      CUDA_VISIBLE_DEVICES=0 python scripts/reprobe_cmp/reprobe_probe.py released --model $Q3 \
        --ckpt "$V1/released/$lab" --name "rel_$lab" --sets "${RP[@]}" --out "$L/rel" \
        $([[ $SMOKE == 0 ]] || echo "--smoke $SMOKE")
      mkdir -p "$RLD/rel_$lab" && cp "$L/rel/scores/rel_$lab"/bon_*.jsonl "$RLD/rel_$lab/"
    done ) > "$LOGS/tts_bonenc_rp-$SLURM_JOB_ID.log" 2>&1 & pids+=($!)
  CK=("$SW/ds_seed0" "$SW/ds_seed1" "$SW/ds_seed2_b")
  for s in 0 1 2; do
    ( set -e
      rt_encode $((s + 1)) "${CK[$s]}/lr0.003_wu0.05/read_token.pt" "$RTD/instep_ds_seed$s" "${RT[@]}"
      rt_encode $((s + 1)) "$V1/runs/tokens/rt_self_seed$s/read_token.pt" "$RVD/rt_self_seed$s" "${RT[@]}"
    ) > "$LOGS/tts_bonenc_rt$s-$SLURM_JOB_ID.log" 2>&1 & pids+=($!)
  done
elif [[ $STAGE == beam ]]; then
  LIM="${BEAM_LIMIT:-300}"; [[ $SMOKE == 0 ]] || LIM=$((SMOKE * 4))
  BD="$V1/beam"; [[ $SMOKE == 0 ]] || BD="$OUT/beam"
  SPEC=(); for n in ${BEAM_ARMS:?arm names}; do SPEC+=("$n=${ARM[$n]}"); done
  for g in 0 1 2 3; do
    ( f=0
      for s in ${BEAM_SETS:?sets}; do
        CUDA_VISIBLE_DEVICES=$g python scripts/reprobe_cmp/beam_search.py --pools "$POOLS" --set "$s" \
          --offset $((g * LIM / 4)) --limit $(( (g + 1) * LIM / 4 - g * LIM / 4 )) --model $Q3 \
          --out "$BD/$s.${BEAM_TAG:-v1}.part$g.jsonl" --arm "${SPEC[@]}" || f=1
      done; exit $f ) > "$LOGS/tts_beam_${BEAM_TAG:-v1}_g$g-$SLURM_JOB_ID.log" 2>&1 & pids+=($!)
  done
elif [[ $STAGE == judge ]]; then
  JD="$OUT/tts/judge_${JUDGE_TAG:?bon or beam}"
  JM="$(ls -d $SCRATCH/hf_cache/hub/models--Qwen--Qwen3-32B/snapshots/*/ | head -1)"
  BB=(); [[ "${JUDGE_WHAT:-bon,val}" != *beam* ]] || BB=(--beam "$V1/beam")
  python scripts/instep_v1/tts_judge.py build --pools "$POOLS" --out "$JD" --what "${JUDGE_WHAT:-bon,val}" \
    --sets "${JUDGE_SETS:-proofnet,trips,meetings}" "${BB[@]}"
  SM=(); [[ $SMOKE == 0 ]] || SM=(--smoke "$SMOKE")
  for g in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$g python scripts/instep_v1/tts_judge.py judge --out "$JD" --model "$JM" --backend hf \
      --shard $g --num_shards 4 "${SM[@]}" > "$LOGS/tts_judge_${JUDGE_TAG}_g$g-$SLURM_JOB_ID.log" 2>&1 & pids+=($!)
  done
else
  echo "unknown STAGE=$STAGE"; exit 2
fi
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
if [[ $STAGE == judge ]]; then
  MB=(); [[ "${JUDGE_WHAT:-bon,val}" != *bon* || $SMOKE != 0 ]] || MB=(--bon "$V1/bon")
  python scripts/instep_v1/tts_judge.py merge --out "$JD" "${MB[@]}" || fail=1
fi
grep -h "^\[bon\]\|^\[beam\].*correct\|^\[judge\]\|^\[encode\] \|Error\|error" "$LOGS"/tts_*-"$SLURM_JOB_ID".log | tail -n 60 || true
exit $fail
