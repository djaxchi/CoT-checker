#!/bin/bash
#SBATCH --job-name=reprobe_cmp
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --output=%x-%j.out
# reprobe_comparison_v1 (docs/reprobe_comparison_v1_plan.md). S0 (data) runs on the login
# node beforehand (scripts/reprobe_cmp/build_{testsets,train}.py, see the plan); every
# stage reads $R/data.
#   STAGE=s1    references: GPU0 released DeepSeek-labeled ReProbe, GPU1 released self-labeled
#               (full trace and future masked, 8 test sets + ProcessBench); GPU2-3 RT-zs, the
#               three RTH-ICS tokens read inline in ReProbe layout
#   STAGE=s2rp  GPU0-2 ReProbe retrained, seeds 0-2, four heads each (rp/rpc x ds/self);
#               GPU3 RT-self seed 0
#   STAGE=s2rt  GPU0-2 RT-ds seeds 0-2; GPU3 RT-self seeds 1, 2 in turn
#   STAGE=bon   v2 B1: every trained and released probe scores the best-of-N candidates in
#               $R/bon/inputs (scripts/reprobe_cmp/build_bon.py): GPU0 released ds, self;
#               GPU1 ReProbe retrained seeds 0-2; GPU2 RT-ds and RT-zs; GPU3 RT-self
#   STAGE=beam  v2 B2: beam search B=5 x N=5 (scripts/reprobe_cmp/beam_search.py), one set per
#               GPU (BEAM_SETS, default gsm8k math strategyqa scienceqa), the first BEAM_LIMIT
#               questions (default 100); arms RT-self seed 0, released self ReProbe, random
# SMOKE=N runs every step on the first N traces / items (pipeline and timing check).
# v2 (docs/reprobe_comparison_v2_plan.md), other backbones: MODEL, LAYOUT, LABELS (the
# label sets of $D/rt, first = primary) and R; e.g. Phi-4:
#   R=$STORE/cot_mech/reprobe_cmp_v2/phi4 MODEL=microsoft/phi-4 LAYOUT=reprobe_phi4 \
#   LABELS="qwen3 self" HF_MODELS=$SCRATCH/hf_cache STAGE=s2rp sbatch slurm/reprobe_cmp_tamia.sh
#   (s2rp: RT on GPU3 uses the second label set; s2rt: GPU0-2 the first, GPU3 the second;
#   with one label set, e.g. Qwen3-32B LABELS=oss: s2rp GPU3 runs the released probe in
#   $R/released/oss, then RT seed 0; s2rt runs RT seeds 1 and 2 on GPU0-1)
set -euo pipefail
STAGE="${STAGE:?s1, s2rp or s2rt}"
SMOKE="${SMOKE:-0}"
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
R="${R:-/project/aip-azouaq/$USER/cot_mech/reprobe_cmp_v1}"
RTV1="${RTV1:-$SCRATCH/cot_mech/read_token_v1}"
D="$R/data"
RUN="$R/runs$([[ $SMOKE == 0 ]] || echo "_smoke$SMOKE")"
LOGS="$RUN/logs"
L="$SLURM_TMPDIR/rpc"
Q3="${MODEL:-Qwen/Qwen3-8B}"
LAYOUT="${LAYOUT:-reprobe}"
read -r -a LABS <<< "${LABELS:-ds self}"
LA="${LABS[0]}"; LB="${LABS[1]:-${LABS[0]}}"
HEADS=(); for l in "${LABS[@]}"; do HEADS+=("rp_$l:$l:visible" "rpc_$l:$l:masked"); done
mkdir -p "$LOGS" "$L"
module purge
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0
export HF_HOME="${HF_MODELS:-/project/aip-azouaq/$USER/hf_cache}"
unset TRANSFORMERS_CACHE
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 MKL_NUM_THREADS=12
cd "$PROJECT_ROOT"
echo "snapshot ${PROJECT_ROOT} stage=$STAGE smoke=$SMOKE run=$RUN"
virtualenv --no-download "$SLURM_TMPDIR/env" >/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip >/dev/null
pip install --no-index torch transformers numpy scipy scikit-learn pyyaml pandas 2>&1 | tail -1

TESTS=(math gsm8k proofnet trips meetings calendar strategyqa scienceqa)
RP_SETS=(); RT_SETS=()
for s in "${TESTS[@]}"; do RP_SETS+=("$s:$D/test/$s.traces.jsonl"); RT_SETS+=("$s:$D/test/$s.items.jsonl"); done
for s in gsm8k math olympiadbench omnimath; do
  RP_SETS+=("pb_$s:$D/pb/$s.traces.jsonl"); RT_SETS+=("pb_$s:$D/pb/$s.items.jsonl")
done
RT_SETS+=("val:$D/rt/all_val.items.jsonl")
RP_SMOKE=(--smoke "$SMOKE")

# rp_released <gpu> <ds|self>
rp_released() {
  CUDA_VISIBLE_DEVICES=$1 python scripts/reprobe_cmp/reprobe_probe.py released --model $Q3 \
    --ckpt "$R/released/$2" --name "rel_$2" --sets "${RP_SETS[@]}" "val:$D/rp/val.traces.jsonl" \
    --out "$RUN/released" "${RP_SMOKE[@]}" > "$LOGS/rel_$2-$SLURM_JOB_ID.log" 2>&1
}

# rt_encode <gpu> <ckpt> <name>
rt_encode() {
  local sm=(); [[ $SMOKE == 0 ]] || sm=(--smoke "$SMOKE")
  CUDA_VISIBLE_DEVICES=$1 python scripts/read_token_inline.py encode --model $Q3 --layout $LAYOUT \
    --ckpt "$2" --out "$L/$3" --prm800k "${RT_SETS[@]}" --passive_blocks "" "${sm[@]}"
  # keep the head scores and labels only (the read and passive states stay node-local)
  mkdir -p "$RUN/rt/$3"
  python - "$L/$3" "$RUN/rt/$3" <<'PY'
import sys, shutil, numpy as np
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
for z in src.glob("*.shard0.npz"):
    a = np.load(z)
    np.savez(dst / z.name, y=a["y"], kept=a["kept"], head_p_correct=a["head_p_correct"])
    shutil.copy(src / z.name.replace(".npz", ".meta.jsonl"), dst)
PY
}

# rt_train <gpu> <ds|self> <seed>
rt_train() {
  local name="rt_$2_seed$3" sm=(); [[ $SMOKE == 0 ]] || sm=(--smoke "$SMOKE")
  (
    set -e
    export CUDA_VISIBLE_DEVICES=$1
    python scripts/read_token_inline.py train --model $Q3 --layout $LAYOUT --init_text " correct" --deep \
      --head linear --chain --start_marker --start_deep --train "$D/rt/$2_train.items.jsonl" --n_train 0 \
      --seed "$3" --lr 3e-3 --soft_attn no_cudnn --out "$L/$name" "${sm[@]}"
    mkdir -p "$RUN/tokens/$name" && cp "$L/$name/read_token.pt" "$RUN/tokens/$name/"
    rt_encode "$1" "$L/$name/read_token.pt" "$name"
  ) > "$LOGS/$name-$SLURM_JOB_ID.log" 2>&1
}

pids=()
if [[ "$STAGE" == s1 ]]; then
  rp_released 0 ds & pids+=($!)
  rp_released 1 self & pids+=($!)
  ( rt_encode 2 "$RTV1/tokens/rthics_seed0/read_token.pt" rtzs_seed0
    rt_encode 2 "$RTV1/tokens/rthics_seed2/read_token.pt" rtzs_seed2 ) > "$LOGS/rtzs_a-$SLURM_JOB_ID.log" 2>&1 & pids+=($!)
  rt_encode 3 "$RTV1/tokens/rthics_seed1/read_token.pt" rtzs_seed1 > "$LOGS/rtzs_b-$SLURM_JOB_ID.log" 2>&1 & pids+=($!)
elif [[ "$STAGE" == s2rp ]]; then
  for s in 0 1 2; do
    CUDA_VISIBLE_DEVICES=$s python scripts/reprobe_cmp/reprobe_probe.py train --model $Q3 \
      --train "$D/rp/train.traces.jsonl" --val "$D/rp/val.traces.jsonl" --val_scores --seed $s --heads "${HEADS[@]}" \
      --sets "${RP_SETS[@]}" --out "$RUN/rp/seed$s" "${RP_SMOKE[@]}" \
      > "$LOGS/rp_seed$s-$SLURM_JOB_ID.log" 2>&1 & pids+=($!)
  done
  # one label set (Qwen3-32B): GPU3 also runs the released probe through our code first
  ( if [[ "$LB" == "$LA" && -d "$R/released/$LA" ]]; then rp_released 3 "$LA"; fi
    rt_train 3 "$LB" 0 ) & pids+=($!)
elif [[ "$STAGE" == bon ]]; then
  RP_SETS=(); RT_SETS=()
  for s in ${BON_SETS:-math gsm8k strategyqa scienceqa calendar}; do
    RP_SETS+=("bon_$s:$R/bon/inputs/$s.traces.jsonl"); RT_SETS+=("bon_$s:$R/bon/inputs/$s.items.jsonl")
  done
  ( rp_released 0 ds; rp_released 0 self ) & pids+=($!)
  ( for s in 0 1 2; do
      CUDA_VISIBLE_DEVICES=1 python scripts/reprobe_cmp/reprobe_probe.py score --model $Q3 \
        --ckpt "$RUN/rp/seed$s/heads.pt" --sets "${RP_SETS[@]}" --out "$RUN/rp/seed$s" "${RP_SMOKE[@]}"
    done ) > "$LOGS/bon_rp-$SLURM_JOB_ID.log" 2>&1 & pids+=($!)
  ( for s in 0 1 2; do rt_encode 2 "$RUN/tokens/rt_${LA}_seed$s/read_token.pt" "rt_${LA}_seed$s"; done
    for s in 0 1 2; do rt_encode 2 "$RTV1/tokens/rthics_seed$s/read_token.pt" "rtzs_seed$s"; done
  ) > "$LOGS/bon_rt_a-$SLURM_JOB_ID.log" 2>&1 & pids+=($!)
  ( for s in 0 1 2; do rt_encode 3 "$RUN/tokens/rt_${LB}_seed$s/read_token.pt" "rt_${LB}_seed$s"; done
  ) > "$LOGS/bon_rt_b-$SLURM_JOB_ID.log" 2>&1 & pids+=($!)
elif [[ "$STAGE" == beam ]]; then
  g=0
  for s in ${BEAM_SETS:-gsm8k math strategyqa scienceqa}; do
    CUDA_VISIBLE_DEVICES=$g python scripts/reprobe_cmp/beam_search.py --pools "$SCRATCH/cot_mech/reprobe_bon/pools" \
      --set "$s" --limit "${BEAM_LIMIT:-100}" --model $Q3 --out "$R/beam/$s.jsonl" \
      --arm "rt_${LB}=rt:$RUN/tokens/rt_${LB}_seed0/read_token.pt" "rp_${LB}=rp:$R/released/$LB" random=random \
      > "$LOGS/beam_$s-$SLURM_JOB_ID.log" 2>&1 & pids+=($!)
    g=$((g + 1))
  done
elif [[ "$STAGE" == s2rt ]]; then
  if [[ "$LB" == "$LA" ]]; then  # one label set: seed 0 ran in s2rp
    for s in 1 2; do rt_train $((s - 1)) "$LA" $s & pids+=($!); done
  else
    for s in 0 1 2; do rt_train $s "$LA" $s & pids+=($!); done
    ( rt_train 3 "$LB" 1; rt_train 3 "$LB" 2 ) & pids+=($!)
  fi
else
  echo "unknown STAGE=$STAGE"; exit 2
fi
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
tail -n 4 "$LOGS"/*-"$SLURM_JOB_ID".log || true
exit $fail
