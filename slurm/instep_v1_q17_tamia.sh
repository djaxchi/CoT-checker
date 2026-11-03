#!/bin/bash
#SBATCH --job-name=instep_v1_q17
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
# InStep V1 on Qwen3-1.7B in native thinking (ReProbe Table 4), the small end of the size
# ablation (1.7B, Qwen3-8B, Phi-4, Qwen3-32B). Data: BB=qwen3_1_7b slurm/reprobe_cmp_v2_s0_login.sh
# ($R/data: 8 test sets from the 7-epoch managers that reproduce Table 4, GPT-OSS-120B-labeled
# training traces, steps = sentences widened to token boundaries, layout reprobe_think).
#   STAGE=main    GPU0-2 InStep (read token with chained reads and the start marker, lr 3e-3,
#                 5% warm-up, EPOCHS epochs, default 1) seeds 0-2, then scored on the test sets
#                 and validation; GPU3 the released ReProbe probe through our code (gate G0:
#                 its scores against the manager's stored ones), then ReProbe retrained in our
#                 harness, seeds 0-2 (visible and future-masked heads)
#   STAGE=states  passive linear baseline: text states at blocks 3..27 (every 4th) and the final
#                 norm on the test sets, validation and 40k training steps (read with InStep
#                 seed 0; text never attends to the read token), then passive_fit.py
# Outputs: tokens $OUT/simp/17b_<name>/, InStep scores $OUT/rt_simp/17b/<name>/, ReProbe
# $R/runs/{released,rp/seed<s>}, passive $OUT/scores/qwen3_1_7b/passive/. SMOKE=N: first N items/traces.
set -euo pipefail
STAGE="${STAGE:?main or states}"
SMOKE="${SMOKE:-0}"
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
CM="/project/aip-azouaq/$USER/cot_mech"
SUF="$([[ $SMOKE == 0 ]] || echo "_smoke$SMOKE")"
OUT="${OUT:-$CM/instep_v1}$SUF"
R="$CM/reprobe_cmp_v2/qwen3_1_7b"
D="$R/data"
RUN="$R/runs$SUF"
ST="$SCRATCH/cot_mech/instep_v1$SUF/states/qwen3_1_7b"
LOGS="$OUT/logs"
L="$SLURM_TMPDIR/q17"
MODEL=Qwen/Qwen3-1.7B
EPOCHS="${EPOCHS:-1}"
mkdir -p "$LOGS" "$L" "$OUT/simp" "$RUN"
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
pip install --no-index torch transformers numpy scipy scikit-learn pyyaml pandas 2>&1 | tail -1
SM=(); [[ $SMOKE == 0 ]] || SM=(--smoke "$SMOKE")
TESTS=(math gsm8k proofnet trips meetings calendar strategyqa scienceqa)
RT_SETS=(); RP_SETS=()
for s in "${TESTS[@]}"; do RT_SETS+=("$s:$D/test/$s.items.jsonl"); RP_SETS+=("$s:$D/test/$s.traces.jsonl"); done
RT_SETS+=("val:$D/rt/all_val.items.jsonl")

# keep <src> <dst>: head scores and labels of an encode, without the states
keep() {
  mkdir -p "$2"
  python - "$1" "$2" <<'PY'
import sys, shutil, numpy as np
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
for z in src.glob("*.shard0.npz"):
    a = np.load(z)
    np.savez(dst / z.name, y=a["y"], kept=a["kept"], head_p_correct=a["head_p_correct"])
    shutil.copy(src / z.name.replace(".npz", ".meta.jsonl"), dst)
PY
}

# instep <gpu> <seed>
instep() {
  local g=$1 s=$2 name="instep_oss_ep${EPOCHS}_seed$2"
  local tok="$OUT/simp/17b_$name" tmp="$L/enc_$name"
  CUDA_VISIBLE_DEVICES=$g python scripts/read_token_inline.py train --model $MODEL --layout reprobe_think \
    --init_text " correct" --head linear --deep --chain --start_marker --start_deep \
    --train "$D/rt/oss_train.items.jsonl" --n_train 0 --val "$D/rt/oss_val.items.jsonl" \
    --sweep "3e-3:0.05" --epochs "$EPOCHS" --seed "$s" --gate_tol "${GATE_TOL:-0.05}" --soft_attn no_cudnn \
    --out "$tok" "${SM[@]}" || return 1
  local ck="$tok/read_token.pt"; [[ -f "$ck" ]] || ck="$tok/lr0.003_wu0.05/read_token.pt"
  CUDA_VISIBLE_DEVICES=$g python scripts/read_token_inline.py encode --model $MODEL --layout reprobe_think \
    --ckpt "$ck" --out "$tmp" --prm800k "${RT_SETS[@]}" --passive_blocks "" "${SM[@]}" || return 1
  keep "$tmp" "$OUT/rt_simp/17b/$name" && rm -rf "$tmp"
}

pids=()
if [[ $STAGE == main ]]; then
  for s in 0 1 2; do instep $s $s > "$LOGS/q17_instep_seed$s-$SLURM_JOB_ID.log" 2>&1 & pids+=($!); done
  ( set -e
    export CUDA_VISIBLE_DEVICES=3
    python scripts/reprobe_cmp/reprobe_probe.py released --model $MODEL --ckpt "$R/released/oss" --name rel_oss \
      --sets "${RP_SETS[@]}" "val:$D/rp/val.traces.jsonl" --out "$RUN/released" "${SM[@]}"
    for s in 0 1 2; do
      python scripts/reprobe_cmp/reprobe_probe.py train --model $MODEL --train "$D/rp/train.traces.jsonl" \
        --val "$D/rp/val.traces.jsonl" --val_scores --seed $s --heads rp_oss:oss:visible rpc_oss:oss:masked \
        --sets "${RP_SETS[@]}" --out "$RUN/rp/seed$s" "${SM[@]}"
    done ) > "$LOGS/q17_reprobe-$SLURM_JOB_ID.log" 2>&1 & pids+=($!)
elif [[ $STAGE == states ]]; then
  TOK="$OUT/simp/17b_instep_oss_ep${EPOCHS}_seed0"; [[ -f $TOK/read_token.pt ]] && TOK=$TOK/read_token.pt || TOK=$TOK/lr0.003_wu0.05/read_token.pt
  mkdir -p "$ST"
  ( set -e
    CUDA_VISIBLE_DEVICES=0 python scripts/read_token_inline.py encode --model $MODEL --layout reprobe_think \
      --ckpt "$TOK" --out "$ST" --prm800k "${RT_SETS[@]}" "train40k:$D/rt/oss_train.items.jsonl:40000" \
      --passive_blocks 3,7,11,15,19,23,27 --read_blocks 3,7,11,15,19,23,27 "${SM[@]}"
    part=(); [[ $SMOKE == 0 ]] || part=(--allow_partial)
    CUDA_VISIBLE_DEVICES=0 python scripts/instep_v1/passive_fit.py --states "$ST" --data "$D" --val_label oss \
      --train_items "$D/rt/oss_train.items.jsonl" --out "$OUT/scores/qwen3_1_7b/passive" \
      --per_layer_out "$OUT/analysis/per_layer_qwen3_1_7b.json" "${part[@]}"
  ) > "$LOGS/q17_states-$SLURM_JOB_ID.log" 2>&1 & pids+=($!)
else
  echo "unknown STAGE=$STAGE"; exit 2
fi
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
grep -h "^\[val\]\|^\[encode\] val\|^\[released\]\|^\[train\].*best\|Error\|error" "$LOGS"/q17_*-"$SLURM_JOB_ID".log | tail -n 40 || true
exit $fail
