#!/bin/bash
#SBATCH --job-name=instep_v1_states
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
# InStep paper V1, runs B1 (passive linear baseline) and S1 (per-layer separability),
# docs/instep_v1_runbook.md.
#   STAGE=encode  text (passive) and read states of Phi-4 (GPU0) and Qwen3-32B (GPU1), read
#                 with each backbone's existing InStep token, at every 4th block for Phi-4
#                 (3..39 of 40) and every 8th for Qwen3-32B (7..63 of 64), on the 8 test sets,
#                 ReProbe validation and a 40k-step draw of the training split, written to
#                 $ST/<backbone>/ (the qwen3_8b states come from instep_v1_tamia.sh STAGE=e1).
#                 Text states do not depend on the token: text never attends to read tokens.
#   STAGE=fit     scripts/instep_v1/passive_fit.py for every backbone whose states are complete
#                 (one GPU each): scores to $OUT/scores/<backbone>/passive/, per-layer val AP
#                 and AUROC of passive and read states to $OUT/analysis/per_layer_<backbone>.json.
# SMOKE=N runs every step on the first N items, under *_smokeN output roots.
set -euo pipefail
STAGE="${STAGE:?encode or fit}"
SMOKE="${SMOKE:-0}"
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
P="/project/aip-azouaq/$USER/cot_mech"
SUF="$([[ $SMOKE == 0 ]] || echo "_smoke$SMOKE")"
OUT="${OUT:-$P/instep_v1}$SUF"
ST="${ST:-$SCRATCH/cot_mech/instep_v1}$SUF/states"
ST8="${ST8:-$ST}"  # qwen3_8b states (e1); a smoke can point this at the real ones
LOGS="$OUT/logs"
mkdir -p "$LOGS" "$ST"
module purge
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0
export HF_HOME=$SCRATCH/hf_cache
unset TRANSFORMERS_CACHE
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 MKL_NUM_THREADS=12
cd "$PROJECT_ROOT"
echo "git $(git rev-parse --short HEAD 2>/dev/null || echo none) stage=$STAGE smoke=$SMOKE out=$OUT st=$ST"
virtualenv --no-download "$SLURM_TMPDIR/env" >/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip >/dev/null
pip install --no-index torch transformers numpy scipy scikit-learn 2>&1 | tail -1
SM=(); [[ $SMOKE == 0 ]] || SM=(--smoke "$SMOKE")

# per backbone: data root, model, layout, train label, token, blocks
declare -A R MODEL LAYOUT LAB TOK BLOCKS
R[qwen3_8b]=$P/reprobe_cmp_v1;           LAB[qwen3_8b]=ds
R[phi4]=$P/reprobe_cmp_v2/phi4;          LAB[phi4]=qwen3
R[qwen3_32b]=$P/reprobe_cmp_v2/qwen3_32b; LAB[qwen3_32b]=oss
MODEL[phi4]=microsoft/phi-4;   LAYOUT[phi4]=reprobe_phi4; TOK[phi4]=${R[phi4]}/runs/tokens/rt_qwen3_seed0/read_token.pt
MODEL[qwen3_32b]=Qwen/Qwen3-32B; LAYOUT[qwen3_32b]=reprobe; TOK[qwen3_32b]=${R[qwen3_32b]}/runs/tokens/rt_oss_seed0/read_token.pt
BLOCKS[phi4]=3,7,11,15,19,23,27,31,35,39
BLOCKS[qwen3_32b]=7,15,23,31,39,47,55,63
TESTS=(math gsm8k proofnet trips meetings calendar strategyqa scienceqa)

# encode <gpu> <backbone>
encode() {
  local g=$1 bb=$2 D="${R[$2]}/data" sets=()
  for s in "${TESTS[@]}"; do sets+=("$s:$D/test/$s.items.jsonl"); done
  sets+=("val:$D/rt/all_val.items.jsonl" "train40k:$D/rt/${LAB[$bb]}_train.items.jsonl:40000")
  mkdir -p "$ST/$bb"
  CUDA_VISIBLE_DEVICES=$g python scripts/read_token_inline.py encode --model "${MODEL[$bb]}" \
    --layout "${LAYOUT[$bb]}" --ckpt "${TOK[$bb]}" --out "$ST/$bb" --prm800k "${sets[@]}" \
    --passive_blocks "${BLOCKS[$bb]}" --read_blocks "${BLOCKS[$bb]}" "${SM[@]}"
}

# fit <gpu> <backbone> <states dir>
fit() {
  local g=$1 bb=$2 st=$3 D="${R[$2]}/data" part=()
  [[ $SMOKE == 0 ]] || part=(--allow_partial)
  for s in "${TESTS[@]}" val train40k; do
    [[ -f $st/$s.shard0.npz && -f $st/$s.shard0.meta.jsonl ]] || { echo "$bb: no $st/$s states, skipped"; return 3; }
  done
  CUDA_VISIBLE_DEVICES=$g python scripts/instep_v1/passive_fit.py --states "$st" --data "$D" \
    --val_label "${LAB[$bb]}" --train_items "$D/rt/${LAB[$bb]}_train.items.jsonl" \
    --out "$OUT/scores/$bb/passive" --per_layer_out "$OUT/analysis/per_layer_$bb.json" "${part[@]}"
}

pids=()
if [[ $STAGE == encode ]]; then
  for bb in phi4 qwen3_32b; do [[ -f ${TOK[$bb]} ]] || { echo "missing token ${TOK[$bb]}"; exit 2; }; done
  encode 0 phi4 > "$LOGS/enc_phi4-$SLURM_JOB_ID.log" 2>&1 & pids+=($!)
  encode 1 qwen3_32b > "$LOGS/enc_qwen3_32b-$SLURM_JOB_ID.log" 2>&1 & pids+=($!)
elif [[ $STAGE == fit ]]; then
  fit 0 qwen3_8b "$ST8/qwen3_8b" > "$LOGS/fit_qwen3_8b-$SLURM_JOB_ID.log" 2>&1 & pids+=($!)
  fit 1 phi4 "$ST/phi4" > "$LOGS/fit_phi4-$SLURM_JOB_ID.log" 2>&1 & pids+=($!)
  fit 2 qwen3_32b "$ST/qwen3_32b" > "$LOGS/fit_qwen3_32b-$SLURM_JOB_ID.log" 2>&1 & pids+=($!)
else
  echo "unknown STAGE=$STAGE"; exit 2
fi
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
grep -h "^\[encode\]\|^\[passive_fit\]\|skipped\|Error\|error" "$LOGS"/*-"$SLURM_JOB_ID".log | tail -n 120 || true
exit $fail
