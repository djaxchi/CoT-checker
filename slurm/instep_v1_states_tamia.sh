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
#   STAGE=curve   audit of B1 on qwen3_8b (scripts/instep_v1/passive_curve.py): passive_final
#                 learning curve on 5k-40k training steps and the late-layer concatenation,
#                 to $OUT/analysis/passive_curve_qwen3_8b.json.
#   STAGE=fullpassive  B1 on the full DS training split (167k steps, not 40k): text states at
#                 blocks 23,27,31,35 + final of all of ds_train (4 trace-disjoint parts, one per
#                 GPU, read with the offsets-only DS token; text states do not depend on it), then
#                 passive_curve.py on 40k / 80k / all, scores of the all-data fits (passive_final and
#                 the 5-state concatenation) to $OUT/scores/qwen3_8b/passive_full/{final,concat5}/.
# SMOKE=N runs every step on the first N items, under *_smokeN output roots.
set -euo pipefail
STAGE="${STAGE:?encode, fit, curve or fullpassive}"
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
elif [[ $STAGE == curve ]]; then
  CUDA_VISIBLE_DEVICES=0 python scripts/instep_v1/passive_curve.py --states "$ST8/qwen3_8b" \
    --data "${R[qwen3_8b]}/data" --val_label ds --out "$OUT/analysis/passive_curve_qwen3_8b.json" \
    > "$LOGS/curve_qwen3_8b-$SLURM_JOB_ID.log" 2>&1 & pids+=($!)
elif [[ $STAGE == fullpassive ]]; then
  D8="${R[qwen3_8b]}/data"; FS="$ST/qwen3_8b_fulltrain"; mkdir -p "$FS" "$SLURM_TMPDIR/parts"
  TOK8="$OUT/rungsweep/off_ds_seed0/lr0.003_wu0.05/read_token.pt"
  python - "$D8/rt/ds_train.items.jsonl" "$SLURM_TMPDIR/parts" "$SMOKE" <<'PY'
import json, sys, zlib
src, out, smoke = sys.argv[1], sys.argv[2], int(sys.argv[3])
fs = [open(f"{out}/train_p{g}.jsonl", "w") for g in range(4)]
for i, l in enumerate(open(src)):
    if smoke and i >= 4 * smoke: break
    tid = json.loads(l)["uid"].rsplit("::", 1)[0]
    fs[zlib.crc32(tid.encode()) % 4].write(l)
PY
  for g in 0 1 2 3; do
    ( HF_HOME=/project/aip-azouaq/$USER/hf_cache CUDA_VISIBLE_DEVICES=$g python scripts/read_token_inline.py encode --model Qwen/Qwen3-8B --layout reprobe \
        --ckpt "$TOK8" --out "$FS" --prm800k "train_p$g:$SLURM_TMPDIR/parts/train_p$g.jsonl" \
        --passive_blocks 23,27,31,35 "${SM[@]}" ) > "$LOGS/fullenc_p$g-$SLURM_JOB_ID.log" 2>&1 & pids+=($!)
  done
  fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done; pids=()
  [[ $fail == 0 ]] || { grep -h "Error\|error" "$LOGS"/fullenc_p*-"$SLURM_JOB_ID".log | tail; exit 1; }
  part=(--sizes 40000,80000,0 --concat_sizes 40000,0); [[ $SMOKE == 0 ]] || part=(--sizes 0 --concat_sizes 0)
  CUDA_VISIBLE_DEVICES=0 python scripts/instep_v1/passive_curve.py --states "$ST8/qwen3_8b" \
    --data "$D8" --val_label ds --train_states "$FS" --train_names train_p0,train_p1,train_p2,train_p3 \
    "${part[@]}" --scores_out "$OUT/scores/qwen3_8b/passive_full" \
    --out "$OUT/analysis/passive_full_qwen3_8b.json" > "$LOGS/fullfit-$SLURM_JOB_ID.log" 2>&1 & pids+=($!)
else
  echo "unknown STAGE=$STAGE"; exit 2
fi
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
grep -h "^\[encode\]\|^\[curve\]\|^\[passive_fit\]\|skipped\|Error\|error" "$LOGS"/*-"$SLURM_JOB_ID".log | tail -n 120 || true
exit $fail
