#!/bin/bash
#SBATCH --job-name=instep_v1_signal
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=01:45:00
#SBATCH --output=%x-%j.out
# InStep paper, "where the signal lives" (Qwen3-8B, ReProbe DS data).
# 1. scripts/instep_v1/signal_states.py on the steps of the stored InStep states ($ST8: text
#    and step-token states of the DS seed-0 token at every 4th block, on train40k, val and the
#    8 test sets): GPU0 Qwen2.5-Math-PRM-7B separator states on val + test, GPU1 the same on
#    train40k, GPU2 ReProbe's released DS head (pooled step representation) on val + test,
#    GPU3 the same on train40k; written to $SS/prm and $SS/reprobe, each gated against the
#    stored step scores. METHODS="prm" (or "reprobe") reruns one method only.
# 2. scripts/instep_v1/signal_geometry.py: one linear probe and one set of geometry measures
#    per representation and layer, to $OUT/signal/.
# SMOKE=N: states of the first N traces per set under ${SS}_smokeN (gates printed, not
#    enforced), then the geometry of passive_final and sep only (real states, no extra);
#    submit it with --time=00:25:00.
# --time: PRM and ReProbe passes ~20 min each on 22k train traces (estimate from the V1 PRM
#    and ReProbe scoring jobs); geometry ~30 keys at ~1 min each (passive_fit S1 took 21 min
#    for 19 keys), in-set ceilings on the 4 final keys.
set -euo pipefail
SMOKE="${SMOKE:-0}"
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
CM="/project/aip-azouaq/$USER/cot_mech"
V1="$CM/reprobe_cmp_v1"
D="$V1/data"
OUT="${OUT:-$CM/instep_v1}"
ST8="${ST8:-$SCRATCH/cot_mech/instep_v1/states/qwen3_8b}"
SUF="$([[ $SMOKE == 0 ]] || echo "_smoke$SMOKE")"
SS="${SS:-$SCRATCH/cot_mech/instep_v1/signal_states}$SUF"
LOGS="$OUT/logs"
mkdir -p "$LOGS" "$SS"
module purge
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0
export HF_HOME=/project/aip-azouaq/$USER/hf_cache
unset TRANSFORMERS_CACHE
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 MKL_NUM_THREADS=12
cd "$PROJECT_ROOT"
echo "git $(git rev-parse --short HEAD 2>/dev/null || echo none) smoke=$SMOKE ss=$SS"
virtualenv --no-download "$SLURM_TMPDIR/env" >/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip >/dev/null
# transformers < 5: the Qwen PRMs' remote code scores garbage under 5 (commit 41156c6)
pip install --no-index torch "transformers<5" numpy scipy scikit-learn pyyaml 2>&1 | tail -1
SM=(); [[ $SMOKE == 0 ]] || SM=(--smoke "$SMOKE")
METHODS="${METHODS:-prm reprobe}"

TESTS=(math gsm8k proofnet trips meetings calendar strategyqa scienceqa)
PI=(); PK=(); RI=(); RK=()
for s in "${TESTS[@]}"; do
  PI+=("$s:$D/test/$s.items.jsonl"); RI+=("$s:$D/test/$s.traces.jsonl"); PK+=("$s:$ST8/$s.shard0.meta.jsonl")
done
PI+=("val:$D/rt/all_val.items.jsonl"); RI+=("val:$D/rp/val.traces.jsonl"); PK+=("val:$ST8/val.shard0.meta.jsonl")
TK=("train40k:$ST8/train40k.shard0.meta.jsonl")
PRM=(python scripts/instep_v1/signal_states.py prm --out "$SS" --gate_scores "$OUT/scores/qwen3_8b/prm7b" "${SM[@]}")
RP=(python scripts/instep_v1/signal_states.py reprobe --ckpt "$V1/released/ds" --out "$SS"
    --gate_scores "$V1/runs/released/scores/rel_ds" "${SM[@]}")
pids=()
if [[ " $METHODS " == *" prm "* ]]; then
  CUDA_VISIBLE_DEVICES=0 "${PRM[@]}" --sets "${PI[@]}" --keep "${PK[@]}" > "$LOGS/signal_prm_test$SUF-$SLURM_JOB_ID.log" 2>&1 & pids+=($!)
  CUDA_VISIBLE_DEVICES=1 "${PRM[@]}" --sets "train40k:$D/rt/ds_train.items.jsonl" --keep "${TK[@]}" \
    > "$LOGS/signal_prm_train$SUF-$SLURM_JOB_ID.log" 2>&1 & pids+=($!)
fi
if [[ " $METHODS " == *" reprobe "* ]]; then
  CUDA_VISIBLE_DEVICES=2 "${RP[@]}" --sets "${RI[@]}" --keep "${PK[@]}" > "$LOGS/signal_rp_test$SUF-$SLURM_JOB_ID.log" 2>&1 & pids+=($!)
  CUDA_VISIBLE_DEVICES=3 "${RP[@]}" --sets "train40k:$D/rp/train.traces.jsonl" --keep "${TK[@]}" \
    > "$LOGS/signal_rp_train$SUF-$SLURM_JOB_ID.log" 2>&1 & pids+=($!)
fi
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
grep -h "^\[prm\]\|^\[reprobe\]\|Error\|error" "$LOGS"/signal_*"$SUF-$SLURM_JOB_ID".log | tail -n 40 || true
[[ $fail == 0 || $SMOKE != 0 ]] || exit 1

if [[ $SMOKE == 0 ]]; then
  CUDA_VISIBLE_DEVICES=0 python scripts/instep_v1/signal_geometry.py --states "$ST8" --extra "$SS/prm" "$SS/reprobe" --data "$D" \
    --out "$OUT/signal"
else
  CUDA_VISIBLE_DEVICES=0 python scripts/instep_v1/signal_geometry.py --states "$ST8" --data "$D" \
    --keys passive_final,sep --out "$OUT/signal$SUF"
fi
