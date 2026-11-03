#!/bin/bash
#SBATCH --job-name=instep_v1_conf
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=00:45:00
#SBATCH --output=%x-%j.out
# InStep V1 runs U1 and T1 (docs/instep_v1_runbook.md), scripts/instep_v1/score_confidence.py.
#   U1  token confidence (maxprob, maxent, ppl) of each backbone on its own traces: the 8
#       test sets and validation; ProcessBench too for qwen3_8b.
#   T1  P(True) with Qwen3-8B on test, validation and ProcessBench items, in N_SHARDS shards
#       that the four GPUs claim as they free up.
# GPU0 qwen3_8b U1, GPU1 phi4 U1, GPU2 qwen3_32b U1, GPU3 straight to P(True); every GPU then
# claims P(True) shards. Then merge and gate ($OUT/scores/gate_confidence.json).
# SMOKE=N runs everything on the first N traces / items of every set.
set -euo pipefail
SMOKE="${SMOKE:-0}"
N_SHARDS="${N_SHARDS:-16}"
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
PROJ=/project/aip-azouaq/$USER
SUF="$([[ $SMOKE == 0 ]] || echo "_smoke$SMOKE")"
OUT="${OUT:-$PROJ/cot_mech/instep_v1}$SUF"
LOGS="$OUT/logs"
mkdir -p "$LOGS"
module purge
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0
unset TRANSFORMERS_CACHE
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 MKL_NUM_THREADS=12
cd "$PROJECT_ROOT"
echo "git $(git rev-parse --short HEAD 2>/dev/null || echo none) smoke=$SMOKE out=$OUT"
virtualenv --no-download "$SLURM_TMPDIR/env" >/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip >/dev/null
pip install --no-index torch transformers numpy scipy scikit-learn 2>&1 | tail -1

S=scripts/instep_v1/score_confidence.py
SM=(); [[ $SMOKE == 0 ]] || SM=(--smoke "$SMOKE")
TESTS="math gsm8k proofnet trips meetings calendar strategyqa scienceqa"
PBS="gsm8k math olympiadbench omnimath"
R8=$PROJ/cot_mech/reprobe_cmp_v1/data
RP4=$PROJ/cot_mech/reprobe_cmp_v2/phi4/data
R32=$PROJ/cot_mech/reprobe_cmp_v2/qwen3_32b/data

# u1 sets: <R> [pb]
u1_sets() {
  local r=$1 out=()
  for s in $TESTS; do out+=("$s:$r/test/$s.traces.jsonl"); done
  out+=("val:$r/rp/val.traces.jsonl")
  if [[ ${2:-} == pb ]]; then for s in $PBS; do out+=("pb_$s:$r/pb/$s.traces.jsonl"); done; fi
  echo "${out[@]}"
}
PT=()
for s in $TESTS; do PT+=("$s:$R8/test/$s.items.jsonl"); done
PT+=("val:$R8/rt/all_val.items.jsonl")
for s in $PBS; do PT+=("pb_$s:$R8/pb/$s.items.jsonl"); done

CLAIM="$OUT/scores/qwen3_8b/ptrue/_claims_${SLURM_JOB_ID}"
ptrue() {  # <gpu>
  CUDA_VISIBLE_DEVICES=$1 HF_HOME=$PROJ/hf_cache python $S ptrue --model Qwen/Qwen3-8B --backbone qwen3_8b \
    --sets "${PT[@]}" --shard "0/$N_SHARDS" --claim_dir "$CLAIM" --out "$OUT" "${SM[@]}"
}
u1() {  # <gpu> <hf_home> <model> <backbone> <max_tokens> <sets...>
  local g=$1 hf=$2 m=$3 bb=$4 mt=$5; shift 5
  CUDA_VISIBLE_DEVICES=$g HF_HOME=$hf python $S u1 --model "$m" --backbone "$bb" --max_tokens "$mt" \
    --sets "$@" --out "$OUT" "${SM[@]}"
}

T0=$SECONDS
pids=()
( f=0; u1 0 "$PROJ/hf_cache" Qwen/Qwen3-8B qwen3_8b 32768 $(u1_sets "$R8" pb) || f=1; ptrue 0 || f=1; exit $f
) > "$LOGS/conf_g0-$SLURM_JOB_ID.log" 2>&1 & pids+=($!)
( f=0; u1 1 "$SCRATCH/hf_cache" microsoft/phi-4 phi4 32768 $(u1_sets "$RP4") || f=1; ptrue 1 || f=1; exit $f
) > "$LOGS/conf_g1-$SLURM_JOB_ID.log" 2>&1 & pids+=($!)
( f=0; u1 2 "$SCRATCH/hf_cache" Qwen/Qwen3-32B qwen3_32b 24576 $(u1_sets "$R32") || f=1; ptrue 2 || f=1; exit $f
) > "$LOGS/conf_g2-$SLURM_JOB_ID.log" 2>&1 & pids+=($!)
ptrue 3 > "$LOGS/conf_g3-$SLURM_JOB_ID.log" 2>&1 & pids+=($!)
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "[time] GPU phase $((SECONDS - T0))s"
python $S merge --backbone qwen3_8b --tag ptrue --out "$OUT" || fail=1
python $S gate --out "$OUT" --data "qwen3_8b:$R8:ds" "phi4:$RP4:qwen3" "qwen3_32b:$R32:oss" || fail=1
grep -h "^\[load\]\|^\[u1\]\|shard .* done\|Error\|error" "$LOGS"/conf_g*-"$SLURM_JOB_ID".log | tail -n 80 || true
echo "[time] total $((SECONDS - T0))s"
exit $fail
