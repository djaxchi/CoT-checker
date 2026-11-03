#!/bin/bash
#SBATCH --job-name=instep_v1_prm
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=00:20:00
#SBATCH --output=%x-%j.out
# InStep V1 P1 (docs/instep_v1_runbook.md): Qwen2.5-Math-PRM-7B (prm7b) and Qwen2.5-Math-7B-PRM800K
# (prm800k) step scores on the 8 ReProbe test sets, validation and ProcessBench, for the three
# backbones' traces. GPUs 0,1: prm7b shards 0,1; GPUs 2,3: prm800k shards 0,1. ProcessBench items
# are the same file for all backbones, so they are scored once (qwen3_8b) and copied after a
# byte-identity check. The merge step runs the gate against the PRM scores stored in ReProbe's
# released test files and writes $OUT/scores/gate_prm.json.
# SMOKE=N: first N traces per set, outputs under ${OUT}_smokeN.
set -euo pipefail
SMOKE="${SMOKE:-0}"
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
CM=/project/aip-azouaq/$USER/cot_mech
SUF="$([[ $SMOKE == 0 ]] || echo "_smoke$SMOKE")"
OUT="${OUT:-$CM/instep_v1}$SUF"
SC="$OUT/scores"
LOGS="$OUT/logs"
mkdir -p "$LOGS" "$SC"
module purge
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0
export HF_HOME=/project/aip-azouaq/$USER/hf_cache
unset TRANSFORMERS_CACHE
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 MKL_NUM_THREADS=12
cd "$PROJECT_ROOT"
echo "git $(git rev-parse --short HEAD 2>/dev/null || echo none) smoke=$SMOKE out=$OUT"
virtualenv --no-download "$SLURM_TMPDIR/env" >/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip >/dev/null
pip install --no-index torch "transformers<5" numpy scipy scikit-learn 2>&1 | tail -1

declare -A ROOTS=([qwen3_8b]=$CM/reprobe_cmp_v1 [phi4]=$CM/reprobe_cmp_v2/phi4 [qwen3_32b]=$CM/reprobe_cmp_v2/qwen3_32b)
IN=(); DATA=(); COPY=()
for bb in qwen3_8b phi4 qwen3_32b; do
  R=${ROOTS[$bb]}
  DATA+=("$bb=$R")
  for s in math gsm8k proofnet trips meetings calendar strategyqa scienceqa; do IN+=("$bb:$s:$R/data/test/$s.items.jsonl"); done
  IN+=("$bb:val:$R/data/rt/all_val.items.jsonl")
  for s in gsm8k math olympiadbench omnimath; do
    if [[ $bb == qwen3_8b ]]; then IN+=("$bb:pb_$s:$R/data/pb/$s.items.jsonl")
    else COPY+=("qwen3_8b:pb_$s:$bb:$R/data/pb/$s.items.jsonl"); fi
  done
done
SM=(); [[ $SMOKE == 0 ]] || SM=(--smoke "$SMOKE")

pids=()
g=0
for tag in prm7b prm800k; do
  for sh in 0 1; do
    CUDA_VISIBLE_DEVICES=$g python scripts/instep_v1/score_prm.py score --tag $tag --out "$SC" \
      --inputs "${IN[@]}" --shard $sh --num_shards 2 "${SM[@]}" \
      > "$LOGS/prm_${tag}_s$sh-$SLURM_JOB_ID.log" 2>&1 & pids+=($!)
    g=$((g + 1))
  done
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
grep -h "^\[prm\] .*\(done\|tokens\|FAILED\)" "$LOGS"/prm_*-"$SLURM_JOB_ID".log || true
if [[ $fail != 0 ]]; then tail -n 30 "$LOGS"/prm_*-"$SLURM_JOB_ID".log; exit 1; fi
python scripts/instep_v1/score_prm.py merge --out "$SC" --inputs "${IN[@]}" --tags prm7b prm800k \
  --num_shards 2 --data "${DATA[@]}" --copy "${COPY[@]}" "${SM[@]}"
