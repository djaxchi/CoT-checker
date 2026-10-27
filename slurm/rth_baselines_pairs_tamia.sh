#!/bin/bash
#SBATCH --job-name=rth_bl_pairs
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=01:45:00
#SBATCH --output=%x-%j.out
# read_token_v1 Stage G (docs/read_token_v1_plan.md): the confidence-wrong pair test for
# the baselines BL1-BL4. Every token in TOKENS scores the MATH-500 T=1.0 Instruct pool
# (instruct_t10), restricted to the 158 questions that hold a mixed pair
# (results/read_token_v1/baselines/pair_questions_math500.json, copied to $QIDS): the
# pairs, and so wrong_conf_question_weighted, only come from those questions, and RTH
# scored on them alone reproduces its full-pool 0.530 exactly. Branched reading,
# segmentation checked against the PRM's score files. Tokens go round-robin on 4 GPUs.
# First (DO_VAL=1, the default) every token re-encodes PRM800K val 5k and its head's val
# AUROC goes to $OUT/baselines/val_auroc-<job>.json: the learning-rate rule for BL3/BL4
# and the per-baseline seed pick read it (ProcessBench is never read). DO_PAIRS=0 stops
# after that.
#   TOKENS="bl1_correct_seed0 bl1_correct_seed1 ..." sbatch --export=ALL,PROJECT_ROOT=<snapshot>,TOKENS=... \
#     <snapshot>/slurm/rth_baselines_pairs_tamia.sh
set -euo pipefail
: "${TOKENS:?set TOKENS to token dir names under \$OUT/tokens}"
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
OUT="${OUT:-$SCRATCH/cot_mech/read_token_v1}"
DS="${DS:-$OUT/downstream}"
QIDS="${QIDS:-$OUT/baselines/pair_questions_math500.json}"
DATA="${DATA:-$SCRATCH/cot_mech/dense_full_7b_v1/data}"
DO_VAL="${DO_VAL:-1}"
DO_PAIRS="${DO_PAIRS:-1}"
POOL="$DS/pools/instruct_t10"
Q3=Qwen/Qwen3-8B
LOGS="$DS/logs"
mkdir -p "$LOGS" "$DS/scores/instruct_t10" "$OUT/baselines"
[[ "$DO_PAIRS" == 0 || -s "$QIDS" ]] || { echo "missing $QIDS"; exit 1; }
read -r -a toks <<< "$TOKENS"
for n in "${toks[@]}"; do
  [[ -s "$OUT/tokens/$n/read_token.pt" ]] || { echo "missing $OUT/tokens/$n/read_token.pt"; exit 1; }
done
cd "$PROJECT_ROOT"
echo "git $(git rev-parse --short HEAD 2>/dev/null || echo snapshot) tokens=${toks[*]}"

module purge
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0
export HF_HOME=/project/aip-azouaq/$USER/hf_cache
unset TRANSFORMERS_CACHE
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 MKL_NUM_THREADS=12
virtualenv --no-download "$SLURM_TMPDIR/env" >/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip >/dev/null
pip install --no-index torch transformers numpy scipy scikit-learn pyyaml 2>&1 | tail -1

fail=0
if [[ "$DO_VAL" == 1 ]]; then
  LOCAL="$SLURM_TMPDIR/val"
  pids=()
  for g in 0 1 2 3; do
    (
      set -e
      export CUDA_VISIBLE_DEVICES=$g
      for ((k = g; k < ${#toks[@]}; k += 4)); do
        n=${toks[$k]}
        python scripts/read_token_probe.py encode --model $Q3 --ckpt "$OUT/tokens/$n/read_token.pt" \
          --out "$LOCAL/$n" --prm800k "val:$DATA/prm800k_val_5k.jsonl"
      done
    ) > "$LOGS/blval_gpu$g-$SLURM_JOB_ID.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  (( fail == 0 )) || { tail -n 5 "$LOGS"/blval_gpu*-"$SLURM_JOB_ID".log; exit 1; }
  vals=(); for n in "${toks[@]}"; do vals+=("$n:1:$LOCAL/$n"); done
  python scripts/analysis/read_token_pick.py --val "${vals[@]}" --out "$OUT/baselines/val_auroc-$SLURM_JOB_ID.json" >/dev/null
  python -c "import json,sys; [print(f'[val] {k:<28} {v:.4f}') for k, v in json.load(open(sys.argv[1]))['val_auroc'].items()]" \
    "$OUT/baselines/val_auroc-$SLURM_JOB_ID.json"
fi
[[ "$DO_PAIRS" == 1 ]] || exit 0

pids=()
for g in 0 1 2 3; do
  (
    set -e
    export CUDA_VISIBLE_DEVICES=$g
    for ((k = g; k < ${#toks[@]}; k += 4)); do
      n=${toks[$k]}
      python scripts/onpolicy/score_traces_with_read_token.py --model $Q3 --branched \
        --ckpt "$OUT/tokens/$n/read_token.pt" --cell_name "$n" --fork_ids "$QIDS" \
        --trajectories "$POOL"/tts_math500.shard*_trajectories.jsonl \
        --verify_against "$POOL"/scores/tts_math500__prm_qwen25_math_7b.shard*.jsonl \
        --out "$DS/scores/instruct_t10/tts_math500__$n.shard0.jsonl"
    done
  ) > "$LOGS/blpairs_gpu$g-$SLURM_JOB_ID.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || fail=1; done
tail -n 2 "$LOGS"/blpairs_gpu*-"$SLURM_JOB_ID".log
echo "[$(date)] done, fail=$fail"
exit $fail
