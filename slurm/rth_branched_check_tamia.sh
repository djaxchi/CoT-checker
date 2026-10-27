#!/bin/bash
#SBATCH --job-name=rth_branched
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=00:45:00
#SBATCH --output=%x-%j.out
# RTH with shared prefixes (read_token_probe.read_branched): the same scores as one read
# per step, computed from the trajectory's longest item plus each step's own tail.
#   GPU0-3  the T=1.0 MATH-500 pool re-scored with --branched, compared with the
#           per-step scores of job 645067 (max / mean |difference|, Pearson)
#   GPU0    then the cost table again with the branched read added (rth_cost.py)
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
OUT="${OUT:-$SCRATCH/cot_mech/read_token_v1}"
DS="${DS:-$OUT/downstream}"
FTOKEN="${FTOKEN:-rth_correct_seed1}"
CELL="rth_$FTOKEN"
CKPT="$OUT/tokens/$FTOKEN/read_token.pt"
POOL="$DS/pools/instruct_t10"
Q3=Qwen/Qwen3-8B
LOGS="$DS/logs"
mkdir -p "$LOGS" "$DS/branched"
cd "$PROJECT_ROOT"

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

pids=()
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g python scripts/onpolicy/score_traces_with_read_token.py --model $Q3 \
    --ckpt "$CKPT" --cell_name "$CELL" --branched \
    --trajectories "$POOL"/tts_math500.shard*_trajectories.jsonl \
    --out "$DS/branched/tts_math500__${CELL}__branched.shard$g.jsonl" --shard_idx $g --num_shards 4 \
    --compare_to "$DS/scores/instruct_t10/tts_math500__${CELL}".shard*.jsonl \
    > "$LOGS/branched_gpu$g-$SLURM_JOB_ID.log" 2>&1 &
  pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
grep -h -E "\] wrote|against|tokens \(" "$LOGS"/branched_gpu*-"$SLURM_JOB_ID".log | tail -n 12
(( fail == 0 )) || exit 1

CUDA_VISIBLE_DEVICES=0 python scripts/analysis/rth_cost.py --model $Q3 --ckpt "$CKPT" \
  --trajectories "$POOL/tts_math500.shard00_trajectories.jsonl" --n 200 \
  --out "$DS/e1_cost_branched-$SLURM_JOB_ID.json" > "$LOGS/e1b-$SLURM_JOB_ID.log" 2>&1 || fail=1
grep "^\[e1\]" "$LOGS/e1b-$SLURM_JOB_ID.log"
exit $fail
