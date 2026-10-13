#!/bin/bash
#SBATCH --job-name=fsl_t1
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=01:00:00

# factorized_step_labels_v1 track 1 (downstream risk, free generation): the frozen
# fork split sharded across the node's 4 GPUs, one process per GPU
# (docs/factorized_step_labels_v1_plan.md section 5). Shards go to $SCRATCH.
# Several nodes split one run with GROUP (0..) and NUM_SHARDS (4 x nodes): this
# node runs shards 4*GROUP .. 4*GROUP+3. RESUME=1 continues shards cut by walltime.
set -euo pipefail
: "${SNAP:?code snapshot}" "${R:?project result dir}" "${W:?scratch work dir}"
SPLIT="$R/split/${SPLIT_NAME:-fork_split_v1.json}"
HF_CACHE="${HF_CACHE:-/project/aip-azouaq/$USER/hf_cache}"
M="${M:-4}"; K="${K:-8}"; LIMIT="${LIMIT:-0}"
GROUP="${GROUP:-0}"; NUM_SHARDS="${NUM_SHARDS:-4}"
RES=(); [[ "${RESUME:-0}" == 1 ]] && RES=(--resume)
OUT="$W/track1"
mkdir -p "$OUT" "$W/logs"

module purge
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0
export HF_HOME="$HF_CACHE"
unset TRANSFORMERS_CACHE
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 TORCH_NUM_THREADS=4
cd "$SNAP"
grep -E "snapshot|git_head" SNAPSHOT_INFO
virtualenv --no-download "$SLURM_TMPDIR/env" >/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip >/dev/null
pip install --no-index torch transformers numpy sympy 2>&1 | tail -1
sha256sum "$SPLIT"

pids=()
for g in 0 1 2 3; do
  sh=$((4 * GROUP + g))
  CUDA_VISIBLE_DEVICES=$g python scripts/factorized_step_labels/track1_generate.py \
    --fork_split "$SPLIT" --out_dir "$OUT" --local_files_only "${RES[@]}" \
    --m "$M" --k "$K" --limit "$LIMIT" --shard_idx $sh --num_shards "$NUM_SHARDS" \
    > "$W/logs/track1_shard$(printf %02d $sh)-${SLURM_JOB_ID}.log" 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
tail -n 3 "$W"/logs/track1_shard*-"${SLURM_JOB_ID}".log
[[ "$fail" == 0 ]] || { echo "[FATAL] a track-1 shard failed" >&2; exit 1; }
echo "[$(date)] track 1 done"
