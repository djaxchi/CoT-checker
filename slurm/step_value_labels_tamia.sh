#!/bin/bash
#SBATCH --job-name=svl
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=01:00:00

# step_value_labels_v1: k free rollouts from every cut point of the frozen trace
# set, sharded across the node's 4 GPUs, one process per GPU
# (docs/step_value_labels_v1_plan.md section 5). Several nodes split one run with
# GROUP (0..) and NUM_SHARDS (4 x nodes): this node runs shards 4*GROUP .. +3.
# RESUME=1 continues shards cut by the walltime. SHARDS (colon-separated, at most
# 4, e.g. 6:12:14:23) runs those shard indices instead of the GROUP block. Shards go to $OUT on project
# storage (small: one JSON row per trace).
set -euo pipefail
: "${SNAP:?code snapshot}" "${R:?project result dir}" "${SET_NAME:?trace set file}" "${OUT:?output dir}"
SET="$R/split/$SET_NAME"
HF_CACHE="${HF_CACHE:-/project/aip-azouaq/$USER/hf_cache}"
K="${K:-16}"; LIMIT="${LIMIT:-0}"; GEN_BATCH="${GEN_BATCH:-192}"; CHUNK="${CHUNK:-256}"
MAX_BATCH_TOKENS="${MAX_BATCH_TOKENS:-160000}"; POOL_ROWS="${POOL_ROWS:-2048}"
GROUP="${GROUP:-0}"; NUM_SHARDS="${NUM_SHARDS:-4}"; SHARDS="${SHARDS:-}"
RES=(); [[ "${RESUME:-0}" == 1 ]] && RES=(--resume)
mkdir -p "$OUT/logs"

module purge
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0
export HF_HOME="$HF_CACHE"
unset TRANSFORMERS_CACHE
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 TORCH_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$SNAP"
grep -E "snapshot|git_head" SNAPSHOT_INFO
virtualenv --no-download "$SLURM_TMPDIR/env" >/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip >/dev/null
pip install --no-index torch transformers numpy sympy 2>&1 | tail -1
sha256sum "$SET"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv -l 60 > "$OUT/logs/gpu_util-${SLURM_JOB_ID}.csv" &

if [[ -n "$SHARDS" ]]; then LIST=(${SHARDS//:/ }); else LIST=($((4 * GROUP)) $((4 * GROUP + 1)) $((4 * GROUP + 2)) $((4 * GROUP + 3))); fi
(( ${#LIST[@]} <= 4 )) || { echo "[FATAL] at most 4 shards per node" >&2; exit 2; }
pids=()
for g in "${!LIST[@]}"; do
  sh=${LIST[$g]}
  CUDA_VISIBLE_DEVICES=$g python scripts/step_value_labels/label_values.py \
    --trace_set "$SET" --out_dir "$OUT" --local_files_only "${RES[@]}" \
    --k "$K" --gen_batch "$GEN_BATCH" --chunk "$CHUNK" --limit "$LIMIT" \
    --max_batch_tokens "$MAX_BATCH_TOKENS" --pool_rows "$POOL_ROWS" \
    --shard_idx $sh --num_shards "$NUM_SHARDS" \
    > "$OUT/logs/shard$(printf %02d $sh)-${SLURM_JOB_ID}.log" 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
tail -n 2 "$OUT"/logs/shard*-"${SLURM_JOB_ID}".log
[[ "$fail" == 0 ]] || { echo "[FATAL] a labeler shard failed" >&2; exit 1; }
echo "[$(date)] labels done"
