#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00

# factorized_step_labels_v1 Rorqual mirror (per-GPU allocation, backfill-friendly),
# same code, split and seeds as the TamIA pipeline:
#   TASK=t1     track 1 shard SHARD of NUM_SHARDS
#   TASK=t2ref  track 2 reference means (once, before the t2 shards)
#   TASK=t2     track 2 shard SHARD of NUM_SHARDS
#   TASK=sum    merge + gate + summary (CPU is enough; submit without a GPU)
set -euo pipefail
: "${SNAP:?}" "${R:?}" "${TASK:?}"
SHARD="${SHARD:-0}"; NUM_SHARDS="${NUM_SHARDS:-4}"; LIMIT="${LIMIT:-0}"; REF_LIMIT="${REF_LIMIT:-0}"
LAYERS="${LAYERS:-1 9 18 27 35}"
SPLIT="$R/split/fork_split_v1.json"; REF="$R/split/reference_set_v1.json"
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
unset TRANSFORMERS_CACHE
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 TORCH_NUM_THREADS=4
mkdir -p "$R/logs"
cd "$SNAP"
grep -E "snapshot|git_head" SNAPSHOT_INFO
virtualenv --no-download "$SLURM_TMPDIR/env" >/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip >/dev/null
pip install --no-index torch transformers numpy sympy 2>&1 | tail -1
case "$TASK" in
  t1)
    python scripts/factorized_step_labels/track1_generate.py --fork_split "$SPLIT" --out_dir "$R/track1" \
      --local_files_only --limit "$LIMIT" --shard_idx "$SHARD" --num_shards "$NUM_SHARDS" ;;
  t2ref)
    python scripts/factorized_step_labels/track2_patch.py --mode ref --reference_set "$REF" \
      --out_dir "$R/track2" --layers $LAYERS --limit "$REF_LIMIT" --local_files_only ;;
  t2)
    python scripts/factorized_step_labels/track2_patch.py --mode patch --fork_split "$SPLIT" \
      --ref_means "$R/track2/ref_means.pt" --out_dir "$R/track2" --layers $LAYERS --limit "$LIMIT" \
      --shard_idx "$SHARD" --num_shards "$NUM_SHARDS" --local_files_only ;;
  sum)
    python scripts/factorized_step_labels/summarize_pilot.py --fork_split "$SPLIT" \
      --track1_dir "$R/track1" --track2_dir "$R/track2" --out_dir "$R/summary" ;;
  *) echo "unknown TASK $TASK" >&2; exit 2 ;;
esac
echo "[$(date)] $TASK shard $SHARD done"
