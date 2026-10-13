#!/bin/bash
#SBATCH --job-name=fsl_t2
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=01:00:00

# factorized_step_labels_v1 track 2 (necessity, activation patching): GPU 0 first
# writes the position-matched reference means at every layer, then the fork split
# is sharded across the 4 GPUs (docs/factorized_step_labels_v1_plan.md section 5).
set -euo pipefail
: "${SNAP:?code snapshot}" "${R:?project result dir}" "${W:?scratch work dir}"
SPLIT="$R/split/${SPLIT_NAME:-fork_split_v1.json}"
REF="$R/split/reference_set_v1.json"
HF_CACHE="${HF_CACHE:-/project/aip-azouaq/$USER/hf_cache}"
LAYERS="${LAYERS:-1 9 18 27 35}"; LIMIT="${LIMIT:-0}"; REF_LIMIT="${REF_LIMIT:-0}"
OUT="$W/track2"
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
sha256sum "$SPLIT" "$REF"

CUDA_VISIBLE_DEVICES=0 python scripts/factorized_step_labels/track2_patch.py --mode ref \
  --reference_set "$REF" --out_dir "$OUT" --layers $LAYERS --limit "$REF_LIMIT" --local_files_only \
  2>&1 | tee "$W/logs/track2_ref-${SLURM_JOB_ID}.log" | grep -v "^Loading"
pids=()
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g python scripts/factorized_step_labels/track2_patch.py --mode patch \
    --fork_split "$SPLIT" --ref_means "$OUT/ref_means.pt" --out_dir "$OUT" --layers $LAYERS \
    --limit "$LIMIT" --shard_idx $g --num_shards 4 --local_files_only \
    > "$W/logs/track2_shard0$g-${SLURM_JOB_ID}.log" 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
tail -n 3 "$W"/logs/track2_shard0*-"${SLURM_JOB_ID}".log
[[ "$fail" == 0 ]] || { echo "[FATAL] a track-2 shard failed" >&2; exit 1; }
echo "[$(date)] track 2 done"
