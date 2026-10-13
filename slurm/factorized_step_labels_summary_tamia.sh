#!/bin/bash
#SBATCH --job-name=fsl_sum
#SBATCH --account=aip-azouaq
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

# factorized_step_labels_v1: merge both tracks, apply the no-op gate and the
# track-1 sanity check, and copy shards plus summary from $SCRATCH to $STORE.
set -euo pipefail
: "${SNAP:?code snapshot}" "${R:?project result dir}" "${W:?scratch work dir}"
module purge
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0
virtualenv --no-download "$SLURM_TMPDIR/env" >/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index numpy sympy 2>&1 | tail -1
cd "$SNAP"
DEST="$R/${DEST_NAME:-pilot}"
python scripts/factorized_step_labels/summarize_pilot.py --fork_split "$R/split/${SPLIT_NAME:-fork_split_v1.json}" \
  --track1_dir "$W/track1" --track2_dir "$W/track2" --out_dir "$W/summary"
mkdir -p "$DEST"
cp -r "$W/track1" "$W/track2" "$W/summary" "$W/logs" "$DEST/"
du -sh "$DEST"
echo "[$(date)] summary done"
