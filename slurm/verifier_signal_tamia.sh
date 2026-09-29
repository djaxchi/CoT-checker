#!/bin/bash
#SBATCH --job-name=verifier_signal_v1
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out

# Validate the exact frozen diagnostic roster before using checkpoints.
set -euo pipefail
module purge
module load StdEnv/2023 gcc arrow/24.0.0 python/3.12

: "${PROJECT_ROOT:?Set an isolated source snapshot containing this experiment}"
: "${GRID_ROOT:?Set the completed Instruct leaderboard root}"
: "${SIGNAL_OUT:?Set a fresh persistent output directory}"
: "${PRM_STORE:?Set the original Instruct training representation store}"
: "${SIGNAL_MODEL_REVISION:?Set the verified cached Qwen snapshot revision}"
export SIGNAL_MODEL_REVISION
export HF_HOME="${HF_CACHE_ROOT:-/project/aip-azouaq/$USER/hf_cache}"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
cd "$PROJECT_ROOT"
[[ ! -e "$SIGNAL_OUT" ]] || { echo "Output already exists: $SIGNAL_OUT" >&2; exit 2; }
mkdir -p "$SIGNAL_OUT"
# Match the versions recorded by Instruct extraction job 487175.
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index 'torch==2.14.0' 'transformers==5.14.1' 'numpy==2.5.3' 'pyyaml==6.0.3'
pip freeze > "$SIGNAL_OUT/environment.txt"
printf '%s\n' "$SIGNAL_MODEL_REVISION" > "$SIGNAL_OUT/backbone_revision.txt"
date -u +%FT%TZ > "$SIGNAL_OUT/started_at.txt"
BUNDLE="${SIGNAL_BUNDLE:-$PROJECT_ROOT/experiments/verifier_signal_v1}"
SCRIPT="$PROJECT_ROOT/scripts/analysis/verifier_signal_experiment.py"

# Missing runs outside this prespecified roster do not block the diagnostic.
python "$SCRIPT" preflight --bundle "$BUNDLE" --cells-root "$GRID_ROOT/cells" \
  --reference "$GRID_ROOT/reference.json" --prm-store "$PRM_STORE" \
  > "$SIGNAL_OUT/preflight.log" 2>&1
cp "$BUNDLE/config.json" "$BUNDLE/manifest.json" "$BUNDLE/probes.cells" "$SIGNAL_OUT/"
sha256sum "$SCRIPT" src/analysis/verifier_signal.py scripts/encode_prm800k_token_store.py \
  scripts/encode_prm800k_hidden_states.py scripts/onpolicy/score_cells_on_split.py \
  scripts/train_rep_learner_cell.py scripts/derive_delta_from_token_store.py \
  src/harness/learners.py src/harness/spanloader.py src/repstore/store.py \
  > "$SIGNAL_OUT/source_sha256.txt"
if ! git rev-parse HEAD > "$SIGNAL_OUT/source_revision.txt" 2>/dev/null; then
  printf '%s\n' "${SOURCE_REVISION:-unavailable; see source_sha256.txt}" > "$SIGNAL_OUT/source_revision.txt"
fi

pids=()
for gpu in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$gpu" python "$SCRIPT" extract --bundle "$BUNDLE" \
    --out "$SIGNAL_OUT" --shard-idx "$gpu" --num-shards 4 --batch-size 8 \
    > "$SIGNAL_OUT/extract_$gpu.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
[[ "$failed" == 0 ]] || { echo "Extraction failed; inspect extract logs" >&2; exit 1; }
CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" score --bundle "$BUNDLE" --out "$SIGNAL_OUT" \
  --cells-root "$GRID_ROOT/cells" --reference "$GRID_ROOT/reference.json" \
  --prm-store "$PRM_STORE" --num-shards 4 --batch-size 128 \
  > "$SIGNAL_OUT/score.log" 2>&1
python "$SCRIPT" analyze --bundle "$BUNDLE" --out "$SIGNAL_OUT" \
  > "$SIGNAL_OUT/analysis.log" 2>&1
echo "Complete: $SIGNAL_OUT/analysis.json"
