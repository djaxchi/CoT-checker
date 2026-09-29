#!/bin/bash
#SBATCH --job-name=verifier_repair_v1
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
: "${PROJECT_ROOT:?Set isolated snapshot}"
: "${REPAIR_OUT:?Set fresh persistent output directory}"
module purge
module load StdEnv/2023 gcc arrow/24.0.0 python/3.12
export HF_HOME=/project/aip-azouaq/$USER/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
[[ ! -e "$REPAIR_OUT" ]] || { echo 'Output already exists' >&2; exit 2; }
mkdir -p "$REPAIR_OUT"
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index 'torch==2.14.0' 'transformers==5.14.1' 'numpy==2.5.3' 'pyyaml==6.0.3'
pip freeze > "$REPAIR_OUT/environment.txt"
cd "$PROJECT_ROOT"
BUNDLE="$PROJECT_ROOT/experiments/verifier_repair_v1"
SCRIPT="$PROJECT_ROOT/scripts/analysis/verifier_repair_experiment.py"
python "$SCRIPT" validate --bundle "$BUNDLE"
cp "$BUNDLE/config.json" "$BUNDLE/manifest.json" "$REPAIR_OUT/"
sha256sum "$SCRIPT" src/analysis/verifier_repair.py scripts/encode_prm800k_token_store.py scripts/encode_prm800k_hidden_states.py > "$REPAIR_OUT/source_sha256.txt"
cp SOURCE_REVISION "$REPAIR_OUT/source_revision.txt"
date -u +%FT%TZ > "$REPAIR_OUT/started_at.txt"
pids=()
for gpu in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$gpu" python "$SCRIPT" extract --bundle "$BUNDLE" --out "$REPAIR_OUT" --shard-idx "$gpu" > "$REPAIR_OUT/extract_$gpu.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
[[ "$failed" == 0 ]] || { echo 'Extraction failed; inspect logs' >&2; exit 1; }
date -u +%FT%TZ > "$REPAIR_OUT/completed_at.txt"
echo "Extraction complete: $REPAIR_OUT"
