#!/bin/bash
#SBATCH --job-name=prg-02-post
#SBATCH --account=aip-azouaq
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=results/logs/%x-%j.out
#SBATCH --error=results/logs/%x-%j.err

# ---------------------------------------------------------------------------
# parametric_retrieval_geometry_v0 stages 2-4 (CPU only): merge shards, grade
# generations + assign retrieval classes, compute unsupervised geometry
# (incl. 1000-shuffle p-values), build the explorer payload.
#
# Submit with a dependency on the generate array (see submit_pipeline.sh).
# ---------------------------------------------------------------------------

set -euo pipefail

PROJECT_DIR="$HOME/CoT-checker"
RUN_DIR="${RUN_DIR:-runs/parametric_retrieval_geometry_v0}"
NUM_SHARDS="${NUM_SHARDS:-4}"

cd "$PROJECT_DIR"
mkdir -p results/logs

module purge
module load StdEnv/2023 gcc arrow/24.0.0 python/3.11
source "$HOME/venvs/cot/bin/activate"
export TOKENIZERS_PARALLELISM=false

echo "--- merge shards ---"
python scripts/parametric_retrieval/prg_generate.py \
    --merge --num_shards "$NUM_SHARDS" --out_dir "$RUN_DIR"

echo "--- grade + labels ---"
python scripts/parametric_retrieval/prg_grade.py --out_dir "$RUN_DIR"

echo "--- geometry ---"
python scripts/parametric_retrieval/prg_geometry.py --out_dir "$RUN_DIR" \
    --n_shuffles 1000

echo "--- explorer payload ---"
python scripts/parametric_retrieval/prg_build_explorer_payload.py \
    --out_dir "$RUN_DIR"

echo "--- done ---"
ls -la "$RUN_DIR" "$RUN_DIR/geometry" "$RUN_DIR/explorer_payload"
