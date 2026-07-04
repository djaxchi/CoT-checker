#!/bin/bash
# Submit the parametric_retrieval_geometry_v0 pipeline on TamIA:
#   01 generate+extract (array 0-3, one H100 each) -> 02 post (CPU, afterok)
#
# Prerequisites (login node):
#   - runs/parametric_retrieval_geometry_v0/metadata.parquet synced from local
#     (produced by scripts/parametric_retrieval/prg_sample_facts.py)
#   - HF_HOME=$HF_CACHE_ROOT huggingface-cli download Qwen/Qwen2.5-7B-Instruct
#
# Smoke first (recommended, ~10 min):
#   sbatch --array=0-0 --time=00:30:00 \
#     --export=ALL,NUM_SHARDS=4 scripts/tamia/jobs/prg/01_generate.sbatch
#   (add --limit via LIMIT env if desired, then check results/logs/)

set -euo pipefail
cd "$HOME/CoT-checker"

gen_id=$(sbatch --parsable scripts/tamia/jobs/prg/01_generate.sbatch)
echo "01_generate array: $gen_id"

post_id=$(sbatch --parsable --dependency=afterok:"$gen_id" \
    scripts/tamia/jobs/prg/02_post.sbatch)
echo "02_post: $post_id (afterok:$gen_id)"
