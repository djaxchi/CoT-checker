#!/bin/bash
# Submit the parametric_retrieval_geometry_v0 pipeline on TamIA:
#   01 generate+extract (ONE whole node, h100:4, shards in parallel)
#   -> 02 post (CPU, afterok)
#
# Prerequisites (login node):
#   - runs/parametric_retrieval_geometry_v0/metadata.parquet (committed)
#   - HF_HOME=$HF_CACHE_ROOT hf download Qwen/Qwen2.5-7B-Instruct
#
# Smoke first (recommended, ~10 min):
#   sbatch --time=00:30:00 --export=ALL,LIMIT=40 \
#       scripts/tamia/jobs/prg/01_generate.sbatch
#   then check results/logs/ and rm the shard outputs before the real run
#   (or rerun with FORCE; prg_generate refuses to overwrite otherwise).

set -euo pipefail
cd "$HOME/CoT-checker"

gen_id=$(sbatch --parsable scripts/tamia/jobs/prg/01_generate.sbatch)
echo "01_generate (whole node): $gen_id"

post_id=$(sbatch --parsable --dependency=afterok:"$gen_id" \
    scripts/tamia/jobs/prg/02_post.sbatch)
echo "02_post: $post_id (afterok:$gen_id)"
