#!/bin/bash
# Submit context_ablation_v3 from the TamIA login node: build -> 4 context jobs in parallel -> eval.
#   bash submit_context_ablation_v3.sh <snapshot_dir>
set -euo pipefail
SNAP="${1:?snapshot dir}"
R=/project/aip-azouaq/$USER/cot_mech/context_ablation_v3
mkdir -p "$R/slurm_out"; cd "$R/slurm_out"
bd=$(sbatch --parsable --export=ALL,SNAP=$SNAP,R=$R "$SNAP/slurm/context_ablation_v3_build_tamia.sh")
jobs=""
for ctx in full prev1 q none; do
  j=$(sbatch --parsable --dependency=afterok:$bd --time=03:00:00 --job-name=cav3_$ctx \
    --export=ALL,SNAP=$SNAP,MANIFEST=$R/views/$ctx,OUT_ROOT=$R/prod,FITS_FILE=$SNAP/experiments/context_ablation_v3/$ctx.fits,NUM_SHARDS=16,PROCS_PER_GPU=2 \
    "$SNAP/slurm/bidirectional_token_probe_train_tamia.sh")
  jobs="$jobs:$j"
done
ev=$(sbatch --parsable --dependency=afterok$jobs --export=ALL,SNAP=$SNAP,R=$R "$SNAP/slurm/context_ablation_v3_eval_tamia.sh")
echo "build=$bd ctx_jobs=${jobs#:} eval=$ev"
