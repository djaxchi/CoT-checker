#!/bin/bash
# Submit the bidirectional_token_probe_v2 DAG from the TamIA login node.
#   bash submit_bidirectional_token_probe_v2.sh <snapshot_dir> [build_dependency_jobid|none]
# build -> {smoke, search} -> {final_s42, final_s43, final_s44, self} -> diag -> eval
#                                                                self -> eval_self
# Values passed through --export must not contain spaces or commas.
set -euo pipefail
SNAP="${1:?snapshot dir}"
R=/project/aip-azouaq/$USER/cot_mech/bidirectional_token_probe_v2
X=$SNAP/experiments/bidirectional_token_probe_v2
T=$SNAP/slurm/bidirectional_token_probe_train_tamia.sh
P=$SNAP/slurm/bidirectional_token_probe_post_tamia.sh
B=ALL,SNAP=$SNAP,MANIFEST=$R/manifest_v2
mkdir -p "$R/slurm_out"; cd "$R/slurm_out"
sub() { sbatch --parsable "$@"; }

if [[ "${2:-}" == none ]]; then dep=""; else
  bd=$(sub --export=ALL,SNAP=$SNAP,R=$R "$SNAP/slurm/bidirectional_token_probe_v2_build_tamia.sh")
  dep="--dependency=afterok:$bd"; echo "build=$bd"
fi
sm=$(sub $dep --time=01:00:00 --gpus-per-node=h200:8 --cpus-per-task=64 --mem=0 --job-name=btp2_smoke \
  --export=$B,OUT_ROOT=$R/smoke,FITS_FILE=$X/smoke.fits,LIMIT_PER_SHARD=60,SMOKE_CHECKS=1 "$T")
se=$(sub $dep --time=02:00:00 --gpus-per-node=h200:8 --cpus-per-task=64 --mem=0 --job-name=btp2_search \
  --export=$B,OUT_ROOT=$R/prod,FITS_FILE=$X/search.fits,PROCS_PER_GPU=2 "$T")
fin=""
for s in 42 43 44; do
  j=$(sub --dependency=afterok:$se --time=03:00:00 --job-name=btp2_final_s$s \
    --export=$B,OUT_ROOT=$R/prod,ROSTER_FROM_SELECTION=^final_s${s}_,SEL_EXTRA_FILE=$X/final_extra.txt "$T")
  fin="$fin:$j"
done
sf=$(sub --dependency=afterok:$se --time=03:00:00 --job-name=btp2_self \
  --export=$B,OUT_ROOT=$R/prod,ROSTER_FROM_SELECTION=^self_s,SEL_PREFIX=self_s,SEL_ARMS=causal:full,SEL_EXTRA_FILE=$X/self_extra.txt,PROCS_PER_GPU=2 "$T")
dg=$(sub --dependency=afterok$fin --time=01:00:00 --gpus-per-node=h100:4 --cpus-per-task=48 --mem=0 \
  --job-name=btp2_diag --export=$B,OUT_ROOT=$R/prod,MODE=diag "$P")
ev=$(sub --dependency=afterok:$dg --time=01:30:00 --cpus-per-task=16 --mem=64G --job-name=btp2_eval \
  --export=$B,OUT_ROOT=$R/prod,MODE=eval,V2=1 "$P")
es=$(sub --dependency=afterok:$sf --time=01:30:00 --cpus-per-task=16 --mem=64G --job-name=btp2_eval_self \
  --export=$B,OUT_ROOT=$R/prod,MODE=eval,V2=1,FIT_PREFIX=self_s,RES_NAME=results_self,SKIP_REPORT=1 "$P")
echo "smoke=$sm search=$se finals=${fin#:} self=$sf diag=$dg eval=$ev eval_self=$es"
