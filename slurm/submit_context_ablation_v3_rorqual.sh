#!/bin/bash
# context_ablation_v3 on Rorqual (per-GPU allocation, backfill-friendly). Login node:
#   bash submit_context_ablation_v3_rorqual.sh <snapshot_dir>
# Expects $RUN/views/<ctx>/, $RUN/manifest_v1/{meta}, $RUN/manifest_v2/{meta} and
# Qwen3-8B under $SCRATCH/hf_cache. DAG per context: 4 encode jobs (4 shards each)
# -> 6 fit jobs (2 sources x 3 seeds); all 24 fits -> 1 CPU eval.
set -euo pipefail
SNAP="${1:?snapshot dir}"
ACCOUNT="${ACCOUNT:-def-azouaq}"
RUN="${RUN:-$SCRATCH/cot_mech/context_ablation_v3}"
mkdir -p "$RUN/logs"
sb() { sbatch --parsable --account="$ACCOUNT" --output="$RUN/logs/%x-%j.out" "$@"; }
fits=""
for ctx in full prev1 q none; do
  enc=""
  for g in 0 1 2 3; do
    sh="$((4*g)):$((4*g+1)):$((4*g+2)):$((4*g+3))"
    t=01:00:00; [[ $ctx == full ]] && t=01:30:00
    j=$(sb --time=$t --job-name=cav3_enc_${ctx}_$g --export=ALL,SNAP=$SNAP,RUN=$RUN,CTX=$ctx,SHARDS=$sh \
          "$SNAP/slurm/context_ablation_v3_encode_rorqual.sh")
    enc="$enc:$j"
  done
  for line in 1 2 3 4 5 6; do
    j=$(sb --dependency=afterok$enc --job-name=cav3_fit_${ctx}_$line \
          --export=ALL,SNAP=$SNAP,RUN=$RUN,CTX=$ctx,LINE=$line "$SNAP/slurm/context_ablation_v3_fit_rorqual.sh")
    fits="$fits:$j"
  done
  echo "$ctx encode=${enc#:}"
done
ev=$(sb --dependency=afterok$fits --export=ALL,SNAP=$SNAP,RUN=$RUN "$SNAP/slurm/context_ablation_v3_eval_rorqual.sh")
echo "fits=${fits#:} eval=$ev"
