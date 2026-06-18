#!/bin/bash
# Fetch the S1 model-size DenseLinear 7B run (Qwen2.5-7B, oracle macro F1_PB ~0.413)
# from TamIA to the local repo for the S3 failure-mode clustering work.
#
# Pulls the probe, thresholds, configs, logs, and the ProcessBench per-step
# hidden states + meta + scores for all 4 subsets. By default it SKIPS the large
# PRM800K encode/merged shards (not needed for ProcessBench clustering); pass
# WITH_PRM=1 to include them.
#
# Usage:
#   scripts/fetch_s1ms_7b.sh                 # default fetch
#   TAMIA_USER=dchikhi scripts/fetch_s1ms_7b.sh
#   REMOTE_RUN='/project/aip-azouaq/dchikhi/CoT-checker/runs/s1_model_size_dense/qwen2_5_7b' \
#       scripts/fetch_s1ms_7b.sh             # if RUNS_ROOT was redirected to $STORE
#   WITH_PRM=1 scripts/fetch_s1ms_7b.sh      # also pull PRM800K encodings
set -euo pipefail

# Default to the `tamia` ssh-config alias (HostName tamia.alliancecan.ca).
TAMIA_HOST="${TAMIA_HOST:-tamia}"
# Default output root per slurm/s1_model_size/models.env (RUNS_ROOT). Override
# REMOTE_RUN if the sweep was launched with RUNS_ROOT pointed at $STORE.
REMOTE_RUN="${REMOTE_RUN:-\$HOME/CoT-checker/runs/s1_model_size_dense/qwen2_5_7b}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_RUN="$REPO_ROOT/runs/s1_model_size_dense/qwen2_5_7b"
mkdir -p "$LOCAL_RUN"

# The ssh alias already carries the user; prefix user@ only if TAMIA_USER is set.
REMOTE="${TAMIA_USER:+${TAMIA_USER}@}${TAMIA_HOST}"

# Sanity: locate the run on the cluster (checks $HOME and $STORE) before pulling.
echo "[fetch] locating 7B run on $REMOTE ..."
ssh "$REMOTE" 'for d in \
    $HOME/CoT-checker/runs/s1_model_size_dense/qwen2_5_7b \
    /project/aip-azouaq/$USER/CoT-checker/runs/s1_model_size_dense/qwen2_5_7b; do \
    [ -e "$d/linear_probe.pt" ] && echo "FOUND: $d"; done; \
    echo "---"; ls -d '"$REMOTE_RUN"' 2>/dev/null && ls "'"$REMOTE_RUN"'"' || true

# Pull ONLY ProcessBench performance data (processbench_eval_shards/, logs/, and
# the small root probe/threshold/config files). Everything else in the run dir is
# either PRM800K training encodings (merged/, prm800k_encode_shards/, multilayer/)
# or auxiliary experiments (pb_multilayer/, forks/, layer_sweep/, steering/) and is
# excluded by default. Opt back in: WITH_PRM=1 (training), WITH_PB_MULTILAYER=1
# (per-layer ProcessBench states, ~1.8G, for layer-wise probing).
EXCLUDES=(--exclude 'processbench_eval_shards/*/pb_step_h.npy.tmp'
          --exclude 'forks' --exclude 'layer_sweep' --exclude 'steering')
if [[ "${WITH_PRM:-0}" != "1" ]]; then
  EXCLUDES+=(--exclude 'merged' --exclude 'prm800k_encode_shards' --exclude 'multilayer')
  echo "[fetch] skipping PRM800K training encodings (set WITH_PRM=1 to include)"
fi
if [[ "${WITH_PB_MULTILAYER:-0}" != "1" ]]; then
  EXCLUDES+=(--exclude 'pb_multilayer')
  echo "[fetch] skipping per-layer ProcessBench states (set WITH_PB_MULTILAYER=1 to include)"
fi

echo "[fetch] rsync $REMOTE:$REMOTE_RUN/ -> $LOCAL_RUN/"
rsync -avz --partial --progress "${EXCLUDES[@]}" \
  "$REMOTE:$REMOTE_RUN/" "$LOCAL_RUN/"

echo "[fetch] done. Contents:"
find "$LOCAL_RUN" -maxdepth 2 -type f | sort
