#!/bin/bash
# Shared setup + helpers for the S1 model-size DenseLinear ablation stages.
# Source this AFTER sourcing models.env. Provides:
#   s1ms_env_setup           module load + HF cache contract + offline flags
#   s1ms_venv                build the per-job virtualenv
#   s1ms_ensure_model_cached symlink scratch snapshot into the unified hub or fail
#   s1ms_resolve_pb_raw      resolve a ProcessBench subset raw file path
#   run_with_oom_retry       run a command, auto-halving __BS__ on CUDA OOM

set -euo pipefail

s1ms_env_setup() {
  module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

  # Unified HuggingFace cache contract (overridable via HF_CACHE_ROOT).
  export HF_HOME="$HF_CACHE_ROOT"
  export HF_HUB_CACHE="$HF_CACHE_ROOT/hub"
  export TRANSFORMERS_CACHE="$HF_CACHE_ROOT"
  export HF_DATASETS_CACHE="$HF_CACHE_ROOT/datasets"
  mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"

  # Fully offline on the compute nodes.
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
  export TOKENIZERS_PARALLELISM=false

  cd "$PROJECT_ROOT"
}

s1ms_venv() {
  virtualenv --no-download "$SLURM_TMPDIR/env"
  # shellcheck disable=SC1091
  source "$SLURM_TMPDIR/env/bin/activate"
  pip install --no-index --upgrade pip
  # torch+transformers+numpy for encoding; pyyaml is pulled in transitively by
  # train_easy_probe_method.py (Stage B imports its probe/threshold functions).
  pip install --no-index torch transformers numpy pyyaml
}

# Map "Qwen/Qwen2.5-1.5B" -> "models--Qwen--Qwen2.5-1.5B".
s1ms_hub_dirname() {
  local model_id="$1"
  echo "models--${model_id//\//--}"
}

# Ensure the model is present in the unified hub. If absent but a snapshot
# exists in the legacy scratch hub, symlink it in. Otherwise FAIL clearly.
s1ms_ensure_model_cached() {
  local model_id="$1"
  local name; name="$(s1ms_hub_dirname "$model_id")"
  local dst="$HF_HUB_CACHE/$name"
  if [[ -e "$dst" ]]; then
    echo "[cache] $model_id present at $dst"
    return 0
  fi
  local src="$HF_SCRATCH_HUB/$name"
  if [[ -e "$src" ]]; then
    echo "[cache] symlinking $src -> $dst"
    ln -s "$src" "$dst"
    return 0
  fi
  echo "[cache] FATAL: $model_id not found in unified hub ($dst) and no scratch" >&2
  echo "[cache] snapshot at $src to symlink. Pre-fetch the model into" >&2
  echo "[cache] $HF_HUB_CACHE before launching this model size." >&2
  exit 1
}

# Resolve a ProcessBench subset raw file under $PB_DIR.
s1ms_resolve_pb_raw() {
  local subset="$1"
  local c
  for c in "$PB_DIR/$subset.json" "$PB_DIR/$subset.jsonl" "$PB_DIR/processbench_$subset.jsonl"; do
    if [[ -f "$c" ]]; then echo "$c"; return 0; fi
  done
  echo "[pb] FATAL: no raw file for subset '$subset' under $PB_DIR" >&2
  echo "[pb] tried: $PB_DIR/$subset.json|.jsonl, $PB_DIR/processbench_$subset.jsonl" >&2
  exit 1
}

# run_with_oom_retry <start_bs> <log_file> <cmd...>
# The command must contain the literal token __BS__ where the batch size goes.
# On CUDA OOM the batch size is halved (floor 1) and the command retried.
# Non-OOM failures are NOT retried. Never truncates, never skips examples:
# only the per-forward batch size changes.
run_with_oom_retry() {
  local bs="$1"; shift
  local logf="$1"; shift
  local attempt=0
  while true; do
    attempt=$((attempt + 1))
    local tmp="${logf}.attempt${attempt}_bs${bs}"
    local cmd=( "${@/__BS__/$bs}" )
    echo "[run] attempt=$attempt bs=$bs :: ${cmd[*]}" | tee -a "$logf"
    set +e
    "${cmd[@]}" 2>&1 | tee "$tmp"
    local rc=${PIPESTATUS[0]}
    set -e
    cat "$tmp" >> "$logf"
    if [[ $rc -eq 0 ]]; then
      echo "[run] success at bs=$bs (attempt $attempt)" | tee -a "$logf"
      return 0
    fi
    if grep -qiE "out of memory|CUDA out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED" "$tmp"; then
      if [[ $bs -le 1 ]]; then
        echo "[oom] FATAL: OOM at bs=1; cannot reduce further. model=${S1MS_MODEL_ID_CUR:-?} stage=${S1MS_STAGE:-?} shard=${S1MS_SHARD:-?}" | tee -a "$logf" >&2
        return "$rc"
      fi
      local nbs=$(( bs / 2 )); [[ $nbs -lt 1 ]] && nbs=1
      echo "[oom] OOM at bs=$bs (model=${S1MS_MODEL_ID_CUR:-?} stage=${S1MS_STAGE:-?} shard=${S1MS_SHARD:-?}); retrying at bs=$nbs" | tee -a "$logf" >&2
      bs=$nbs
      continue
    fi
    echo "[run] FATAL: non-OOM failure rc=$rc (not retrying). See $tmp" | tee -a "$logf" >&2
    return "$rc"
  done
}
