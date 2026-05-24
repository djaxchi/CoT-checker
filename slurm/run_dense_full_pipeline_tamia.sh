#!/usr/bin/env bash
# Convenience launcher: submit build -> encode (PRM + PB in parallel) -> train.
# Prints every sbatch command and does not hide logs.
#
# Usage:
#   bash slurm/run_dense_full_pipeline_tamia.sh [build|encode|train|all]
#
# Default action is 'all'. Stages depend on each other via sbatch
# --dependency=afterok.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
SLURM_DIR="$PROJECT_ROOT/slurm"

ACTION="${1:-all}"

submit() {
  local script="$1"; shift
  local extras=("$@")
  echo "[submit] sbatch ${extras[*]} $script"
  sbatch "${extras[@]}" "$script"
}

build_id=""
prm_id=""
pb_id=""
train_id=""

if [[ "$ACTION" == "build" || "$ACTION" == "all" ]]; then
  out=$(submit "$SLURM_DIR/build_prm800k_full_dense_tamia.sh" --parsable)
  build_id="$out"
  echo "build_id=$build_id"
fi

if [[ "$ACTION" == "encode" || "$ACTION" == "all" ]]; then
  dep_prm=()
  if [[ -n "$build_id" ]]; then dep_prm=(--dependency=afterok:$build_id); fi
  prm_id=$(submit "$SLURM_DIR/encode_prm800k_full_dense_tamia.sh" --parsable "${dep_prm[@]}")
  # PB encoding does not depend on the PRM build.
  pb_id=$(submit "$SLURM_DIR/encode_processbench_full_dense_tamia.sh" --parsable)
  echo "prm_encode_id=$prm_id pb_encode_id=$pb_id"
fi

if [[ "$ACTION" == "train" || "$ACTION" == "all" ]]; then
  deps=()
  if [[ -n "$prm_id" && -n "$pb_id" ]]; then
    deps=(--dependency=afterok:$prm_id:$pb_id)
  elif [[ -n "$prm_id" ]]; then
    deps=(--dependency=afterok:$prm_id)
  elif [[ -n "$pb_id" ]]; then
    deps=(--dependency=afterok:$pb_id)
  fi
  train_id=$(submit "$SLURM_DIR/train_dense_linear_full_tamia.sh" --parsable "${deps[@]}")
  echo "train_id=$train_id"
fi

echo "Submitted: build=$build_id prm_encode=$prm_id pb_encode=$pb_id train=$train_id"
echo "Monitor:  squeue -u \$USER -o '%.10i %.20j %.8T %.10M %R'"
