#!/bin/bash
#SBATCH --job-name=gptoss_label
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=07:00:00
#SBATCH --output=%x-%j.out

# Phase 4: annotate the on-policy pool with GPT-OSS-120B under the ReProbe
# protocol. Submit once per shard with SHARD=<i> NUM_SHARDS=<n>.
#
# Sized from the smoke runs. Job 443124 measured 227 trajectories/hour at batch
# 8, max_new_tokens 1024 and reasoning_effort low, with every reply parsing; the
# earlier 270/hour at batch 2 was faster per trajectory and useless, since only
# 19% of those replies reached the harmony final channel. 5,686 trajectories at
# 227/hour is about 25 GPU-hours, so four shards land inside a 7-hour walltime
# with room for the five-minute CPU build each pays.
#
# The pool is written incorrect-first, so --shuffle mixes it once under a fixed
# seed: without it a shard would annotate wrong solutions for hours before
# reaching a correct one, and the false-alarm rate on correct trajectories is
# the label-quality number worth watching early.
#
# Every trajectory is appended as it finishes and a rerun skips what is already
# there, so a walltime kill costs the trajectory in flight.
#
# NO INTERNET on compute nodes.

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH/cot_mech/reprobe_v1}"
MODEL_PATH="${MODEL_PATH:-$SCRATCH/shared_models/gpt-oss-120b}"
TRACES="${TRACES:-$RUN_ROOT/reprobe_train_judge_traces.jsonl}"
OUT_DIR="${OUT_DIR:-$RUN_ROOT/labels}"
SHARD="${SHARD:-0}"
NUM_SHARDS="${NUM_SHARDS:-4}"
BATCH="${BATCH:-8}"
MAX_NEW="${MAX_NEW:-1024}"
REASONING="${REASONING:-low}"
MAX_MEM_GIB="${MAX_MEM_GIB:-68}"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
# torch's own suggestion from the shard 1 OOM: the failure reserved 3.24 GiB it
# could not use, which is fragmentation rather than genuine demand.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HOME="${HF_HOME:-/project/aip-azouaq/$USER/hf_cache}"
mkdir -p "$OUT_DIR" "$RUN_ROOT/logs"

[[ -d "$MODEL_PATH" ]] || { echo "[FATAL] no model at $MODEL_PATH" >&2; exit 2; }
[[ -f "$TRACES" ]] || { echo "[FATAL] no traces at $TRACES" >&2; exit 2; }

cd "$PROJECT_ROOT"
cat <<BANNER
================================================================
job        : ${SLURM_JOB_NAME:-gptoss_label}  id: ${SLURM_JOB_ID:-N/A}
git_commit : $(git rev-parse --short HEAD 2>/dev/null || echo unknown)
model      : $MODEL_PATH (offline, MXFP4 dequantised to bf16)
traces     : $TRACES ($(wc -l <"$TRACES") total)
shard      : $SHARD of $NUM_SHARDS   batch $BATCH   reasoning $REASONING
out        : $OUT_DIR/labels.shard${SHARD}.jsonl
================================================================
BANNER

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch transformers numpy accelerate triton kernels safetensors 2>&1 | tail -1

python scripts/onpolicy/judge_local_reprobe.py \
  --traces "$TRACES" \
  --out "$OUT_DIR/labels.shard${SHARD}.jsonl" \
  --report "$OUT_DIR/labels.shard${SHARD}_report.json" \
  --model_path "$MODEL_PATH" \
  --shard_idx "$SHARD" --num_shards "$NUM_SHARDS" \
  --batch_size "$BATCH" --max_new_tokens "$MAX_NEW" \
  --reasoning_effort "$REASONING" \
  --dtype bfloat16 --max_memory_gib "$MAX_MEM_GIB" \
  --dequantize_mxfp4 --cpu_then_dispatch --shuffle

echo "[$(date)] gptoss_label shard $SHARD done"
