#!/bin/bash
#SBATCH --job-name=prga-01-gen
#SBATCH --account=aip-azouaq
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --gpus-per-node=h100:4
#SBATCH --output=results/logs/%x-%j.out
#SBATCH --error=results/logs/%x-%j.err

# ---------------------------------------------------------------------------
# parametric_retrieval_access_v1 stage 1: greedy generation only (no hidden
# states). One whole node, 4 fact-sharded workers, one per GPU; fails if any
# shard fails; merges on success.
#
# Requires runs/parametric_retrieval_access_v1/metadata.parquet (from
# prga_build_prompts.py, committed/synced) and Qwen/Qwen2.5-7B-Instruct in
# $HF_CACHE_ROOT.
#
# Smoke (whole node, ~10 min):
#   sbatch --time=00:30:00 --export=ALL,LIMIT=60 \
#       scripts/tamia/jobs/prga/01_generate.sbatch
# --time above is a pre-smoke estimate; tighten on submit after the smoke.
# ---------------------------------------------------------------------------

set -euo pipefail

PROJECT_DIR="$HOME/CoT-checker"
STORE="/project/aip-azouaq/$USER"
RUN_DIR="${RUN_DIR:-runs/parametric_retrieval_access_v1}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
NUM_SHARDS="${NUM_SHARDS:-4}"

cd "$PROJECT_DIR"
mkdir -p results/logs

module purge
module load StdEnv/2023 gcc arrow/24.0.0 python/3.11 cuda/12.2
source "$HOME/venvs/cot/bin/activate"

# Qwen weights live in HF_CACHE_ROOT (models.env), not $SCRATCH/hf_cache.
if [[ -f slurm/s1_model_size/models.env ]]; then
    source slurm/s1_model_size/models.env
fi
export HF_HOME="${HF_CACHE_ROOT:-$STORE/hf_cache}"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

EXTRA=()
[[ -n "${LIMIT:-}" ]] && EXTRA+=(--limit "$LIMIT")
[[ -n "${FORCE:-}" ]] && EXTRA+=(--force)

echo "======================================================="
echo "  PRGA stage 01: greedy generation, $NUM_SHARDS shards on one node"
echo "  run dir : $RUN_DIR"
echo "  model   : $MODEL"
echo "  limit   : ${LIMIT:-none}"
echo "======================================================="

pids=()
for s in $(seq 0 $((NUM_SHARDS - 1))); do
    CUDA_VISIBLE_DEVICES=$s python scripts/parametric_retrieval/prga_generate.py \
        --out_dir "$RUN_DIR" \
        --model_name_or_path "$MODEL" \
        --local_files_only \
        --model_dtype bfloat16 \
        --shard_idx "$s" \
        --num_shards "$NUM_SHARDS" \
        "${EXTRA[@]}" \
        > "results/logs/prga-01-gen-${SLURM_JOB_ID}_shard${s}.out" 2>&1 &
    pids+=($!)
done

rc=0
for s in $(seq 0 $((NUM_SHARDS - 1))); do
    if ! wait "${pids[$s]}"; then
        echo "shard $s FAILED (see results/logs/prga-01-gen-${SLURM_JOB_ID}_shard${s}.out)"
        rc=1
    else
        echo "shard $s done"
    fi
done
[[ $rc -ne 0 ]] && exit $rc

python scripts/parametric_retrieval/prga_generate.py --merge \
    --num_shards "$NUM_SHARDS" --out_dir "$RUN_DIR"

# Malformed-output gate: downstream jobs depend on afterok, so fail loudly
# when the merged generations do not cover metadata exactly.
python - "$RUN_DIR" <<'EOF'
import json, sys
import pandas as pd
run = sys.argv[1]
meta = pd.read_parquet(f"{run}/metadata.parquet")
ids, n_empty = set(), 0
for ln in open(f"{run}/generations.jsonl"):
    if not ln.strip():
        continue
    r = json.loads(ln)
    ids.add(r["instance_id"])
    n_empty += not r["greedy_ids"]
missing = set(meta.instance_id) - ids
extra = ids - set(meta.instance_id)
assert not missing and not extra, (len(missing), len(extra))
print(f"[check] generations complete: {len(ids)} records, "
      f"{n_empty} empty greedy outputs")
assert n_empty < 0.01 * len(ids), f"too many empty generations: {n_empty}"
EOF
