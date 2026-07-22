#!/bin/bash
#SBATCH --job-name=prgc-02-experiment
#SBATCH --account=aip-azouaq
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --gpus-per-node=h100:4
#SBATCH --output=results/logs/%x-%j.out
#SBATCH --error=results/logs/%x-%j.err

# ---------------------------------------------------------------------------
# parametric_retrieval_component_v1: depth x component same-fact patching.
#   baseline (4 shards) -> merge
#   val      (4 shards by cell) -> select best layer per mode
#   test     (4 shards) -> analyze (control battery)
#   capture  (4 shards) -> merge (readout vectors for the spatial figure)
#
# Requires: component_states_v1 (from prgc-01), plus the existing access_v1
# artifacts (hidden_states_v1 symlink, pairs.parquet, candidates.json,
# metadata.parquet, grading.jsonl).
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

if [[ -f slurm/s1_model_size/models.env ]]; then
    source slurm/s1_model_size/models.env
fi
export HF_HOME="${HF_CACHE_ROOT:-$STORE/hf_cache}"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

EXP=scripts/parametric_retrieval/prgc_component.py
FORCEFLAG=()
[[ -n "${FORCE:-}" ]] && FORCEFLAG+=(--force)

run_shards () {  # phase
    local phase=$1 pids=() rc=0
    for s in $(seq 0 $((NUM_SHARDS - 1))); do
        CUDA_VISIBLE_DEVICES=$s python "$EXP" \
            --out_dir "$RUN_DIR" --model_name_or_path "$MODEL" \
            --local_files_only --phase "$phase" \
            --shard_idx "$s" --num_shards "$NUM_SHARDS" "${FORCEFLAG[@]}" \
            > "results/logs/prgc-02-${phase}-${SLURM_JOB_ID}_shard${s}.out" 2>&1 &
        pids+=($!)
    done
    for s in $(seq 0 $((NUM_SHARDS - 1))); do
        wait "${pids[$s]}" || { echo "$phase shard $s FAILED"; rc=1; }
    done
    return $rc
}

echo "== step 1: baseline =="
run_shards baseline
python "$EXP" --out_dir "$RUN_DIR" --phase baseline --merge_only \
    --num_shards "$NUM_SHARDS"

echo "== step 2: val depth x component sweep =="
run_shards val
python "$EXP" --out_dir "$RUN_DIR" --num_shards "$NUM_SHARDS" --select

echo "== step 3: test control battery =="
run_shards test
python "$EXP" --out_dir "$RUN_DIR" --num_shards "$NUM_SHARDS" --analyze

echo "== step 4: capture readout vectors for the figure =="
run_shards capture
python "$EXP" --out_dir "$RUN_DIR" --phase capture --merge_only \
    --num_shards "$NUM_SHARDS"

echo "ALL PRGC EXPERIMENTS COMPLETE"
