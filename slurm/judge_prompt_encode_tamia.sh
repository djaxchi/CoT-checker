#!/bin/bash
#SBATCH --job-name=jp_enc
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=01:15:00

# judge_prompt_v1, TamIA encode group GROUP (0..3): shards 4*GROUP..4*GROUP+3 of
# NUM_SHARDS=16, one per GPU, under the judge prompt (docs/judge_prompt_v1_plan.md).
#
# The span store (~165 GB) does not fit TamIA's shared scratch (126 GB free on
# 2026-10-08), so spans go to node-local disk and only what the TamIA cells read
# is kept on shared storage:
#   $RUN/vec/span/<rep>/<split>/shard_XX     last_token, step_mean, boundary_stats
#   $RUN/vec/pb_span/<rep>/<subset>/shard_XX
#   $RUN/judge_token, $RUN/pb_judge_token    boundary + verdict token (small)
# The step_tokens cell, which needs the spans themselves, runs on Rorqual.
# Each shard's PB and PRM800K parts are encoded by the same process.

set -euo pipefail
: "${GROUP:?0..3}" "${RUN:?}"
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
DATA_DIR="${DATA_DIR:-$SCRATCH/cot_mech/dense_full_7b_v1/data}"
PB_DIR="${PB_DIR:-/scratch/d/dchikhi/cot-checker/processbench_full}"
HF_CACHE="${HF_CACHE:-/project/aip-azouaq/$USER/hf_cache}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
LAYER="${LAYER:-35}"
NUM_SHARDS=16
LOCAL="$SLURM_TMPDIR/jp"
mkdir -p "$LOCAL" "$RUN/logs" "$RUN/vec" "$RUN/judge_token" "$RUN/pb_judge_token"

module purge
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0
export HF_HOME="$HF_CACHE"
unset TRANSFORMERS_CACHE
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
cd "$PROJECT_ROOT"
echo "git $(git rev-parse --short HEAD) group=$GROUP model=$MODEL layer=$LAYER"
virtualenv --no-download "$SLURM_TMPDIR/env" >/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip >/dev/null
pip install --no-index torch transformers numpy 2>&1 | tail -1

common=(--model_name_or_path "$MODEL" --local_files_only --span_only --layer "$LAYER"
        --batch_size 8 --model_dtype bfloat16 --num_shards $NUM_SHARDS --prompt_style judge)
REPS=(last_token step_mean boundary_stats)

one_shard() {  # gpu shard
  local g=$1 s=$2 sh; sh=$(printf "shard_%02d" "$2")
  CUDA_VISIBLE_DEVICES=$g python scripts/encode_processbench_token_store.py \
    --raw_specs gsm8k:$PB_DIR/processbench_gsm8k.jsonl math:$PB_DIR/processbench_math.jsonl \
                olympiadbench:$PB_DIR/processbench_olympiadbench.jsonl \
                omnimath:$PB_DIR/processbench_omnimath.jsonl \
    --rep_root "$LOCAL/pb_step_spans" --judge_rep_root "$LOCAL/pb_judge_token" \
    --max_seq_len 2048 --shard_idx "$s" "${common[@]}"
  CUDA_VISIBLE_DEVICES=$g python scripts/encode_prm800k_token_store.py \
    --data_dir "$DATA_DIR" \
    --splits prm800k_val_5k.jsonl:val_5k prm800k_test_2k.jsonl:test_2k \
             prm800k_probe_train_full.jsonl:probe_train_full \
    --rep_root "$LOCAL/step_spans" --judge_rep_root "$LOCAL/judge_token" \
    --max_seq_len 2304 --shard_idx "$s" "${common[@]}"
  # vectors straight to shared storage, then the small judge shards
  python scripts/derive_vector_store.py --out_root "$RUN/vec/span" --reps "${REPS[@]}" \
    --shard_dir "$LOCAL"/step_spans/*/"$sh"
  python scripts/derive_vector_store.py --out_root "$RUN/vec/pb_span" --reps "${REPS[@]}" \
    --shard_dir "$LOCAL"/pb_step_spans/*/"$sh"
  for kind in judge_token pb_judge_token; do
    for d in "$LOCAL/$kind"/*/"$sh"; do
      split=$(basename "$(dirname "$d")")
      mkdir -p "$RUN/$kind/$split"
      rm -rf "$RUN/$kind/$split/.tmp_$sh"
      cp -r "$d" "$RUN/$kind/$split/.tmp_$sh"
      rm -rf "$RUN/$kind/$split/$sh"
      mv "$RUN/$kind/$split/.tmp_$sh" "$RUN/$kind/$split/$sh"
    done
  done
  rm -rf "$LOCAL"/*/*/"$sh"
}

pids=()
for g in 0 1 2 3; do
  s=$((4 * GROUP + g))
  one_shard "$g" "$s" > "$RUN/logs/encode_shard$(printf %02d $s)-${SLURM_JOB_ID}.log" 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
if [[ "$fail" == 1 ]]; then tail -20 "$RUN"/logs/encode_shard*-"${SLURM_JOB_ID}".log >&2; exit 1; fi
du -sh "$RUN"/vec/* "$RUN"/judge_token "$RUN"/pb_judge_token
echo "[$(date)] encode group $GROUP done"
