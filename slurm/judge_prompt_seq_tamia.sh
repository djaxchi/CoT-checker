#!/bin/bash
#SBATCH --job-name=jp_seq_t
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=04:00:00

# judge_prompt_v1 FALLBACK for the step_tokens x d512 cell, in case Rorqual never
# starts it. The spans do not fit TamIA's shared scratch, so this job re-encodes
# all 16 shards to node-local disk (4 per GPU, ~1 h), then runs seed 42 and,
# after it, seeds 43 and 44 side by side. Not submitted by default:
#   sbatch --export=ALL,RUN=$SCRATCH/cot_mech/judge_prompt_v1 slurm/judge_prompt_seq_tamia.sh
set -euo pipefail
: "${RUN:?}"
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
DATA_DIR="${DATA_DIR:-$SCRATCH/cot_mech/dense_full_7b_v1/data}"
PB_DIR="${PB_DIR:-/scratch/d/dchikhi/cot-checker/processbench_full}"
export HF_HOME="${HF_CACHE:-/project/aip-azouaq/$USER/hf_cache}"
LEARNER="transformer:d512,l2,f2048,h8"
L="$SLURM_TMPDIR/repstore"; mkdir -p "$L" "$RUN/logs" "$RUN/cells/span"
module purge
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0
unset TRANSFORMERS_CACHE
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
cd "$PROJECT_ROOT"
echo "git $(git rev-parse --short HEAD)"
virtualenv --no-download "$SLURM_TMPDIR/env" >/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip >/dev/null
pip install --no-index torch transformers numpy pyyaml 2>&1 | tail -1
common=(--model_name_or_path Qwen/Qwen3-8B --local_files_only --span_only --layer 35
        --batch_size 8 --model_dtype bfloat16 --num_shards 16 --prompt_style judge)
pids=()
for g in 0 1 2 3; do
  ( for s in $g $((g+4)) $((g+8)) $((g+12)); do
      CUDA_VISIBLE_DEVICES=$g python scripts/encode_processbench_token_store.py \
        --raw_specs gsm8k:$PB_DIR/processbench_gsm8k.jsonl math:$PB_DIR/processbench_math.jsonl \
                    olympiadbench:$PB_DIR/processbench_olympiadbench.jsonl \
                    omnimath:$PB_DIR/processbench_omnimath.jsonl \
        --rep_root "$L/pb_step_spans" --judge_rep_root "$L/pb_judge_token" \
        --max_seq_len 2048 --shard_idx $s "${common[@]}"
      CUDA_VISIBLE_DEVICES=$g python scripts/encode_prm800k_token_store.py --data_dir "$DATA_DIR" \
        --splits prm800k_val_5k.jsonl:val_5k prm800k_test_2k.jsonl:test_2k \
                 prm800k_probe_train_full.jsonl:probe_train_full \
        --rep_root "$L/step_spans" --judge_rep_root "$L/judge_token" \
        --max_seq_len 2304 --shard_idx $s "${common[@]}"
    done ) > "$RUN/logs/seq_tamia_encode_gpu$g-${SLURM_JOB_ID}.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
echo "[$(date)] encode done"
cell() {  # gpu seed [hp args]
  local g=$1 s=$2; shift 2
  CUDA_VISIBLE_DEVICES=$g python scripts/train_rep_learner_cell.py --rep step_tokens \
    --learner "$LEARNER" --prm_store "$L/step_spans" --pb_store "$L/pb_step_spans" \
    --out_dir "$RUN/cells/span/step_tokens__transformer_d512_l2_f2048_h8__seed$s" \
    --train_stem probe_train_full --seed $s --epochs 30 --patience 3 --batch_size 256 \
    --hp_search_cap 100000 --rescale none --preload_budget_gb 1 "$@" \
    > "$RUN/logs/seq_tamia_seed$s-${SLURM_JOB_ID}.log" 2>&1
}
H="$RUN/cells/span/step_tokens__transformer_d512_l2_f2048_h8__seed42/results.json"
[[ -f "$H" ]] || cell 0 42
cell 0 43 --hp_from "$H" & a=$!
cell 1 44 --hp_from "$H" & b=$!
wait $a; wait $b
echo "[$(date)] d512 done"
