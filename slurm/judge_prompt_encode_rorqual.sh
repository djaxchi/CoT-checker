#!/bin/bash
#SBATCH --job-name=jp_enc_r
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00

# judge_prompt_v1 on Rorqual (per-GPU allocation): encode shard SHARD of 16 under
# the judge prompt, ProcessBench and PRM800K, into the shared span stores that
# the step_tokens cell reads. Same encoder, model, layer and prompt as TamIA's
# encode jobs; the cells' input fingerprints record what each side read.
set -euo pipefail
: "${SHARD:?0..15}" "${RUN:?}"
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
DATA_DIR="${DATA_DIR:-$SCRATCH/cot_mech/prm_backbone_v1/data}"
PB_DIR="${PB_DIR:-$SCRATCH/cot_mech/prm_backbone_v1/processbench}"
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
unset TRANSFORMERS_CACHE
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
cd "$PROJECT_ROOT"
echo "git $(git rev-parse --short HEAD) shard=$SHARD"
virtualenv --no-download "$SLURM_TMPDIR/env" >/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip >/dev/null
pip install --no-index torch transformers numpy 2>&1 | tail -1
R="$RUN/repstore"
common=(--model_name_or_path Qwen/Qwen3-8B --local_files_only --span_only --layer 35
        --batch_size 8 --model_dtype bfloat16 --num_shards 16 --shard_idx "$SHARD"
        --prompt_style judge)
python scripts/encode_processbench_token_store.py \
  --raw_specs gsm8k:$PB_DIR/processbench_gsm8k.jsonl math:$PB_DIR/processbench_math.jsonl \
              olympiadbench:$PB_DIR/processbench_olympiadbench.jsonl \
              omnimath:$PB_DIR/processbench_omnimath.jsonl \
  --rep_root "$R/pb_step_spans" --judge_rep_root "$R/pb_judge_token" \
  --max_seq_len 2048 "${common[@]}"
python scripts/encode_prm800k_token_store.py --data_dir "$DATA_DIR" \
  --splits prm800k_val_5k.jsonl:val_5k prm800k_test_2k.jsonl:test_2k \
           prm800k_probe_train_full.jsonl:probe_train_full \
  --rep_root "$R/step_spans" --judge_rep_root "$R/judge_token" \
  --max_seq_len 2304 "${common[@]}"
echo "[$(date)] shard $SHARD done"
