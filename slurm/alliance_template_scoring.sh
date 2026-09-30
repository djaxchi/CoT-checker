#!/bin/bash
#SBATCH --job-name=tmpl_score
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

# Score a saved pool with every probe under the verifier template it was trained
# on, one GPU per job (Alliance per-GPU clusters). Same code path as the
# leaderboard: a span store of each step, then score_cells_on_split.
#   TASK=encode   verifier-template span store for SHARD of the pool's steps
#   TASK=score    every cell in CELLS_FILE on the store (one representation
#                 group per job via CELL_REGEX), per dataset
#   TASK=convert  pb_step_scores -> frontier score files; fails on any missing trace
#   TASK=pbgate   the same encode+score path on ProcessBench gsm8k (REF holds the raw
#                 file and each cell's training-time pb_step_scores_gsm8k.jsonl);
#                 reports how closely it reproduces them. POOL is the gate's workdir.
# NO INTERNET on compute nodes.

set -euo pipefail
: "${TASK:?}" "${POOL:?}"
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
unset TRANSFORMERS_CACHE
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
STORE="${STORE:-$POOL/store}"; OUT="$POOL/template_cells"
mkdir -p "$POOL/logs" "$POOL/scores" "$OUT"
cd "$PROJECT_ROOT"
echo "git $(git rev-parse --short HEAD) task $TASK pool $POOL"
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch transformers numpy sympy pyyaml 2>&1 | tail -1

case "$TASK" in
encode)
  # Each shard builds its own (deterministic) copy of the traces, so concurrent
  # shards never read a file another one is rewriting.
  TR="$POOL/traces/shard$SHARD"; mkdir -p "$TR"; SPECS=()
  for STEM in tts_gsm8k tts_math500; do
    python scripts/onpolicy/build_pb_traces.py \
      --trajectories "$POOL"/"$STEM".shard*_trajectories.jsonl --out_dir "$TR" --stem "$STEM" \
      --unlabelled "$TR/${STEM}_traces.jsonl" --min_steps 1 --force | tail -2
    SPECS+=("${STEM}:$TR/${STEM}_traces.jsonl")
  done
  python scripts/encode_processbench_token_store.py --raw_specs "${SPECS[@]}" --rep_root "$STORE" \
    --model_name_or_path Qwen/Qwen3-8B --local_files_only --span_only --prompt_style verifier \
    --layer 35 --max_seq_len "${MAX_SEQ_LEN:-20480}" --batch_size 64 --max_batch_tokens 32768 --sort_by_length \
    --model_dtype bfloat16 --shard_idx "${SHARD:?}" --num_shards "${NUM_SHARDS:?}" 2>&1 | tee "$POOL/logs/encode_$SHARD.log" | tail -4
  if grep -q "\[pb_tokstore\] skip" "$POOL/logs/encode_$SHARD.log"; then echo "[FATAL] steps skipped" >&2; exit 1; fi ;;
score)
  mapfile -t CELLS < <(grep -v '^#' "${CELLS_FILE:?}" | sed '/^\s*$/d' | grep -E "${CELL_REGEX:-.}")
  for STEM in tts_gsm8k tts_math500; do
    python scripts/onpolicy/score_cells_on_split.py --cells "${CELLS[@]}" --split_dir "$STORE/$STEM" \
      --split_name "$STEM" --out_dir "$OUT" --summary "$OUT/summary_${STEM}_${SLURM_JOB_ID:-x}.json"
  done ;;
convert)
  for STEM in tts_gsm8k tts_math500; do
    python scripts/onpolicy/pb_scores_to_tts.py --cells_root "$OUT" --split "$STEM" \
      --trajectories "$POOL"/"$STEM".shard*_trajectories.jsonl --stem "$STEM" --out_dir "$POOL/scores"
  done ;;
pbgate)
  : "${REF:?}" "${CELLS_FILE:?}"
  python scripts/encode_processbench_token_store.py --raw_specs "gsm8k:$REF/processbench_gsm8k.jsonl" \
    --rep_root "$STORE" --model_name_or_path Qwen/Qwen3-8B --local_files_only --span_only \
    --prompt_style verifier --layer 35 --max_seq_len "${MAX_SEQ_LEN:-20480}" --batch_size 64 \
    --max_batch_tokens 32768 --sort_by_length --model_dtype bfloat16 2>&1 | grep -v CUDACaching | tail -3
  mapfile -t CELLS < <(grep -v '^#' "$CELLS_FILE" | sed '/^\s*$/d')
  python scripts/onpolicy/score_cells_on_split.py --cells "${CELLS[@]}" --split_dir "$STORE/gsm8k" \
    --split_name gsm8k --out_dir "$OUT" --summary "$OUT/summary_gsm8k.json" | grep -v "^\[score\] .*AUROC" | tail -2
  python scripts/onpolicy/check_scoring_reproduction.py --cells_root "$REF" --rescored_root "$OUT" \
    --split gsm8k --out "$POOL/reproduction.json" ;;
esac
