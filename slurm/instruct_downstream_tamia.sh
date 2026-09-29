#!/bin/bash
#SBATCH --job-name=instruct_downstream
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=10:00:00
#SBATCH --output=%x-%j.out

# Downstream test-time scaling on Qwen3-8B Instruct, every leaderboard checkpoint
# plus the PRM, on the saved ten-candidate pools (docs/instruct_leaderboard_v1_plan.md,
# "Downstream comparison"). N <= 10 is what the pools hold.
#
#   stage 0  reproduction gate: rescore ProcessBench gsm8k through the scoring path
#            and compare with each checkpoint's training-time scores
#   stage 1  per pool: verifier-template span store of every step (4 GPUs, sharded,
#            length-sorted batches under a token budget)
#   stage 2  per pool: score every checkpoint (4 GPUs, cells split by representation
#            so each process derives a representation's vectors once)
#   stage 3  per pool: convert to frontier score files, which fails on any missing
#            trace or step; the store is then deleted (~70 GB each)
#   stage 4  PRM (Qwen2.5-Math-PRM-7B) on any pool that lacks it, 4 GPUs
#
# Selection rules and the frontier are CPU work on the saved atoms and run after.
# NO INTERNET on compute nodes.

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
HF_CACHE="${HF_CACHE:-/project/aip-azouaq/$USER/hf_cache}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
LAYER="${LAYER:-35}"
CELLS_ROOT="${CELLS_ROOT:-/project/aip-azouaq/$USER/cot_mech/instruct_leaderboard_v1/cells}"
PB_STORE="${PB_STORE:-$SCRATCH/cot_mech/qwen3_8b_instruct_v1/repstore/pb_step_spans}"
OUT_ROOT="${OUT_ROOT:-/project/aip-azouaq/$USER/cot_mech/instruct_downstream_v1}"
STORE_ROOT="${STORE_ROOT:-$SCRATCH/cot_mech/instruct_downstream_v1/stores}"
# name:pool_dir pairs. t10 is the primary pool, t07 the frozen sensitivity arm.
POOLS="${POOLS:-t10:$SCRATCH/cot_mech/tts_instruct_v1 t07:$SCRATCH/cot_mech/tts_instruct_t07_v1}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-8192}"
MAX_BATCH_TOKENS="${MAX_BATCH_TOKENS:-32768}"
KEEP_STORE="${KEEP_STORE:-0}"
CELLS_GLOB="${CELLS_GLOB:-*__seed4?}"   # a smoke run narrows this

export HF_HOME="$HF_CACHE"
unset TRANSFORMERS_CACHE
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT_ROOT/logs" "$STORE_ROOT"
LOG="$OUT_ROOT/logs/downstream-${SLURM_JOB_ID:-$$}.log"

cd "$PROJECT_ROOT"
echo "git_commit: $(git rev-parse HEAD)" | tee -a "$LOG"
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
# transformers < 5: the PRM's remote code breaks on 5.x (see slurm/tts_sota_tamia.sh);
# 4.x runs Qwen3 as well, so one environment serves every stage.
pip install --no-index torch "transformers<5" numpy sympy pyyaml 2>&1 | tail -1

# Checkpoints, grouped four ways by representation. Each group is one process on
# one GPU, so a representation's vectors are derived once per pool. Sequence
# cells preload a split (~35 GB each); two such groups fit the node's memory.
mapfile -t ALL < <(ls -d "$CELLS_ROOT"/$CELLS_GLOB | sort)
group () { printf '%s\n' "${ALL[@]}" | grep -E "$1" || true; }
G0=$(group '/(last_token|step_mean)__')
G1=$(group '/(step_delta|step_stats|lengthfree_geom)__')
G2=$(group '/(boundary_stats__|step_tokens__attn_query)')
G3=$(group '/step_tokens__transformer')
echo "checkpoints: $(printf '%s\n' "${ALL[@]}" | wc -l)  groups: $(echo "$G0" | wc -l) $(echo "$G1" | wc -l) $(echo "$G2" | wc -l) $(echo "$G3" | wc -l)" | tee -a "$LOG"

score_groups () {  # score_groups <split_dir> <split_name> <out_dir>
  local pids=() i=0
  for G in "$G0" "$G1" "$G2" "$G3"; do
    [[ -n "$G" ]] || { i=$((i+1)); continue; }
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$i python scripts/onpolicy/score_cells_on_split.py \
      --cells $G --split_dir "$1" --split_name "$2" --out_dir "$3" \
      --summary "$3/summary_${2}_g$i.json" >>"$LOG" 2>&1 &
    pids+=($!); i=$((i+1))
  done
  local fail=0
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  [[ "$fail" == "0" ]] || { echo "[FATAL] scoring $2 failed" >&2; tail -60 "$LOG" >&2; exit 1; }
}

echo "=== stage 0: reproduction gate on ProcessBench gsm8k ===" | tee -a "$LOG"
score_groups "$PB_STORE/gsm8k" gsm8k "$OUT_ROOT/gate"
python scripts/onpolicy/check_scoring_reproduction.py --cells_root "$CELLS_ROOT" \
  --rescored_root "$OUT_ROOT/gate" --split gsm8k --out "$OUT_ROOT/gate/reproduction.json" \
  | tee -a "$LOG" | tail -12

for P in $POOLS; do
  NAME="${P%%:*}"; POOL="${P#*:}"; OUT="$OUT_ROOT/$NAME"; STORE="$STORE_ROOT/$NAME"
  mkdir -p "$OUT/scores"
  echo "=== pool $NAME: $POOL ===" | tee -a "$LOG"

  echo "--- stage 1: verifier-template span store ---" | tee -a "$LOG"
  SPECS=()
  for STEM in tts_gsm8k tts_math500; do
    python scripts/onpolicy/build_pb_traces.py --trajectories "$POOL"/"$STEM".shard*_trajectories.jsonl \
      --out_dir "$OUT" --stem "$STEM" --unlabelled "$OUT/${STEM}_traces.jsonl" \
      --min_steps 1 --force >>"$LOG" 2>&1
    SPECS+=("${STEM}:$OUT/${STEM}_traces.jsonl")
  done
  pids=()
  for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i python scripts/encode_processbench_token_store.py \
      --raw_specs "${SPECS[@]}" --rep_root "$STORE" \
      --model_name_or_path "$MODEL" --local_files_only --span_only --prompt_style verifier \
      --layer "$LAYER" --max_seq_len "$MAX_SEQ_LEN" --batch_size 64 \
      --max_batch_tokens "$MAX_BATCH_TOKENS" --sort_by_length --model_dtype bfloat16 \
      --shard_idx "$i" --num_shards 4 >>"$LOG" 2>&1 &
    pids+=($!)
  done
  fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
  [[ "$fail" == "0" ]] || { echo "[FATAL] encode $NAME" >&2; tail -60 "$LOG" >&2; exit 1; }
  if grep -q "\[pb_tokstore\] skip" "$LOG"; then
    echo "[FATAL] the encoder skipped steps; raise MAX_SEQ_LEN" >&2
    grep "\[pb_tokstore\] skip" "$LOG" | head >&2; exit 1
  fi
  du -sh "$STORE" | tee -a "$LOG"

  for STEM in tts_gsm8k tts_math500; do
    echo "--- stage 2: score all checkpoints on $NAME/$STEM ---" | tee -a "$LOG"
    score_groups "$STORE/$STEM" "$STEM" "$OUT/cells"
    echo "--- stage 3: convert $NAME/$STEM ---" | tee -a "$LOG"
    python scripts/onpolicy/pb_scores_to_tts.py --cells_root "$OUT/cells" --split "$STEM" \
      --trajectories "$POOL"/"$STEM".shard*_trajectories.jsonl --stem "$STEM" \
      --out_dir "$OUT/scores" | tee -a "$LOG"
  done
  [[ "$KEEP_STORE" == "1" ]] || { rm -rf "$STORE"; echo "[store] deleted $STORE" | tee -a "$LOG"; }

  echo "--- stage 4: PRM on $NAME ---" | tee -a "$LOG"
  for STEM in tts_gsm8k tts_math500; do
    if ls "$POOL"/scores/"$STEM"__prm_qwen25_math_7b.shard*.jsonl >/dev/null 2>&1; then
      cp "$POOL"/scores/"$STEM"__prm_qwen25_math_7b.shard*.jsonl "$OUT/scores/"
      echo "[prm] $NAME/$STEM: reused existing scores" | tee -a "$LOG"; continue
    fi
    REF=("$OUT"/scores/"$STEM"__step_tokens__transformer_d512_l2_f2048_h8__seed42.shard00.jsonl)
    pids=()
    for i in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=$i python scripts/onpolicy/score_traces_with_prm.py \
        --prm_name_or_path Qwen/Qwen2.5-Math-PRM-7B --local_files_only \
        --verify_against "${REF[@]}" \
        --trajectories "$POOL"/"$STEM".shard*_trajectories.jsonl \
        --shard_idx "$i" --num_shards 4 \
        --out "$OUT/scores/${STEM}__prm_qwen25_math_7b.shard${i}.jsonl" >>"$LOG" 2>&1 &
      pids+=($!)
    done
    fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
    [[ "$fail" == "0" ]] || { echo "[FATAL] PRM $NAME/$STEM" >&2; tail -60 "$LOG" >&2; exit 1; }
    echo "[prm] $NAME/$STEM scored" | tee -a "$LOG"
  done
done
echo "[$(date)] instruct_downstream done" | tee -a "$LOG"
