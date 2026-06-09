#!/bin/bash
#SBATCH --job-name=s2-06-densepair
#SBATCH --account=aip-azouaq
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gpus-per-node=h100:4
#SBATCH --output=results/logs/%x-%j.out
#SBATCH --error=results/logs/%x-%j.err

# ---------------------------------------------------------------------------
# Sprint 2, Stage 06: classifier-level fork supervision (no representation).
# Train ONLY the final dense linear scorer s(h)=w.h+b on PRM800K fork pairs
# over frozen Qwen2.5-1.5B hidden states. No AE/SAE/SSAE, no BCE.
#
#   dense_rank           : pairwise rank, l_neg > l_pos by margin
#   dense_anchor_rank    : anchor-relative score loss (preserve anchor for pos,
#                          raise score above anchor for neg) - NOT a latent triplet
#   dense_rank_absmargin : absolute-margin constraints (l_pos < -1, l_neg > +1, gap)
#
# Threshold selection stays on PRM800K val_1k; ProcessBench eval is the locked
# 4-subset pipeline (fixed 0.5 / val-selected / oracle). Quick: 3 methods, one
# per GPU, concurrent; fails if any worker exits non-zero.
#
# Optional margin sweep (disabled by default): MARGIN_SWEEP=1 sbatch ... runs
# each method at rank_margin in {0.5, 1.0, 2.0} (9 jobs, round-robin on 4 GPUs).
#
# Prereqs: stages build (forks) + 02 (encode) complete. No representation step.
#
# Submit:  sbatch scripts/tamia/jobs/s2/06_train_dense_pair_scorers.sbatch
# ---------------------------------------------------------------------------

set -euo pipefail

PROJECT_DIR="$HOME/CoT-checker"
STORE="/project/aip-azouaq/$USER"
SCRATCH_BASE="$SCRATCH/cot_mech"

: "${CACHE_DIR:=$SCRATCH_BASE/prestudy_v1/cache/qwen2_5_1_5b}"                  # probe_train_40k / val_1k
: "${ENC_DIR:=$SCRATCH_BASE/s2_forks/encoded}"
: "${FORKS_DIR:=$SCRATCH_BASE/s2_forks/data}"
: "${PB_BASE:=$SCRATCH_BASE/prestudy_v1/cache/qwen2_5_1_5b_processbench_full}"  # per-subset pb_step_*
OUT_BASE="$PROJECT_DIR/runs/s2/dense_pair_scorer"

PAIR_EPOCHS="${PAIR_EPOCHS:-20}"
BATCH="${BATCH:-512}"
THRESHOLD_GRID="${THRESHOLD_GRID:-0.005}"
PB_SUBSETS="${PB_SUBSETS:-gsm8k math olympiadbench omnimath}"
RANK_MARGIN="${RANK_MARGIN:-1.0}"
LOW_MARGIN="${LOW_MARGIN:-1.0}"
HIGH_MARGIN="${HIGH_MARGIN:-1.0}"
MARGIN_SWEEP="${MARGIN_SWEEP:-0}"
MAX_PAIRS="${MAX_PAIRS:-}"

METHODS=(dense_rank dense_anchor_rank dense_rank_absmargin)

cd "$PROJECT_DIR"
mkdir -p results/logs

module purge
module load StdEnv/2023 gcc arrow/24.0.0 python/3.11 cuda/12.2
source "$HOME/venvs/cot/bin/activate"

export HF_HOME="$STORE/hf_cache"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

LOG_DIR="$OUT_BASE/logs"
mkdir -p "$OUT_BASE" "$LOG_DIR"
rm -f "$LOG_DIR"/.fails_gpu* 2>/dev/null || true

# Build the --pb_specs argument from the requested subsets.
PB_SPECS=""
for sub in $PB_SUBSETS; do
    PB_SPECS="$PB_SPECS ${sub}:${PB_BASE}/${sub}/pb_step_h.npy:${PB_BASE}/${sub}/pb_step_meta.jsonl"
done

# Build the job list as "method:margin:tag" specs.
declare -a JOBS
if [ "$MARGIN_SWEEP" = "1" ]; then
    for m in "${METHODS[@]}"; do
        for mg in 0.5 1.0 2.0; do
            JOBS+=("${m}:${mg}:${m}_m${mg}")
        done
    done
else
    for m in "${METHODS[@]}"; do
        JOBS+=("${m}:${RANK_MARGIN}:${m}")
    done
fi

run_job() {  # gpu method margin tag
    local gpu=$1 method=$2 margin=$3 tag=$4
    local maxpairs=""
    [ -n "$MAX_PAIRS" ] && maxpairs="--max_pairs $MAX_PAIRS"

    CUDA_VISIBLE_DEVICES=$gpu python scripts/train_easy_probe_method.py \
        --method "$method" \
        --cache_dir "$CACHE_DIR" \
        --out_dir "$OUT_BASE/$tag" \
        --fork_items_h "$ENC_DIR/forks_train_items_h.npy" \
        --fork_items_meta "$ENC_DIR/forks_train_items_meta.jsonl" \
        --fork_pairs "$FORKS_DIR/forks_train_pairs.jsonl" \
        --pair_epochs "$PAIR_EPOCHS" \
        --batch_size "$BATCH" --threshold_grid "$THRESHOLD_GRID" \
        --rank_margin "$margin" --low_margin "$LOW_MARGIN" --high_margin "$HIGH_MARGIN" \
        $maxpairs \
        --pb_specs $PB_SPECS \
        > "$LOG_DIR/$tag.log" 2>&1
}

gpu_runner() {  # gpu spec1 spec2 ...
    local gpu=$1; shift
    local fails=0 spec method margin tag
    for spec in "$@"; do
        method="${spec%%:*}"
        local rest="${spec#*:}"
        margin="${rest%%:*}"
        tag="${rest#*:}"
        echo "  [GPU$gpu] start $tag (margin=$margin)"
        if run_job "$gpu" "$method" "$margin" "$tag"; then
            echo "  [GPU$gpu] OK   $tag"
        else
            echo "  [GPU$gpu] FAIL $tag  (see $LOG_DIR/$tag.log)"
            fails=$((fails + 1))
        fi
    done
    echo "$fails" > "$LOG_DIR/.fails_gpu$gpu"
    return "$fails"
}

echo "======================================================="
echo "  S2 Stage 06 : dense pair scorers (no representation)"
echo "  out=$OUT_BASE  jobs=${#JOBS[@]}  pair_epochs=$PAIR_EPOCHS batch=$BATCH"
echo "  pb_subsets=[$PB_SUBSETS]  margin_sweep=$MARGIN_SWEEP  max_pairs=${MAX_PAIRS:-<all>}"
echo "  logs -> $LOG_DIR/<tag>.log"
echo "======================================================="

# Round-robin assignment to the 4 GPUs.
declare -a G0 G1 G2 G3
idx=0
for job in "${JOBS[@]}"; do
    case $((idx % 4)) in
        0) G0+=("$job") ;;
        1) G1+=("$job") ;;
        2) G2+=("$job") ;;
        3) G3+=("$job") ;;
    esac
    idx=$((idx + 1))
done

gpu_runner 0 ${G0[@]+"${G0[@]}"} & P0=$!
gpu_runner 1 ${G1[@]+"${G1[@]}"} & P1=$!
gpu_runner 2 ${G2[@]+"${G2[@]}"} & P2=$!
gpu_runner 3 ${G3[@]+"${G3[@]}"} & P3=$!

for p in "$P0" "$P1" "$P2" "$P3"; do wait "$p" || true; done

FAILED=0
for g in 0 1 2 3; do
    FAILED=$((FAILED + $(cat "$LOG_DIR/.fails_gpu$g" 2>/dev/null || echo 0)))
done

echo ""
echo "======================================================="
echo "  Summary: FAILED=$FAILED / ${#JOBS[@]} jobs"
if [ "$FAILED" -gt 0 ]; then
    echo "  One or more jobs failed; see per-job logs above."
    exit 1
fi
for job in "${JOBS[@]}"; do
    t="${job##*:}"
    f="$OUT_BASE/$t/eval_summary.json"
    [ -f "$f" ] && echo "    $t -> $f" || echo "    $t -> MISSING eval_summary.json"
done
echo "  All jobs complete."
echo "======================================================="
