# ---------------------------------------------------------------------------
# Sprint 2 4-cell matrix runner (sourced by the sanity / full launchers).
#
# Launches the 2x2 representation-shaping matrix on a single 4-GPU node, one
# method pinned per GPU, concurrently in the background, then waits and fails
# if ANY worker exits non-zero.
#
#   GPU0 -> ae_rank      GPU1 -> sae_rank
#   GPU2 -> ae_triplet   GPU3 -> sae_triplet
#
# Caller must export: PROJECT_DIR CACHE_DIR ENC_DIR FORKS_DIR PB_BASE
#                     OUT_BASE EPOCHS_SAE EPOCHS_PROBE BATCH THRESHOLD_GRID
#                     PB_SUBSETS  (space-separated)
# Optional: MAX_PAIRS OBJ_WEIGHT RANK_KIND RANK_MARGIN TRIPLET_METRIC TRIPLET_MARGIN
# ---------------------------------------------------------------------------

: "${OBJ_WEIGHT:=1.0}"
: "${RANK_KIND:=logistic}"
: "${RANK_MARGIN:=1.0}"
: "${TRIPLET_METRIC:=l2}"
: "${TRIPLET_MARGIN:=1.0}"

LOG_DIR="$OUT_BASE/logs"
mkdir -p "$OUT_BASE" "$LOG_DIR"

# Build the --pb_specs argument from the requested subsets.
PB_SPECS=""
for sub in $PB_SUBSETS; do
    PB_SPECS="$PB_SPECS ${sub}:${PB_BASE}/${sub}/pb_step_h.npy:${PB_BASE}/${sub}/pb_step_meta.jsonl"
done

run_worker() {
    local gpu=$1 method=$2
    local extra=""
    case "$method" in
        *_rank)    extra="--rank_kind $RANK_KIND --rank_margin $RANK_MARGIN --obj_weight $OBJ_WEIGHT" ;;
        *_triplet) extra="--triplet_metric $TRIPLET_METRIC --triplet_margin $TRIPLET_MARGIN --obj_weight $OBJ_WEIGHT" ;;
    esac
    local maxpairs=""
    [ -n "${MAX_PAIRS:-}" ] && maxpairs="--max_pairs $MAX_PAIRS"

    CUDA_VISIBLE_DEVICES=$gpu python scripts/train_easy_probe_method.py \
        --method "$method" \
        --cache_dir "$CACHE_DIR" \
        --out_dir "$OUT_BASE/$method" \
        --fork_items_h "$ENC_DIR/forks_train_items_h.npy" \
        --fork_items_meta "$ENC_DIR/forks_train_items_meta.jsonl" \
        --fork_pairs "$FORKS_DIR/forks_train_pairs.jsonl" \
        --epochs_sae "$EPOCHS_SAE" --epochs_probe "$EPOCHS_PROBE" \
        --batch_size "$BATCH" --threshold_grid "$THRESHOLD_GRID" \
        $extra $maxpairs \
        --pb_specs $PB_SPECS \
        > "$LOG_DIR/$method.log" 2>&1 &
}

echo "======================================================="
echo "  S2 matrix : $OUT_BASE"
echo "  epochs_sae=$EPOCHS_SAE epochs_probe=$EPOCHS_PROBE batch=$BATCH"
echo "  pb_subsets=[$PB_SUBSETS]  max_pairs=${MAX_PAIRS:-<all>}"
echo "  logs -> $LOG_DIR/{ae_rank,sae_rank,ae_triplet,sae_triplet}.log"
echo "======================================================="

METHODS=(ae_rank sae_rank ae_triplet sae_triplet)
GPUS=(0 1 2 3)
PIDS=()
for i in 0 1 2 3; do
    run_worker "${GPUS[$i]}" "${METHODS[$i]}"
    PIDS+=($!)
    echo "  launched ${METHODS[$i]} on CUDA_VISIBLE_DEVICES=${GPUS[$i]} (pid $!)"
done

FAILED=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "  OK   ${METHODS[$i]}"
    else
        echo "  FAIL ${METHODS[$i]}  (see $LOG_DIR/${METHODS[$i]}.log)"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "======================================================="
echo "  Summary: FAILED=$FAILED"
if [ "$FAILED" -gt 0 ]; then
    echo "  One or more workers failed; see per-method logs above."
    exit 1
fi
echo "  All 4 cells complete. Outputs under $OUT_BASE/<method>/"
for m in "${METHODS[@]}"; do
    f="$OUT_BASE/$m/eval_summary.json"
    [ -f "$f" ] && echo "    $m: $f" || echo "    $m: MISSING eval_summary.json"
done
echo "======================================================="
