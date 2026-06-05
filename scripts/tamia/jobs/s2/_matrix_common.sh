# ---------------------------------------------------------------------------
# Sprint 2 job runner (sourced by the sanity / full / sweep launchers).
#
# Distributes a JOBS list across the 4 GPUs of a single node (round-robin), one
# job pinned per CUDA_VISIBLE_DEVICES, each GPU running its jobs sequentially in
# the background. Waits for all, and exits non-zero if ANY job failed (other
# jobs still run to completion, so one failure does not abort the rest).
#
# Caller must export:
#   PROJECT_DIR CACHE_DIR ENC_DIR FORKS_DIR PB_BASE OUT_BASE
#   EPOCHS_SAE EPOCHS_PROBE BATCH THRESHOLD_GRID PB_SUBSETS
#   JOBS   (bash array; entries "<method>" for *_recon controls, or
#           "<method>:<obj_weight>" for *_rank / *_triplet)
# Optional: MAX_PAIRS RANK_KIND RANK_MARGIN TRIPLET_METRIC TRIPLET_MARGIN
#
# Method -> objective: *_recon = recon-only control (objective off);
#                      *_rank / *_triplet = recon + objective at obj_weight.
# ---------------------------------------------------------------------------

: "${RANK_KIND:=logistic}"
: "${RANK_MARGIN:=1.0}"
: "${TRIPLET_METRIC:=l2}"
: "${TRIPLET_MARGIN:=1.0}"

if [ "${#JOBS[@]}" -eq 0 ]; then
    echo "ERROR: JOBS array is empty; nothing to run." >&2
    exit 2
fi

LOG_DIR="$OUT_BASE/logs"
mkdir -p "$OUT_BASE" "$LOG_DIR"
rm -f "$LOG_DIR"/.fails_gpu* 2>/dev/null || true

# Build the --pb_specs argument from the requested subsets.
PB_SPECS=""
for sub in $PB_SUBSETS; do
    PB_SPECS="$PB_SPECS ${sub}:${PB_BASE}/${sub}/pb_step_h.npy:${PB_BASE}/${sub}/pb_step_meta.jsonl"
done

job_tag() {  # method weight -> output/log tag
    local method=$1 weight=$2
    case "$method" in
        *_recon) echo "$method" ;;
        *)       echo "${method}_w${weight}" ;;
    esac
}

run_job() {  # gpu tag method weight
    local gpu=$1 tag=$2 method=$3 weight=$4
    local extra=""
    case "$method" in
        *_rank)    extra="--rank_kind $RANK_KIND --rank_margin $RANK_MARGIN --obj_weight $weight" ;;
        *_triplet) extra="--triplet_metric $TRIPLET_METRIC --triplet_margin $TRIPLET_MARGIN --obj_weight $weight" ;;
        *_recon)   extra="" ;;
    esac
    local maxpairs=""
    [ -n "${MAX_PAIRS:-}" ] && maxpairs="--max_pairs $MAX_PAIRS"

    CUDA_VISIBLE_DEVICES=$gpu python scripts/train_easy_probe_method.py \
        --method "$method" \
        --cache_dir "$CACHE_DIR" \
        --out_dir "$OUT_BASE/$tag" \
        --fork_items_h "$ENC_DIR/forks_train_items_h.npy" \
        --fork_items_meta "$ENC_DIR/forks_train_items_meta.jsonl" \
        --fork_pairs "$FORKS_DIR/forks_train_pairs.jsonl" \
        --epochs_sae "$EPOCHS_SAE" --epochs_probe "$EPOCHS_PROBE" \
        --batch_size "$BATCH" --threshold_grid "$THRESHOLD_GRID" \
        $extra $maxpairs \
        --pb_specs $PB_SPECS \
        > "$LOG_DIR/$tag.log" 2>&1
}

gpu_runner() {  # gpu spec1 spec2 ...
    local gpu=$1; shift
    local fails=0 spec method weight tag
    for spec in "$@"; do
        method="${spec%%:*}"
        weight="${spec#*:}"
        [ "$weight" = "$spec" ] && weight=""        # no ":weight" present
        tag="$(job_tag "$method" "$weight")"
        echo "  [GPU$gpu] start $tag"
        if run_job "$gpu" "$tag" "$method" "$weight"; then
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
echo "  S2 runner : $OUT_BASE"
echo "  jobs=${#JOBS[@]}  epochs_sae=$EPOCHS_SAE epochs_probe=$EPOCHS_PROBE batch=$BATCH"
echo "  pb_subsets=[$PB_SUBSETS]  max_pairs=${MAX_PAIRS:-<all>}"
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
    m="${job%%:*}"; w="${job#*:}"; [ "$w" = "$job" ] && w=""
    t="$(job_tag "$m" "$w")"
    f="$OUT_BASE/$t/eval_summary.json"
    [ -f "$f" ] && echo "    $t -> $f" || echo "    $t -> MISSING eval_summary.json"
done
echo "  All jobs complete."
echo "======================================================="
