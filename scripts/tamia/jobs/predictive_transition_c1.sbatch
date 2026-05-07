#!/bin/bash
#SBATCH --job-name=ptb-c1
#SBATCH --account=aip-azouaq
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gpus=h100:4
#SBATCH --output=results/logs/%x-%j.out
#SBATCH --error=results/logs/%x-%j.err

# ---------------------------------------------------------------------------
# Predictive Transition Bottleneck (PTB, C=1) — full overnight pipeline
#
# TamIA policy compliance:
#   - h100:4  — full node allocation required (no partial GPU allocation)
#   - All 4 GPUs used: stages 1-2 parallelise encoding across 4 GPUs;
#     stage 3 uses DataParallel across all 4 GPUs.
#   - Max walltime: 24h. This job is budgeted at ~18h with safety margin.
#
# Storage assignment:
#   $HOME/CoT-checker/     CODE ONLY — small, no big writes
#   $SCRATCH/cot-checker/  LARGE TEMP DATA — transition .npz shards, probe data
#                          Subject to auto-purge. Regenerable.
#   $STORE (=$PROJECT)     CHECKPOINTS + FINAL RESULTS — daily backup.
#                          All .pt files, summary tables, figures go here.
#
# Pipeline stages:
#   0.5[CPU    ~5min]  add_ms_solution_ids.py              — patch dense .npz with solution_ids
#   1  [4×GPU ~40min]  cache_transition_hidden_states.py   — 4 parallel shards (train)
#   2  [CPU    ~2min]  merge_transition_shards.py           — merge train shards
#   3  [4×GPU ~15min]  cache_transition_hidden_states.py   — val split (single, no filter)
#   4  [4×GPU ~20min]  train_predictive_transition_bottleneck.py  — DataParallel
#   5  [CPU    ~5min]  extract_predictive_transition_latents.py
#   6  [CPU   ~90min]  eval_predictive_transition_probes.py  — 6 reps × 4 seeds
#   7  [CPU    ~3min]  mechanistic_analysis.py  — PTB z_k
#   8  [CPU    ~3min]  compare_mechanistic_viz.py
#
# Prerequisites (check portail.tamia.ecpia.ca before submitting):
#   $STORE/checkpoints/gsm8k-385k_Qwen2.5-0.5b_spar-10.pt
#   $STORE/data/gsm8k_385K_train.json
#   $STORE/data/gsm8k_385K_valid.json
#   $SCRATCH/cot-checker/probe_data/dense_train_full.npz
#   $SCRATCH/cot-checker/probe_data/dense_eval_held_out.npz
#   $SCRATCH/cot-checker/probe_data/train_final.npz
#   $SCRATCH/cot-checker/probe_data/eval_held_out.npz
#   $SCRATCH/cot-checker/processbench/processbench_dense_gsm8k.npz
#   $SCRATCH/cot-checker/processbench/processbench_gsm8k.npz
#   $STORE/results/dense_linear_probe_seed{42,43,44,45}.pt
#   $STORE/results/linear_probe_seed{42,43,44,45}.pt
#
# Submit:
#   sbatch scripts/tamia/jobs/predictive_transition_c1.sbatch
#
# Fetch results afterward:
#   rsync -avz $USER@tamia.alliancecan.ca:$STORE/results/ptb_probes/ ./results/ptb_probes/
#   rsync -avz $USER@tamia.alliancecan.ca:$STORE/results/mechanistic_comparison_ptb/ ./results/mechanistic_comparison_ptb/
# ---------------------------------------------------------------------------

set -euo pipefail

PROJECT_DIR="$HOME/CoT-checker"
STORE="/project/aip-azouaq/$USER"
SCRATCH_BASE="$SCRATCH/cot-checker"

# --- Paths (follow existing project conventions) ---
CKPT_SSAE="$STORE/checkpoints/gsm8k-385k_Qwen2.5-0.5b_spar-10.pt"
DATA_TRAIN="$STORE/data/gsm8k_385K_train.json"
DATA_VALID="$STORE/data/gsm8k_385K_valid.json"

# SCRATCH: large regenerable data
PROBE_DATA="$SCRATCH_BASE/probe_data"
PB_DIR="$SCRATCH_BASE/processbench"
TRANS_DIR="$SCRATCH_BASE/probe_data/transitions"
PTB_REP_DIR="$SCRATCH_BASE/probe_data/ptb_representations"

# STORE: checkpoints + final results (backed up daily)
PTB_CKPT_DIR="$STORE/results/checkpoints/ptb_c1"
PTB_CKPT="$PTB_CKPT_DIR/best.pt"
PTB_PROBE_DIR="$STORE/results/ptb_probes"
VIZ_OUT="$STORE/results/mechanistic_comparison_ptb"

# Symlink viz into project dir so rsync from repo root works too
LOCAL_VIZ="$PROJECT_DIR/results/mechanistic_comparison_ptb"

cd "$PROJECT_DIR"
mkdir -p results/logs \
         "$TRANS_DIR" \
         "$PTB_REP_DIR" \
         "$PTB_CKPT_DIR" \
         "$PTB_PROBE_DIR" \
         "$VIZ_OUT"
ln -sfn "$VIZ_OUT" "$LOCAL_VIZ" 2>/dev/null || true

module purge
module load StdEnv/2023 gcc arrow/24.0.0 python/3.11 cuda/12.2
source "$HOME/venvs/cot/bin/activate"

export HF_HOME="$STORE/hf_cache"
export TRANSFORMERS_CACHE="$STORE/hf_cache"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
echo "=== Pre-flight checks ==="
MISSING=0
for f in "$CKPT_SSAE" "$DATA_TRAIN" "$DATA_VALID" \
          "$PROBE_DATA/dense_train_full.npz" \
          "$PROBE_DATA/dense_eval_held_out.npz" \
          "$PROBE_DATA/train_final.npz" \
          "$PROBE_DATA/eval_held_out.npz" \
          "$PB_DIR/processbench_dense_gsm8k.npz" \
          "$PB_DIR/processbench_gsm8k.npz"; do
    if [ ! -f "$f" ]; then
        echo "  ERROR: required file not found: $f"
        MISSING=$((MISSING + 1))
    fi
done
[ "$MISSING" -gt 0 ] && exit 1
echo "  All required files found."

# Warn but don't fail if existing probe checkpoints are missing
for seed in 42 43 44 45; do
    for probe in "$STORE/results/dense_linear_probe_seed${seed}.pt" \
                 "$STORE/results/linear_probe_seed${seed}.pt"; do
        [ ! -f "$probe" ] && echo "  WARNING: existing probe not found (will skip in viz): $probe"
    done
done
echo ""

# ---------------------------------------------------------------------------
# Stage 0.5: Patch solution_ids into dense .npz files (CPU, idempotent)
#
# The dense files were generated with an older generate_probe_data.py that
# did not save solution_ids / step_positions. add_ms_solution_ids.py
# re-streams Math-Shepherd (no GPU) and patches the files in-place.
# If the arrays already exist the script exits immediately.
# ---------------------------------------------------------------------------
echo "=== [Stage 0.5] Patch solution_ids into dense .npz files ==="
python scripts/add_ms_solution_ids.py \
    --npz       "$PROBE_DATA/dense_eval_held_out.npz" \
    --offset    0 \
    --max-steps 90000

python scripts/add_ms_solution_ids.py \
    --npz       "$PROBE_DATA/dense_train_full.npz" \
    --offset    90000 \
    --max-steps 360000
echo ""

# ---------------------------------------------------------------------------
# Stage 1: Cache training transitions — 4 parallel shards (one per GPU)
# ---------------------------------------------------------------------------
echo "=== [Stage 1] Cache training transitions — 4 parallel GPU shards ==="
TRANS_TRAIN="$TRANS_DIR/transition_train_positive.npz"

if [ -f "$TRANS_TRAIN" ]; then
    echo "  Already exists, skipping: $TRANS_TRAIN"
else
    SHARD_PIDS=()
    for i in 0 1 2 3; do
        echo "  GPU $i: shard $i/4 ..."
        CUDA_VISIBLE_DEVICES=$i python scripts/cache_transition_hidden_states.py \
            --checkpoint   "$CKPT_SSAE" \
            --data-file    "$DATA_TRAIN" \
            --output       "$TRANS_DIR/trans_train_shard${i}.npz" \
            --positive-only \
            --shard-idx    $i \
            --n-shards     4 \
            --batch-size   32 \
            --max-seq-len  256 \
            --device       cuda \
            > "$TRANS_DIR/shard${i}_train.log" 2>&1 &
        SHARD_PIDS+=($!)
    done

    FAILED=0
    for i in "${!SHARD_PIDS[@]}"; do
        if wait "${SHARD_PIDS[$i]}"; then
            echo "  Shard $i: OK"
        else
            echo "  ERROR: shard $i failed. Check $TRANS_DIR/shard${i}_train.log"
            FAILED=$((FAILED + 1))
        fi
    done
    [ "$FAILED" -gt 0 ] && exit 1

    echo ""
    echo "  Merging shards ..."
    python scripts/merge_transition_shards.py \
        --inputs "$TRANS_DIR/trans_train_shard0.npz" \
                 "$TRANS_DIR/trans_train_shard1.npz" \
                 "$TRANS_DIR/trans_train_shard2.npz" \
                 "$TRANS_DIR/trans_train_shard3.npz" \
        --output "$TRANS_TRAIN"

    # Remove shards to free SCRATCH space
    rm -f "$TRANS_DIR/trans_train_shard"*.npz
fi
echo ""

# ---------------------------------------------------------------------------
# Stage 2: Cache validation transitions — 4 parallel GPU shards
# ---------------------------------------------------------------------------
echo "=== [Stage 2] Cache validation transitions ==="
TRANS_VAL="$TRANS_DIR/transition_val.npz"

if [ -f "$TRANS_VAL" ]; then
    echo "  Already exists, skipping: $TRANS_VAL"
else
    SHARD_PIDS=()
    for i in 0 1 2 3; do
        echo "  GPU $i: shard $i/4 ..."
        CUDA_VISIBLE_DEVICES=$i python scripts/cache_transition_hidden_states.py \
            --checkpoint  "$CKPT_SSAE" \
            --data-file   "$DATA_VALID" \
            --output      "$TRANS_DIR/trans_val_shard${i}.npz" \
            --shard-idx   $i \
            --n-shards    4 \
            --batch-size  32 \
            --max-seq-len 256 \
            --device      cuda \
            > "$TRANS_DIR/shard${i}_val.log" 2>&1 &
        SHARD_PIDS+=($!)
    done

    FAILED=0
    for i in "${!SHARD_PIDS[@]}"; do
        if wait "${SHARD_PIDS[$i]}"; then
            echo "  Shard $i: OK"
        else
            echo "  ERROR: shard $i failed."
            FAILED=$((FAILED + 1))
        fi
    done
    [ "$FAILED" -gt 0 ] && exit 1

    echo "  Merging ..."
    python scripts/merge_transition_shards.py \
        --inputs "$TRANS_DIR/trans_val_shard0.npz" \
                 "$TRANS_DIR/trans_val_shard1.npz" \
                 "$TRANS_DIR/trans_val_shard2.npz" \
                 "$TRANS_DIR/trans_val_shard3.npz" \
        --output "$TRANS_VAL"
    rm -f "$TRANS_DIR/trans_val_shard"*.npz
fi
echo ""

# ---------------------------------------------------------------------------
# Stage 3: Train PTB — DataParallel across all 4 GPUs
# ---------------------------------------------------------------------------
echo "=== [Stage 3] Train PTB (C=1, d=896) — DataParallel x4 ==="
if [ -f "$PTB_CKPT" ]; then
    echo "  Checkpoint exists, skipping: $PTB_CKPT"
else
    python scripts/train_predictive_transition_bottleneck.py \
        --train-data   "$TRANS_TRAIN" \
        --val-data     "$TRANS_VAL" \
        --output-dir   "$PTB_CKPT_DIR" \
        --d            896 \
        --l1-target    3.0 \
        --epochs       30 \
        --batch-size   2048 \
        --lr           1e-3 \
        --device       cuda
fi
echo ""

# ---------------------------------------------------------------------------
# Stage 4: Extract all representations (CPU — no GPU needed)
# ---------------------------------------------------------------------------
echo "=== [Stage 4] Extract PTB + baseline representations ==="
python scripts/extract_predictive_transition_latents.py \
    --ptb-checkpoint "$PTB_CKPT" \
    --ms-dense-eval  "$PROBE_DATA/dense_eval_held_out.npz" \
    --pb-dense       "$PB_DIR/processbench_dense_gsm8k.npz" \
    --ms-dense-train "$PROBE_DATA/dense_train_full.npz" \
    --output-dir     "$PTB_REP_DIR" \
    --batch-size     4096 \
    --device         cpu
echo ""

# ---------------------------------------------------------------------------
# Stage 5: Linear probes on all representations
# ---------------------------------------------------------------------------
echo "=== [Stage 5] Linear probes — 6 representations × 4 seeds ==="
python scripts/eval_predictive_transition_probes.py \
    --rep-dir      "$PTB_REP_DIR" \
    --dense-train  "$PROBE_DATA/dense_train_full.npz" \
    --dense-eval   "$PROBE_DATA/dense_eval_held_out.npz" \
    --ssae-train   "$PROBE_DATA/train_final.npz" \
    --ssae-eval    "$PROBE_DATA/eval_held_out.npz" \
    --pb-dense     "$PB_DIR/processbench_dense_gsm8k.npz" \
    --pb-ssae      "$PB_DIR/processbench_gsm8k.npz" \
    --output-dir   "$PTB_PROBE_DIR" \
    --epochs       50 \
    --device       cpu
echo ""

# ---------------------------------------------------------------------------
# Stage 6: Mechanistic analysis — PTB z_k
# ---------------------------------------------------------------------------
echo "=== [Stage 6] Mechanistic analysis — PTB z_k ==="
PTB_PROBES=()
for seed in 42 43 44 45; do
    p="$PTB_PROBE_DIR/ptb_z_linear_probe_seed${seed}.pt"
    [ -f "$p" ] && PTB_PROBES+=("$p")
done

if [ ${#PTB_PROBES[@]} -gt 0 ]; then
    python scripts/mechanistic_analysis.py \
        --eval-data  "$PTB_REP_DIR/ptb_z_ms_eval.npz" \
        --train-data "$PTB_REP_DIR/ptb_z_ms_train.npz" \
        --probes     "${PTB_PROBES[@]}" \
        --output-dir "$VIZ_OUT/ptb_mechanistic"
else
    echo "  WARNING: no PTB probe checkpoints found, skipping mechanistic analysis."
fi
echo ""

# ---------------------------------------------------------------------------
# Stage 7: Full encoding comparison figure
# ---------------------------------------------------------------------------
echo "=== [Stage 7] Full encoding comparison figure ==="
CONFIG_JSON="$VIZ_OUT/config.json"

# Build probe list strings (space-separated paths → JSON array)
_probe_array() {
    local dir=$1 prefix=$2
    local arr=""
    for seed in 42 43 44 45; do
        # If prefix is non-empty, insert underscore separator; otherwise use bare name
        local name="${prefix:+${prefix}_}linear_probe_seed${seed}.pt"
        local p="$dir/$name"
        [ -f "$p" ] && arr+="\"$p\","
    done
    echo "[${arr%,}]"
}

dense_probes=$(_probe_array "$STORE/results" "dense")
ssae_probes=$(_probe_array  "$STORE/results" "")
ptb_probes=$(_probe_array   "$PTB_PROBE_DIR" "ptb_z")
delta_probes=$(_probe_array "$PTB_PROBE_DIR" "dense_delta")
concat_probes=$(_probe_array "$PTB_PROBE_DIR" "dense_concat")
rand_probes=$(_probe_array  "$PTB_PROBE_DIR" "random_bln")

# Only include encodings whose .npz files exist
_entry() {
    local label=$1 ms=$2 pb=$3 probes=$4
    [ -f "$ms" ] || return
    cat <<JSON
    {
      "label":        "$label",
      "ms_npz":       "$ms",
      "pb_npz":       "$pb",
      "probes":       $probes,
      "ms_label_key": "correctness",
      "pb_label_key": "step_labels"
    }
JSON
}

{
  echo '{"n_samples_ms": 5000, "n_samples_pb": 0, "encodings": ['
  entries=""
  add() { entries+="$1,"; }

  add "$(_entry "Dense h_k"       "$PROBE_DATA/dense_eval_held_out.npz" "$PB_DIR/processbench_dense_gsm8k.npz" "$dense_probes")"
  add "$(_entry "SSAE recon z_k"  "$PROBE_DATA/eval_held_out.npz"       "$PB_DIR/processbench_gsm8k.npz"       "$ssae_probes")"
  add "$(_entry "Dense Δh_k"      "$PTB_REP_DIR/dense_delta_ms_eval.npz" "$PTB_REP_DIR/dense_delta_pb.npz"      "$delta_probes")"
  add "$(_entry "Dense [h_k;Δh]"  "$PTB_REP_DIR/dense_concat_ms_eval.npz" "$PTB_REP_DIR/dense_concat_pb.npz"   "$concat_probes")"
  add "$(_entry "PTB z_k (C=1)"   "$PTB_REP_DIR/ptb_z_ms_eval.npz"      "$PTB_REP_DIR/ptb_z_pb.npz"           "$ptb_probes")"
  add "$(_entry "Random Bottleneck" "$PTB_REP_DIR/random_bln_ms_eval.npz" "$PTB_REP_DIR/random_bln_pb.npz"     "$rand_probes")"

  echo "${entries%,}"
  echo "]}"
} > "$CONFIG_JSON"

echo "  Config: $CONFIG_JSON"

python scripts/compare_mechanistic_viz.py \
    --config     "$CONFIG_JSON" \
    --output-dir "$VIZ_OUT" \
    --reducer    pca \
    --dpi        150

echo ""
echo "======================================================="
echo "  PTB C=1 experiment complete"
echo "======================================================="
echo ""
echo "  Summary table : $PTB_PROBE_DIR/summary_table.md"
echo "  Figures       : $VIZ_OUT/"
echo "  PTB checkpoint: $PTB_CKPT"
echo "  Train log     : $PTB_CKPT_DIR/train_log.jsonl"
echo ""
echo "  Fetch with:"
echo "    rsync -avz $USER@tamia.alliancecan.ca:$PTB_PROBE_DIR/ ./results/ptb_probes/"
echo "    rsync -avz $USER@tamia.alliancecan.ca:$VIZ_OUT/ ./results/mechanistic_comparison_ptb/"
