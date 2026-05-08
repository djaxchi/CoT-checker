#!/bin/bash
#SBATCH --job-name=ptb-robust
#SBATCH --account=aip-azouaq
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gpus=h100:4
#SBATCH --output=results/logs/%x-%j.out
#SBATCH --error=results/logs/%x-%j.err

# ---------------------------------------------------------------------------
# PTB Robustness Experiment -- full multi-variant pipeline
#
# Goal: distinguish implementation failure from representation failure.
#
# Pipeline overview:
#   Stage 0: patch solution_ids into dense .npz files (idempotent)
#   Stage 1: cache training transitions (4 parallel GPU shards, positive-only)
#   Stage 2: cache validation transitions (4 parallel GPU shards)
#   Stage 3: train 6 PTB variants in parallel (one per GPU where possible)
#             - no_l1             (transition-only baseline)
#             - dwa_original      (current behavior, l1_target=3.0)
#             - dwa_calibrated    (l1 target from calibration pass)
#             - fixed_lambda      (fixed lambda, no DWA)
#             - active_fraction   (DWA on active-dim fraction)
#             - topk              (exact k-sparse, no L1)
#   Stage 4: extract representations for all trained variants
#   Stage 5: linear probe benchmark (all representations + controls + balance sweep)
#   Stage 6: mechanistic viz (compare_mechanistic_viz.py)
#
# Submit:
#   sbatch scripts/tamia/jobs/ptb_robust.sbatch
# ---------------------------------------------------------------------------

set -euo pipefail

PROJECT_DIR="$HOME/CoT-checker"
STORE="/project/aip-azouaq/$USER"
SCRATCH_BASE="$SCRATCH/cot-checker"

CKPT_SSAE="$STORE/checkpoints/gsm8k-385k_Qwen2.5-0.5b_spar-10.pt"
DATA_TRAIN="$STORE/data/gsm8k_385K_train.json"
DATA_VALID="$STORE/data/gsm8k_385K_valid.json"

PROBE_DATA="$SCRATCH_BASE/probe_data"
PB_DIR="$SCRATCH_BASE/processbench"
TRANS_DIR="$SCRATCH_BASE/probe_data/transitions"
PTB_REP_DIR="$SCRATCH_BASE/probe_data/ptb_representations_robust"

PTB_CKPT_BASE="$STORE/results/checkpoints/ptb_robust"
PTB_PROBE_DIR="$STORE/results/ptb_robust_probes"
VIZ_OUT="$STORE/results/mechanistic_comparison_ptb_robust"

cd "$PROJECT_DIR"
mkdir -p results/logs \
         "$TRANS_DIR" \
         "$PTB_REP_DIR" \
         "$PTB_CKPT_BASE" \
         "$PTB_PROBE_DIR" \
         "$VIZ_OUT"

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
echo ""

# ---------------------------------------------------------------------------
# Stage 0: Patch solution_ids (idempotent)
# ---------------------------------------------------------------------------
echo "=== [Stage 0] Patch solution_ids ==="
python scripts/add_ms_solution_ids.py \
    --npz "$PROBE_DATA/dense_eval_held_out.npz" --offset 0     --max-steps 90000
python scripts/add_ms_solution_ids.py \
    --npz "$PROBE_DATA/dense_train_full.npz"    --offset 90000 --max-steps 360000
echo ""

# ---------------------------------------------------------------------------
# Stage 1: Cache training transitions
# ---------------------------------------------------------------------------
echo "=== [Stage 1] Cache training transitions ==="
TRANS_TRAIN="$TRANS_DIR/transition_train_positive.npz"

if [ -f "$TRANS_TRAIN" ]; then
    echo "  Already exists, skipping."
else
    SHARD_PIDS=()
    for i in 0 1 2 3; do
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
        wait "${SHARD_PIDS[$i]}" || FAILED=$((FAILED+1))
    done
    [ "$FAILED" -gt 0 ] && { echo "ERROR: training shard(s) failed"; exit 1; }
    python scripts/merge_transition_shards.py \
        --inputs "$TRANS_DIR/trans_train_shard0.npz" "$TRANS_DIR/trans_train_shard1.npz" \
                 "$TRANS_DIR/trans_train_shard2.npz" "$TRANS_DIR/trans_train_shard3.npz" \
        --output "$TRANS_TRAIN"
    rm -f "$TRANS_DIR/trans_train_shard"*.npz
fi
echo ""

# ---------------------------------------------------------------------------
# Stage 2: Cache validation transitions
# ---------------------------------------------------------------------------
echo "=== [Stage 2] Cache validation transitions ==="
TRANS_VAL="$TRANS_DIR/transition_val.npz"

if [ -f "$TRANS_VAL" ]; then
    echo "  Already exists, skipping."
else
    SHARD_PIDS=()
    for i in 0 1 2 3; do
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
        wait "${SHARD_PIDS[$i]}" || FAILED=$((FAILED+1))
    done
    [ "$FAILED" -gt 0 ] && { echo "ERROR: val shard(s) failed"; exit 1; }
    python scripts/merge_transition_shards.py \
        --inputs "$TRANS_DIR/trans_val_shard0.npz" "$TRANS_DIR/trans_val_shard1.npz" \
                 "$TRANS_DIR/trans_val_shard2.npz" "$TRANS_DIR/trans_val_shard3.npz" \
        --output "$TRANS_VAL"
    rm -f "$TRANS_DIR/trans_val_shard"*.npz
fi
echo ""

# ---------------------------------------------------------------------------
# Stage 3: Train PTB variants
#
# 4 variants that answer the two research questions:
#   Q1/Q2: PTB_no_l1 vs random_bln / dense_h  (transition objective value)
#   Q3:    DWA variants vs PTB_no_l1           (sparsity effect)
#   Q3b:   TopK vs DWA variants                (DWA vs hard sparsity)
#
# GPU assignment (4 GPUs available):
#   GPU 0: no_l1            (central baseline, runs first)
#   GPU 1: dwa_calibrated   (calibrated L1 target)
#   GPU 2: active_fraction  (DWA on active-dim fraction)
#   GPU 3: topk             (exact k-sparse)
# ---------------------------------------------------------------------------
echo "=== [Stage 3] Train PTB variants ==="

_train() {
    local name=$1; local gpu=$2; shift 2
    local ckpt_dir="$PTB_CKPT_BASE/${name}"
    mkdir -p "$ckpt_dir"
    if [ -f "$ckpt_dir/best.pt" ]; then
        echo "  [$name] Checkpoint exists, skipping."
        return 0
    fi
    echo "  [$name] Training on GPU $gpu ..."
    CUDA_VISIBLE_DEVICES=$gpu python scripts/train_predictive_transition_bottleneck.py \
        --train-data    "$TRANS_TRAIN" \
        --val-data      "$TRANS_VAL" \
        --output-dir    "$ckpt_dir" \
        --d             896 \
        --epochs        30 \
        --batch-size    2048 \
        --lr            1e-3 \
        --device        cuda \
        --calib-batches 50 \
        "$@" \
        > "$ckpt_dir/train.log" 2>&1 &
    echo $!
}

# All 4 variants in parallel -- one per GPU
PID_NO_L1=$( _train "no_l1"           0  --dwa-mode no_l1                                            )
PID_DWA_C=$( _train "dwa_calibrated"  1  --dwa-mode dwa_calibrated_mean_l1                           )
PID_ACTF=$(  _train "active_fraction"  2  --dwa-mode dwa_active_fraction --l1-target-active-frac 0.25 )
PID_TOPK=$(  _train "topk"             3  --dwa-mode topk               --l1-k 50                    )

for pid in $PID_NO_L1 $PID_DWA_C $PID_ACTF $PID_TOPK; do
    [ -n "$pid" ] && wait "$pid" || true
done
echo "  All variants done."
echo ""

# Print validity summaries
echo "=== Validity reports ==="
for name in no_l1 dwa_calibrated active_fraction topk; do
    vr="$PTB_CKPT_BASE/$name/validity_report.json"
    if [ -f "$vr" ]; then
        echo "  [$name] flags: $(python3 -c "import json; d=json.load(open('$vr')); print(d['flags'])")"
    fi
done
echo ""

# ---------------------------------------------------------------------------
# Stage 4: Extract all representations
# ---------------------------------------------------------------------------
echo "=== [Stage 4] Extract representations ==="

# Build --ptb-checkpoints argument string (only the 4 focused variants)
PTB_CKPTS=""
for name in no_l1 dwa_calibrated active_fraction topk; do
    ckpt="$PTB_CKPT_BASE/$name/best.pt"
    [ -f "$ckpt" ] && PTB_CKPTS="$PTB_CKPTS $ckpt:$name"
done

python scripts/extract_predictive_transition_latents.py \
    --ptb-checkpoints $PTB_CKPTS \
    --ms-dense-eval   "$PROBE_DATA/dense_eval_held_out.npz" \
    --pb-dense        "$PB_DIR/processbench_dense_gsm8k.npz" \
    --ms-dense-train  "$PROBE_DATA/dense_train_full.npz" \
    --output-dir      "$PTB_REP_DIR" \
    --batch-size      4096 \
    --device          cpu
echo ""

# ---------------------------------------------------------------------------
# Stage 5: Linear probe benchmark
# ---------------------------------------------------------------------------
echo "=== [Stage 5] Linear probe benchmark ==="
python scripts/eval_predictive_transition_probes.py \
    --rep-dir      "$PTB_REP_DIR" \
    --dense-train  "$PROBE_DATA/dense_train_full.npz" \
    --dense-eval   "$PROBE_DATA/dense_eval_held_out.npz" \
    --pb-dense     "$PB_DIR/processbench_dense_gsm8k.npz" \
    --output-dir   "$PTB_PROBE_DIR" \
    --epochs       50 \
    --device       cpu \
    --controls \
    --balance-sweep
echo ""

# ---------------------------------------------------------------------------
# Stage 6: Mechanistic viz
# ---------------------------------------------------------------------------
echo "=== [Stage 6] Mechanistic comparison figure ==="
CONFIG_JSON="$VIZ_OUT/config.json"

_entry() {
    local label=$1 ms=$2 pb=$3
    [ -f "$ms" ] || return
    cat <<JSON
    {"label": "$label", "ms_npz": "$ms", "pb_npz": "$pb",
     "probes": [], "ms_label_key": "correctness", "pb_label_key": "step_labels"}
JSON
}

{
  echo '{"n_samples_ms": 5000, "n_samples_pb": 0, "encodings": ['
  entries=""
  # Baselines first
  e="$(_entry "dense_h"      "$PROBE_DATA/dense_eval_held_out.npz"   "$PB_DIR/processbench_dense_gsm8k.npz")"
  [ -n "$e" ] && entries+="$e,"
  e="$(_entry "dense_delta"  "$PTB_REP_DIR/dense_delta_ms_eval.npz"   "$PTB_REP_DIR/dense_delta_pb.npz")"
  [ -n "$e" ] && entries+="$e,"
  e="$(_entry "dense_concat" "$PTB_REP_DIR/dense_concat_ms_eval.npz" "$PTB_REP_DIR/dense_concat_pb.npz")"
  [ -n "$e" ] && entries+="$e,"
  e="$(_entry "random_bln"   "$PTB_REP_DIR/random_bln_ms_eval.npz"   "$PTB_REP_DIR/random_bln_pb.npz")"
  [ -n "$e" ] && entries+="$e,"
  # PTB variants (focused set only)
  for name in no_l1 dwa_calibrated active_fraction topk; do
      e="$(_entry "ptb_${name}" "$PTB_REP_DIR/ptb_${name}_ms_eval.npz" "$PTB_REP_DIR/ptb_${name}_pb.npz")"
      [ -n "$e" ] && entries+="$e,"
  done
  echo "${entries%,}"
  echo "]}"
} > "$CONFIG_JSON"

python scripts/compare_mechanistic_viz.py \
    --config     "$CONFIG_JSON" \
    --output-dir "$VIZ_OUT" \
    --reducer    pca \
    --dpi        150

echo ""
echo "======================================================="
echo "  PTB Robustness Experiment complete"
echo "======================================================="
echo "  Probe benchmark : $PTB_PROBE_DIR/summary_table.md"
echo "  Figures         : $VIZ_OUT/"
echo "  Validity reports: $PTB_CKPT_BASE/{no_l1,dwa_calibrated,active_fraction,topk}/validity_report.json"
echo "  Calibration     : $PTB_CKPT_BASE/{no_l1,dwa_calibrated,active_fraction,topk}/calibration.json"
echo ""
echo "  Fetch with:"
echo "    rsync -avz $USER@tamia.alliancecan.ca:$PTB_PROBE_DIR/ ./results/ptb_robust_probes/"
echo "    rsync -avz $USER@tamia.alliancecan.ca:$PTB_CKPT_BASE/ ./results/checkpoints/ptb_robust/"
echo "    rsync -avz $USER@tamia.alliancecan.ca:$VIZ_OUT/ ./results/mechanistic_comparison_ptb_robust/"
