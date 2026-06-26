#!/bin/bash
#SBATCH --job-name=s1ms_steercaus7b
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
#
# S3 Stage-5: causal validation of the correctness direction w on Qwen2.5-7B.
# Steer the residual stream by +/- alpha * s_layer * w_hat DURING generation and ask
# whether final-answer correctness moves with a dose-response (and matched-norm controls
# do not). Inject at hidden_states L20 (upstream) and L28 (deployed probe space).
#
# Pipeline:
#   1. build_steering_directions.py  per layer -> directions_L{20,28}.npz
#      (probe w; sparse_restricted; meandiff; top_pc; random x3; +surface/perplexity if
#       a row-feature file is supplied). L20 trains+saves its own linear_probe_L20.pt.
#   2. s1ms_steer_forks.py --directions_npz  Tier-0 teacher-forced fork battery (no
#      generation): pins sign/usable-alpha range + early readout-vs-behaviour read.
#   3. s1ms_steer_generate.py  Tier-1 generation, 4 GPU shards (one per H100) over the
#      flat (condition, problem) grid, then merge.
#   4. analyze_steer_causality.py  dose-response, paired repair/corrupt, fluency, dense-vs-
#      sparse, causal-vs-diagnostic -> steer_causality_report.md + PNGs.
#
# SMOKE / CALIBRATION FIRST: alpha scales by s_layer (~100), so keep it a small fraction.
#   N_PROBLEMS=20 DIRECTIONS=probe ALPHAS_GEN="-0.4 -0.2 -0.1 -0.05 0.05 0.1 0.2 0.4" \
#   MAX_NEW_TOKENS=512  -> find the band where P(correct) bends but gradeable stays high.
# Phase the budget by invoking twice: Phase A DIRECTIONS="probe sparse_restricted" with a
# full ALPHAS_GEN; Phase B DIRECTIONS="meandiff top_pc random_0 ..." at the +/- peak alphas.
#
# Knobs: TAG, LAYERS, DIRECTIONS, ALPHAS_TIER0, ALPHAS_GEN, N_PROBLEMS, N_SAMPLES,
#        TEMPERATURE, MAX_NEW_TOKENS.
set -uo pipefail

HERE="${S1MS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)}"
if [[ ! -f "$HERE/models.env" && -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  HERE="$SLURM_SUBMIT_DIR/slurm/s1_model_size"
fi
# shellcheck disable=SC1091
source "$HERE/models.env"
# shellcheck disable=SC1091
source "$HERE/_common.sh"
set +e

TAG="${TAG:-qwen2_5_7b}"
LAYERS="${LAYERS:-20 28}"
DEPLOYED_LAYER="${DEPLOYED_LAYER:-28}"
DIRECTIONS="${DIRECTIONS:-}"                 # empty = all directions in the npz
# alpha is a fraction of the median residual norm s_layer (~100 at 7B/L20), so alpha>=1
# DOUBLES the residual and destroys generation. Operating range is small fractions; the
# smoke is a calibration sweep to find where the margin bends but text stays gradeable.
ALPHAS_TIER0="${ALPHAS_TIER0:-0.05 0.1 0.2 0.4}"                      # magnitudes; +/- mirrored
ALPHAS_GEN="${ALPHAS_GEN:--0.4 -0.2 -0.1 -0.05 0.05 0.1 0.2 0.4}"     # signed; 0 baseline added
N_PROBLEMS="${N_PROBLEMS:-150}"
N_SAMPLES="${N_SAMPLES:-4}"
TEMPERATURE="${TEMPERATURE:-0.8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"

MODEL_ID="${S1MS_MODEL_ID[$TAG]}"
MDIR="$RUNS_ROOT/$TAG"
MLDIR="$MDIR/multilayer"
FORKS="$MDIR/steering/steering_forks.jsonl"
SAMPLE="$RUNS_ROOT/_forks_sample/forks_val_items.jsonl"
PROBE="$MDIR/linear_probe.pt"
STEER_DIR="$MDIR/steering"
GEN_DIR="$STEER_DIR/gen"
ANALYSIS_OUT="$RUNS_ROOT/../fork_rep_audit/$TAG/steer_causality"
LOG_DIR="$MDIR/logs"
mkdir -p "$STEER_DIR" "$GEN_DIR" "$ANALYSIS_OUT" "$LOG_DIR"
LOG="$LOG_DIR/steer_causality.log"

s1ms_env_setup
echo "[steercaus] TAG=$TAG layers=$LAYERS N_PROBLEMS=$N_PROBLEMS N_SAMPLES=$N_SAMPLES" | tee "$LOG"
for need in "$MLDIR" "$FORKS" "$SAMPLE" "$PROBE"; do
  [[ -e "$need" ]] || { echo "[steercaus] FATAL missing $need" | tee -a "$LOG" >&2; exit 1; }
done
s1ms_ensure_model_cached "$MODEL_ID" 2>&1 | tee -a "$LOG" || exit 1
s1ms_venv
pip install --no-index matplotlib scikit-learn 2>&1 | tail -2 | tee -a "$LOG"

# ---- 1. build steering directions per layer -------------------------------
NPZ_ARGS=()
for li in $LAYERS; do
  NPZ="$STEER_DIR/directions_L${li}.npz"
  echo "[steercaus] === 1/4 build directions L$li ===" | tee -a "$LOG"
  python scripts/build_steering_directions.py \
    --cache_dir "$MLDIR" --layer_index "$li" \
    --probe_path "$PROBE" --deployed_layer "$DEPLOYED_LAYER" \
    --out_path "$NPZ" 2>&1 | tee -a "$LOG"
  [[ ${PIPESTATUS[0]} -ne 0 ]] && { echo "[steercaus] FATAL build dir L$li" | tee -a "$LOG" >&2; exit 1; }
  NPZ_ARGS+=("$NPZ")
done

# ---- 2. Tier-0 fork battery per layer (1 GPU, no generation) ---------------
BATT_ARGS=()
for li in $LAYERS; do
  echo "[steercaus] === 2/4 Tier-0 fork battery L$li ===" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=0 python scripts/s1ms_steer_forks.py \
    --model_name_or_path "$MODEL_ID" --local_files_only \
    --directions_npz "$STEER_DIR/directions_L${li}.npz" --layer_index "$li" \
    --steering_forks "$FORKS" --alphas $ALPHAS_TIER0 \
    --out_dir "$STEER_DIR" 2>&1 | tee -a "$LOG"
  [[ ${PIPESTATUS[0]} -ne 0 ]] && { echo "[steercaus] FATAL battery L$li" | tee -a "$LOG" >&2; exit 1; }
  BATT_ARGS+=("$STEER_DIR/steer_forks_battery_L${li}.json")
done

# ---- 3. Tier-1 generation, 4 GPU shards -----------------------------------
echo "[steercaus] === 3/4 Tier-1 generation (4 shards) ===" | tee -a "$LOG"
DIR_FLAG=(); [[ -n "$DIRECTIONS" ]] && DIR_FLAG=(--directions $DIRECTIONS)
pids=()
for g in 0 1 2 3; do
  (
    export CUDA_VISIBLE_DEVICES="$g"
    python scripts/s1ms_steer_generate.py \
      --fork_items "$SAMPLE" --directions_npz "${NPZ_ARGS[@]}" "${DIR_FLAG[@]}" \
      --alphas $ALPHAS_GEN --out_dir "$GEN_DIR" --stem steer_gen \
      --model_name_or_path "$MODEL_ID" --local_files_only \
      --run_name "s1ms_${TAG}_steergen" \
      --n_samples "$N_SAMPLES" --temperature "$TEMPERATURE" \
      --max_new_tokens "$MAX_NEW_TOKENS" --max_problems "$N_PROBLEMS" \
      --shard_idx "$g" --n_shards 4 --force \
      2>&1 | tee "$LOG_DIR/steergen_shard${g}.log"
  ) &
  pids+=("$!")
done
fail=0
for g in "${!pids[@]}"; do
  wait "${pids[$g]}" || { echo "[steercaus] gen shard $g FAILED" | tee -a "$LOG" >&2; fail=1; }
done
[[ $fail -ne 0 ]] && { echo "[steercaus] FATAL: a generation shard failed" | tee -a "$LOG" >&2; exit 1; }

MERGED="$GEN_DIR/steer_gen_merged.jsonl"
cat "$GEN_DIR"/steer_gen_shard*.jsonl > "$MERGED"
echo "[steercaus] merged -> $MERGED ($(wc -l < "$MERGED") rows)" | tee -a "$LOG"

# ---- 4. analysis ----------------------------------------------------------
echo "[steercaus] === 4/4 analysis ===" | tee -a "$LOG"
python scripts/analyze_steer_causality.py \
  --gen "$MERGED" --battery "${BATT_ARGS[@]}" --out_dir "$ANALYSIS_OUT" 2>&1 | tee -a "$LOG"
[[ ${PIPESTATUS[0]} -ne 0 ]] && { echo "[steercaus] FATAL analysis" | tee -a "$LOG" >&2; exit 1; }

mkdir -p "$RUNS_ROOT/figures"
cp "$ANALYSIS_OUT"/*.png "$STEER_DIR"/steer_forks_battery_L*.png "$RUNS_ROOT/figures/" 2>/dev/null
echo "[steercaus] DONE. Report: cat $ANALYSIS_OUT/steer_causality_report.md" | tee -a "$LOG"
cat "$ANALYSIS_OUT/steer_causality_report.md" 2>/dev/null