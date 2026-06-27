#!/bin/bash
#SBATCH --job-name=gemma_sae_audit_9b
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=02:30:00
#SBATCH --output=%x-%j.out
#
# GemmaScope arm of the public-SAE audit (clean public-SAE test of the
# correctness hypothesis on a model with a first-class SAE suite).
# Pipeline, offline on one 4xH100 node:
#   1. extract Gemma-2-9B (base) residuals at GemmaScope layer_20 -> hs[21] (L20)
#      and layer_31 -> hs[32] (L31), 4-GPU fan-out, attn_impl=eager
#   2. merge -> merged/heldout_{L20,L31}_h.npy + y + meta
#   3. encode through GemmaScope canonical JumpReLU SAEs (width_16k)
#   4. probes: dense h / SAE z / h_hat / residual / controls / null
#   5. figures (A-E + decoder-map F) + auto report
#
# PRE-FETCH (internet node, GATED - needs HF_TOKEN + accepted Gemma licences):
#   see scripts/public_sae/download_public_sae.md (Gemma section)
#
# Usage:
#   sbatch slurm/s1_model_size/run_gemma_sae_audit_9b.sh
#   LIMIT=200 sbatch ...                       # quick smoke
#   WIDTH=width_131k sbatch ...                # bigger dictionary

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
HF_HOME="${HF_HOME:-/scratch/d/dchikhi/hf}"
HF_MODELS="${HF_MODELS:-$HF_HOME/models}"
MODEL_DIR="${MODEL_DIR:-$HF_MODELS/google/gemma-2-9b}"
SAE_ROOT="${SAE_ROOT:-$HF_MODELS/google/gemma-scope-9b-pt-res-canonical}"
DATA_DIR="${DATA_DIR:-/scratch/d/dchikhi/cot_mech/prestudy_v1/data}"
STEM="${STEM:-prm800k_heldout_test}"
JSONL="${JSONL:-$DATA_DIR/$STEM.jsonl}"
OUT="${OUT:-$PROJECT_ROOT/runs/public_sae_audit/gemma2_9b}"
WIDTH="${WIDTH:-width_16k}"
GLAYERS="${GLAYERS:-20 31}"          # GemmaScope layer indices for the two readouts
BATCH_SIZE="${BATCH_SIZE:-8}"
LIMIT="${LIMIT:-}"

export HF_HOME TRANSFORMERS_CACHE="$HF_HOME/transformers" HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$PROJECT_ROOT"

# Build the readout name <-> hidden_states index and SAE-folder maps from GLAYERS.
read -r -a GL <<< "$GLAYERS"
LAYER_NAMES=(); LAYER_MAP=(); SAE_LAYER_MAP=()
for g in "${GL[@]}"; do
  LAYER_NAMES+=("L${g}")
  LAYER_MAP+=("L${g}:$((g+1))")                              # GemmaScope layer_g = hidden_states[g+1]
  SAE_LAYER_MAP+=("L${g}:layer_${g}/${WIDTH}/canonical")
done

LOG="$OUT/logs"; SHARDS="$OUT/shards"; MERGED="$OUT/merged"
SAE_OUT="$OUT/sae"; PROBE="$OUT/probe"; FIGS="$OUT/figures"
REPORT="$PROJECT_ROOT/runs/public_sae_audit/gemma_sae_audit_report.md"
mkdir -p "$LOG" "$SHARDS" "$MERGED" "$SAE_OUT" "$PROBE" "$FIGS"

cat <<BANNER
================================================================
job        : ${SLURM_JOB_NAME:-gemma_sae_audit_9b}  id=${SLURM_JOB_ID:-NA}
model      : $MODEL_DIR
sae_root   : $SAE_ROOT  ($WIDTH)
readouts   : ${LAYER_NAMES[*]}  (layer_map ${LAYER_MAP[*]})
jsonl      : $JSONL
out        : $OUT
git_commit : $(git rev-parse HEAD 2>/dev/null || echo unknown)
================================================================
BANNER

CHECK=("$MODEL_DIR" "$JSONL")
for g in "${GL[@]}"; do CHECK+=("$SAE_ROOT/layer_${g}/${WIDTH}/canonical/params.npz"); done
for p in "${CHECK[@]}"; do
  [[ -e "$p" ]] || { echo "FATAL missing (pre-fetch it): $p" >&2; exit 1; }
done

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch transformers numpy scipy scikit-learn matplotlib

LIM=(); [[ -n "$LIMIT" ]] && LIM=(--limit "$LIMIT")

# ---- 1. extract: 4-GPU fan-out -------------------------------------------
echo "[1] extract Gemma-2-9B residuals (4-GPU fan-out, eager attn)"
pids=()
for sh in 0 1 2 3; do
  ( export CUDA_VISIBLE_DEVICES="$sh"
    python scripts/public_sae/extract_instruct_residuals.py \
      --jsonl "$JSONL" --out "$SHARDS/shard_${sh}.npz" \
      --model_name_or_path "$MODEL_DIR" --local_files_only --attn_impl eager \
      --layer_map "${LAYER_MAP[@]}" \
      --max_seq_len -1 --batch_size "$BATCH_SIZE" \
      --shard_idx "$sh" --num_shards 4 "${LIM[@]}" \
      > "$LOG/extract_shard_${sh}.log" 2>&1 ) &
  pids+=("$!")
done
fail=0; for i in "${!pids[@]}"; do wait "${pids[$i]}" || { echo "[1] shard $i FAILED" >&2; fail=1; }; done
[[ $fail -eq 0 ]] || { tail -20 "$LOG"/extract_shard_*.log >&2; exit 1; }

# ---- 2. merge -------------------------------------------------------------
echo "[2] merge shards"
python scripts/public_sae/extract_instruct_residuals.py --merge \
  --shard_dir "$SHARDS" --merged_out "$MERGED" --num_shards 4 2>&1 | tee "$LOG/merge.log"

# ---- 3. encode through GemmaScope SAEs ------------------------------------
echo "[3] encode GemmaScope JumpReLU SAEs ($WIDTH)"
CUDA_VISIBLE_DEVICES=0 python scripts/public_sae/encode_gemma_sae.py \
  --enc_dir "$MERGED" --sae_root "$SAE_ROOT" --out_dir "$SAE_OUT" \
  --layers "${LAYER_NAMES[@]}" --sae_layer_map "${SAE_LAYER_MAP[@]}" \
  --trainer 0 --batch_size 256 2>&1 | tee "$LOG/encode_sae.log"

# ---- 4. probe -------------------------------------------------------------
echo "[4] probe representations"
python scripts/public_sae/probe_public_sae.py \
  --enc_dir "$MERGED" --sae_dir "$SAE_OUT" --out_dir "$PROBE" \
  --layers "${LAYER_NAMES[@]}" --trainer 0 --jsonl "$JSONL" 2>&1 | tee "$LOG/probe.log"

# ---- 5. figures + report --------------------------------------------------
echo "[5] figures + report"
python scripts/public_sae/plot_public_sae_audit.py \
  --enc_dir "$MERGED" --sae_dir "$SAE_OUT" --probe_dir "$PROBE" \
  --out_dir "$FIGS" --report "$REPORT" --layers "${LAYER_NAMES[@]}" --trainer 0 \
  --title "GemmaScope public-SAE audit" --model_label "google/gemma-2-9b (base)" \
  --sae_label "\`google/gemma-scope-9b-pt-res-canonical\` ($WIDTH), JumpReLU" \
  --case_note "GemmaScope SAE + Gemma-2-9B base (matched). Same base-style prompt and last-token-of-step readout as the Qwen arm; GemmaScope trains on all positions/full norm dist, so the recon FVU gate tells us if our readout is in-distribution (contrast with the Qwen2.5-Instruct arm's FVU)." \
  2>&1 | tee "$LOG/plot.log"

echo "[done] report -> $REPORT ; metrics -> $PROBE/metrics.csv ; figures -> $FIGS"
