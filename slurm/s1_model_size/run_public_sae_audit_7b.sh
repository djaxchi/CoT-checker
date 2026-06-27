#!/bin/bash
#SBATCH --job-name=public_sae_audit_7b
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#
# Public-SAE representation audit (Instruct-matched arm).
# Pipeline, all offline on one 4xH100 node:
#   1. extract Qwen2.5-7B-INSTRUCT residuals (L20=hs[20], L28=hs[28]) for the
#      PRM800K held-out steps, 4-GPU fan-out (one shard per GPU)
#   2. merge shards -> merged/heldout_{L20,L28}_h.npy + y + meta
#   3. encode through the public BatchTopK SAE (L20<-layer19, L28<-layer27, trainer 1=k64)
#   4. linear probes: dense h / SAE z / h_hat / residual / controls / null
#   5. figures (A-E + decoder-map F) + auto report
#
# PRE-FETCH (internet node) BEFORE sbatch: see scripts/public_sae/download_public_sae.md
#
# Usage:
#   sbatch slurm/s1_model_size/run_public_sae_audit_7b.sh
#   LIMIT=200 sbatch ...                      # quick smoke (first 200 steps)
#   TRAINER=2 sbatch ...                      # k=128 instead of k=64 (must be pre-fetched)

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
HF_HOME="${HF_HOME:-/scratch/d/dchikhi/hf}"
HF_MODELS="${HF_MODELS:-$HF_HOME/models}"
MODEL_DIR="${MODEL_DIR:-$HF_MODELS/Qwen/Qwen2.5-7B-Instruct}"
SAE_ROOT="${SAE_ROOT:-$HF_MODELS/andyrdt/saes-qwen2.5-7b-instruct}"
DATA_DIR="${DATA_DIR:-/scratch/d/dchikhi/cot_mech/prestudy_v1/data}"
STEM="${STEM:-prm800k_heldout_test}"
JSONL="${JSONL:-$DATA_DIR/$STEM.jsonl}"
OUT="${OUT:-$PROJECT_ROOT/runs/public_sae_audit/qwen2_5_7b_instruct}"
TRAINER="${TRAINER:-1}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LIMIT="${LIMIT:-}"

export HF_HOME TRANSFORMERS_CACHE="$HF_HOME/transformers" HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$PROJECT_ROOT"

LOG="$OUT/logs"; SHARDS="$OUT/shards"; MERGED="$OUT/merged"
SAE_OUT="$OUT/sae"; PROBE="$OUT/probe"; FIGS="$OUT/figures"
REPORT="$PROJECT_ROOT/runs/public_sae_audit/public_sae_audit_report.md"
mkdir -p "$LOG" "$SHARDS" "$MERGED" "$SAE_OUT" "$PROBE" "$FIGS"

cat <<BANNER
================================================================
job        : ${SLURM_JOB_NAME:-public_sae_audit_7b}  id=${SLURM_JOB_ID:-NA}
model      : $MODEL_DIR
sae_root   : $SAE_ROOT  (trainer_$TRAINER)
jsonl      : $JSONL
out        : $OUT
git_commit : $(git rev-parse HEAD 2>/dev/null || echo unknown)
================================================================
BANNER

for p in "$MODEL_DIR" "$SAE_ROOT/resid_post_layer_19/trainer_$TRAINER/ae.pt" \
         "$SAE_ROOT/resid_post_layer_27/trainer_$TRAINER/ae.pt" "$JSONL"; do
  [[ -e "$p" ]] || { echo "FATAL missing (pre-fetch it): $p" >&2; exit 1; }
done

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch transformers numpy scipy scikit-learn matplotlib

LIM=(); [[ -n "$LIMIT" ]] && LIM=(--limit "$LIMIT")

# ---- 1. extract: 4-GPU fan-out -------------------------------------------
echo "[1] extract residuals (4-GPU fan-out)"
pids=()
for g in 0 1 2 3; do
  ( export CUDA_VISIBLE_DEVICES="$g"
    python scripts/public_sae/extract_instruct_residuals.py \
      --jsonl "$JSONL" --out "$SHARDS/shard_${g}.npz" \
      --model_name_or_path "$MODEL_DIR" --local_files_only \
      --max_seq_len -1 --batch_size "$BATCH_SIZE" \
      --shard_idx "$g" --num_shards 4 "${LIM[@]}" \
      > "$LOG/extract_shard_${g}.log" 2>&1 ) &
  pids+=("$!")
done
fail=0; for i in "${!pids[@]}"; do wait "${pids[$i]}" || { echo "[1] shard $i FAILED" >&2; fail=1; }; done
[[ $fail -eq 0 ]] || { tail -20 "$LOG"/extract_shard_*.log >&2; exit 1; }

# ---- 2. merge -------------------------------------------------------------
echo "[2] merge shards"
python scripts/public_sae/extract_instruct_residuals.py --merge \
  --shard_dir "$SHARDS" --merged_out "$MERGED" --num_shards 4 2>&1 | tee "$LOG/merge.log"

# ---- 3. encode through the public SAE ------------------------------------
echo "[3] encode public SAE (trainer_$TRAINER)"
CUDA_VISIBLE_DEVICES=0 python scripts/public_sae/encode_public_sae.py \
  --enc_dir "$MERGED" --sae_root "$SAE_ROOT" --out_dir "$SAE_OUT" \
  --layers L20 L28 --trainer "$TRAINER" --batch_size 256 2>&1 | tee "$LOG/encode_sae.log"

# ---- 4. probe -------------------------------------------------------------
echo "[4] probe representations"
python scripts/public_sae/probe_public_sae.py \
  --enc_dir "$MERGED" --sae_dir "$SAE_OUT" --out_dir "$PROBE" \
  --layers L20 L28 --trainer "$TRAINER" --sae_root "$SAE_ROOT" \
  --jsonl "$JSONL" 2>&1 | tee "$LOG/probe.log"

# ---- 5. figures + report --------------------------------------------------
echo "[5] figures + report"
python scripts/public_sae/plot_public_sae_audit.py \
  --enc_dir "$MERGED" --sae_dir "$SAE_OUT" --probe_dir "$PROBE" \
  --out_dir "$FIGS" --report "$REPORT" --layers L20 L28 --trainer "$TRAINER" 2>&1 | tee "$LOG/plot.log"

echo "[done] report -> $REPORT ; metrics -> $PROBE/metrics.csv ; figures -> $FIGS"
