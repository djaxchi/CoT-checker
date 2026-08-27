#!/bin/bash
#SBATCH --job-name=derive_sae
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out

# Derive sparse SAE representations from the step-span store, so "does sparsity
# help?" becomes cells in the same grid rather than a separate study.
#
#   sae_last  <-> last_token      sae_mean <-> step_mean
#
# Same layer, same activations, same protocol: only the raw-state vs sparse-code
# question changes. The SAE is a matmul over states already stored, so this is an
# offline derive like the other readouts, just on a GPU.
#
# Layer pairing: the store holds hidden_states[35] = resid_post of block 34,
# which pairs with layer34.sae.pt (measured FVU 0.336, against 0.542 for a
# rank-50 PCA on the same rows). The script re-checks FVU before deriving and
# refuses if it is >= 1.
#
# NO INTERNET on compute nodes: the SAE must already be in $HF_CACHE.

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH/cot_mech/qwen3_8b_v1}"
PRM_STORE="${PRM_STORE:-$RUN_ROOT/repstore/step_spans}"
PB_STORE="${PB_STORE:-$RUN_ROOT/repstore/pb_step_spans}"
OUT_DIR="${OUT_DIR:-$RUN_ROOT/sae_reps}"
HF_CACHE="${HF_CACHE:-/project/aip-azouaq/$USER/hf_cache}"
REPO_ID="${REPO_ID:-Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50}"
SAE_LAYER="${SAE_LAYER:-34}"
READOUTS="${READOUTS:-sae_last sae_mean}"
PRM_SPLITS="${PRM_SPLITS:-val_5k test_2k probe_train_full}"
PB_SUBSETS="${PB_SUBSETS:-gsm8k math olympiadbench omnimath}"

export HF_HOME="$HF_CACHE" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p "$OUT_DIR"
cd "$PROJECT_ROOT"

cat <<BANNER
================================================================
job      : ${SLURM_JOB_NAME:-derive_sae}  id: ${SLURM_JOB_ID:-N/A}
sae      : $REPO_ID  layer$SAE_LAYER
store    : $PRM_STORE  (hidden_states[35] = resid_post of block 34)
readouts : $READOUTS
out      : $OUT_DIR
================================================================
BANNER

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch numpy

# One readout per GPU; each is an independent pass over the store.
pids=(); tags=()
gpu=0
for R in $READOUTS; do
  echo "[launch] gpu$gpu $R (PRM800K)"
  CUDA_VISIBLE_DEVICES=$gpu python scripts/public_sae/derive_sae_rep.py \
    --store_root "$PRM_STORE" --splits $PRM_SPLITS \
    --out_dir "$OUT_DIR" --readout "$R" --mode prm \
    --hf_cache "$HF_CACHE" --repo_id "$REPO_ID" --sae_layer "$SAE_LAYER" \
    > "$OUT_DIR/${R}_prm.log" 2>&1 &
  pids+=($!); tags+=("$R:prm"); gpu=$((gpu+1))
  echo "[launch] gpu$gpu $R (ProcessBench)"
  CUDA_VISIBLE_DEVICES=$gpu python scripts/public_sae/derive_sae_rep.py \
    --store_root "$PB_STORE" --splits $PB_SUBSETS \
    --out_dir "$OUT_DIR" --readout "$R" --mode pb \
    --hf_cache "$HF_CACHE" --repo_id "$REPO_ID" --sae_layer "$SAE_LAYER" \
    > "$OUT_DIR/${R}_pb.log" 2>&1 &
  pids+=($!); tags+=("$R:pb"); gpu=$((gpu+1))
done

fail=0
for j in "${!pids[@]}"; do
  if wait "${pids[$j]}"; then echo "[ok] ${tags[$j]}"; else
    echo "[FAIL] ${tags[$j]}"; tail -20 "$OUT_DIR/$(echo ${tags[$j]} | tr ':' '_').log" 2>/dev/null
    fail=1
  fi
done
[[ "$fail" == "1" ]] && exit 1

echo "=== derived ==="
grep -hE "FVU|mean nnz" "$OUT_DIR"/*.log | head -20
du -sh "$OUT_DIR"
echo "[$(date)] derive_sae done"
