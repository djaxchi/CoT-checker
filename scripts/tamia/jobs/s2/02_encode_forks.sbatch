#!/bin/bash
#SBATCH --job-name=s2-02-encode
#SBATCH --account=aip-azouaq
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gpus=h100:1
#SBATCH --output=results/logs/%x-%j.out
#SBATCH --error=results/logs/%x-%j.err

# ---------------------------------------------------------------------------
# Sprint 2: encode fork items (anchor / positive / negative) to
# Qwen2.5-1.5B last-token hidden states. Train then val, sequentially, on one
# GPU. Writes {forks_train,forks_val}_items_{h.npy,meta.jsonl} to $ENC_DIR.
#
# Submit:  sbatch scripts/tamia/jobs/s2/02_encode_forks.sbatch
# ---------------------------------------------------------------------------

set -euo pipefail

PROJECT_DIR="$HOME/CoT-checker"
STORE="/project/aip-azouaq/$USER"
SCRATCH_BASE="$SCRATCH/cot_mech"
FORKS_DIR="$SCRATCH_BASE/s2_forks/data"
ENC_DIR="$SCRATCH_BASE/s2_forks/encoded"
MODEL="Qwen/Qwen2.5-1.5B"
BATCH="${BATCH:-32}"

cd "$PROJECT_DIR"
mkdir -p "$ENC_DIR" results/logs

module purge
module load StdEnv/2023 gcc arrow/24.0.0 python/3.11 cuda/12.2
source "$HOME/venvs/cot/bin/activate"

export HF_HOME="$STORE/hf_cache"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "======================================================="
echo "  S2 Stage 02: encode fork items"
echo "  in  : $FORKS_DIR"
echo "  out : $ENC_DIR"
echo "======================================================="

for split in train val; do
    echo ""
    echo "--- encoding $split items ---"
    python scripts/encode_prm800k_forks.py \
        --items "$FORKS_DIR/forks_${split}_items.jsonl" \
        --out_dir "$ENC_DIR" \
        --stem "forks_${split}_items" \
        --model_name_or_path "$MODEL" \
        --local_files_only \
        --run_name "s2_enc_${split}" \
        --batch_size "$BATCH" \
        --model_dtype float16 \
        --save_dtype float16
done

echo ""
echo "  Done. Encoded items in $ENC_DIR"
ls -lh "$ENC_DIR"/*_h.npy
