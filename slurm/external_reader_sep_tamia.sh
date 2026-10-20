#!/bin/bash
#SBATCH --job-name=ext_sep
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=01:15:00
#SBATCH --output=%x-%j.out

# external_reader_v1 (docs/external_reader_v1_plan.md): one native-template pass of
# Qwen2.5-Math-PRM-7B over the 60k PRM800K training subsample, val_5k and the
# MATH-500 TTS step items, sharded over the 4 GPUs, vectors on node-local disk;
# then the separator-LDA and head pair analysis on the node's CPUs.
set -euo pipefail
OUT="$SCRATCH/cot_mech/external_reader_v1"
DATA="${DATA:-$SCRATCH/cot_mech/dense_full_7b_v1/data}"
ITEMS="$SCRATCH/cot_mech/external_reader_v0/tts_math500_items.jsonl"
TTS_ROOT="$SCRATCH/cot_mech/external_reader_v0/tts_pool"
LOCAL="$SLURM_TMPDIR/vec"
mkdir -p "$OUT/logs" "$LOCAL"
module purge
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0
export HF_HOME=/project/aip-azouaq/$USER/hf_cache
unset TRANSFORMERS_CACHE
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$HOME/CoT-checker"
echo "git $(git rev-parse --short HEAD)"
virtualenv --no-download "$SLURM_TMPDIR/env" >/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip >/dev/null
pip install --no-index torch transformers numpy scipy scikit-learn sympy 2>&1 | tail -1
pids=()
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i python scripts/encode_pooled_multilayer.py --model Qwen/Qwen2.5-Math-PRM-7B \
    --style prm_native --layers 20 28 --shard_idx $i --num_shards 4 \
    --prm800k "train:$DATA/prm800k_probe_train_full.jsonl:60000" "val:$DATA/prm800k_val_5k.jsonl" "tts:$ITEMS" \
    --out "$LOCAL" --max_seq_len 4096 > "$OUT/logs/encode_shard$i-${SLURM_JOB_ID}.log" 2>&1 &
  pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
for i in 0 1 2 3; do tail -n 1 "$OUT/logs/encode_shard$i-${SLURM_JOB_ID}.log"; done
[ $fail -eq 0 ] || exit 1
du -sh "$LOCAL"
export OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 MKL_NUM_THREADS=32
python scripts/analysis/external_reader_sep.py --tts_root "$TTS_ROOT" --vectors "$LOCAL" \
  --out "$OUT/pairs_sep.json" 2>&1 | tee "$OUT/logs/pairs-${SLURM_JOB_ID}.log"
