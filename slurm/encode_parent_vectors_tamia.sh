#!/bin/bash
#SBATCH --job-name=parent_vec
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=00:45:00
#SBATCH --output=%x-%j.out
# Re-encode Qwen2.5-Math-7B-Instruct pooled vectors exactly as backbone_geometry_v0's
# parent_ver config (same items, seed and layers), sharded over 4 GPUs, and keep
# them on scratch next to prm_ver for the cross-model direction test.
set -euo pipefail
OUT="$SCRATCH/cot_mech/backbone_geometry_v0/vectors/parent_ver"
DATA="$SCRATCH/cot_mech/dense_full_7b_v1/data"
PB=/scratch/d/dchikhi/cot-checker/processbench_full
module purge
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0
export HF_HOME=/project/aip-azouaq/$USER/hf_cache
unset TRANSFORMERS_CACHE
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$HOME/CoT-checker"
virtualenv --no-download "$SLURM_TMPDIR/env" >/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip >/dev/null
pip install --no-index torch transformers numpy 2>&1 | tail -1
pids=()
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i python scripts/encode_pooled_multilayer.py --model Qwen/Qwen2.5-Math-7B-Instruct \
    --style verifier --layers 14 20 24 27 28 --out "$SLURM_TMPDIR/parent" --shard_idx $i --num_shards 4 \
    --max_seq_len 4096 --prm800k "train:$DATA/prm800k_probe_train_full.jsonl:60000" \
    "val:$DATA/prm800k_val_5k.jsonl" "test:$DATA/prm800k_test_2k.jsonl" \
    --pb "gsm8k:$PB/processbench_gsm8k.jsonl" "math:$PB/processbench_math.jsonl" \
         "olympiadbench:$PB/processbench_olympiadbench.jsonl" "omnimath:$PB/processbench_omnimath.jsonl" \
    > "$SCRATCH/cot_mech/backbone_geometry_v0/logs/parent_vec_shard$i-$SLURM_JOB_ID.log" 2>&1 &
  pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
mkdir -p "$OUT" && rsync -a "$SLURM_TMPDIR/parent/" "$OUT/"
du -sh "$OUT"
exit $fail
