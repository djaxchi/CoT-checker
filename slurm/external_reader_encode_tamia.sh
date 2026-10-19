#!/bin/bash
#SBATCH --job-name=ext_reader
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
# external_reader_v0 (docs/external_reader_v0_plan.md): pooled step vectors of the
# MATH-500 TTS candidates under four backbones, one per GPU, written to scratch.
set -euo pipefail
OUT="$SCRATCH/cot_mech/external_reader_v0"
ITEMS="$OUT/tts_math500_items.jsonl"
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
pip install --no-index torch transformers numpy 2>&1 | tail -1
configs=("q3|Qwen/Qwen3-8B|35" "q3base|Qwen/Qwen3-8B-Base|35" "parent|Qwen/Qwen2.5-Math-7B-Instruct|27 28"
         "prm|Qwen/Qwen2.5-Math-PRM-7B|27 28")
pids=()
for i in 0 1 2 3; do
  IFS='|' read -r name model layers <<<"${configs[$i]}"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES=$i python scripts/encode_pooled_multilayer.py --model "$model" --style verifier \
    --layers $layers --prm800k "tts:$ITEMS" --out "$OUT/vectors/$name" --max_seq_len 4096 \
    > "$OUT/logs/encode_${name}-${SLURM_JOB_ID}.log" 2>&1 &
  pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
for f in "$OUT"/logs/encode_*-"${SLURM_JOB_ID}".log; do tail -n 1 "$f"; done
du -sh "$OUT"/vectors/*
exit $fail
