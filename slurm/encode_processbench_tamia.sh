#!/bin/bash
#SBATCH --job-name=pb_encode
#SBATCH --account=aip-${PI_NAME}
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

module load StdEnv/2023 python/3.12

PROJECT_ROOT="$HOME/cot-mech-benchmark"
RUN_ROOT="$SCRATCH/cot_mech/prestudy_v1"
RAW_PB_FILE="$SCRATCH/cot_mech/raw/processbench/gsm8k.json"
OUT_DIR="$RUN_ROOT/cache/qwen2_5_1_5b_processbench"
MODEL_NAME_OR_PATH="Qwen/Qwen2.5-1.5B"

mkdir -p "$OUT_DIR"

export HF_HOME="$SCRATCH/hf_cache"
export TRANSFORMERS_CACHE="$SCRATCH/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd "$PROJECT_ROOT"

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"

pip install --no-index --upgrade pip
pip install --no-index torch transformers numpy

echo "[$(date)] Encoding ProcessBench-GSM8K hidden states"
python scripts/encode_processbench_hidden_states.py \
  --raw_file "$RAW_PB_FILE" \
  --out_dir "$OUT_DIR" \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --local_files_only \
  --run_name "prestudy_v1_qwen2_5_1_5b_processbench_gsm8k" \
  --max_seq_len 2048 \
  --batch_size 16 \
  --model_dtype float16 \
  --save_dtype float16

echo "[$(date)] Done. Output in $OUT_DIR"
