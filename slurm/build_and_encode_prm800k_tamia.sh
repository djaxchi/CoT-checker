#!/bin/bash
#SBATCH --job-name=prm800k_encode
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

module load StdEnv/2023 python/3.12

PROJECT_ROOT="$HOME/CoT-checker"
RUN_NAME="prestudy_v1_qwen2_5_1_5b_prm800k_40k_seed42"

RAW_PRM800K_DIR="$SCRATCH/cot_mech/raw/prm800k"
OUT_ROOT="$SCRATCH/cot_mech/prestudy_v1"
DATA_DIR="$OUT_ROOT/data"
CACHE_DIR="$OUT_ROOT/cache/qwen2_5_1_5b"
LOG_DIR="$OUT_ROOT/logs"

# Path to pre-cached Qwen2.5-1.5B weights -- update if stored elsewhere
MODEL_NAME_OR_PATH="Qwen/Qwen2.5-1.5B"
HF_CACHE="$SCRATCH/hf_cache"

mkdir -p "$DATA_DIR" "$CACHE_DIR" "$LOG_DIR" "$HF_CACHE"

export HF_HOME="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_CACHE"
export HF_DATASETS_CACHE="$HF_CACHE/datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd "$PROJECT_ROOT"

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"

pip install --no-index --upgrade pip
pip install --no-index torch transformers datasets accelerate numpy tqdm scikit-learn

echo "[$(date)] Building PRM800K prestudy dataset"
python scripts/build_prm800k_prestudy.py \
  --raw_dir "$RAW_PRM800K_DIR" \
  --out_dir "$DATA_DIR" \
  --tokenizer_name_or_path "$MODEL_NAME_OR_PATH" \
  --local_files_only \
  --run_name "$RUN_NAME" \
  --seed 42 \
  --max_seq_len 2048 \
  --n_pos_base 20000 \
  --n_neg_base 20000 \
  --n_pos_val 500 \
  --n_neg_val 500 \
  --n_forks 20

echo "[$(date)] Encoding hidden states"
python scripts/encode_prm800k_hidden_states.py \
  --data_dir "$DATA_DIR" \
  --out_dir "$CACHE_DIR" \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --local_files_only \
  --run_name "$RUN_NAME" \
  --max_seq_len 2048 \
  --batch_size 16 \
  --model_dtype float16 \
  --save_dtype float16

echo "[$(date)] Checking artifacts"
python scripts/check_prestudy_artifacts.py \
  --data_dir "$DATA_DIR" \
  --cache_dir "$CACHE_DIR"

echo "[$(date)] Done"
