#!/bin/bash
#SBATCH --job-name=easy_probes
#SBATCH --account=aip-${PI_NAME}
#SBATCH --nodes=1
#SBATCH --gpus=h100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

module load StdEnv/2023 python/3.12

PROJECT_ROOT="$HOME/cot-mech-benchmark"
RUN_ROOT="$SCRATCH/cot_mech/prestudy_v1"
CACHE_DIR="$RUN_ROOT/cache/qwen2_5_1_5b"
PB_CACHE_DIR="$RUN_ROOT/cache/qwen2_5_1_5b_processbench"
OUT_ROOT="$RUN_ROOT/runs"

mkdir -p "$OUT_ROOT"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd "$PROJECT_ROOT"

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"

pip install --no-index --upgrade pip
pip install --no-index torch numpy tqdm scikit-learn pyyaml

run_method () {
  local gpu_id="$1"
  local method="$2"

  echo "[$(date)] Starting $method on GPU $gpu_id"

  CUDA_VISIBLE_DEVICES="$gpu_id" python scripts/train_easy_probe_method.py \
    --method "$method" \
    --cache_dir "$CACHE_DIR" \
    --pb_cache_dir "$PB_CACHE_DIR" \
    --out_dir "$OUT_ROOT/$method" \
    --seed 42 \
    --epochs_sae 20 \
    --epochs_probe 50 \
    --batch_size 512 \
    --lr_sae 1e-3 \
    --lr_probe 1e-3 \
    --l1_weight 1e-4 \
    --bce_weight 0.1

  echo "[$(date)] Finished $method on GPU $gpu_id"
}

# Wave 1: use all 4 H100 GPUs
declare -A wave1_pids
run_method 0 random         & wave1_pids[random]=$!
run_method 1 dense_linear   & wave1_pids[dense_linear]=$!
run_method 2 sae_positive   & wave1_pids[sae_positive]=$!
run_method 3 sae_mixed      & wave1_pids[sae_mixed]=$!

wave1_fail=0
for method in "${!wave1_pids[@]}"; do
  pid="${wave1_pids[$method]}"
  if ! wait "$pid"; then
    echo "[ERROR] Wave 1 method failed: $method (pid $pid)" >&2
    wave1_fail=1
  fi
done
if [[ "$wave1_fail" -ne 0 ]]; then
  echo "[FATAL] Aborting before Wave 2 and leaderboard merge due to Wave 1 failure(s)." >&2
  exit 1
fi

# Wave 2: remaining heavier method
run_method 0 sae_contrastive &
wave2_pid=$!
if ! wait "$wave2_pid"; then
  echo "[ERROR] Wave 2 method failed: sae_contrastive (pid $wave2_pid)" >&2
  echo "[FATAL] Aborting before leaderboard merge." >&2
  exit 1
fi

python scripts/merge_easy_probe_leaderboard.py \
  --runs_dir "$OUT_ROOT" \
  --out_csv "$OUT_ROOT/leaderboard_easy_probes.csv" \
  --out_md "$OUT_ROOT/leaderboard_easy_probes.md"

echo "[$(date)] All easy probes done"
