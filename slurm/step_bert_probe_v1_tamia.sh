#!/bin/bash
#SBATCH --job-name=sbp1
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=/project/aip-azouaq/dchikhi/cot_mech/step_bert_probe_v1/logs/%x-%j.out

# step_bert_probe_v1 (docs/step_bert_probe_v1_plan.md): token-level encoder over
# frozen Qwen3-8B-Base L35 token states (arm tok) and fine-tuned ModernBERT-base
# on raw step text (arm txt), local / causal / bidir, on v0's split and metrics.
#
# 1. copy the code checkout to node-local disk (immutable for the run)
# 2. concatenate the on-policy and ProcessBench span stores into per-collection
#    token arrays on node-local disk (5 collections in parallel, ~38 GB)
# 3. driver_v1.py: both arms concurrently, one fit per GPU per arm; fits write
#    straight to OUT_ROOT and finished fits are skipped on resubmission
set -euo pipefail
OUT_ROOT=/project/aip-azouaq/dchikhi/cot_mech/step_bert_probe_v1
SRC=/scratch/d/dchikhi/cot_mech
MODEL=/project/aip-azouaq/dchikhi/hf_cache/hub/models--answerdotai--ModernBERT-base/snapshots/8949b909ec900327062f0ebf497f51aef5e6f0c8
LOCAL=$SLURM_TMPDIR/sbp1
mkdir -p "$LOCAL/data" "$OUT_ROOT/logs"

module purge
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0
export HF_HOME=/project/aip-azouaq/dchikhi/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

rsync -a --exclude='.git' --exclude='data/' --exclude='results/' --exclude='runs/' --exclude='cot-checker-results/' \
  --exclude='papers/' --exclude='graphify-out/' "$HOME/CoT-checker/" "$LOCAL/code/"
cd "$LOCAL/code"
git -C "$HOME/CoT-checker" log --oneline -1 | tee "$OUT_ROOT/logs/code_${SLURM_JOB_ID}.txt"

virtualenv --no-download "$SLURM_TMPDIR/env" >/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip >/dev/null
pip install --no-index torch transformers numpy scikit-learn 2>&1 | tail -1
python -c "import torch,transformers,sys;print('versions',sys.version.split()[0],torch.__version__,transformers.__version__,torch.cuda.device_count())"
nvidia-smi -L; df -h "$SLURM_TMPDIR" | tail -1; free -g | sed -n 2p

t0=$(date +%s)
pids=()
python scripts/step_bert/build_token_store.py --store_roots $SRC/reprobe_v1/repstore/onpolicy_train_spans/{train,val} \
  --out "$LOCAL/data/onpolicy" > "$OUT_ROOT/logs/build_onpolicy_${SLURM_JOB_ID}.log" 2>&1 &
pids+=($!)
for s in gsm8k math olympiadbench omnimath; do
  python scripts/step_bert/build_token_store.py --store_roots $SRC/qwen3_8b_v1/repstore/pb_step_spans/$s \
    --out "$LOCAL/data/pb_$s" > "$OUT_ROOT/logs/build_pb_${s}_${SLURM_JOB_ID}.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
cat "$LOCAL"/data/*/info.json; echo
echo "[build] $(( $(date +%s) - t0 ))s"; du -sh "$LOCAL/data"

(while true; do nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader \
   | sed "s/^/$(date +%T) /" >> "$OUT_ROOT/logs/gpu_${SLURM_JOB_ID}.csv"; sleep 60; done) &
MON=$!
python scripts/step_bert/driver_v1.py --data "$LOCAL/data" --texts "$OUT_ROOT/texts" --model "$MODEL" \
  --split "$OUT_ROOT/split.json" --out_root "$OUT_ROOT" --gpus 4
kill $MON 2>/dev/null || true
