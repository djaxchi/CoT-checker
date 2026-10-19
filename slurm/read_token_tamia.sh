#!/bin/bash
#SBATCH --job-name=read_token
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
# read_token_v0 (docs/read_token_v0_plan.md): four soft-token configs, one per GPU,
# train then encode then closed-form geometry. Only results and the trained
# tokens leave the node.
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
OUT="${OUT:-$SCRATCH/cot_mech/read_token_v0}"
DATA="${DATA:-$SCRATCH/cot_mech/dense_full_7b_v1/data}"
PB="${PB:-/scratch/d/dchikhi/cot-checker/processbench_full}"
N_TRAIN="${N_TRAIN:-100000}"
SMOKE="${SMOKE:-0}"
LOCAL="$SLURM_TMPDIR/rt"
mkdir -p "$OUT/logs" "$LOCAL"
module purge
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0
export HF_HOME=/project/aip-azouaq/$USER/hf_cache
unset TRANSFORMERS_CACHE
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$PROJECT_ROOT"
echo "git $(git rev-parse --short HEAD) n_train=$N_TRAIN smoke=$SMOKE"
virtualenv --no-download "$SLURM_TMPDIR/env" >/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip >/dev/null
pip install --no-index torch transformers numpy scipy scikit-learn 2>&1 | tail -1
configs=("m1|1|vocab" "m4|4|vocab" "m16|16|vocab" "m4text|4|text")
pids=()
for i in 0 1 2 3; do
  IFS='|' read -r name m init <<<"${configs[$i]}"
  (
    set -e
    export CUDA_VISIBLE_DEVICES=$i
    python scripts/read_token_probe.py train --m "$m" --init "$init" --n_train "$N_TRAIN" \
      --train "$DATA/prm800k_probe_train_full.jsonl" --out "$LOCAL/$name" --smoke "$SMOKE"
    python scripts/read_token_probe.py encode --out "$LOCAL/$name" --smoke "$SMOKE" \
      --prm800k "train:$DATA/prm800k_probe_train_full.jsonl:30000" "val:$DATA/prm800k_val_5k.jsonl" \
                "test:$DATA/prm800k_test_2k.jsonl" \
      --pb "gsm8k:$PB/processbench_gsm8k.jsonl" "math:$PB/processbench_math.jsonl" \
           "olympiadbench:$PB/processbench_olympiadbench.jsonl" "omnimath:$PB/processbench_omnimath.jsonl"
    OMP_NUM_THREADS=12 python scripts/analysis/backbone_geometry.py --configs "$name:$LOCAL/$name" \
      --reps sep --out "$OUT/geometry_$name"
    mkdir -p "$OUT/tokens/$name" && cp "$LOCAL/$name/read_token.pt" "$OUT/tokens/$name/"
  ) > "$OUT/logs/${name}-${SLURM_JOB_ID}.log" 2>&1 &
  pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
for f in "$OUT"/logs/*-"${SLURM_JOB_ID}".log; do echo "== $f"; tail -n 4 "$f"; done
exit $fail
