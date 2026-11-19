#!/bin/bash
#SBATCH --job-name=instep_v1_offset_view
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=00:40:00
#SBATCH --output=%x-%j.out
# InStep paper, "what the offsets do" (Qwen3-8B, ReProbe DS data): the step token's state with and
# without per-layer offsets. GPU0 encodes the embedding-only token (ablation rung emb_ds, seed 0,
# its picked lr 5e-4), GPU1 the offsets token (off_ds, seed 0, lr 3e-3: the only difference with
# emb_ds is the offsets), on val and the 8 test sets, keeping `sep` (the state the head reads) to
# $STV/view_<rung>_seed0. InStep DS seed 0's states already exist ($ST8). Then
# scripts/instep_v1/offset_view.py on the three: head logit, PCA of the val states (no labels)
# projected on the test steps, to $OUT/offset_view.
# --time: an encode of val + 8 tests takes ~4 min (e1 logs: val 35 s, math 25 s); analysis ~5 min.
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
CM="/project/aip-azouaq/$USER/cot_mech"
D="$CM/reprobe_cmp_v1/data"
OUT="${OUT:-$CM/instep_v1}"
ST8="$SCRATCH/cot_mech/instep_v1/states/qwen3_8b"
STV="$SCRATCH/cot_mech/instep_v1/states"
LOGS="$OUT/logs"
mkdir -p "$LOGS"
module purge
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0
export HF_HOME=/project/aip-azouaq/$USER/hf_cache
unset TRANSFORMERS_CACHE
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 MKL_NUM_THREADS=12
cd "$PROJECT_ROOT"
echo "git $(git rev-parse --short HEAD 2>/dev/null || echo none)"
virtualenv --no-download "$SLURM_TMPDIR/env" >/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip >/dev/null
pip install --no-index torch transformers numpy scipy scikit-learn 2>&1 | tail -1

SETS=("val:$D/rt/all_val.items.jsonl")
for s in math gsm8k proofnet trips meetings calendar strategyqa scienceqa; do SETS+=("$s:$D/test/$s.items.jsonl"); done
TOK_EMB="$OUT/rungsweep/emb_ds_seed0/lr0.0005_wu0.05/read_token.pt"
TOK_OFF="$OUT/rungsweep/off_ds_seed0/lr0.003_wu0.05/read_token.pt"
TOK_IN="$CM/instep_lr_sweep/ds_seed0/lr0.003_wu0.05/read_token.pt"
pids=()
g=0
for r in emb off; do
  ck=TOK_${r^^}
  CUDA_VISIBLE_DEVICES=$g python scripts/read_token_inline.py encode --model Qwen/Qwen3-8B --layout reprobe \
    --ckpt "${!ck}" --out "$STV/view_${r}_ds_seed0" --prm800k "${SETS[@]}" --passive_blocks "" \
    > "$LOGS/offset_view_$r-$SLURM_JOB_ID.log" 2>&1 & pids+=($!)
  g=$((g + 1))
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
grep -h "^\[encode\]" "$LOGS"/offset_view_*-"$SLURM_JOB_ID".log || true
[[ $fail == 0 ]] || exit 1
CUDA_VISIBLE_DEVICES=0 python scripts/instep_v1/offset_view.py --data "$D" --out "$OUT/offset_view" \
  --tok "emb=$STV/view_emb_ds_seed0=$TOK_EMB" "off=$STV/view_off_ds_seed0=$TOK_OFF" "instep=$ST8=$TOK_IN"
