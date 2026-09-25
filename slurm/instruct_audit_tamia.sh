#!/bin/bash
#SBATCH --job-name=instruct_audit
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --output=%x-%j.out

# instruct_arm_v1 §4: the activation-artifact audit for ONE arm (ARM=base or
# ARM=instruct), both parts concurrently on two GPUs of the whole node:
#   GPU 0  probe part: outlier mass, massive tokens, occlusion, residualisation
#   GPU 1  attention part: step-token attention mass by category, eager attention
# When both arms' outputs exist, the compare table is written as well.
#
# The Base reference cell predates protocol.rescale; its log and the code at the
# time (3804f76, before 983c890 added --rescale) show it trained on raw states,
# hence --assume_rescale none for that arm only.
#
# NO INTERNET ON COMPUTE NODES: both backbones must already be in $HF_CACHE.

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

ARM="${ARM:?set ARM=base or ARM=instruct}"
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
HF_CACHE="${HF_CACHE:-/project/aip-azouaq/$USER/hf_cache}"
DATA_DIR="${DATA_DIR:-$SCRATCH/cot_mech/dense_full_7b_v1/data}"
AUDIT_ROOT="${AUDIT_ROOT:-$SCRATCH/cot_mech/qwen3_8b_instruct_v1/audit}"
CELL_NAME="${CELL_NAME:-step_tokens__transformer_d512_l2_f2048_h8__seed42}"
case "$ARM" in
  base)
    RUN_ROOT="$SCRATCH/cot_mech/qwen3_8b_v1"; MODEL="Qwen/Qwen3-8B-Base"
    RESCALE_ARG=(--assume_rescale none) ;;
  instruct)
    RUN_ROOT="$SCRATCH/cot_mech/qwen3_8b_instruct_v1"; MODEL="Qwen/Qwen3-8B"
    RESCALE_ARG=() ;;
  *) echo "[FATAL] ARM=$ARM" >&2; exit 2 ;;
esac
CELL="$RUN_ROOT/runs/rep_grid_q3/$CELL_NAME"
OUT="$AUDIT_ROOT/$ARM"
[[ -f "$CELL/model.pt" ]] || { echo "[FATAL] no trained cell at $CELL" >&2; exit 2; }
mkdir -p "$OUT"

export HF_HOME="$HF_CACHE" TRANSFORMERS_CACHE="$HF_CACHE"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd "$PROJECT_ROOT"
cat <<BANNER
================================================================
job        : ${SLURM_JOB_NAME:-instruct_audit}  id: ${SLURM_JOB_ID:-N/A}
git_commit : $(git rev-parse HEAD 2>/dev/null || echo unknown)
arm        : $ARM   model: $MODEL
cell       : $CELL
out        : $OUT
================================================================
BANNER

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch transformers numpy pyyaml

CUDA_VISIBLE_DEVICES=0 python scripts/analysis/instruct_artifact_audit.py --part probe \
  --cell_dir "$CELL" "${RESCALE_ARG[@]}" \
  --prm_store "$RUN_ROOT/repstore/step_spans" --pb_store "$RUN_ROOT/repstore/pb_step_spans" \
  --out "$OUT/probe.json" > "$OUT/probe.log" 2>&1 &
P1=$!
CUDA_VISIBLE_DEVICES=1 python scripts/analysis/instruct_artifact_audit.py --part attention \
  --model_name_or_path "$MODEL" --data "$DATA_DIR/prm800k_test_2k.jsonl" \
  --out "$OUT/attention.json" > "$OUT/attention.log" 2>&1 &
P2=$!
fail=0
wait $P1 || { echo "[FAIL] probe part"; tail -30 "$OUT/probe.log"; fail=1; }
wait $P2 || { echo "[FAIL] attention part"; tail -30 "$OUT/attention.log"; fail=1; }
[[ "$fail" == "0" ]] || exit 1
tail -8 "$OUT/probe.log"; tail -3 "$OUT/attention.log"

if [[ -f "$AUDIT_ROOT/base/probe.json" && -f "$AUDIT_ROOT/instruct/probe.json" && \
      -f "$AUDIT_ROOT/base/attention.json" && -f "$AUDIT_ROOT/instruct/attention.json" ]]; then
  python scripts/analysis/instruct_artifact_audit.py --part compare \
    --base "$AUDIT_ROOT/base" --instruct "$AUDIT_ROOT/instruct" \
    --out "$AUDIT_ROOT/compare.json"
fi
echo "[$(date)] instruct_audit $ARM done"
