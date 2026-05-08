#!/bin/bash
#SBATCH --job-name=ptb-00-sanity
#SBATCH --account=aip-azouaq
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=results/logs/%x-%j.out
#SBATCH --error=results/logs/%x-%j.err

# ---------------------------------------------------------------------------
# Stage 00: Sanity check
#
# Verifies that all upstream data files exist and have the expected structure,
# then patches solution_ids into the dense .npz files (idempotent).
#
# Artifact: results/logs/ptb-00-sanity-<JOBID>.out
# Decision: job exits non-zero on any FAIL -- do not proceed to Stage 01.
#
# Submit:
#   sbatch scripts/tamia/jobs/00_ptb_sanity.sbatch
# ---------------------------------------------------------------------------

set -euo pipefail

PROJECT_DIR="$HOME/CoT-checker"
STORE="/project/aip-azouaq/$USER"
SCRATCH_BASE="$SCRATCH/cot-checker"

CKPT_SSAE="$STORE/checkpoints/gsm8k-385k_Qwen2.5-0.5b_spar-10.pt"
DATA_TRAIN="$STORE/data/gsm8k_385K_train.json"
DATA_VALID="$STORE/data/gsm8k_385K_valid.json"

PROBE_DATA="$STORE/probe_data"
PB_DIR="$SCRATCH_BASE/processbench"
TRANS_DIR="$SCRATCH_BASE/probe_data/transitions"

cd "$PROJECT_DIR"
mkdir -p results/logs "$TRANS_DIR"

module purge
module load StdEnv/2023 gcc arrow/24.0.0 python/3.11 cuda/12.2
source "$HOME/venvs/cot/bin/activate"

export HF_HOME="$STORE/hf_cache"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PROBE_DATA PB_DIR

echo "======================================================="
echo "  Stage 00: Pre-flight checks"
echo "======================================================="

PASS=0
FAIL=0

_check_file() {
    local f="$1"
    if [ -f "$f" ]; then
        local sz
        sz=$(du -sh "$f" | cut -f1)
        echo "  PASS  $f  ($sz)"
        PASS=$((PASS + 1))
    else
        echo "  FAIL  $f  (not found)"
        FAIL=$((FAIL + 1))
    fi
}

echo ""
echo "--- Required input files ---"
_check_file "$CKPT_SSAE"
_check_file "$DATA_TRAIN"
_check_file "$DATA_VALID"
_check_file "$PROBE_DATA/dense_train_full.npz"
_check_file "$PROBE_DATA/dense_eval_held_out.npz"
_check_file "$PROBE_DATA/train_final.npz"
_check_file "$PROBE_DATA/eval_held_out.npz"
_check_file "$PB_DIR/processbench_dense_gsm8k.npz"
_check_file "$PB_DIR/processbench_gsm8k.npz"

echo ""
echo "--- .npz structure check ---"
python3 - <<'PYEOF'
import sys
import numpy as np
from pathlib import Path

import os
PROBE_DATA = os.environ.get("PROBE_DATA", "")
PB_DIR     = os.environ.get("PB_DIR", "")

checks = [
    (f"{PROBE_DATA}/dense_train_full.npz",        ["latents", "correctness"]),
    (f"{PROBE_DATA}/dense_eval_held_out.npz",     ["latents", "correctness"]),
    (f"{PB_DIR}/processbench_dense_gsm8k.npz",    ["latents", "step_labels", "solution_ids"]),
]
all_ok = True
for path, required_keys in checks:
    if not Path(path).exists():
        print(f"  SKIP  {path}  (missing)")
        continue
    try:
        d = np.load(path)
        missing = [k for k in required_keys if k not in d.files]
        shape   = d["latents"].shape if "latents" in d.files else "N/A"
        if missing:
            print(f"  FAIL  {path}  shape={shape}  missing keys: {missing}")
            all_ok = False
        else:
            extra_keys = [k for k in d.files if k not in required_keys]
            print(f"  PASS  {path}  shape={shape}  keys={list(d.files)}")
    except Exception as e:
        print(f"  FAIL  {path}  error: {e}")
        all_ok = False

sys.exit(0 if all_ok else 1)
PYEOF

export PROBE_DATA PB_DIR

echo ""
echo "--- Transition cache presence ---"
TRANS_TRAIN="$TRANS_DIR/transition_train_positive.npz"
TRANS_VAL="$TRANS_DIR/transition_val.npz"
if [ -f "$TRANS_TRAIN" ]; then
    echo "  PRESENT  $TRANS_TRAIN"
else
    echo "  ABSENT   $TRANS_TRAIN  (will be built in Stage 01)"
fi
if [ -f "$TRANS_VAL" ]; then
    echo "  PRESENT  $TRANS_VAL"
else
    echo "  ABSENT   $TRANS_VAL  (will be built in Stage 01)"
fi

echo ""
echo "--- Patch solution_ids (idempotent) ---"
python scripts/add_ms_solution_ids.py \
    --npz "$PROBE_DATA/dense_eval_held_out.npz" --offset 0     --max-steps 90000
python scripts/add_ms_solution_ids.py \
    --npz "$PROBE_DATA/dense_train_full.npz"    --offset 90000 --max-steps 360000

echo ""
echo "======================================================="
echo "  Summary: PASS=$PASS  FAIL=$FAIL"
if [ "$FAIL" -gt 0 ]; then
    echo "  RESULT: FAIL -- fix missing files before submitting Stage 01."
    exit 1
fi
echo "  RESULT: PASS -- safe to submit Stage 01."
echo "======================================================="
