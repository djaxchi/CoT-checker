#!/bin/bash
#SBATCH --job-name=btp_post
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out

# bidirectional_token_probe_v1 post-processing on completed final fits.
#   MODE=diag   GPU: future-perturbation diagnostics (re-encodes with Qwen3-8B)
#               submit with --gpus-per-node=h100:4 (whole node) --cpus-per-task=48 --mem=0
#   MODE=eval   CPU: metrics, paired bootstrap, structural baseline, figures, audit
#               submit with --cpus-per-task=16 --mem=64G
set -euo pipefail
: "${SNAP:?}" "${MANIFEST:?}" "${OUT_ROOT:?}" "${MODE:?diag or eval}"
export HF_HOME="${HF_CACHE:-/project/aip-azouaq/$USER/hf_cache}"
unset TRANSFORMERS_CACHE
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
module purge
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0
virtualenv --no-download "$SLURM_TMPDIR/env" >/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip >/dev/null
pip install --no-index torch transformers numpy scikit-learn matplotlib 2>&1 | tail -1
cd "$SNAP"
LOGS="$OUT_ROOT/logs/${SLURM_JOB_ID}"
mkdir -p "$LOGS" "$OUT_ROOT/results"
if [[ "$MODE" == diag ]]; then
  python scripts/diagnose_contextual_token_probe.py --manifest "$MANIFEST" --fits_root "$OUT_ROOT/fits" \
    --out "$OUT_ROOT/results/diagnostics.json" 2>&1 | tee "$LOGS/diag.log"
else
  V2F=(); [[ "${V2:-0}" == 1 ]] && V2F=(--v2)
  RES="$OUT_ROOT/${RES_NAME:-results}"
  python scripts/eval_contextual_token_probe.py --fits_root "$OUT_ROOT/fits" --manifest "$MANIFEST" \
    --fit_prefix "${FIT_PREFIX:-final_s}" --out_dir "$RES" "${V2F[@]}" 2>&1 | tee "$LOGS/eval.log"
  [[ "${SKIP_REPORT:-0}" == 1 ]] && { echo "[$(date)] eval done (no report)"; exit 0; }
  DIAG=(); [[ -f "$OUT_ROOT/results/diagnostics.json" ]] && DIAG=(--diag "$OUT_ROOT/results/diagnostics.json")
  python scripts/analysis/bidirectional_token_probe_report.py --res_dir "$RES" \
    --fits_root "$OUT_ROOT/fits" --manifest "$MANIFEST" "${DIAG[@]}" "${V2F[@]}" 2>&1 | tee "$LOGS/report.log"
fi
echo "[$(date)] $MODE done"
