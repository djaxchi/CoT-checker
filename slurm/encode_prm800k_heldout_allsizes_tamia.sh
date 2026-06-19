#!/bin/bash
#SBATCH --job-name=prm800k_heldout_allsizes
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out

# Encode the PRM800K test set with EACH model-size backbone (last layer / last
# token = the deployed readout) and eval each size's dense probe in ITS OWN hidden
# space, then aggregate into a cross-size table -- ALL IN ONE JOB.
#
# TamIA allocates h100 only by whole node, so we take one 4-GPU node and run the
# sizes in PARALLEL, one (or two small ones) per GPU:
#   GPU0 -> 32B   GPU1 -> 14B   GPU2 -> 7B   GPU3 -> 3B then 1.5B
# Largest models get a GPU to themselves; the long pole (32B) sets wall time.
#
# Writes <RUNS_ROOT>/<tag>/merged/<stem>_{h,y,meta} and
#   results/prm800k_test_full_eval/<size>.json  (+ table.csv from the aggregator).
#
# Usage:
#   STEM=prm800k_test_full sbatch slurm/encode_prm800k_heldout_allsizes_tamia.sh
#   STEM=prm800k_test_full FORCE=1 sbatch ...                 # re-encode
#   GPU3_TAGS="qwen2_5_3b" sbatch ...                         # override a GPU's tags
set -uo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
cd "$PROJECT_ROOT"
source slurm/s1_model_size/models.env   # S1MS_MODEL_ID, S1MS_BATCH, S1MS_PARAMS_LABEL, RUNS_ROOT, PRM_SPLIT_DIR, HF_CACHE_ROOT

STEM="${STEM:-prm800k_test_full}"
DATA_DIR="${DATA_DIR:-$PRM_SPLIT_DIR}"
EVAL_OUT="${EVAL_OUT:-$PROJECT_ROOT/results/prm800k_test_full_eval}"

# GPU -> tag list (override any via env). Big models alone; small ones share.
GPU0_TAGS="${GPU0_TAGS:-qwen2_5_32b}"
GPU1_TAGS="${GPU1_TAGS:-qwen2_5_14b}"
GPU2_TAGS="${GPU2_TAGS:-qwen2_5_7b}"
GPU3_TAGS="${GPU3_TAGS:-qwen2_5_3b qwen2_5_1_5b}"

export HF_HOME="${HF_HOME:-$HF_CACHE_ROOT}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"   # so `import src` works under the venv
mkdir -p "$EVAL_OUT"

cat <<BANNER
================================================================
job      : ${SLURM_JOB_NAME:-allsizes}  id ${SLURM_JOB_ID:-N/A}
host     : $(hostname)   date $(date -Iseconds)
git      : $(git rev-parse --short HEAD 2>/dev/null || echo unknown)
data     : $DATA_DIR/${STEM}.jsonl
gpus     : 0[$GPU0_TAGS] 1[$GPU1_TAGS] 2[$GPU2_TAGS] 3[$GPU3_TAGS]
eval_out : $EVAL_OUT
================================================================
BANNER

if [[ ! -f "$DATA_DIR/${STEM}.jsonl" ]]; then
  echo "[FATAL] missing $DATA_DIR/${STEM}.jsonl (build it first)"; exit 1
fi

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch transformers numpy scikit-learn

EXTRA=()
if [[ "${FORCE:-0}" == "1" ]]; then EXTRA+=(--force); fi

# Encode + eval one size on a given GPU. Errors are logged but never abort the
# node (so the other sizes and the final aggregation still run).
run_one () {
  local gpu="$1" tag="$2"
  local model_id="${S1MS_MODEL_ID[$tag]}" batch="${S1MS_BATCH[$tag]}"
  local out_dir="$RUNS_ROOT/$tag/merged" run_dir="$RUNS_ROOT/$tag"
  local label="${S1MS_PARAMS_LABEL[$tag]:-$tag}"
  mkdir -p "$out_dir"
  if [[ "${FORCE:-0}" != "1" && -f "$out_dir/${STEM}_h.npy" ]]; then
    echo "[gpu$gpu][$(date -Iseconds)] $tag encode SKIP (exists $out_dir/${STEM}_h.npy)"
  else
  echo "[gpu$gpu][$(date -Iseconds)] $tag ($model_id, batch=$batch) encode start"
  if ! CUDA_VISIBLE_DEVICES="$gpu" python scripts/encode_prm800k_hidden_states.py \
       --data_dir "$DATA_DIR" --out_dir "$out_dir" \
       --model_name_or_path "$model_id" --local_files_only \
       --run_name "fulltest_${tag}" --max_seq_len -1 \
       --batch_size "$batch" --model_dtype float16 --save_dtype float16 \
       --splits "${STEM}.jsonl:${STEM}" "${EXTRA[@]}"; then
    echo "[gpu$gpu][ERR] $tag encode FAILED; skipping eval"; return 1
  fi
  echo "[gpu$gpu][$(date -Iseconds)] $tag encode done -> $out_dir/${STEM}_{h,y,meta}"
  fi
  if [[ -f "$run_dir/linear_probe.pt" ]]; then
    if ! python scripts/eval_prm800k_heldout_probe.py \
         --run_dir "$run_dir" --enc_dir "$out_dir" --stem "$STEM" \
         --tag "$label" --out_dir "$EVAL_OUT"; then
      echo "[gpu$gpu][ERR] $tag eval FAILED"; return 1
    fi
  else
    echo "[gpu$gpu][WARN] $tag: no $run_dir/linear_probe.pt; eval skipped (encoding kept)"
  fi
  echo "[gpu$gpu][$(date -Iseconds)] $tag done"
}

# Run a GPU's tag list sequentially; the four lists run in parallel.
run_gpu () {
  local gpu="$1"; shift
  for tag in "$@"; do run_one "$gpu" "$tag"; done
}

run_gpu 0 $GPU0_TAGS &
run_gpu 1 $GPU1_TAGS &
run_gpu 2 $GPU2_TAGS &
run_gpu 3 $GPU3_TAGS &
wait
echo "[$(date -Iseconds)] all GPU lanes finished; aggregating"

python scripts/aggregate_heldout_eval.py --in_dir "$EVAL_OUT" || \
  echo "[WARN] aggregation failed (per-size JSONs still in $EVAL_OUT)"

echo "[$(date -Iseconds)] allsizes full-test encode+eval complete -> $EVAL_OUT"
