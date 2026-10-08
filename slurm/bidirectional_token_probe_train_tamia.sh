#!/bin/bash
#SBATCH --job-name=btp_train
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out

# bidirectional_token_probe_v1: encode the frozen manifests onto NODE-LOCAL disk,
# then run a roster of probe fits on every GPU of the node.
#
# Why node-local: the full-token store is ~293 GiB (38.5M tokens x 4096 x fp16)
# and on 2026-10-05 shared /scratch had 126 GB free and /project 154 GB. TamIA
# nodes have multi-TB $SLURM_TMPDIR. Encoding is a deterministic function of the
# frozen manifest, the pinned Qwen3-8B snapshot and the encoder; every shard
# records a fingerprint (encode_stats.json), and each job copies its fingerprints
# to STORE so copies across jobs can be compared.
#
# Required env:
#   SNAP       isolated source snapshot (code)
#   MANIFEST   frozen manifest dir (encode_manifest.jsonl, inputs/, meta/)
#   OUT_ROOT   durable output root under /project (fits write here directly)
#   FITS_FILE  one fit per line: <name> <train_contextual_token_probe.py args...>
# Optional:
#   SELECT_AND_FINAL=1  after FITS_FILE (the search), select the shared LR/WD and
#                       run the final roster it writes, reusing this job's store
#   PROCS_PER_GPU (default 1), LIMIT_PER_SHARD (smoke), SMOKE_CHECKS=1,
#   NUM_SHARDS (default 32), BATCH_TOKENS (default 32768), MAX_HOURS (per fit),
#   SKIP_ENCODE_SPLITS (unused), EXTRA_STEPS (shell snippet run after the fits)

set -euo pipefail
: "${SNAP:?}" "${MANIFEST:?}" "${OUT_ROOT:?}"
[[ -n "${FITS_FILE:-}" || -n "${ROSTER_FROM_SELECTION:-}" ]] || { echo "[FATAL] FITS_FILE or ROSTER_FROM_SELECTION" >&2; exit 2; }
HF_CACHE="${HF_CACHE:-/project/aip-azouaq/$USER/hf_cache}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
LAYER="${LAYER:-35}"
NUM_SHARDS="${NUM_SHARDS:-32}"
BATCH_TOKENS="${BATCH_TOKENS:-32768}"
PROCS_PER_GPU="${PROCS_PER_GPU:-1}"
LOCAL="$SLURM_TMPDIR/btp"
STORE_DIR="$LOCAL/store"
LOGS="$OUT_ROOT/logs/${SLURM_JOB_ID}"
mkdir -p "$STORE_DIR" "$LOGS"

module purge
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0
export HF_HOME="$HF_CACHE"
unset TRANSFORMERS_CACHE
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MANIFEST OUT_ROOT
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} TORCH_THREADS=${TORCH_THREADS:-4}
cd "$SNAP"
NG=$(nvidia-smi -L | wc -l)
echo "snap=$SNAP manifest=$MANIFEST out=$OUT_ROOT gpus=$NG procs/gpu=$PROCS_PER_GPU fits=${FITS_FILE:-from-selection:${ROSTER_FROM_SELECTION:-}}"
cat "$SNAP/SNAPSHOT_INFO" 2>/dev/null || true
df -h "$SLURM_TMPDIR" | tail -1; free -g | head -2

virtualenv --no-download "$SLURM_TMPDIR/env" >/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip >/dev/null
pip install --no-index torch transformers numpy scikit-learn pytest matplotlib 2>&1 | tail -1
python -c "import torch,transformers,sys;print('versions',sys.version.split()[0],torch.__version__,torch.version.cuda,transformers.__version__)" | tee "$LOGS/versions.txt"

# ------------------------------------------------------------------ encode
t0=$(date +%s)
LIMIT_ARG=(); [[ -n "${LIMIT_PER_SHARD:-}" ]] && LIMIT_ARG=(--limit_per_shard "$LIMIT_PER_SHARD")
SHARDS_TO_ENCODE="${SHARDS_TO_ENCODE:-$(seq -s ' ' 0 $((NUM_SHARDS-1)))}"
pids=()
for g in $(seq 0 $((NG-1))); do
  mine=(); i=0
  for s in $SHARDS_TO_ENCODE; do [[ $((i % NG)) == "$g" ]] && mine+=("$s"); i=$((i+1)); done
  [[ ${#mine[@]} == 0 ]] && continue
  SC=(); [[ "${SMOKE_CHECKS:-0}" == 1 && "$g" == 0 ]] && SC=(--smoke_checks_out "$LOGS/smoke_checks.json")
  CUDA_VISIBLE_DEVICES=$g python scripts/encode_trajectory_token_store.py \
    --manifest_dir "$MANIFEST" --rep_root "$STORE_DIR" --model_name_or_path "$MODEL" \
    --local_files_only --layer "$LAYER" --shards "${mine[@]}" --batch_tokens "$BATCH_TOKENS" \
    "${LIMIT_ARG[@]}" "${SC[@]}" > "$LOGS/encode_gpu$g.log" 2>&1 &
  pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
if [[ $fail == 1 ]]; then tail -30 "$LOGS"/encode_gpu*.log >&2; exit 1; fi
n_done=$(ls -d "$STORE_DIR"/shard_*/DONE 2>/dev/null | wc -l)
n_want=$(echo $SHARDS_TO_ENCODE | wc -w)
[[ "$n_done" == "$n_want" ]] || { echo "[FATAL] $n_done/$n_want shards complete" >&2; exit 1; }
mkdir -p "$LOGS/encode_stats"
for d in "$STORE_DIR"/shard_*; do cp "$d/encode_stats.json" "$LOGS/encode_stats/$(basename $d).json"; done
cp "$STORE_DIR"/provenance_shard*.json "$LOGS/encode_stats/" 2>/dev/null || true
du -sh "$STORE_DIR" | tee "$LOGS/store_size.txt"
echo "[encode] $(( $(date +%s) - t0 ))s" | tee "$LOGS/encode_time.txt"

# ------------------------------------------------------------------ fits
NSLOT=$((NG * PROCS_PER_GPU))
(while true; do nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader >> "$LOGS/gpu_util.csv"; free -g | sed -n 2p >> "$LOGS/ram.txt"; sleep 60; done) &
MON=$!

run_roster() {  # $1 = fits file; fits already complete (done.json for every arm) are re-run cheaply (skip)
  local file="$1"
  mapfile -t FITS < <(grep -v '^\s*#' "$file" | grep -v '^\s*$')
  echo "[fits] ${#FITS[@]} fits from $file on $NSLOT slots $(date +%T)"
  local slot_pids=() k
  for k in $(seq 0 $((NSLOT-1))); do
    (
      g=$((k % NG)); rc=0
      for ((j=k; j<${#FITS[@]}; j+=NSLOT)); do
        read -r name args <<< "${FITS[$j]}"
        args=$(envsubst <<< "$args")
        echo "[slot $k gpu $g] start $name $(date +%T)"
        if CUDA_VISIBLE_DEVICES=$g python scripts/train_contextual_token_probe.py \
             --store "$STORE_DIR" --out_dir "$OUT_ROOT/fits/$name" ${MAX_HOURS:+--max_hours $MAX_HOURS} $args \
             >> "$LOGS/fit_$name.log" 2>&1; then
          echo "[slot $k] ok $name $(date +%T)"
        else
          echo "[slot $k] FAIL($?) $name"; tail -5 "$LOGS/fit_$name.log"; rc=1
        fi
      done
      exit $rc
    ) &
    slot_pids+=($!)
  done
  local f=0 p
  for p in "${slot_pids[@]}"; do wait "$p" || f=1; done
  return $f
}

fail=0
if [[ -n "${ROSTER_FROM_SELECTION:-}" ]]; then
  SEL_ARMS="${SEL_ARMS:-future1:full:causal:local}"
  # finals fanned out over several jobs: each derives the frozen selection from the
  # completed search fits (deterministic) and runs the lines matching the regex
  python scripts/select_contextual_probe_hparams.py --fits_root "$OUT_ROOT/fits" \
    --out "$LOGS/selection.json" --final_fits "$LOGS/selected_all.fits" \
    --name_prefix "${SEL_PREFIX:-final_s}" --arms ${SEL_ARMS//:/ } \
    --extra "$( [[ -n "${SEL_EXTRA_FILE:-}" ]] && envsubst < "$SEL_EXTRA_FILE" )" | tee "$LOGS/selection.txt"
  grep -E "$ROSTER_FROM_SELECTION" "$LOGS/selected_all.fits" > "$LOGS/roster.fits" || true
  [[ -s "$LOGS/roster.fits" ]] || { echo "[FATAL] empty roster for $ROSTER_FROM_SELECTION" >&2; exit 2; }
  FITS_FILE="$LOGS/roster.fits"
fi
run_roster "$FITS_FILE" || fail=1
if [[ "$fail" == 0 && -n "${SELECT_AND_FINAL:-}" ]]; then
  # search -> frozen shared setting -> final roster, in the same allocation
  python scripts/select_contextual_probe_hparams.py --fits_root "$OUT_ROOT/fits" \
    --out "$OUT_ROOT/selection.json" --final_fits "$OUT_ROOT/final.fits" | tee "$LOGS/selection.txt"
  run_roster "$OUT_ROOT/final.fits" || fail=1
fi
kill $MON 2>/dev/null || true

if [[ -n "${EXTRA_STEPS:-}" ]]; then
  echo "[extra] $EXTRA_STEPS"
  STORE_DIR="$STORE_DIR" LOGS="$LOGS" OUT_ROOT="$OUT_ROOT" MANIFEST="$MANIFEST" bash -c "$EXTRA_STEPS" || fail=1
fi
[[ $fail == 0 ]] || { echo "[FATAL] some fits failed" >&2; exit 1; }
echo "[$(date)] all fits done"
