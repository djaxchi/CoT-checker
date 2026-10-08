#!/bin/bash
#SBATCH --job-name=cav3_fg
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:30:00

# context_ablation_v3 on Rorqual: all 3 seeds of one (context, source) on one GPU.
# The store is first copied to node-local disk: random memmap reads from Lustre
# starved the per-fit jobs (data_wait 5,274 s of 5,280 s). Resumable per fit.
set -euo pipefail
: "${SNAP:?}" "${RUN:?}" "${CTX:?}" "${SRC:?v1human or v2ds}"
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 TORCH_THREADS=4
virtualenv --no-download "$SLURM_TMPDIR/env" >/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index torch numpy 2>&1 | tail -1
LOCAL="$SLURM_TMPDIR/store"
mkdir -p "$LOCAL"
t0=$(date +%s)
ls -d "$RUN/stores/$CTX"/shard_?? | xargs -P 8 -I{} cp -r {} "$LOCAL/"
n=$(ls -d "$LOCAL"/shard_??/DONE 2>/dev/null | wc -l)
[[ "$n" == 16 ]] || { echo "[FATAL] staged $n/16 shards" >&2; exit 1; }
echo "[stage] $(du -sh "$LOCAL" | cut -f1) in $(( $(date +%s) - t0 ))s"
cd "$SNAP"
pids=()
while read -r name args; do
  python scripts/train_contextual_token_probe.py --store "$LOCAL" --out_dir "$RUN/prod/fits/$name" $args \
    > "$RUN/logs/fitlog_$name.log" 2>&1 &
  pids+=($!)
done < <(grep -E "^${CTX}_${SRC}_s" "experiments/context_ablation_v3/$CTX.fits")
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
[[ $fail == 0 ]] || { tail -5 "$RUN"/logs/fitlog_${CTX}_${SRC}_s*.log >&2; exit 1; }
echo "[$(date)] done ${CTX}_${SRC}"
