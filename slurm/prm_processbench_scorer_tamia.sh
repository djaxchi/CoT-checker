#!/bin/bash
#SBATCH --job-name=prm_pb_scorer
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH --output=%x-%j.out

# Qwen2.5-Math-PRM-7B's own head on ProcessBench, one subset per GPU, in its
# native template. The reference for prm_backbone_v1's probes on its states.
# transformers < 5: the PRM's remote code breaks on 5.x (see slurm/tts_sota_tamia.sh).
# HF_HOME only: transformers 4.x treats TRANSFORMERS_CACHE as the hub dir itself.

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH/cot_mech/prm_backbone_v1}"
PB_DIR="${PB_DIR:-/scratch/d/dchikhi/cot-checker/processbench_full}"
OUT="$RUN_ROOT/prm_head_scores"
module purge
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0
export HF_HOME="${HF_CACHE:-/project/aip-azouaq/$USER/hf_cache}"
unset TRANSFORMERS_CACHE
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$PROJECT_ROOT"
mkdir -p "$OUT" "$RUN_ROOT/logs"
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch "transformers<5" numpy 2>&1 | tail -1

pids=(); i=0
for s in gsm8k math olympiadbench omnimath; do
  CUDA_VISIBLE_DEVICES=$i python scripts/score_processbench_with_prm.py \
    --raw "$PB_DIR/processbench_$s.jsonl" --subset "$s" --out_dir "$OUT" \
    --local_files_only > "$RUN_ROOT/logs/prm_head_$s-${SLURM_JOB_ID}.log" 2>&1 &
  pids+=($!); i=$((i+1))
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
tail -q -n 2 "$RUN_ROOT"/logs/prm_head_*-"${SLURM_JOB_ID}".log
[[ "$fail" == 0 ]] || { echo "[FATAL] a subset failed" >&2; exit 1; }
echo "[$(date)] done"
