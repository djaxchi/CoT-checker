#!/bin/bash
# judge_prompt_v1 on Rorqual, per-GPU jobs (login node: bash slurm/submit_judge_prompt_rorqual.sh)
#   16 encode shards (h100:1, ~30 min each) -> step_tokens x d512 seed 42 -> seeds 43, 44
# The span stores (~165 GB) live on Rorqual scratch; TamIA keeps only vectors.
set -euo pipefail
ACCOUNT="${ACCOUNT:-def-azouaq}"
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN="${RUN:-$SCRATCH/cot_mech/judge_prompt_v1}"
cd "$PROJECT_ROOT"
snap="$SCRATCH/hf_cache/hub/models--Qwen--Qwen3-8B/snapshots"
[[ -n "$(ls -A "$snap" 2>/dev/null)" ]] || { echo "[FATAL] Qwen3-8B missing under $snap" >&2; exit 2; }
D="$SCRATCH/cot_mech/prm_backbone_v1"
for f in data/prm800k_val_5k data/prm800k_test_2k data/prm800k_probe_train_full \
         processbench/processbench_gsm8k processbench/processbench_math \
         processbench/processbench_olympiadbench processbench/processbench_omnimath; do
  [[ -s "$D/$f.jsonl" ]] || { echo "[FATAL] $D/$f.jsonl missing" >&2; exit 2; }
done
[[ -e "$RUN/repstore" && -n "$(ls -A "$RUN/repstore" 2>/dev/null)" ]] && {
  echo "[FATAL] $RUN/repstore is not empty; never submit a second DAG into it" >&2; exit 2; }
mkdir -p "$RUN/logs"
JOBS="$RUN/submitted_jobs_rorqual.txt"; echo "source $(git rev-parse HEAD)" > "$JOBS"
sb() { sbatch --parsable --account="$ACCOUNT" --output="$RUN/logs/%x-%j.out" "$@"; }
enc=""
for s in $(seq 0 15); do
  j=$(sb --job-name=jp_enc_r$s --export=ALL,SHARD=$s,RUN=$RUN,PROJECT_ROOT=$PROJECT_ROOT \
        slurm/judge_prompt_encode_rorqual.sh)
  enc="$enc:$j"; echo "encode shard $s $j" | tee -a "$JOBS"
done
s42=$(sb --dependency=afterok$enc --job-name=jp_d512_s42 \
        --export=ALL,SEED=42,RUN=$RUN,PROJECT_ROOT=$PROJECT_ROOT slurm/judge_prompt_seq_rorqual.sh)
echo "d512 seed42 $s42" | tee -a "$JOBS"
for s in 43 44; do
  j=$(sb --dependency=afterok:$s42 --time=01:30:00 --job-name=jp_d512_s$s \
        --export=ALL,SEED=$s,RUN=$RUN,PROJECT_ROOT=$PROJECT_ROOT slurm/judge_prompt_seq_rorqual.sh)
  echo "d512 seed$s $j" | tee -a "$JOBS"
done
