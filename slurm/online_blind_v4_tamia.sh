#!/bin/bash
#SBATCH --job-name=online_v4
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=05:00:00
#SBATCH --output=%x-%j.out

# online_reject_v4 (REPORT §21.27 next step): the MATH-500 control that v3 never
# ran. Same sampler, prompt and settings as v3 (scripts/onpolicy/online_bon.py,
# prm800k few-shot prompt, Qwen3-8B, T=1.0, top-p 0.95, top-k 50, two retries),
# three arms on the same 500 problems, round-robin sharded over the 4 GPUs, each
# GPU running plain, then blind, then reject on its shard:
#
#   plain   one draw per step
#   blind   the retry loop with a coin, at the extra-draw rate v3's
#           reject_boundary_stats_q50 arm measured (0.618 per step)
#   reject  boundary_stats x mlp:h1024x2 seed 42 at v3's q50 threshold 0.128792
#
# Rollouts are appended as they finish, so a resubmit resumes where it stopped.
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
OUT="${OUT:-$SCRATCH/cot_mech/online_v4}"
CELLS="${CELLS:-/project/aip-azouaq/$USER/cot_mech/instruct_leaderboard_v1/cells}"
PROBLEMS="${PROBLEMS:-$OUT/problems/online_math500.jsonl}"
TAU="${TAU:-0.128792}"
EXTRA_DRAWS="${EXTRA_DRAWS:-0.618}"
# reject_length (REPORT §21.36 follow-up): condemn drafts longer than this many
# characters; 110 is the 0.667 quantile of plain-arm step lengths, so it
# condemns the share of first drafts the probe arm condemned (33%).
LENGTH_TAU="${LENGTH_TAU:-110}"
ARMS="${ARMS:-plain reject_blind reject}"
mkdir -p "$OUT/runs" "$OUT/logs"
module purge
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0
export HF_HOME=/project/aip-azouaq/$USER/hf_cache
unset TRANSFORMERS_CACHE
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$PROJECT_ROOT"
echo "git $(git rev-parse --short HEAD) arms=$ARMS tau=$TAU extra=$EXTRA_DRAWS"
virtualenv --no-download "$SLURM_TMPDIR/env" >/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip >/dev/null
# transformers < 5: the PRM's remote code needs it, and 4.x runs Qwen3.
pip install --no-index torch "transformers<5" numpy sympy 2>&1 | tail -1

common=(--checker panel --gen_cells "$CELLS/boundary_stats__mlp_h1024x2__seed42" "$CELLS/step_stats__mlp_h1024__seed43"
        --prm_name_or_path Qwen/Qwen2.5-Math-PRM-7B --traces "$PROBLEMS"
        --model_name_or_path Qwen/Qwen3-8B --local_files_only --layer 35
        --prompt_style prm800k --probe_context verifier
        --plain_temperature 1.0 --reject_temperature 1.0 --top_p 0.95 --top_k 50
        --max_steps 1000 --max_new_tokens 4096 --max_retries 2 --max_problems 100000
        --seed 42 --num_shards 4)
pids=()
for i in 0 1 2 3; do
  (
    for arm in $ARMS; do
      case $arm in
        plain) extra=(--active none) ;;
        reject_blind) extra=(--active none --reject_tau 1.0 --blind_target_extra_draws "$EXTRA_DRAWS") ;;
        reject) extra=(--active boundary_stats__mlp_h1024x2__seed42 --reject_tau "$TAU") ;;
        reject_length) extra=(--active none --reject_tau 1.0 --length_tau "$LENGTH_TAU" --score_all_drafts) ;;
      esac
      mkdir -p "$OUT/runs/$arm"
      CUDA_VISIBLE_DEVICES=$i python scripts/onpolicy/online_bon.py "${common[@]}" --arms "$arm" \
        --shard_idx $i --out "$OUT/runs/$arm/math500.shard$i.jsonl" "${extra[@]}"
    done
  ) > "$OUT/logs/shard$i-${SLURM_JOB_ID}.log" 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
tail -n 2 "$OUT"/logs/shard*-"${SLURM_JOB_ID}".log
exit $fail
