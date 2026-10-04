#!/bin/bash
# prm_backbone_v1 on Rorqual: the Instruct leaderboard roster, retrained on
# Qwen2.5-Math-PRM-7B states. Run on the Rorqual login node, not as a job.
#
# Question: does the PRM's residual stream carry a richer correct/incorrect
# signal than Qwen3-8B (Instruct)? Only the backbone changes: same frozen
# PRM800K splits and ProcessBench files (sha256-checked copies of TamIA's), same
# verifier template, same span store, same 22 cells, same trainer protocol
# (RESCALE=none, 30 epochs, patience 3, batch 256, seeds 42 43 44, lr x wd
# search at seed 42 reused by 43 and 44).
#
# Rorqual rather than TamIA: the PRM stores need ~155 GB and TamIA scratch had
# 127 GB free on 2026-10-02. Rorqual allocates by GPU, so every cell x seed is
# its own single-GPU job and starts through backfill.
#
# DAG: 4 PRM800K shard encodes + 1 ProcessBench encode
#   -> 6 vector-rep jobs (3 learners x 3 seeds each)
#   -> 4 sequence cells at seed 42 -> seeds 43 and 44 (reuse 42's selection)

set -euo pipefail
ACCOUNT="${ACCOUNT:-def-azouaq}"
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH/cot_mech/prm_backbone_v1}"
JOBS="$RUN_ROOT/submitted_jobs.txt"
cd "$PROJECT_ROOT"

snap="$SCRATCH/hf_cache/hub/models--Qwen--Qwen2.5-Math-PRM-7B/snapshots"
[[ -n "$(ls -A "$snap" 2>/dev/null)" ]] || { echo "[FATAL] PRM weights missing under $snap" >&2; exit 2; }
for f in prm800k_val_5k prm800k_test_2k prm800k_probe_train_full; do
  [[ -s "$RUN_ROOT/data/$f.jsonl" ]] || { echo "[FATAL] $RUN_ROOT/data/$f.jsonl missing" >&2; exit 2; }
done
for s in gsm8k math olympiadbench omnimath; do
  [[ -s "$RUN_ROOT/processbench/processbench_$s.jsonl" ]] || { echo "[FATAL] PB $s missing" >&2; exit 2; }
done
[[ -e "$RUN_ROOT/cells" && -n "$(ls -A "$RUN_ROOT/cells" 2>/dev/null)" ]] && {
  echo "[FATAL] $RUN_ROOT/cells is not empty; never submit a second DAG into it" >&2; exit 2; }
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/cells" "$RUN_ROOT/cellfiles"
echo "source $(git rev-parse HEAD)" > "$JOBS"

sb() { sbatch --parsable --account="$ACCOUNT" --output="$RUN_ROOT/logs/%x-%j.out" "$@"; }

enc=()
for i in 0 1 2 3; do
  j=$(sb --job-name=prmbb_enc_prm$i --time=02:00:00 \
        --export=ALL,WHAT=prm,SHARD=$i,RUN_ROOT="$RUN_ROOT" slurm/prm_backbone_encode_rorqual.sh)
  enc+=("$j"); echo "encode prm shard $i $j" | tee -a "$JOBS"
done
j=$(sb --job-name=prmbb_enc_pb --time=01:00:00 \
      --export=ALL,WHAT=pb,RUN_ROOT="$RUN_ROOT" slurm/prm_backbone_encode_rorqual.sh)
enc+=("$j"); echo "encode pb $j" | tee -a "$JOBS"
after_enc="afterok:$(IFS=:; echo "${enc[*]}")"

# Trainer overrides: TamIA's #SBATCH header asks for a whole node; on Rorqual
# the command line takes one GPU instead.
train_env="ALL,PROJECT_ROOT=$PROJECT_ROOT,RUN_ROOT=$RUN_ROOT,PRM_STORE=$RUN_ROOT/repstore/step_spans,PB_STORE=$RUN_ROOT/repstore/pb_step_spans,OUT_ROOT=$RUN_ROOT/cells,VEC_CACHE=$RUN_ROOT/cache/grid_vectors,RESCALE=none,EPOCHS=30,BATCH_SIZE=256,HP_SEARCH_CAP=100000,N_GPUS=1"
tsb() { sb --gpus-per-node=h100:1 --cpus-per-task=12 "$@" slurm/train_rep_grid_7b_tamia.sh; }

for rep in last_token step_mean step_delta step_stats boundary_stats lengthfree_geom; do
  cf="$RUN_ROOT/cellfiles/$rep.cells"
  printf '%s linear\n%s mlp:h1024\n%s mlp:h1024x2\n' "$rep" "$rep" "$rep" > "$cf"
  j=$(tsb --job-name=prmbb_vec_$rep --time=06:00:00 --mem=160G --dependency="$after_enc" \
        --export="$train_env",CELLS_FILE="$cf",SEEDS="42 43 44",PRELOAD_BUDGET_GB=100)
  echo "vectors $rep $j" | tee -a "$JOBS"
done

seq_cells=("attn_query" "transformer:d128,l1,f512,h4" "transformer:d256,l2,f1024,h4" "transformer:d512,l2,f2048,h8")
for learner in "${seq_cells[@]}"; do
  tag="$(echo "$learner" | tr ':,' '__')"
  cf="$RUN_ROOT/cellfiles/step_tokens__$tag.cells"
  echo "step_tokens $learner" > "$cf"
  j42=$(tsb --job-name=prmbb_seq_${tag}_s42 --time=16:00:00 --mem=240G --dependency="$after_enc" \
          --export="$train_env",CELLS_FILE="$cf",SEEDS=42,PRELOAD_BUDGET_GB=200)
  echo "seq $tag seed42 $j42" | tee -a "$JOBS"
  for s in 43 44; do
    # SEEDS="42 s": phase 1 skips the finished seed 42, phase 2 reuses its search.
    j=$(tsb --job-name=prmbb_seq_${tag}_s$s --time=14:00:00 --mem=240G --dependency="afterok:$j42" \
          --export="$train_env",CELLS_FILE="$cf",SEEDS="42 $s",PRELOAD_BUDGET_GB=200)
    echo "seq $tag seed$s $j" | tee -a "$JOBS"
  done
done
cat "$JOBS"
