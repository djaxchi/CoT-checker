#!/bin/bash
#SBATCH --job-name=read_token_f
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out
# read_token_v1 Stage F (docs/read_token_v1_plan.md): leaderboard learners on a
# trained deep read token. STAGE picks the job:
#   store  pick the token on PRM800K val AUROC, write its rt_layers and rt_append
#          stores for every item of the Qwen3-8B Instruct step-span stores (train,
#          val, test, four ProcessBench subsets) on 4 GPUs, then train F-A, F-B and
#          F-C in parallel at seed 42, one lane per GPU, each cell searching lr / wd
#          as the leaderboard's seed-42 cells do. rt_layers (~45 GB) stays on
#          node-local disk, since shared scratch is near quota; rt_append (~4 GB)
#          is copied out so `fc` can add seeds later.
#   fc     optional extra F-C seeds ($SEEDS, default "43 44"), reusing the seed-42
#          F-C selection (--hp_from), one GPU per seed
set -euo pipefail
STAGE="${STAGE:?set STAGE=store or fc}"
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
OUT="${OUT:-$SCRATCH/cot_mech/read_token_v1}"
F="${F:-$OUT/stage_f}"
DATA="${DATA:-$SCRATCH/cot_mech/dense_full_7b_v1/data}"
PB="${PB:-/scratch/d/dchikhi/cot-checker/processbench_full}"
REF="${REF:-/scratch/d/dchikhi/cot_mech/qwen3_8b_instruct_v1/repstore}"
Q3=Qwen/Qwen3-8B
STEMS=(probe_train_full val_5k test_2k)
SUBSETS=(gsm8k math olympiadbench omnimath)
LOGS="$F/logs"
mkdir -p "$LOGS"
cd "$PROJECT_ROOT"
echo "git $(git rev-parse --short HEAD) stage=$STAGE"

module purge
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0
export HF_HOME=/project/aip-azouaq/$USER/hf_cache
unset TRANSFORMERS_CACHE
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 MKL_NUM_THREADS=12
virtualenv --no-download "$SLURM_TMPDIR/env" >/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip >/dev/null
pip install --no-index torch transformers numpy scipy scikit-learn pyyaml 2>&1 | tail -1

FC_TAG=step_tokens__transformer_d512_l2_f2048_h8
# one leaderboard cell, the arguments of slurm/train_rep_grid_7b_tamia.sh's run_cell
cell() {  # gpu out_root prm_store pb_store rep learner seed [extra args...]
  local gpu="$1" root="$2" prm="$3" pb="$4" rep="$5" learner="$6" seed="$7"; shift 7
  local tag="${rep}__$(echo "$learner" | tr ':,' '__')__seed${seed}"
  mkdir -p "$root"
  [[ -f "$root/$tag/results.json" ]] && { echo "[skip] $tag"; return 0; }
  echo "[launch] gpu$gpu $tag $*"
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/train_rep_learner_cell.py \
    --rep "$rep" --learner "$learner" --prm_store "$prm" --pb_store "$pb" \
    --out_dir "$root/$tag" --vec_cache_dir "$SLURM_TMPDIR/rt_vectors" \
    --train_stem probe_train_full --seed "$seed" --epochs 30 --batch_size 256 \
    --hp_search_cap 100000 --rescale none "$@" > "$root/$tag.log" 2>&1 \
    && echo "[ok] $tag" || { echo "[FAIL] $tag"; tail -n 5 "$root/$tag.log"; return 1; }
}

if [[ "$STAGE" == fc ]]; then
  HP_FROM="${HP_FROM:-$F/cells_fc/${FC_TAG}__seed42/results.json}"
  [[ -s "$HP_FROM" ]] || { echo "missing $HP_FROM"; exit 1; }
  pids=(); i=0
  for s in ${SEEDS:-43 44}; do
    cell $i "$F/cells_fc" "$REF/step_spans" "$REF/pb_step_spans" step_tokens \
      transformer:d512,l2,f2048,h8 "$s" --append_store "$F/prm/rt_append" \
      --append_pb_store "$F/pb/rt_append" --preload_budget_gb 185 --hp_from "$HP_FROM" &
    pids+=($!); i=$((i + 1))
  done
  fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
  exit $fail
fi

[[ "$STAGE" == store ]] || { echo "unknown STAGE=$STAGE"; exit 1; }
# --- pick the token: seed-0 deep read tokens, PRM800K val AUROC of the head ---------
CANDS=(d_deep_linear_seed0:4 dq_is_step_correct:4 dq_correct_q:2 dq_q:1 dq_correct:1)
LOCAL="$SLURM_TMPDIR/rtf"
mkdir -p "$LOCAL"
vals=(); pids=(); i=0
for c in "${CANDS[@]}"; do
  n=${c%%:*}
  [[ -s "$OUT/tokens/$n/read_token.pt" ]] || { echo "missing $OUT/tokens/$n/read_token.pt"; exit 1; }
  CUDA_VISIBLE_DEVICES=$((i % 4)) python scripts/read_token_probe.py encode --model $Q3 \
    --ckpt "$OUT/tokens/$n/read_token.pt" --out "$LOCAL/val_$n" \
    --prm800k "val:$DATA/prm800k_val_5k.jsonl" > "$LOGS/pick_val_$n-$SLURM_JOB_ID.log" 2>&1 &
  pids+=($!); vals+=("$c:$LOCAL/val_$n"); i=$((i + 1))
  if (( i % 4 == 0 )); then for p in "${pids[@]}"; do wait "$p"; done; pids=(); fi
done
for p in "${pids[@]}"; do wait "$p"; done
line=$(python scripts/analysis/read_token_pick.py --val "${vals[@]}" --out "$F/pick-$SLURM_JOB_ID.json" | tail -1)
echo "[pick] $line"
eval "$line"
CKPT="$OUT/tokens/$FTOKEN/read_token.pt"

# --- every reference item must match its text before any GPU work ------------------
refs=(); srcs=(); outs=()
for s in "${STEMS[@]}"; do
  refs+=("$REF/step_spans/$s"); srcs+=("prm800k:$DATA/prm800k_$s.jsonl"); outs+=(prm)
done
for s in "${SUBSETS[@]}"; do
  refs+=("$REF/pb_step_spans/$s"); srcs+=("pb:$s:$PB/processbench_$s.jsonl"); outs+=(pb)
done
L="$SLURM_TMPDIR/stage_f"
for k in "${!refs[@]}"; do
  python scripts/read_token_store.py --check --ref "${refs[$k]}" --source "${srcs[$k]}"
done

# --- write the stores, each split over 4 GPUs --------------------------------------
pids=()
for g in 0 1 2 3; do
  (
    set -e
    export CUDA_VISIBLE_DEVICES=$g
    for k in "${!refs[@]}"; do
      python scripts/read_token_store.py --model $Q3 --ckpt "$CKPT" --ref "${refs[$k]}" \
        --source "${srcs[$k]}" --out_root "$L/${outs[$k]}" --shards "$g/4"
    done
  ) > "$LOGS/store_gpu$g-$SLURM_JOB_ID.log" 2>&1 &
  pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
(( fail == 0 )) || { tail -n 5 "$LOGS"/store_gpu*-"$SLURM_JOB_ID".log; exit 1; }
du -sh "$L/prm/rt_layers" "$L/prm/rt_append" "$L/pb/rt_layers"
for side in prm pb; do
  mkdir -p "$F/$side"
  rm -rf "$F/$side/rt_append"
  cp -r "$L/$side/rt_append" "$F/$side/rt_append"
done
echo "$FTOKEN" > "$F/token.txt"

# --- F-A, F-B, F-C in parallel at seed 42, one lane per GPU --------------------------
# Memory: F-C preloads 163 GB of step spans, each F-B lane ~42 GB of rt_layers,
# on a 500 GB node. F-A's three cells are minutes each and share one lane.
FAB="$F/cells_fab"
lane_fc() { cell 0 "$F/cells_fc" "$REF/step_spans" "$REF/pb_step_spans" step_tokens \
  transformer:d512,l2,f2048,h8 42 --append_store "$L/prm/rt_append" \
  --append_pb_store "$L/pb/rt_append" --preload_budget_gb 185; }
lane_fb_d512() { cell 1 "$FAB" "$L/prm/rt_layers" "$L/pb/rt_layers" step_tokens \
  transformer:d512,l2,f2048,h8 42 --preload_budget_gb 100; }
lane_fb_attn() { cell 2 "$FAB" "$L/prm/rt_layers" "$L/pb/rt_layers" step_tokens \
  attn_query 42 --preload_budget_gb 100; }
lane_fa() {
  local fail=0 l
  for l in linear mlp:h1024 mlp:h1024x2; do
    cell 3 "$FAB" "$L/prm/rt_layers" "$L/pb/rt_layers" last_token "$l" 42 \
      --preload_budget_gb 100 || fail=1
  done
  return $fail
}
pids=()
for lane in lane_fc lane_fb_d512 lane_fb_attn lane_fa; do "$lane" & pids+=($!); done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "[$(date)] stage F cells done, fail=$fail"
exit $fail
