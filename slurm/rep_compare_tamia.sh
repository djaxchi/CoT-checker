#!/bin/bash
#SBATCH --job-name=rep_compare
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --output=%x-%j.out
# read_token_v1 Stage K (docs/read_token_v1_plan.md): the read token against its four
# comparison points, one vector per PRM800K step each, and the weak PRM's scores.
#   Qwen2.5-Math-7B-PRM800K (weak PRM): native pass, separator state and head on PRM800K
#       train 30k / val / test and ProcessBench (geometry + head calib-20), then its step
#       scores on the four Instruct answer pools (same scorer as the strong PRM)
#   Qwen2.5-Math-PRM-7B (strong PRM): native separator state, val / test
#   Qwen3-8B last step token, layer 35, verifier template, val / test
#   best leaderboard probe (step_tokens x transformer d512, seed 42): pooled embedding
#   offline read token (rth_correct_seed1) and online read token (rthics_seed1), val / test
# then scripts/analysis/rep_views.py: PCA and LDA fit on val, applied to test.
# Submit from a writable dir (snapshots are read-only), with -o pointing at $OUT/compare/logs.
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
OUT="${OUT:-$SCRATCH/cot_mech/read_token_v1}"
DATA="${DATA:-$SCRATCH/cot_mech/dense_full_7b_v1/data}"
PB="${PB:-/scratch/d/dchikhi/cot-checker/processbench_full}"
REF="${REF:-/scratch/d/dchikhi/cot_mech/qwen3_8b_instruct_v1/repstore/step_spans}"
CELL="${CELL:-/project/aip-azouaq/$USER/cot_mech/instruct_leaderboard_v1/cells/step_tokens__transformer_d512_l2_f2048_h8__seed42}"
SMOKE="${SMOKE:-0}"
CMP="$OUT/compare"
LOGS="$CMP/logs"
L="$SLURM_TMPDIR/cmp"
WEAK=Qwen/Qwen2.5-Math-7B-PRM800K
STRONG=Qwen/Qwen2.5-Math-PRM-7B
Q3=Qwen/Qwen3-8B
mkdir -p "$LOGS" "$L" "$CMP/enc"
cd "$PROJECT_ROOT"
echo "snapshot ${PROJECT_ROOT} smoke=$SMOKE"

module purge
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0
export HF_HOME=/project/aip-azouaq/$USER/hf_cache
unset TRANSFORMERS_CACHE
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 MKL_NUM_THREADS=12
virtualenv --no-download "$SLURM_TMPDIR/env" >/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip >/dev/null
pip install --no-index torch transformers numpy scipy scikit-learn pyyaml sympy safetensors huggingface_hub matplotlib 2>&1 | tail -1
deactivate
# the PRMs' released remote code needs transformers < 5 (scripts/onpolicy/score_traces_with_prm.py)
virtualenv --no-download "$SLURM_TMPDIR/env4" >/dev/null
source "$SLURM_TMPDIR/env4/bin/activate"
pip install --no-index --upgrade pip >/dev/null
pip install --no-index torch "transformers<5" numpy sympy pyyaml 2>&1 | tail -1
deactivate
E5="$SLURM_TMPDIR/env/bin/python"
E4="$SLURM_TMPDIR/env4/bin/python"

VT=(--prm800k "val:$DATA/prm800k_val_5k.jsonl" "test:$DATA/prm800k_test_2k.jsonl")
PBS=(--pb "gsm8k:$PB/processbench_gsm8k.jsonl" "math:$PB/processbench_math.jsonl"
          "olympiadbench:$PB/processbench_olympiadbench.jsonl" "omnimath:$PB/processbench_omnimath.jsonl")

# pool <gpu> <temp tag> <dataset>: the weak PRM's step scores on one Instruct answer pool,
# segmentation checked against the strong PRM's score files of the same pool
pool() {
  local g=$1 t=$2 ds=$3 P="$OUT/downstream/pools/instruct_$2"
  CUDA_VISIBLE_DEVICES=$g $E4 scripts/onpolicy/score_traces_with_prm.py \
    --prm_name_or_path $WEAK --local_files_only --cell_name prm800k_qwen25_math_7b --orientation_traces 40 \
    --orientation_metric "${ORIENT:-mean}" \
    --verify_against "$P"/scores/tts_${ds}__prm_qwen25_math_7b.shard*.jsonl \
    --trajectories "$P"/tts_${ds}.shard*_trajectories.jsonl \
    --max_traces "$([[ $SMOKE == 0 ]] && echo 0 || echo 40)" --force \
    --out "$OUT/downstream/scores/instruct_$t/tts_${ds}__prm800k_qwen25_math_7b.shard0.jsonl"
}

# ONLY_POOL="t10 math500": rescore one pool and stop (with ORIENT=worst: on the T = 1.0
# MATH-500 pool the weak PRM's mean step score ties across classes, 0.942 vs 0.944, while
# the same model and template pass the gate on the other three pools)
if [[ -n "${ONLY_POOL:-}" ]]; then
  pool 0 $ONLY_POOL > "$LOGS/pool_${ONLY_POOL// /_}-${SLURM_JOB_ID}.log" 2>&1
  tail -n 4 "$LOGS/pool_${ONLY_POOL// /_}-${SLURM_JOB_ID}.log"
  exit 0
fi


# VIEWS=1: the class-separation figure (read tokens at hidden_states[35], the policy's layer) (scripts/analysis/plot_candidate_views.py) with every
# projection fit on PRM800K train: one frozen 30k train subsample for all six candidates
if [[ "${VIEWS:-0}" == 1 ]]; then
  T30="$CMP/train_30k.jsonl"
  [[ -s "$T30" ]] || $E5 -c "
import numpy as np, sys
rows = open(sys.argv[1]).readlines()
pick = np.sort(np.random.default_rng(0).choice(len(rows), 30000, replace=False))
open(sys.argv[2], 'w').writelines(rows[i] for i in pick)" "$DATA/prm800k_probe_train_full.jsonl" "$T30"
  TVT=(--prm800k "train:$T30" "val:$DATA/prm800k_val_5k.jsonl" "test:$DATA/prm800k_test_2k.jsonl")
  v0() { $E5 scripts/encode_pooled_multilayer.py --model $WEAK --style prm_native --layers 28 "${TVT[@]}" --out "$L/weak"; }
  v1() { $E5 scripts/encode_pooled_multilayer.py --model $STRONG --style prm_native --layers 28 "${TVT[@]}" --out "$L/strong"
         $E5 scripts/encode_pooled_multilayer.py --model $Q3 --style verifier --layers 35 "${TVT[@]}" --out "$L/q3"; }
  v2() { $E5 scripts/read_token_probe.py encode --model $Q3 --ckpt "$OUT/tokens/rth_correct_seed1/read_token.pt" \
           --read_blocks 34 "${TVT[@]}" --out "$L/offline"
         $E5 scripts/read_token_inline.py encode --model $Q3 --ckpt "$OUT/tokens/rthics_seed1/read_token.pt" \
           --read_blocks 34 "${TVT[@]}" --out "$L/online"; }
  v3() { $E5 scripts/analysis/cell_embed.py --cell_dir "$CELL" --store "$REF" \
           --splits "train:probe_train_full:$T30" val:val_5k test:test_2k --out "$L/probe"; }
  vl() { local g=$1 name=$2; shift 2; ( set -e; export CUDA_VISIBLE_DEVICES=$g; "$@" ) > "$LOGS/views_${name}-${SLURM_JOB_ID}.log" 2>&1; }
  pids=(); vl 0 weak v0 & pids+=($!); vl 1 strong_q3 v1 & pids+=($!); vl 2 read_tokens v2 & pids+=($!); vl 3 probe v3 & pids+=($!)
  fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
  (( fail == 0 )) || { tail -n 5 "$LOGS"/views_*-"$SLURM_JOB_ID".log; exit 1; }
  $E5 scripts/analysis/plot_candidate_views.py --out "$CMP/figure" --cand \
    "Online read token (L35)=$L/online:read_block34:-1" "Offline read token (L35)=$L/offline:read_block34:-1" \
    "Strong PRM, Qwen2.5-Math-PRM-7B (separator, L28)=$L/strong:sep:28" \
    "Weak PRM, Qwen2.5-Math-7B-PRM800K (separator, L28)=$L/weak:sep:28" \
    "Best probe, d512 transformer (pooled 512-d)=$L/probe:z:-1" \
    "Qwen3-8B last step token (L35)=$L/q3:last:35"
  exit $?
fi

lane() {  # gpu name commands...: one GPU's work in order, logged
  local g=$1 name=$2; shift 2
  ( set -e; export CUDA_VISIBLE_DEVICES=$g; "$@" ) > "$LOGS/${name}-${SLURM_JOB_ID}.log" 2>&1
}

w0() {
  $E5 scripts/encode_pooled_multilayer.py --model $WEAK --style prm_native --layers 28 \
    --prm800k "train:$DATA/prm800k_probe_train_full.jsonl:30000" "${VT[@]:1}" "${PBS[@]}" \
    --out "$L/weak" --smoke "$SMOKE"
  if [[ $SMOKE == 0 ]]; then  # calib-20 needs whole ProcessBench subsets
    $E5 scripts/analysis/backbone_geometry.py --configs "weak_prm:$L/weak" --reps sep \
      --out "$CMP/geometry_weak_prm" > "$LOGS/geometry_weak-${SLURM_JOB_ID}.log" 2>&1 &
  fi
  pool 0 t10 math500
  wait
}
w1() {
  $E5 scripts/encode_pooled_multilayer.py --model $STRONG --style prm_native --layers 28 "${VT[@]}" \
    --out "$L/strong" --smoke "$SMOKE"
  $E5 scripts/encode_pooled_multilayer.py --model $Q3 --style verifier --layers 35 "${VT[@]}" \
    --out "$L/q3" --smoke "$SMOKE"
  pool 1 t10 gsm8k
}
w2() {
  $E5 scripts/read_token_probe.py encode --model $Q3 --ckpt "$OUT/tokens/rth_correct_seed1/read_token.pt" \
    "${VT[@]}" --out "$L/offline" --smoke "$SMOKE"
  $E5 scripts/read_token_inline.py encode --model $Q3 --ckpt "$OUT/tokens/rthics_seed1/read_token.pt" \
    "${VT[@]}" --out "$L/online" --smoke "$SMOKE"
  pool 2 t07 math500
}
w3() {
  $E5 scripts/analysis/cell_embed.py --cell_dir "$CELL" --store "$REF" --splits val:val_5k test:test_2k \
    --out "$L/probe"
  pool 3 t07 gsm8k
}
pids=()
lane 0 weak w0 & pids+=($!)
lane 1 strong_q3 w1 & pids+=($!)
lane 2 read_tokens w2 & pids+=($!)
lane 3 probe w3 & pids+=($!)
fail=0
names=(weak strong_q3 read_tokens probe)
for i in 0 1 2 3; do
  wait "${pids[$i]}" || { echo "[FAIL] ${names[$i]}"; fail=1; }
  grep -vE "Loading weights|UNEXPECTED|MISSING|Notes:|can be ignored|LOAD REPORT|^Key|^-+\+|^ *$" \
    "$LOGS/${names[$i]}-${SLURM_JOB_ID}.log" | tail -8
done
tail -n 4 "$LOGS/geometry_weak-${SLURM_JOB_ID}.log" || true

$E5 scripts/analysis/rep_views.py --out "$CMP/views-${SLURM_JOB_ID}.json" --cand \
  "Weak PRM (PRM800K)=$L/weak:sep:28" "Strong PRM (PRM-7B)=$L/strong:sep:28" \
  "Instruct last step token=$L/q3:last:35" "Best probe (d512)=$L/probe:z:-1" \
  "Offline read token=$L/offline:sep:-1" "Online read token=$L/online:sep:-1" \
  > "$LOGS/views-${SLURM_JOB_ID}.log" 2>&1 || [[ $SMOKE != 0 ]] || fail=1  # smoke items need not overlap
tail -n 9 "$LOGS/views-${SLURM_JOB_ID}.log"
# keep the small val / test encodes for re-plotting
for d in weak strong q3 probe offline online; do
  mkdir -p "$CMP/enc/$d" && cp "$L/$d"/{val,test}.shard*.{npz,meta.jsonl} "$CMP/enc/$d/" 2>/dev/null || true
done
echo "[$(date)] done, fail=$fail"
exit $fail
