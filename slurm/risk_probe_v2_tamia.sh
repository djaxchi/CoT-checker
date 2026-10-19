#!/bin/bash
#SBATCH --job-name=rp2
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=03:00:00

# risk_probe_v2 (docs/risk_probe_v2_plan.md, section 9), one H100:4 node:
#   0. label.jsonl from risk_probe_v1's aligned views (build_label_jsonl.py), unless frozen
#   1. encode the label set with the leaderboard's span encoder (verifier template,
#      Qwen3-8B, hidden_states[35]) onto node-local disk, 4 shards
#   2. score the label set and ProcessBench with the three leaderboard d512 seeds,
#      checking ProcessBench against their saved scores (gate 2)
#   3. hyperparameter search on fold 0 (soft and harmful), then folds 1-4 and the full
#      fits (seeds 42, 43, 44) with the selected setting, 4 GPUs
#   4. analyze.py
# Required env: SNAP, OUT. Optional: SMOKE=1 (400 label items per shard, 1 epoch,
# one grid point, 4 bootstrap resamples).
set -euo pipefail
: "${SNAP:?}" "${OUT:?}"
C=/project/aip-azouaq/$USER/cot_mech
R1=$C/risk_probe_v1
R2=$C/risk_probe_v2
SV=$C/step_value_labels_v1/label/summary/step_values.jsonl
PB_STORE=/scratch/d/dchikhi/cot_mech/qwen3_8b_instruct_v1/repstore/pb_step_spans
CELLS=$C/instruct_leaderboard_v1/cells/step_tokens__transformer_d512_l2_f2048_h8__seed
HF_CACHE=/project/aip-azouaq/$USER/hf_cache
MODEL=Qwen/Qwen3-8B
STORE=$SLURM_TMPDIR/repstore
DATA=$R2/data
mkdir -p "$OUT/logs" "$DATA"
SMOKE="${SMOKE:-0}"
if [[ "$SMOKE" == 1 ]]; then
  LIMIT=(--limit_per_file 400); TR=(--epochs 1 --lr_grid 3e-4 --wd_grid 0.0); NBOOT=4
else
  LIMIT=(); TR=(); NBOOT=200
fi

module purge
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0
export HF_HOME="$HF_CACHE" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
cd "$SNAP"
grep -E "snapshot|git_head" SNAPSHOT_INFO
virtualenv --no-download "$SLURM_TMPDIR/env" >/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip >/dev/null
pip install --no-index torch transformers numpy scipy scikit-learn 2>&1 | tail -1
t0=$(date +%s)

# ---- 0. label jsonl (frozen once; its audit holds the sha256)
if [[ ! -s "$DATA/prm800k_label.jsonl" ]]; then
  python scripts/risk_probe_v2/build_label_jsonl.py --views "$R1/views" --step_values "$SV" \
    --out "$DATA/prm800k_label.jsonl"
fi
sha256sum "$DATA/prm800k_label.jsonl" | tee "$OUT/label_jsonl.sha256"

# ---- 1. encode
pids=()
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i python scripts/encode_prm800k_token_store.py --data_dir "$DATA" --rep_root "$STORE" \
    --splits prm800k_label.jsonl:label --model_name_or_path "$MODEL" --local_files_only --span_only \
    --layer 35 --max_seq_len 4096 --batch_size 16 --shard_idx $i --num_shards 4 "${LIMIT[@]}" \
    > "$OUT/logs/encode_$i.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || { tail -20 "$OUT"/logs/encode_*.log >&2; exit 1; }; done
du -sh "$STORE" | tee "$OUT/store_size.txt"
echo "[encode] $(( $(date +%s) - t0 ))s" | tee "$OUT/encode_time.txt"

# ---- 2. validity cells (GPU 2, alongside the search on GPUs 0 and 1)
CUDA_VISIBLE_DEVICES=2 python scripts/risk_probe_v2/score_cells.py \
  --cells v42=${CELLS}42 v43=${CELLS}43 v44=${CELLS}44 --label_store "$STORE" --pb_store "$PB_STORE" \
  --check_saved --out_dir "$OUT/validity" > "$OUT/logs/validity.log" 2>&1 &
VPID=$!

# ---- 3. fits
fit() {  # gpu target fold seed [hp_from]
  local name; [[ "$3" == full ]] && name="full_$2_s$4" || name="f$3_$2"
  [[ -s "$OUT/fits/$name/results.json" ]] && { echo "[skip] $name done"; return 0; }   # a resubmitted run resumes
  CUDA_VISIBLE_DEVICES=$1 python scripts/risk_probe_v2/train_risk_cell.py --store "$STORE" \
    --labels "$DATA/prm800k_label.jsonl" --target "$2" --fold "$3" --seed "$4" --pb_store "$PB_STORE" \
    --out_dir "$OUT/fits/$name" "${TR[@]}" ${5:+--hp_from $5} > "$OUT/logs/fit_$name.log" 2>&1
}
fit 0 soft 0 42 & a=$!
fit 1 harmful 0 42 & b=$!
wait $a && wait $b || { tail -20 "$OUT"/logs/fit_f0_*.log >&2; exit 1; }
JOBS=()
for t in soft harmful; do
  for k in 1 2 3 4; do JOBS+=("$t $k 42"); done
  for s in 42 43 44; do JOBS+=("$t full $s"); done
done
slot() {  # gpu, first job index, stride
  local g=$1 rc=0 j
  for ((j=$2; j<${#JOBS[@]}; j+=$3)); do
    read -r t k s <<< "${JOBS[$j]}"
    fit "$g" "$t" "$k" "$s" "$OUT/fits/f0_$t/results.json" || { echo "[FAIL] ${JOBS[$j]}"; rc=1; }
  done
  return $rc
}
slot 0 0 4 & p0=$!; slot 1 1 4 & p1=$!; slot 2 2 4 & p2=$!; slot 3 3 4 & p3=$!
fail=0
wait $VPID || { echo "[FAIL] validity scoring"; tail -20 "$OUT/logs/validity.log"; fail=1; }
for p in $p0 $p1 $p2 $p3; do wait "$p" || fail=1; done
[[ $fail == 0 ]] || exit 1
cat "$OUT"/validity/*__gate.json
echo "[fits] $(( $(date +%s) - t0 ))s"

# ---- 4. analysis
python scripts/risk_probe_v2/analyze.py --labels "$DATA/prm800k_label.jsonl" --fits "$OUT/fits" \
  --validity "$OUT/validity" --n_boot "$NBOOT" --workers 32 --out_dir "$OUT/analysis" \
  > "$OUT/logs/analyze.log" 2>&1 || { tail -30 "$OUT/logs/analyze.log" >&2; exit 1; }
cat "$OUT/analysis/summary.md"
echo "[$(date)] risk_probe_v2 done in $(( $(date +%s) - t0 ))s"
