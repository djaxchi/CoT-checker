#!/bin/bash
#SBATCH --job-name=rep_separation
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
# Class separation of end-of-step states on PRM800K test and ProcessBench
# (scripts/analysis/plot_rep_separation.py). Main comparison, one vector per step:
# Qwen3-8B Instruct last step token and the read token placed right after it, both
# hidden_states[35] (block 34 of 36), and Qwen2.5-Math-PRM-7B last step token at
# hidden_states[27] (block 26 of 28). Also shown: the read token's final post-norm
# state, and DRV's pooled 512-d embedding (a trained learner's output, not a step
# state). Closed-form geometry (scripts/analysis/prm_geometry.py, rep last_token, no
# kNN), every projection fit on the full PRM800K train split; test and ProcessBench
# are only projected.
#
# Kept for a mechanistic study ($MECH on /project, ~55 GB):
#   bundle/   scripts/read_token_mech_bundle.py: every layer of the last step token
#             (Qwen3-8B and PRM) and of the read token, the read token's head logit,
#             DRV's logit and embedding; all of test, val and ProcessBench, a seeded
#             20k subsample of train
#   vectors/  every split, all of train (scripts/analysis/save_split_vectors.py):
#             rt_L35, rt_final, drv_z, prm_L27
#
# Node-local: the read token's stores (as in read_token_leaderboard_tamia.sh; the
# rt_append store is its block-34 output) and the PRM span store (as in
# prm_backbone_tamia.sh, ~155 GB), rebuilt here because the jobs that made them
# kept them on their own node. Alongside on CPU: the Stage F paired trace bootstrap.
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
OUT="${OUT:-$SCRATCH/cot_mech/read_token_v1}"
F="$OUT/stage_f"
RS="${RS:-$OUT/rep_separation}"
DATA="${DATA:-$SCRATCH/cot_mech/dense_full_7b_v1/data}"
PB="${PB:-/scratch/d/dchikhi/cot-checker/processbench_full}"
REF="${REF:-/scratch/d/dchikhi/cot_mech/qwen3_8b_instruct_v1/repstore}"
DRV_CELL="${DRV_CELL:-$F/cells_fab/step_tokens__transformer_d512_l2_f2048_h8__seed42}"
MECH="${MECH:-/project/aip-azouaq/$USER/cot_mech/read_token_v1/mech}"
TRAIN_SAMPLE="${TRAIN_SAMPLE:-20000}"
PRM_MODEL=Qwen/Qwen2.5-Math-PRM-7B
PRM_LAYER=27
Q3=Qwen/Qwen3-8B
STEMS=(probe_train_full val_5k test_2k)
SUBSETS=(gsm8k math olympiadbench omnimath)
LOGS="$RS/logs"
GEO="$RS/geometry"
L="$SLURM_TMPDIR/rs"
mkdir -p "$LOGS" "$GEO" "$L"
cd "$PROJECT_ROOT"
echo "git $(git rev-parse --short HEAD)"
TOKEN="$(cat "$F/token.txt")"
CKPT="$OUT/tokens/$TOKEN/read_token.pt"
[[ -s "$CKPT" && -s "$DRV_CELL/model.pt" ]] || { echo "missing $CKPT or $DRV_CELL/model.pt"; exit 1; }
echo "token $TOKEN, DRV $DRV_CELL, mech -> $MECH"

module purge
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0
export HF_HOME=/project/aip-azouaq/$USER/hf_cache
unset TRANSFORMERS_CACHE
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
virtualenv --no-download "$SLURM_TMPDIR/env" >/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip >/dev/null
pip install --no-index torch transformers numpy scipy scikit-learn pyyaml 2>&1 | tail -1

# --- CPU, alongside everything: Stage F paired trace bootstrap -----------------------
python scripts/analysis/stage_f_bootstrap.py --stage_f "$F" \
  --instruct /scratch/d/dchikhi/cot_mech/qwen3_8b_instruct_v1/runs/rep_grid_q3 \
  --prm /scratch/d/dchikhi/cot_mech/prm_backbone_v1/cells --out "$F/bootstrap.json" \
  > "$LOGS/bootstrap-$SLURM_JOB_ID.log" 2>&1 &
boot_pid=$!

# --- read token stores on 4 GPUs -------------------------------------------------
refs=(); srcs=(); outs=()
for s in "${STEMS[@]}"; do
  refs+=("$REF/step_spans/$s"); srcs+=("prm800k:$DATA/prm800k_$s.jsonl"); outs+=(prm)
done
for s in "${SUBSETS[@]}"; do
  refs+=("$REF/pb_step_spans/$s"); srcs+=("pb:$s:$PB/processbench_$s.jsonl"); outs+=(pb)
done
pids=()
for g in 0 1 2 3; do
  (
    set -e
    export CUDA_VISIBLE_DEVICES=$g
    for k in "${!refs[@]}"; do
      python scripts/read_token_store.py --model $Q3 --ckpt "$CKPT" --ref "${refs[$k]}" \
        --source "${srcs[$k]}" --out_root "$L/${outs[$k]}" --shards "$g/4"
    done
  ) > "$LOGS/rt_store_gpu$g-$SLURM_JOB_ID.log" 2>&1 &
  pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
(( fail == 0 )) || { tail -n 5 "$LOGS"/rt_store_gpu*-"$SLURM_JOB_ID".log; exit 1; }
echo "[$(date)] read token stores done"

# --- PRM span store on 4 GPUs (background), meanwhile DRV embedding + 4 geometries --
common=(--model_name_or_path "$PRM_MODEL" --local_files_only --span_only --layer "$PRM_LAYER"
        --max_seq_len 2048 --batch_size 8 --model_dtype bfloat16 --num_shards 4)
enc=()
for i in 0 1 2 3; do
  ( CUDA_VISIBLE_DEVICES=$i python scripts/encode_processbench_token_store.py \
      --raw_specs gsm8k:$PB/processbench_gsm8k.jsonl math:$PB/processbench_math.jsonl \
                  olympiadbench:$PB/processbench_olympiadbench.jsonl \
                  omnimath:$PB/processbench_omnimath.jsonl \
      --rep_root "$L/prm_states/pb_step_spans" --shard_idx $i "${common[@]}" &&
    CUDA_VISIBLE_DEVICES=$i python scripts/encode_prm800k_token_store.py \
      --data_dir "$DATA" --rep_root "$L/prm_states/step_spans" \
      --splits prm800k_val_5k.jsonl:val_5k prm800k_test_2k.jsonl:test_2k \
               prm800k_probe_train_full.jsonl:probe_train_full \
      --shard_idx $i "${common[@]}" ) > "$LOGS/prm_encode_gpu$i-$SLURM_JOB_ID.log" 2>&1 &
  enc+=($!)
done

for side in prm pb; do
  CUDA_VISIBLE_DEVICES=3 python scripts/analysis/drv_embed.py --cell_dir "$DRV_CELL" \
    --layers_root "$L/$side/rt_layers" --out_root "$L/$side/drv_embed"
done
geo() {  # gpu name prm_store pb_store
  CUDA_VISIBLE_DEVICES=$1 python scripts/analysis/prm_geometry.py --out_dir "$GEO" \
    --reps last_token --no_knn --backbone "$2=$3,$4" > "$LOGS/geometry_$2-$SLURM_JOB_ID.log" 2>&1
}
gp=()
geo 1 instruct "$REF/step_spans" "$REF/pb_step_spans" & gp+=($!)
geo 2 rt "$L/prm/rt_append" "$L/pb/rt_append" & gp+=($!)
( geo 3 rt_final "$L/prm/rt_layers" "$L/pb/rt_layers" && geo 3 drv "$L/prm/drv_embed" "$L/pb/drv_embed" ) & gp+=($!)

fail=0; for p in "${enc[@]}"; do wait "$p" || fail=1; done
(( fail == 0 )) || { tail -n 5 "$LOGS"/prm_encode_gpu*-"$SLURM_JOB_ID".log; exit 1; }
echo "[$(date)] PRM encode done"
geo 0 prm "$L/prm_states/step_spans" "$L/prm_states/pb_step_spans" & gp+=($!)

# --- mechanistic bundle: every layer at the end of each step, on 4 GPUs -------------
# all of test_2k, val_5k and ProcessBench; a seeded 20k subsample of train
bpids=()
for g in 0 1 2 3; do
  (
    set -e
    export CUDA_VISIBLE_DEVICES=$g
    for k in "${!refs[@]}"; do
      sample=0; [[ "$(basename "${refs[$k]}")" == probe_train_full ]] && sample=$TRAIN_SAMPLE
      python scripts/read_token_mech_bundle.py --model $Q3 --prm_model $PRM_MODEL --ckpt "$CKPT" \
        --drv_cell "$DRV_CELL" --ref "${refs[$k]}" --source "${srcs[$k]}" --out "$L/bundle" \
        --sample $sample --shards "$g/4"
    done
  ) > "$LOGS/bundle_gpu$g-$SLURM_JOB_ID.log" 2>&1 &
  bpids+=($!)
done

# --- vectors of every split to shared storage: the geometry inputs and DRV's embedding
python scripts/analysis/save_split_vectors.py --out "$MECH/vectors" \
  rt_final="$L/prm/rt_layers,$L/pb/rt_layers" rt_L35="$L/prm/rt_append,$L/pb/rt_append" \
  drv_z="$L/prm/drv_embed,$L/pb/drv_embed" \
  prm_L27="$L/prm_states/step_spans,$L/prm_states/pb_step_spans"

for p in "${bpids[@]}"; do wait "$p" || fail=1; done
tail -n 2 "$LOGS"/bundle_gpu*-"$SLURM_JOB_ID".log
# written node-local (scattered row writes), then copied in one pass
rm -rf "$MECH/bundle" && cp -r "$L/bundle" "$MECH/bundle"
for p in "${gp[@]}"; do wait "$p" || fail=1; done
tail -n 3 "$LOGS"/geometry_*-"$SLURM_JOB_ID".log
wait $boot_pid || fail=1
cat "$LOGS/bootstrap-$SLURM_JOB_ID.log"

# --- checks: the re-encoded stores against earlier runs, the bundle against the stores
python scripts/analysis/check_rep_separation.py --geo "$GEO" --mech "$MECH" --ref "$REF" || fail=1
du -sh "$MECH"/bundle "$MECH"/vectors/*
echo "[$(date)] done, fail=$fail"
exit $fail
