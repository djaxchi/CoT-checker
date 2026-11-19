#!/bin/bash
#SBATCH --job-name=instep_rev
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=02:15:00
#SBATCH --output=%x-%j.out
# InStep paper, reviewer runs on Qwen3-8B. Everything under $RV = $OUT/review.
#   STAGE=w1   reviewer W1 + Q1 (retrain).
#              GPU0-2: three other per-layer parameterisations of the step token, each as a rung
#              next to "+ offsets" (embedding + the parameterisation, no chain, no marker), on the
#              ReProbe DS data, with the ladder's recipe search (5 lrs x 5% warm-up, 3 seeds,
#              validation only, as instep_v1_rungsweep_tamia.sh):
#                kvp_ds      2 learned key/value prefix slots per layer, seen by the step token only
#                            (147,456 parameters, the offsets' count)
#                lora_ds     rank-1 LoRA on q_proj and v_proj, active at the step-token position only
#                scaleoff_ds per-layer scale and offset at the step-token position
#              GPU3: InStep (PRM800K), same recipe as the paper's tokens (lr 3e-3, 5% warm-up,
#              100,000-item draw per seed), with every ProcessBench problem removed from the draw
#              (--exclude_problems), seeds 0-2, each encoded on ProcessBench: $RV/rt/instep_prm_nopb_seed<s>.
#   STAGE=pick picked config per W1 rung (instep_lr_select.py), its 3 seeds encoded on the 8 test
#              sets, ProcessBench and validation: $RV/rt/<rung>_lrpick_seed<s>.
#   STAGE=w2   reviewer W2: GPU0-2 ReProbe's hidden-state head (released recipe, future visible,
#              and the mask-corrected variant, future masked; one backbone pass for both) trained on
#              InStep (PRM800K) seed s's 100,000 steps and labels, best epoch on PRM800K validation,
#              scored on ProcessBench: $RV/rp_prm/seed<s>/scores/{rp_prm,rpc_prm}/.
#              GPU3: last-layer text states of the union of the three draws and of the validation
#              items (chat layout, as InStep trains), then STEP (step_prm.py) per seed: $RV/step_prm.
#   Data (login node, before w1/w2): prm_review_data.py draws -> $RV/data.
# SMOKE=N: first N items / traces everywhere, outputs under review_smokeN.
set -euo pipefail
STAGE="${STAGE:?w1, pick or w2}"
SMOKE="${SMOKE:-0}"
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-review}"
CM="/project/aip-azouaq/$USER/cot_mech"
OUT="$CM/instep_v1"
V1="$CM/reprobe_cmp_v1"; D="$V1/data"
PRMD="$SCRATCH/cot_mech/dense_full_7b_v1/data"
SUF="$([[ $SMOKE == 0 ]] || echo "_smoke$SMOKE")"
RV="${RV:-$OUT/review$SUF}"
W="$OUT/review/data"   # draws, shared with the smoke
ST="$SCRATCH/cot_mech/instep_v1/states"
STU="$SCRATCH/cot_mech/instep_v1/review$SUF/prm_union"
LOGS="$RV/logs"
L="$SLURM_TMPDIR/rev"
GRID="${GRID:-5e-4:0.05,1e-3:0.05,1.5e-3:0.05,2e-3:0.05,3e-3:0.05}"
Q3=Qwen/Qwen3-8B
mkdir -p "$LOGS" "$L" "$RV/rungsweep" "$RV/rt" "$RV/tokens"
module purge
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0
export HF_HOME=/project/aip-azouaq/$USER/hf_cache
unset TRANSFORMERS_CACHE
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 MKL_NUM_THREADS=12
cd "$PROJECT_ROOT"
echo "worktree $(git rev-parse --short HEAD) + review files; stage=$STAGE smoke=$SMOKE rv=$RV"
virtualenv --no-download "$SLURM_TMPDIR/env" >/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip >/dev/null
pip install --no-index torch transformers numpy scipy scikit-learn pyyaml 2>&1 | tail -1
SM=(); [[ $SMOKE == 0 ]] || SM=(--smoke "$SMOKE")
[[ -f $W/draws.json && -f $W/pb_problems.json ]] || { echo "run prm_review_data.py draws first"; exit 2; }

flags() {
  case $1 in
    kvp)      echo "--kv_prefix 2" ;;
    lora)     echo "--interv lora --rank 1" ;;
    scaleoff) echo "--deep --scale" ;;
  esac
}

# sweep <gpu> <rung> <seed>: the rung's lr grid on ReProbe DS data, validation only
sweep() {
  local g=$1 r=$2 s=$3
  CUDA_VISIBLE_DEVICES=$g python scripts/read_token_inline.py train --model $Q3 --init_text " correct" \
    --head linear $(flags "$r") --layout reprobe --train "$D/rt/ds_train.items.jsonl" --n_train 0 \
    --val "$D/rt/ds_val.items.jsonl" --sweep "$GRID" --seed "$s" --soft_attn no_cudnn \
    --out "$RV/rungsweep/${r}_ds_seed$s" "${SM[@]}" > "$LOGS/rung_${r}_ds_seed$s-$SLURM_JOB_ID.log" 2>&1
}

# encode <gpu> <ckpt> <dst> <sets...>: head scores only (as the paper's runs keep)
encode() {
  local g=$1 ck=$2 dst=$3; shift 3
  local tmp="$L/enc_$(basename "$dst")"
  CUDA_VISIBLE_DEVICES=$g python scripts/read_token_inline.py encode --model $Q3 --layout reprobe \
    --ckpt "$ck" --out "$tmp" --prm800k "$@" --passive_blocks "" "${SM[@]}" || return 1
  mkdir -p "$dst"
  python - "$tmp" "$dst" <<'PY' || return 1
import sys, shutil, numpy as np
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
for z in src.glob("*.shard0.npz"):
    a = np.load(z)
    np.savez(dst / z.name, y=a["y"], kept=a["kept"], head_p_correct=a["head_p_correct"])
    shutil.copy(src / z.name.replace(".npz", ".meta.jsonl"), dst)
PY
  rm -rf "$tmp"
}
PBSETS=(); for s in gsm8k math olympiadbench omnimath; do PBSETS+=("pb_$s:$D/pb/$s.items.jsonl"); done
ALLSETS=(); for s in math gsm8k proofnet trips meetings calendar strategyqa scienceqa; do ALLSETS+=("$s:$D/test/$s.items.jsonl"); done
ALLSETS+=("${PBSETS[@]}" "val:$D/rt/all_val.items.jsonl")

pids=()
if [[ $STAGE == w1 ]]; then
  g=0
  for r in kvp lora scaleoff; do
    ( f=0; for s in 0 1 2; do sweep $g $r $s || f=1; done; exit $f ) & pids+=($!)
    g=$((g + 1))
  done
  (
    f=0
    for s in 0 1 2; do
      name=instep_prm_nopb_seed$s
      CUDA_VISIBLE_DEVICES=3 python scripts/read_token_inline.py train --model $Q3 --init_text " correct" \
        --deep --head linear --chain --start_marker --start_deep --soft_attn no_cudnn \
        --layout chat --train "$PRMD/prm800k_probe_train_full.jsonl" --n_train 100000 \
        --exclude_problems "$W/pb_problems.json" --lr 3e-3 --warmup 0.05 --seed $s \
        --out "$L/tok_$name" "${SM[@]}" > "$LOGS/$name-$SLURM_JOB_ID.log" 2>&1 || { f=1; continue; }
      mkdir -p "$RV/tokens/$name" && cp "$L/tok_$name/read_token.pt" "$RV/tokens/$name/"
      encode 3 "$L/tok_$name/read_token.pt" "$RV/rt/$name" "${PBSETS[@]}" >> "$LOGS/$name-$SLURM_JOB_ID.log" 2>&1 || f=1
    done
    exit $f
  ) & pids+=($!)
elif [[ $STAGE == pick ]]; then
  : > "$RV/rungsweep/picks.txt"
  for r in kvp lora scaleoff; do
    n=${r}_ds
    python scripts/analysis/instep_lr_select.py --source "$n:ap:$RV/rungsweep/${n}_seed*/sweep.json" \
      --out "$RV/rungsweep/selection_$n" > /dev/null || { echo "$n none FAILED" >> "$RV/rungsweep/picks.txt"; continue; }
    python - "$RV/rungsweep/selection_$n/selection.json" "$n" >> "$RV/rungsweep/picks.txt" <<'PY'
import json, sys
j = json.load(open(sys.argv[1])); n = sys.argv[2]; t = j["tables"][n]; p = j["picks"][n]
if p is None:
    p = min(t, key=lambda k: (t[k]["n_unstable"], -(t[k]["mean"] or 0)))
    print(n, p, "NO_STABLE_CONFIG", f"unstable {t[p]['n_unstable']}/3 mean {t[p]['mean']:.4f}")
else:
    print(n, p, "rule", f"mean {t[p]['mean']:.4f}")
PY
  done
  cat "$RV/rungsweep/picks.txt"
  tag() { awk -v n="$1" '$1 == n {print $2}' "$RV/rungsweep/picks.txt"; }
  # 9 encodes over 4 GPUs: GPU g takes encodes g, g+4, g+8
  jobs=()
  for r in kvp lora scaleoff; do t=$(tag ${r}_ds); [[ -z $t || $t == none ]] && continue
    for s in 0 1 2; do jobs+=("$RV/rungsweep/${r}_ds_seed$s/$t/read_token.pt|$RV/rt/${r}_ds_lrpick_seed$s"); done; done
  for g in 0 1 2 3; do
    ( f=0; for ((i = g; i < ${#jobs[@]}; i += 4)); do IFS='|' read -r ck dst <<< "${jobs[$i]}"
        encode $g "$ck" "$dst" "${ALLSETS[@]}" || f=1; done; exit $f
    ) > "$LOGS/pick_g$g-$SLURM_JOB_ID.log" 2>&1 & pids+=($!)
  done
elif [[ $STAGE == w2 ]]; then
  mkdir -p "$W/traces" "$STU"
  RPS=(); for s in gsm8k math olympiadbench omnimath; do RPS+=("pb_$s:$D/pb/$s.traces.jsonl"); done
  VALT="$W/traces/val5k${SUF}.traces.jsonl"
  [[ -f $VALT ]] || python scripts/instep_v1/prm_review_data.py traces --items "$PRMD/prm800k_val_5k.jsonl" \
    --out "$VALT" "${SM[@]}"
  for s in 0 1 2; do
    (
      tr="$W/traces/train_seed$s${SUF}.traces.jsonl"
      [[ -f $tr ]] || python scripts/instep_v1/prm_review_data.py traces --items "$PRMD/prm800k_probe_train_full.jsonl" \
        --uids "$W/draw_seed$s.uids.json" --out "$tr" "${SM[@]}" || exit 1
      CUDA_VISIBLE_DEVICES=$s python scripts/reprobe_cmp/reprobe_probe.py train --model $Q3 --train "$tr" --val "$VALT" \
        --heads rp_prm:prm:visible rpc_prm:prm:masked --seed $s --out "$RV/rp_prm/seed$s" --sets "${RPS[@]}" \
        --val_scores "${SM[@]}"
    ) > "$LOGS/rp_prm_seed$s-$SLURM_JOB_ID.log" 2>&1 & pids+=($!)
  done
  (
    set -e
    CUDA_VISIBLE_DEVICES=3 python scripts/read_token_inline.py encode --model $Q3 --layout chat \
      --ckpt "$OUT/tokens/emb_prm_seed0/read_token.pt" --out "$STU" --passive_blocks "" \
      --prm800k "union:$W/union.items.jsonl" "val5k:$PRMD/prm800k_val_5k.jsonl" "${SM[@]}"
    TN=union
    if [[ $SMOKE != 0 ]]; then
      # a smoke encodes only the first N union rows: train every seed on the rows that exist
      python - "$STU" "$W" "$RV/step_draws" <<'PY'
import json, sys
from pathlib import Path
st, w, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
have = [json.loads(l)["uid"] for l in open(st / "union.shard0.meta.jsonl")]
out.mkdir(parents=True, exist_ok=True)
for s in range(3):
    (out / f"draw_seed{s}.uids.json").write_text(json.dumps(have))
PY
      DR="$RV/step_draws"
    else
      DR="$W"
    fi
    CUDA_VISIBLE_DEVICES=3 python scripts/instep_v1/step_prm.py --train_states "$STU" --train_names $TN \
      --draws "$DR" --val_states "$STU" --val_name val5k --pb_states "$ST/qwen3_8b" --out "$RV/step_prm"
  ) > "$LOGS/step_prm-$SLURM_JOB_ID.log" 2>&1 & pids+=($!)
else
  echo "unknown STAGE=$STAGE"; exit 2
fi
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
grep -h "^\[val\]\|^\[exclude\]\|^\[encode\] pb\|^\[step_prm\]\|^\[score\] pb\|Error" "$LOGS"/*-"$SLURM_JOB_ID".log | tail -n 60 || true
exit $fail
