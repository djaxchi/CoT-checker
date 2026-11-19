#!/bin/bash
#SBATCH --job-name=instep_nopb
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=02:45:00
#SBATCH --output=%x-%j.out
# InStep paper, no ProcessBench problem in any training data: InStep (PRM800K) trained without the
# 2,842 ProcessBench problems ("no-overlap", seeds 0-2 from job 701011, $RV/tokens/instep_prm_nopb_seed<s>).
# Everything under $NP = $RV/nopb unless said.
#   CPU   nopb_tools.py filter: each seed's 100,000-item draw minus ProcessBench problems ($NP/data),
#         and ReProbe traces of the filtered draws of seeds 0-2.
#   GPU0-2 ReProbe's head (released recipe, future visible, and the mask-corrected variant, future
#         masked) on the filtered draw of seed s, as job 701015 on the unfiltered one: $NP/rp_prm/seed<s>.
#   GPU3  (1) seeds 3 and 4 of the no-overlap token, job 701011's recipe (own 100,000-item draw,
#         --exclude_problems on the 2,842 problems, lr 3e-3, warm-up 0.05, one epoch) plus --val
#         val5k (sweep.json for the stability rule): $RV/tokens/instep_prm_nopb_seed<s>, encoded on the
#         8 test sets, ProcessBench and validation: $RV/rt/instep_prm_nopb_seed<s>;
#         (2) seeds 0-2 encoded on the 8 test sets and validation (ProcessBench exists, 701011);
#         (3) every seed's P(incorrect) on PRM800K val5k (chat layout, as score_val): $NP/val5k/seed<s>;
#         (4) STEP on the filtered draws of seeds 0-2 (union states of job 701015): $NP/step_prm.
#   CPU   sweep.json of seeds 0-2 rebuilt from the 701011 logs + val5k encodes (seeds 3-4 rebuilt too,
#         as a check against their own sweep.json); instep_lr_select.py over the five ($NP/selection);
#         paper_v1.py on Qwen3-8B with the arms instep_prm_nopb (rule-kept seeds), _all, _seed<s>
#         ($NP/final); pb_oracle.py (F1 at P = 0.5; baselines vs InStep); nopb_report.py ($NP/report.md).
# --time: ReProbe 104 min on 66,231 traces (701015) -> ~100 min on the filtered ones + scoring;
#   GPU3 ~90 min (two trainings of 17 min + validation, ten encodes); CPU stage ~10 min.
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-review}"
CM="/project/aip-azouaq/$USER/cot_mech"
OUT="$CM/instep_v1"
V1="$CM/reprobe_cmp_v1"; D="$V1/data"
PRMD="$SCRATCH/cot_mech/dense_full_7b_v1/data"
RV="$OUT/review"; W="$RV/data"; NP="$RV/nopb"
ST="$SCRATCH/cot_mech/instep_v1/states"
STU="$SCRATCH/cot_mech/instep_v1/review/prm_union"
LOGS="$RV/logs"
L="$SLURM_TMPDIR/nopb"
Q3=Qwen/Qwen3-8B
J=$SLURM_JOB_ID
mkdir -p "$LOGS" "$L" "$NP"/{data/traces,rp_prm,val5k,sweep,sweep_check,pb} "$RV/rt" "$RV/tokens"
module purge
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0
export HF_HOME=/project/aip-azouaq/$USER/hf_cache
unset TRANSFORMERS_CACHE
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 MKL_NUM_THREADS=12
cd "$PROJECT_ROOT"
echo "worktree $(git rev-parse --short HEAD) + review files; job $J"
virtualenv --no-download "$SLURM_TMPDIR/env" >/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip >/dev/null
pip install --no-index torch transformers numpy scipy scikit-learn pyyaml matplotlib 2>&1 | tail -1

PBSETS=(); RPS=()
for s in gsm8k math olympiadbench omnimath; do PBSETS+=("pb_$s:$D/pb/$s.items.jsonl"); RPS+=("pb_$s:$D/pb/$s.traces.jsonl"); done
TESTS=(); for s in math gsm8k proofnet trips meetings calendar strategyqa scienceqa; do TESTS+=("$s:$D/test/$s.items.jsonl"); done
VAL=("val:$D/rt/all_val.items.jsonl")
VALT="$W/traces/val5k.traces.jsonl"

# encode <gpu> <ckpt> <dst> <layout> <sets...>: head scores only (as the paper's runs keep)
encode() {
  local g=$1 ck=$2 dst=$3 lay=$4; shift 4
  local tmp="$L/enc_$(basename "$dst")_$RANDOM"
  CUDA_VISIBLE_DEVICES=$g python scripts/read_token_inline.py encode --model $Q3 --layout "$lay" \
    --ckpt "$ck" --out "$tmp" --prm800k "$@" --passive_blocks "" || return 1
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

# ---- CPU: filtered draws and ReProbe traces
python scripts/instep_v1/nopb_tools.py filter --full "$PRMD/prm800k_probe_train_full.jsonl" \
  --pb_problems "$W/pb_problems.json" --seeds 0,1,2,3,4 --out "$NP/data"
tp=()
for s in 0 1 2; do
  python scripts/instep_v1/prm_review_data.py traces --items "$PRMD/prm800k_probe_train_full.jsonl" \
    --uids "$NP/data/draw_seed$s.uids.json" --out "$NP/data/traces/train_seed$s.traces.jsonl" \
    > "$LOGS/nopb_traces_seed$s-$J.log" 2>&1 & tp+=($!)
done
for p in "${tp[@]}"; do wait "$p"; done
cat "$LOGS"/nopb_traces_seed*-$J.log

pids=()
for s in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$s python scripts/reprobe_cmp/reprobe_probe.py train --model $Q3 \
    --train "$NP/data/traces/train_seed$s.traces.jsonl" --val "$VALT" \
    --heads rp_prm:prm:visible rpc_prm:prm:masked --seed $s --out "$NP/rp_prm/seed$s" --sets "${RPS[@]}" \
    --val_scores > "$LOGS/nopb_rp_prm_seed$s-$J.log" 2>&1 & pids+=($!)
done
(
  set -e
  for s in 3 4; do
    name=instep_prm_nopb_seed$s
    CUDA_VISIBLE_DEVICES=3 python scripts/read_token_inline.py train --model $Q3 --init_text " correct" \
      --deep --head linear --chain --start_marker --start_deep --soft_attn no_cudnn \
      --layout chat --train "$PRMD/prm800k_probe_train_full.jsonl" --n_train 100000 \
      --exclude_problems "$W/pb_problems.json" --lr 3e-3 --warmup 0.05 --seed $s \
      --val "$PRMD/prm800k_val_5k.jsonl" --out "$L/tok_$name" > "$LOGS/$name-$J.log" 2>&1
    mkdir -p "$RV/tokens/$name" "$NP/sweep/seed$s"
    cp "$L/tok_$name/read_token.pt" "$RV/tokens/$name/"
    cp "$L/tok_$name/sweep.json" "$NP/sweep/seed$s/sweep.json"
    encode 3 "$RV/tokens/$name/read_token.pt" "$RV/rt/$name" reprobe "${TESTS[@]}" "${PBSETS[@]}" "${VAL[@]}"
  done
  for s in 0 1 2; do
    encode 3 "$RV/tokens/instep_prm_nopb_seed$s/read_token.pt" "$RV/rt/instep_prm_nopb_seed$s" reprobe "${TESTS[@]}" "${VAL[@]}"
  done
  for s in 0 1 2 3 4; do
    encode 3 "$RV/tokens/instep_prm_nopb_seed$s/read_token.pt" "$NP/val5k/seed$s" chat "val5k:$PRMD/prm800k_val_5k.jsonl"
  done
  CUDA_VISIBLE_DEVICES=3 python scripts/instep_v1/step_prm.py --train_states "$STU" --train_names union \
    --draws "$NP/data" --val_states "$STU" --val_name val5k --pb_states "$ST/qwen3_8b" --out "$NP/step_prm"
) > "$LOGS/nopb_gpu3-$J.log" 2>&1 & pids+=($!)
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
grep -h "^\[exclude\]\|^\[data\]\|^\[val\]\|^\[encode\] [a-z_0-9]*:\|^\[step_prm\]\|^\[score\]\|Error\|Traceback" \
  "$LOGS"/*-"$J".log | cut -c1-220 | tail -n 80 || true
[[ $fail == 0 ]] || exit 1

# ---- CPU: stability rule, paper numbers, report
for s in 0 1 2 3 4; do
  if [[ $s -le 2 ]]; then lg="$LOGS/instep_prm_nopb_seed$s-701011.log"; dst="$NP/sweep/seed$s"; else lg="$LOGS/instep_prm_nopb_seed$s-$J.log"; dst="$NP/sweep_check/seed$s"; fi
  python scripts/instep_v1/nopb_tools.py sweep --log "$lg" --seed $s --filter "$NP/data/filter.json" \
    --val_npz "$NP/val5k/seed$s/val5k.shard0.npz" --train "$PRMD/prm800k_probe_train_full.jsonl" \
    --val "$PRMD/prm800k_val_5k.jsonl" --out "$dst"
done
python - "$NP" <<'PY'
import json, sys
np_ = sys.argv[1]
for s in (3, 4):
    a = json.load(open(f"{np_}/sweep/seed{s}/sweep.json")); b = json.load(open(f"{np_}/sweep_check/seed{s}/sweep.json"))
    ca, cb = a["configs"][0], b["configs"][0]
    print(f"[check] seed {s}: const {a['const_loss']:.5f} vs rebuilt {b['const_loss']:.5f}; val AUROC "
          f"{ca['val_auroc']:.5f} vs encode {cb['val_auroc']:.5f}; final loss {ca['final_loss']:.4f} vs log {cb['final_loss']:.4f}")
PY
python scripts/analysis/instep_lr_select.py --source "instep_prm_nopb:auroc:$NP/sweep/seed*/sweep.json" --out "$NP/selection"
KEPT=$(python -c "
import json; t = json.load(open('$NP/selection/selection.json'))['tables']['instep_prm_nopb']
(c,) = t.values(); print(','.join(s for s in sorted(c['seeds']) if not c['seeds'][s]['flags']))")
echo "rule keeps seeds: $KEPT"
dirs() { local o=(); for s in ${1//,/ }; do o+=("$RV/rt/instep_prm_nopb_seed$s"); done; (IFS=,; echo "${o[*]}"); }
OV=("qwen3_8b:instep_prm_nopb=$(dirs "$KEPT")" "qwen3_8b:instep_prm_nopb_all=$(dirs 0,1,2,3,4)")
ARMS=(instep_prm_nopb instep_prm_nopb_all)
for s in 0 1 2 3 4; do OV+=("qwen3_8b:instep_prm_nopb_seed$s=$(dirs $s)"); ARMS+=(instep_prm_nopb_seed$s); done
TP=(); PP=()
for k in "${ARMS[@]}"; do
  TP+=("instep_ds-$k" "$k-prm7b")
  for c in prm7b prm800k instep_ds instep_self rp_rel_ds; do PP+=("$k-$c"); done
done
python scripts/instep_v1/paper_v1.py --out_root "$OUT" --v1 "$V1" --v2 "$CM/reprobe_cmp_v2" \
  --backbones qwen3_8b --no_bon --no_figures --n_boot 2000 --pb_boot 2000 --workers 44 \
  --final "$NP/final" --override "${OV[@]}" --extra_pairs "${TP[@]}" --extra_pb_pairs "${PP[@]}"
PA=()
for k in instep_prm "${ARMS[@]}"; do
  PA+=("$k=$(ls -d "$NP"/final/pb_scores/qwen3_8b/$k/seed* | paste -sd, -)")
done
python scripts/instep_v1/pb_oracle.py --arm "${PA[@]}" --out "$NP/pb/pb_oracle.json"
raw() { local o=(); for s in 0 1 2; do o+=("$1/seed$s${2:-}"); done; (IFS=,; echo "${o[*]}"); }
python scripts/instep_v1/pb_oracle.py --data "$D" --n_boot 2000 --out "$NP/pb/baselines.json" \
  --arm "${PA[1]}" "${PA[2]}" \
  --raw "step_prm_nopb=rp:$(raw "$NP/step_prm")" "rp_prm_nopb=rp:$(raw "$NP/rp_prm" /scores/rp_prm)" \
        "rpc_prm_nopb=rp:$(raw "$NP/rp_prm" /scores/rpc_prm)" \
  --pair instep_prm_nopb-step_prm_nopb instep_prm_nopb-rp_prm_nopb instep_prm_nopb-rpc_prm_nopb \
         instep_prm_nopb_all-step_prm_nopb instep_prm_nopb_all-rp_prm_nopb instep_prm_nopb_all-rpc_prm_nopb
python scripts/instep_v1/nopb_report.py --selection "$NP/selection/selection.json" --sweeps "$NP/sweep" \
  --filter "$NP/data/filter.json" --logs "$LOGS" --final "$NP/final/final.json" --reference "$OUT/final/final.json" \
  --pb_oracle "$NP/pb/pb_oracle.json" --baselines "$NP/pb/baselines.json" --out "$NP/report.md" --json "$NP/report.json"
