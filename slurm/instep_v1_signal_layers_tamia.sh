#!/bin/bash
#SBATCH --job-name=instep_v1_signal_layers
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
# InStep paper, "where the signal forms" (one panel, four verifiers, per layer), Qwen3-8B, ReProbe
# DS data, on the steps of the stored DS seed-0 states ($ST8: train40k, val, 8 test sets).
# 1. GPU0-2: scripts/instep_v1/reprobe_layers.py, ReProbe's head retrained on each sampled block
#    (3,7,...,35) with its released recipe, seed 0 / 1 / 2, on the DS labels of the train40k
#    steps; pooled step representations of train40k, val and the tests to $SS/reprobe_layers/seed<s>.
#    GPU3: the InStep DS tokens of seeds 1 and 2 encoded with --read_blocks on the same sets
#    (step-token states per block) to $STB/qwen3_8b_ds_seed<s>.
# 2. scripts/instep_v1/signal_layers.py: the shared linear probe on every per-layer state, on one
#    joint step mask over all these dirs plus $SS/prm and $SS/reprobe; GPU0 InStep + text + PRM +
#    released ReProbe (reproduces the printed last-layer values), GPU1 the retrained ReProbe heads,
#    GPU2 the text states also read by STEP's own MLP (3 seeds), to $OUT/signal/layers_*.json.
# SMOKE=N: step 1 on the first N traces/items per set under *_smokeN with 1 epoch, then step 2
#    only on the existing final keys of GPU0 (checks the printed values); --time=00:30:00.
# --time: ReProbe 9-head training ~70 min (the 4-head 28.7k-trace run took 65 min; backbone
#    forward dominates, ~0.02 s/trace), dump ~12 min; InStep encodes ~10 min each; probes
#    ~30 s per key.
set -euo pipefail
SMOKE="${SMOKE:-0}"
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
CM="/project/aip-azouaq/$USER/cot_mech"
V1="$CM/reprobe_cmp_v1"
D="$V1/data"
SW="$CM/instep_lr_sweep"
OUT="${OUT:-$CM/instep_v1}"
ST8="${ST8:-$SCRATCH/cot_mech/instep_v1/states/qwen3_8b}"
SUF="$([[ $SMOKE == 0 ]] || echo "_smoke$SMOKE")"
SS0="$SCRATCH/cot_mech/instep_v1/signal_states"
RL="$SS0/reprobe_layers$SUF"
STB="$SCRATCH/cot_mech/instep_v1/states$SUF"
LOGS="$OUT/logs"
BLOCKS="3,7,11,15,19,23,27,31,35"
mkdir -p "$LOGS" "$RL" "$STB"
module purge
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0
export HF_HOME=/project/aip-azouaq/$USER/hf_cache
unset TRANSFORMERS_CACHE
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 MKL_NUM_THREADS=12
cd "$PROJECT_ROOT"
echo "git $(git rev-parse --short HEAD 2>/dev/null || echo none) smoke=$SMOKE rl=$RL stb=$STB"
virtualenv --no-download "$SLURM_TMPDIR/env" >/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip >/dev/null
pip install --no-index torch transformers numpy scipy scikit-learn pyyaml 2>&1 | tail -1
SM=(); EP=(); [[ $SMOKE == 0 ]] || { SM=(--smoke "$SMOKE"); EP=(--epochs 1); }

TESTS=(math gsm8k proofnet trips meetings calendar strategyqa scienceqa)
RI=("train40k:$D/rp/train.traces.jsonl" "val:$D/rp/val.traces.jsonl"); RK=()
EI=("train40k:$D/rt/ds_train.items.jsonl:40000" "val:$D/rt/all_val.items.jsonl")
for s in "${TESTS[@]}"; do RI+=("$s:$D/test/$s.traces.jsonl"); EI+=("$s:$D/test/$s.items.jsonl"); done
for s in train40k val "${TESTS[@]}"; do RK+=("$s:$ST8/$s.shard0.meta.jsonl"); done

pids=()
for seed in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$seed python scripts/instep_v1/reprobe_layers.py --train "$D/rp/train.traces.jsonl" \
    --val "$D/rp/val.traces.jsonl" --train_keep "$ST8/train40k.shard0.meta.jsonl" --sets "${RI[@]}" \
    --keep "${RK[@]}" --blocks "$BLOCKS" --seed $seed --out "$RL/seed$seed" "${SM[@]}" "${EP[@]}" \
    > "$LOGS/signal_layers_rp$seed$SUF-$SLURM_JOB_ID.log" 2>&1 & pids+=($!)
done
(
  for seed in ds_seed1 ds_seed2_b; do
    CUDA_VISIBLE_DEVICES=3 python scripts/read_token_inline.py encode --model Qwen/Qwen3-8B --layout reprobe \
      --ckpt "$SW/$seed/lr0.003_wu0.05/read_token.pt" --out "$STB/qwen3_8b_$seed" --prm800k "${EI[@]}" \
      --passive_blocks "" --read_blocks "$BLOCKS" "${SM[@]}" || exit 1
  done
) > "$LOGS/signal_layers_instep$SUF-$SLURM_JOB_ID.log" 2>&1 & pids+=($!)
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
grep -h "^\[rpl\] seed\|^\[rpl val\]\|^\[rpl dump\]\|^\[encode\] [a-z0-9]*:\|Error\|error" \
  "$LOGS"/signal_layers_*"$SUF-$SLURM_JOB_ID".log | tail -n 60 || true
[[ $fail == 0 ]] || exit 1

PROBE=(python scripts/instep_v1/signal_layers.py --data "$D")
if [[ $SMOKE != 0 ]]; then
  CUDA_VISIBLE_DEVICES=0 "${PROBE[@]}" --src "s0=$ST8=passive_final|sep" "prm=$SS0/prm=prm_final" "rp=$SS0/reprobe" \
    --out "$OUT/signal$SUF/layers_check.json"
  exit 0
fi
# every job sees the same sources (same joint mask); the regexes pick what each one fits
NONE='^$'
pids=()
CUDA_VISIBLE_DEVICES=0 "${PROBE[@]}" --src "s0=$ST8=read_block\\d+|sep|passive_final" "prm=$SS0/prm" "rp=$SS0/reprobe" \
  "rpl0=$RL/seed0=$NONE" "rpl1=$RL/seed1=$NONE" "rpl2=$RL/seed2=$NONE" \
  "in1=$STB/qwen3_8b_ds_seed1=read_block\\d+|sep" "in2=$STB/qwen3_8b_ds_seed2_b=read_block\\d+|sep" \
  --out "$OUT/signal/layers_base.json" \
  > "$LOGS/signal_layers_probe0-$SLURM_JOB_ID.log" 2>&1 & pids+=($!)
CUDA_VISIBLE_DEVICES=1 "${PROBE[@]}" --src "s0=$ST8=$NONE" "prm=$SS0/prm=$NONE" "rp=$SS0/reprobe=$NONE" \
  "rpl0=$RL/seed0" "rpl1=$RL/seed1" "rpl2=$RL/seed2" "in1=$STB/qwen3_8b_ds_seed1=$NONE" \
  "in2=$STB/qwen3_8b_ds_seed2_b=$NONE" --out "$OUT/signal/layers_reprobe.json" \
  > "$LOGS/signal_layers_probe1-$SLURM_JOB_ID.log" 2>&1 & pids+=($!)
CUDA_VISIBLE_DEVICES=2 "${PROBE[@]}" --src "s0=$ST8=passive_block\\d+|passive_final" "prm=$SS0/prm=$NONE" \
  "rp=$SS0/reprobe=$NONE" "rpl0=$RL/seed0=$NONE" "rpl1=$RL/seed1=$NONE" "rpl2=$RL/seed2=$NONE" \
  "in1=$STB/qwen3_8b_ds_seed1=$NONE" "in2=$STB/qwen3_8b_ds_seed2_b=$NONE" --mlp_src s0 \
  --out "$OUT/signal/layers_step_mlp.json" > "$LOGS/signal_layers_probe2-$SLURM_JOB_ID.log" 2>&1 & pids+=($!)
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
grep -h "^\[layers\]" "$LOGS"/signal_layers_probe*-"$SLURM_JOB_ID".log | tail -n 120 || true
[[ $fail == 0 ]] || exit 1
