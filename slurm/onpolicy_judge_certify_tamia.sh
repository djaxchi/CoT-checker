#!/bin/bash
#SBATCH --job-name=onpolicy_judge
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --output=%x-%j.out

# Stage 2: pick the judge by measuring it, not by arguing about it.
#
# Three candidates, all already in the HF cache, scored on one shared set of
# human-labelled ProcessBench traces:
#
#   Qwen3-8B-Base        the generator judging its own writing (self-supervised,
#                        the published ReProbe setting, and free)
#   Qwen2.5-32B          four times the size, a different family, so its errors
#                        are not correlated with the generator's
#   Qwen2.5-7B-Instruct  same size class as the first, but instruction-tuned:
#                        separates "bigger" from "tuned to follow instructions"
#
# All three answer the same questions, and three degenerate strategies are scored
# on the same traces underneath them. A judge that has only learned that mistakes
# come late in a solution posts a respectable F1_PB, and the always-last-step row
# is the only thing that makes that visible.
#
# The certification is run in the SAME configuration as deployment: the judge is
# told whether the final answer was right and is not shown the answer itself,
# because ProcessBench carries the first and not the second. A number measured
# under different information than the deployed run describes a different judge.
#
# NO INTERNET on compute nodes.

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH/cot_mech/onpolicy_v1}"
PB_DIR="${PB_DIR:-/scratch/d/dchikhi/cot-checker/processbench_full}"
HF_CACHE="${HF_CACHE:-/project/aip-azouaq/$USER/hf_cache}"
N_PER_SUBSET="${N_PER_SUBSET:-100}"
JUDGE_DIR="$RUN_ROOT/judge"
CERT_SET="$JUDGE_DIR/certification_traces.jsonl"

export HF_HOME="$HF_CACHE" TRANSFORMERS_CACHE="$HF_CACHE"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$JUDGE_DIR" "$RUN_ROOT/logs"
LOG_FILE="$RUN_ROOT/logs/onpolicy_judge-${SLURM_JOB_ID:-$$}.log"

cd "$PROJECT_ROOT"
cat <<BANNER
================================================================
job        : ${SLURM_JOB_NAME:-onpolicy_judge}  id: ${SLURM_JOB_ID:-N/A}
git_commit : $(git rev-parse HEAD 2>/dev/null || echo unknown)
cert set   : $N_PER_SUBSET traces x 4 ProcessBench subsets
out        : $JUDGE_DIR
================================================================
BANNER

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch transformers numpy accelerate

python scripts/onpolicy/build_judge_certification_set.py \
  --pb_dir "$PB_DIR" --n_per_subset "$N_PER_SUBSET" \
  --out "$CERT_SET" | tee -a "$LOG_FILE"

# name:hf_path:flags. Each runs on the whole node (device_map=auto) in turn: the
# 32B does not fit one H100 at bfloat16, and running them one at a time keeps the
# per-process memory budget from being divided by a concurrency that a 32B load
# would blow through anyway.
JUDGES=(
  "qwen3_8b_base:Qwen/Qwen3-8B-Base:"
  "qwen25_32b:Qwen/Qwen2.5-32B:"
  "qwen25_7b_instruct:Qwen/Qwen2.5-7B-Instruct:--chat"
)

for entry in "${JUDGES[@]}"; do
  name="${entry%%:*}"; rest="${entry#*:}"
  model="${rest%%:*}"; flags="${rest#*:}"
  snap="$HF_CACHE/hub/models--$(echo "$model" | sed 's|/|--|g')/snapshots"
  if [[ ! -d "$snap" || -z "$(ls -A "$snap" 2>/dev/null)" ]]; then
    echo "[skip] $name: no local snapshot for $model" | tee -a "$LOG_FILE"; continue
  fi
  echo "=== judge $name ($model) ===" | tee -a "$LOG_FILE"
  python scripts/onpolicy/judge_steps.py \
    --traces "$CERT_SET" \
    --out "$JUDGE_DIR/cert_${name}.jsonl" \
    --report "$JUDGE_DIR/cert_${name}_report.json" \
    --model_name_or_path "$model" --local_files_only --model_dtype bfloat16 \
    --certify $flags 2>&1 | tee -a "$LOG_FILE"
done

echo
echo "=== bake-off ==="
python - <<PY | tee -a "$LOG_FILE"
import json, glob
from pathlib import Path
rows = []
for f in sorted(glob.glob("$JUDGE_DIR/cert_*_report.json")):
    r = json.loads(Path(f).read_text())
    rows.append((Path(f).stem.replace("cert_", "").replace("_report", ""), r))
if not rows:
    raise SystemExit("no reports")
print(f"{'judge':<22}{'F1_PB':>8}{'Acc_err':>9}{'Acc_cor':>9}{'exact':>8}{'parsefail':>11}")
for name, r in sorted(rows, key=lambda x: -x[1]["judge"]["F1_PB"]):
    j = r["judge"]
    print(f"{name:<22}{j['F1_PB']:>8.3f}{j['Acc_error']:>9.3f}{j['Acc_correct']:>9.3f}"
          f"{j['exact_match_all']:>8.3f}{r['parse_failure_rate']:>11.3f}")
print()
for name, b in rows[0][1]["baselines"].items():
    print(f"{name:<22}{b['F1_PB']:>8.3f}{b['Acc_error']:>9.3f}{b['Acc_correct']:>9.3f}"
          f"{b['exact_match_all']:>8.3f}")
print()
for name, r in rows:
    mp = r.get("mean_relative_position")
    if mp:
        print(f"{name:<22} points at {mp['predicted']:.2f} of the way through; "
              f"the true errors sit at {mp['true']:.2f}")
print()
print("Judge-judge agreement (exact first-error index, shared traces):")
preds = {}
for name, _ in rows:
    preds[name] = {x["id"]: (x["first_error"] if x["parse_ok"] else None)
                   for x in (json.loads(l) for l in
                             open(f"$JUDGE_DIR/cert_{name}.jsonl"))}
names = list(preds)
for i, a in enumerate(names):
    for b in names[i+1:]:
        shared = set(preds[a]) & set(preds[b])
        agree = sum(1 for k in shared if preds[a][k] == preds[b][k])
        print(f"  {a} vs {b}: {agree}/{len(shared)} = {agree/max(1,len(shared)):.3f}")
PY

echo "[$(date)] onpolicy_judge done"
