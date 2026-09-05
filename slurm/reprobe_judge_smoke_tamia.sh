#!/bin/bash
#SBATCH --job-name=gptoss_smoke
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out

# Phase 2 smoke test: can GPT-OSS-120B load and annotate offline on one node,
# and how fast?
#
# Nothing here is committed to until this reports. It loads the local snapshot
# with no network, annotates a handful of trajectories under the ReProbe
# protocol, checks that the output parses, and prints a throughput figure that
# decides whether transformers is enough or the run needs vLLM.
#
# The model is MXFP4 (config quantization_config.quant_method = mxfp4), 117B
# total with 4 of 128 experts active. At MXFP4 the weights are ~61 GiB and fit
# one H100; the whole node is requested so device_map can spill if transformers
# dequantises to bf16, which would need ~240 GiB and still fits across four.
#
# NO INTERNET on compute nodes: local_files_only everywhere, and both offline
# flags set so a stray hub call fails loudly instead of hanging.

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH/cot_mech/reprobe_v1}"
MODEL_PATH="${MODEL_PATH:-$SCRATCH/shared_models/gpt-oss-120b}"
TRACES="${TRACES:-$RUN_ROOT/reprobe_train_judge_traces.jsonl}"
N="${N:-16}"
BATCH="${BATCH:-2}"
# Job 443012 died in tensor.to() with CUDA_ERROR_ILLEGAL_ADDRESS after
# device_map="auto" packed GPU 0 to 79.4 of 81.5 GiB. MXFP4 inference needs
# triton_kernels, which is absent from the offline wheelhouse and which the
# kernels library would otherwise fetch from the hub, so the checkpoint
# dequantises to bf16: ~234 GiB of weights against 4 x 81.5 GiB. An explicit
# per-GPU cap spreads them evenly and leaves room for activations.
MAX_MEM_GIB="${MAX_MEM_GIB:-68}"
DTYPE="${DTYPE:-bfloat16}"
MAX_NEW="${MAX_NEW:-320}"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
# Both previous failures reported an async CUDA fault, so the stack trace could
# have been pointing at the wrong call. If this one fails too, the trace is the
# real one.
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"
export HF_HOME="${HF_HOME:-/project/aip-azouaq/$USER/hf_cache}"
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/smoke"

[[ -d "$MODEL_PATH" ]] || { echo "[FATAL] no model at $MODEL_PATH" >&2; exit 2; }
[[ -f "$MODEL_PATH/model.safetensors.index.json" ]] || {
  echo "[FATAL] no weight index at $MODEL_PATH" >&2; exit 2; }
[[ -f "$TRACES" ]] || { echo "[FATAL] no traces at $TRACES" >&2; exit 2; }

cd "$PROJECT_ROOT"
cat <<BANNER
================================================================
job        : ${SLURM_JOB_NAME:-gptoss_smoke}  id: ${SLURM_JOB_ID:-N/A}
git_commit : $(git rev-parse --short HEAD 2>/dev/null || echo unknown)
model      : $MODEL_PATH  ($(du -sh "$MODEL_PATH" | cut -f1), offline)
traces     : $TRACES  ($(wc -l <"$TRACES") available, annotating $N)
batch      : $BATCH   max_new_tokens: $MAX_NEW
dtype      : $DTYPE   per-GPU cap: ${MAX_MEM_GIB}GiB (bf16 dequant, ~234GiB total)
================================================================
BANNER

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch transformers numpy accelerate triton kernels 2>&1 | tail -2
# triton_kernels is what the MXFP4 path needs and it is not in the wheelhouse;
# the attempt is logged rather than hidden so the fallback to bf16 is on record.
pip install --no-index triton_kernels 2>&1 | tail -1 || echo "[note] triton_kernels unavailable offline: MXFP4 path disabled, bf16 dequant it is"

python -c "
import torch, transformers
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(),
      'gpus', torch.cuda.device_count())
print('transformers', transformers.__version__)
"

python scripts/onpolicy/judge_local_reprobe.py \
  --traces "$TRACES" \
  --out "$RUN_ROOT/smoke/labels_smoke.jsonl" \
  --report "$RUN_ROOT/smoke/smoke_report.json" \
  --model_path "$MODEL_PATH" \
  --max_traces "$N" --batch_size "$BATCH" --max_new_tokens "$MAX_NEW" \
  --dtype "$DTYPE" --max_memory_gib "$MAX_MEM_GIB" --dequantize_mxfp4

echo
echo "=== what the judge actually said ==="
python - <<PY
import json
from pathlib import Path
rows = [json.loads(l) for l in
        Path("$RUN_ROOT/smoke/labels_smoke.jsonl").read_text().splitlines() if l.strip()]
print(f"{len(rows)} annotated, {sum(r['parse_ok'] for r in rows)} parsed")
for r in rows[:4]:
    print("-" * 70)
    print(f"traj {r['traj_uid']}  steps {r['n_steps']}  answer_correct {r['traj_correct']}")
    print(f"faulty {r['faulty_steps']}  labels {r['step_labels']}")
    print("raw:", (r["raw"] or "")[:400].replace("\n", " | "))
PY
echo "[$(date)] gptoss_smoke done"
