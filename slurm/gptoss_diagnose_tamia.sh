#!/bin/bash
#SBATCH --job-name=gptoss_diag
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=00:40:00
#SBATCH --output=%x-%j.out

# Three smoke attempts failed identically in transformers'
# _materialize_copy at tensor.to(device) with CUDA_ERROR_ILLEGAL_ADDRESS:
# 443012 (auto device map), 443041 (68GiB cap), 443043 (cap plus
# Mxfp4Config(dequantize=True), both confirmed active in the log). Memory and
# the dequantise switch are therefore both ruled out, and a fourth attempt at
# the same call would tell us nothing new.
#
# This job does not try to load the model. It splits the remaining hypotheses
# with three cheap checks, in increasing cost, and stops at the first failure:
#
#   1. can this stack copy ANY tensor to a GPU at all
#      -> if not, the fault is the driver/torch build and has nothing to do
#         with gpt-oss
#   2. can safetensors read the actual shards, and can one real MXFP4 tensor
#      make the same host-to-device trip
#      -> isolates a corrupt or unreadable checkpoint from the loader
#   3. can transformers build the model on CPU only, with no device placement
#      -> separates the MXFP4 dequantisation MATH from the device copy that
#         reports the error
#
# Whichever check fails first names the next fix. NO INTERNET on compute nodes.

set -uo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

MODEL_PATH="${MODEL_PATH:-$SCRATCH/shared_models/gpt-oss-120b}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=1

virtualenv --no-download "$SLURM_TMPDIR/env" >/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip >/dev/null
pip install --no-index torch transformers numpy accelerate safetensors 2>&1 | tail -1

python - <<'PY'
import os, sys, traceback, json
import torch
M = os.environ.get("MODEL_PATH", os.path.expandvars("$SCRATCH/shared_models/gpt-oss-120b"))
print(f"torch {torch.__version__}  cuda {torch.version.cuda}  gpus {torch.cuda.device_count()}")
print(f"capability {torch.cuda.get_device_capability(0)}  {torch.cuda.get_device_name(0)}")

print("\n[1] plain host-to-device copy")
try:
    t = torch.randn(4096, 4096)
    g = t.to("cuda:0")
    torch.cuda.synchronize()
    print(f"    OK: moved {tuple(g.shape)} {g.dtype}, sum {float(g.sum()):.3f}")
except Exception:
    traceback.print_exc(); print("    VERDICT: the stack cannot copy to GPU; gpt-oss is not the problem")
    sys.exit(0)

print("\n[2] read real shards and move one MXFP4 tensor")
try:
    from safetensors import safe_open
    idx = json.load(open(f"{M}/model.safetensors.index.json"))
    wm = idx["weight_map"]
    picked = []
    for name, shard in wm.items():
        if any(k in name for k in ("blocks", "scales")) and len(picked) < 3:
            picked.append((name, shard))
    if not picked:
        picked = list(wm.items())[:3]
    for name, shard in picked:
        with safe_open(f"{M}/{shard}", framework="pt", device="cpu") as f:
            t = f.get_tensor(name)
        print(f"    {name}: {tuple(t.shape)} {t.dtype} on cpu")
        g = t.to("cuda:0")
        torch.cuda.synchronize()
        print(f"      -> moved to cuda ok ({g.dtype})")
except Exception:
    traceback.print_exc()
    print("    VERDICT: the checkpoint tensors themselves cannot be read or moved")
    sys.exit(0)

print("\n[3] build the model on CPU only, no device placement")
try:
    from transformers import AutoModelForCausalLM, Mxfp4Config
    m = AutoModelForCausalLM.from_pretrained(
        M, local_files_only=True, dtype=torch.bfloat16, device_map=None,
        quantization_config=Mxfp4Config(dequantize=True))
    n = sum(p.numel() for p in m.parameters())
    print(f"    OK: built on CPU, {n/1e9:.1f}B params, dtype {next(m.parameters()).dtype}")
    print("    VERDICT: dequantisation is fine; the fault is device placement, so")
    print("             load on CPU and move shards deliberately, or use vLLM")
except Exception:
    traceback.print_exc()
    print("    VERDICT: dequantisation itself fails; transformers cannot read this")
    print("             checkpoint on this stack and vLLM is the path")
PY
echo "[$(date)] gptoss_diag done"
