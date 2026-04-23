#!/bin/bash
# Run this ONCE on the login node before submitting any jobs.
# TamIA compute nodes have no internet access — everything must be downloaded here.
#
# Usage (on tamia.alliancecan.ca login node):
#   bash scripts/slurm/tamia_setup.sh

set -euo pipefail

PROJECT_DIR="$HOME/CoT-checker"           # adjust if cloned elsewhere
STORE="$HOME/projects/aip-azouaq/$USER"   # backed-up project space

HF_CACHE="$STORE/hf_cache"
DATA_DIR="$STORE/probe_data"
CKPT_DIR="$STORE/checkpoints"

mkdir -p "$HF_CACHE" "$DATA_DIR" "$CKPT_DIR"

echo "=== [1/4] Loading Python environment ==="
module purge
module load StdEnv/2023 python/3.11
virtualenv --no-download "$HOME/venvs/cot" 2>/dev/null || echo "(venv already exists)"
source "$HOME/venvs/cot/bin/activate"

echo "=== [2/4] Installing dependencies ==="
pip install --no-index --upgrade pip 2>/dev/null || pip install --upgrade pip
# Install from wheels available on the cluster first, fall back to PyPI
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121 \
    || pip install torch torchvision
pip install transformers datasets huggingface_hub tqdm numpy

echo "=== [3/4] Downloading SSAE checkpoint ==="
python - <<'PYEOF'
import os, sys
os.environ["HF_HOME"] = os.path.expandvars("$HOME/projects/aip-azouaq/$USER/hf_cache")
from huggingface_hub import hf_hub_download
dest = hf_hub_download(
    repo_id="Miaow-Lab/SSAE-Checkpoints",
    filename="gsm8k-385k_Qwen2.5-0.5b_spar-10.pt",
)
import shutil, pathlib
target = pathlib.Path(os.path.expandvars("$HOME/projects/aip-azouaq/$USER/checkpoints"))
target.mkdir(parents=True, exist_ok=True)
shutil.copy(dest, target / "gsm8k-385k_Qwen2.5-0.5b_spar-10.pt")
print(f"Checkpoint saved to {target}")
PYEOF

echo "=== [4/4] Pre-downloading Qwen2.5-0.5B weights and Math-Shepherd dataset ==="
python - <<'PYEOF'
import os
os.environ["HF_HOME"] = os.path.expandvars("$HOME/projects/aip-azouaq/$USER/hf_cache")
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

print("  Qwen2.5-0.5B tokenizer + model weights...")
AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
AutoModel.from_pretrained("Qwen/Qwen2.5-0.5B")

print("  Math-Shepherd (streaming=False to cache locally)...")
ds = load_dataset("peiyi9979/Math-Shepherd", split="train")
print(f"  Math-Shepherd: {len(ds):,} rows cached")
PYEOF

echo ""
echo "=== Setup complete ==="
echo "  Checkpoint : $CKPT_DIR/gsm8k-385k_Qwen2.5-0.5b_spar-10.pt"
echo "  HF cache   : $HF_CACHE"
echo "  Data dir   : $DATA_DIR"
echo ""
echo "Next steps:"
echo "  1. sbatch scripts/slurm/tamia_generate_data.sh"
echo "  2. (after it finishes) sbatch scripts/slurm/tamia_train_probe.sh"
