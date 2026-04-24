#!/bin/bash
# Run on the LOGIN NODE (compute nodes have no internet).
# Downloads Qwen2.5-0.5B-Instruct and Qwen2.5-Math-7B-Instruct into project space
# so compute jobs can load them with HF_DATASETS_OFFLINE=1, TRANSFORMERS_OFFLINE=1.

set -euo pipefail

STORE="/project/aip-azouaq/$USER"
HF_CACHE="$STORE/hf_cache"

mkdir -p "$HF_CACHE"

# Arrow/Python must be loaded BEFORE venv activation
module purge
module load StdEnv/2023 gcc arrow/24.0.0 python/3.11

source "$HOME/venvs/cot/bin/activate"

export HF_HOME="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_CACHE"

python - <<'PYEOF'
import os
from huggingface_hub import snapshot_download

cache = os.environ["HF_HOME"]
models = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-Math-7B-Instruct",
]
for m in models:
    print(f"\n=== Downloading {m} ===")
    p = snapshot_download(
        repo_id=m,
        cache_dir=cache,
        allow_patterns=[
            "*.json", "*.txt", "*.safetensors", "tokenizer*", "merges.txt",
        ],
    )
    print(f"  -> {p}")

print("\nDone. Models cached in", cache)
PYEOF
