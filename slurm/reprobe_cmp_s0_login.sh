#!/bin/bash
# reprobe_comparison_v1 S0 (docs/reprobe_comparison_v1_plan.md), run on the TamIA LOGIN node
# (it downloads ReProbe's released managers and probes; compute nodes are offline):
#   bash slurm/reprobe_cmp_s0_login.sh            from a snapshot or the checkout
# Writes $R/data (test sets, gate T, training traces/items, ProcessBench in ReProbe layout)
# and $R/released/{ds,self} (config.yaml + weights.pth).
set -euo pipefail
R="${R:-/project/aip-azouaq/$USER/cot_mech/reprobe_cmp_v1}"
V2="${V2:-/project/aip-azouaq/$USER/cot_mech/bidirectional_token_probe_v2}"
PB="${PB:-/scratch/d/dchikhi/cot-checker/processbench_full}"
export HF_HOME=/project/aip-azouaq/$USER/hf_cache
unset TRANSFORMERS_CACHE
mkdir -p "$R/data" "$R/released"
python scripts/reprobe_cmp/build_testsets.py --out "$R/data"
python scripts/reprobe_cmp/build_train.py --labels "ds:$V2/raw/ds.parquet" "self:$V2/raw/self.parquet" \
  --v2_meta "$V2/manifest_v2/meta" --pb_dir "$PB" --out "$R/data"
python - "$R/released" <<'PY'
import sys
from huggingface_hub import snapshot_download
for k, repo in (("ds", "JingweiNi/ReProbe_Qwen3-8B_DeepSeek_anno"), ("self", "JingweiNi/ReProbe_Qwen3-8B_self_anno")):
    print(snapshot_download(repo, local_dir=f"{sys.argv[1]}/{k}"))
PY
ls -la "$R/data/test" "$R/data/rp" "$R/data/rt" "$R/data/pb" "$R/released"/*
