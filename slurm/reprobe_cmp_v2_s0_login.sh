#!/bin/bash
# reprobe_comparison_v2 S0 (docs/reprobe_comparison_v2_plan.md), on the TamIA LOGIN node
# (downloads ReProbe's released managers, training parquets and probes):
#   BB=qwen3_32b bash slurm/reprobe_cmp_v2_s0_login.sh     or BB=phi4
# Writes $R/<backbone>/data (test sets, gate T, training traces/items, ProcessBench) and,
# for Qwen3-32B, $R/<backbone>/released/oss.
set -euo pipefail
BB="${BB:?set BB=qwen3_32b or BB=phi4}"
R="${R:-/project/aip-azouaq/$USER/cot_mech/reprobe_cmp_v2}/$BB"
V2="${V2:-/project/aip-azouaq/$USER/cot_mech/bidirectional_token_probe_v2}"
PB="${PB:-/scratch/d/dchikhi/cot-checker/processbench_full}"
# small downloads only (managers, parquets, tokenizers); /scratch is at its quota
export HF_HOME="${S0_HF_HOME:-/project/aip-azouaq/$USER/hf_cache}"
unset TRANSFORMERS_CACHE
mkdir -p "$R/data" "$R/raw" "$R/released"
python scripts/reprobe_cmp/build_testsets.py --backbone "$BB" --out "$R/data"
python - "$R/raw" "$BB" <<'PY'
import sys
from huggingface_hub import hf_hub_download
out, bb = sys.argv[1], sys.argv[2]
sets = {"qwen3_32b": {"oss": "JingweiNi/train_prm800k_gpt-oss-120b_annotate_qwen3_32b_finished"},
        "phi4": {"qwen3": "JingweiNi/train_prm800k_finished_qwen3_annotate_phi-4_prm",
                 "self": "JingweiNi/train_prm800k_finished_self_annotate_phi-4_prm"}}[bb]
for k, repo in sets.items():
    p = hf_hub_download(repo, "data/train-00000-of-00001.parquet", repo_type="dataset")
    import shutil; shutil.copy(p, f"{out}/{k}.parquet"); print(k, p)
if bb == "qwen3_32b":
    from huggingface_hub import snapshot_download
    print(snapshot_download("JingweiNi/ReProbe_Qwen3-32B_GPT-OSS-120B_anno", local_dir=f"{out}/../released/oss"))
PY
LABELS=$(ls "$R/raw"/*.parquet | sort -r | sed -E 's#.*/([a-z0-9]+)\.parquet#\1:&#')
if [ "$BB" = phi4 ]; then LABELS="qwen3:$R/raw/qwen3.parquet self:$R/raw/self.parquet"; fi
python scripts/reprobe_cmp/build_train.py --backbone "$BB" --labels $LABELS \
  --v2_meta "$V2/manifest_v2/meta" --pb_dir "$PB" --out "$R/data"
ls -la "$R/data/test" "$R/data/rp" "$R/data/rt" "$R/data/pb" "$R/released"
