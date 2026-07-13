#!/bin/bash
# Submit the parametric_retrieval_access_v1 pipeline on TamIA (login node):
#   01 generate (ONE whole node, h100:4, greedy only, merge + output gate)
#   -> 02 pairs + extract + merge + logit lens (afterok, ONE whole node)
#
# Runs a login-node preflight first and refuses to submit when anything is
# missing or stale. Records the model snapshot revision and job ids in
# <run_dir>/submit_manifest.json.
#
# Prerequisites (synced from laptop):
#   runs/parametric_retrieval_access_v1/{metadata.parquet,build_manifest.json}
#   data/wikiprofile/wikiprofile.csv
#   Qwen/Qwen2.5-7B-Instruct in $HF_CACHE_ROOT

set -euo pipefail
cd "$HOME/CoT-checker"

RUN_DIR="${RUN_DIR:-runs/parametric_retrieval_access_v1}"
MODEL_DIR_NAME="models--Qwen--Qwen2.5-7B-Instruct"

if [[ -f slurm/s1_model_size/models.env ]]; then
    source slurm/s1_model_size/models.env
fi
HF_ROOT="${HF_CACHE_ROOT:-/project/aip-azouaq/$USER/hf_cache}"

echo "== PRGA v1 preflight (login node) =="

fail=0
for f in "$RUN_DIR/metadata.parquet" "$RUN_DIR/build_manifest.json" \
         data/wikiprofile/wikiprofile.csv \
         scripts/parametric_retrieval/prga_generate.py \
         scripts/parametric_retrieval/prga_pairs.py \
         scripts/parametric_retrieval/prga_extract.py \
         scripts/parametric_retrieval/prga_logitlens.py; do
    if [[ ! -f "$f" ]]; then echo "MISSING: $f"; fail=1; fi
done

snap_dir="$HF_ROOT/hub/$MODEL_DIR_NAME/snapshots"
[[ -d "$snap_dir" ]] || snap_dir="$HF_ROOT/$MODEL_DIR_NAME/snapshots"
if [[ ! -d "$snap_dir" ]]; then
    echo "MISSING: Qwen2.5-7B-Instruct snapshot under $HF_ROOT"; fail=1
    revision="absent"
else
    revision=$(ls -1 "$snap_dir" | head -1)
    echo "model snapshot revision: $revision"
fi

for stale in "$RUN_DIR/generations.jsonl" "$RUN_DIR"/generations_shard*.jsonl; do
    if [[ -e "$stale" && -z "${FORCE:-}" ]]; then
        echo "STALE OUTPUT: $stale (set FORCE=1 to allow overwrite)"; fail=1
    fi
done

echo "-- home usage (hidden states go to \$SCRATCH, not here):"
df -h "$HOME" | tail -1 || true
echo "-- scratch target:"
df -h "$SCRATCH" | tail -1 || true

[[ $fail -ne 0 ]] && { echo "preflight FAILED, nothing submitted"; exit 1; }
echo "preflight OK"

mkdir -p results/logs
gen_id=$(sbatch --parsable scripts/tamia/jobs/prga/01_generate.sbatch)
echo "01_generate (whole node): $gen_id"
ext_id=$(sbatch --parsable --dependency=afterok:"$gen_id" \
    scripts/tamia/jobs/prga/02_extract.sbatch)
echo "02_extract (pairs+extract+logitlens): $ext_id (afterok:$gen_id)"

python - "$RUN_DIR" "$revision" "$gen_id" "$ext_id" <<'EOF'
import json, sys, datetime
run, rev, gen_id, ext_id = sys.argv[1:5]
json.dump({
    "submitted": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "model_snapshot_revision": rev,
    "job_01_generate": gen_id,
    "job_02_extract": ext_id,
    "dependency": f"afterok:{gen_id}",
}, open(f"{run}/submit_manifest.json", "w"), indent=2)
EOF
echo "wrote $RUN_DIR/submit_manifest.json"
