#!/bin/bash
# Submit parametric_retrieval_steer_v1 on TamIA (login node): a single job that
# clamps the prgm flip-neurons on unrelated prompts and analyzes the result.
#
# Prerequisites (from finished minimal_v1 / component_v1 runs, on cluster):
#   runs/parametric_retrieval_access_v1/expF/run.parquet
#   runs/parametric_retrieval_access_v1/expE/selection.json
#   runs/parametric_retrieval_access_v1/neuron_states_v1/neuron_meta.parquet
#   runs/parametric_retrieval_access_v1/metadata.parquet
#   Qwen/Qwen2.5-7B-Instruct in $HF_CACHE_ROOT

set -euo pipefail
cd "$HOME/CoT-checker"

RUN_DIR="${RUN_DIR:-runs/parametric_retrieval_access_v1}"
MODEL_DIR_NAME="models--Qwen--Qwen2.5-7B-Instruct"

if [[ -f slurm/s1_model_size/models.env ]]; then
    source slurm/s1_model_size/models.env
fi
HF_ROOT="${HF_CACHE_ROOT:-/project/aip-azouaq/$USER/hf_cache}"

echo "== PRGS steer_v1 preflight (login node) =="
fail=0
for f in "$RUN_DIR/expF/run.parquet" "$RUN_DIR/expE/selection.json" \
         "$RUN_DIR/neuron_states_v1/neuron_meta.parquet" \
         "$RUN_DIR/metadata.parquet" \
         scripts/parametric_retrieval/prgs_steer.py; do
    if [[ ! -e "$f" ]]; then echo "MISSING: $f"; fail=1; fi
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

if [[ -e "$RUN_DIR/expG" && -z "${FORCE:-}" ]]; then
    echo "NOTE: $RUN_DIR/expG already exists (set FORCE=1 to overwrite)"
fi

[[ $fail -ne 0 ]] && { echo "preflight FAILED, nothing submitted"; exit 1; }
echo "preflight OK"

mkdir -p results/logs
job_id=$(sbatch --parsable scripts/tamia/jobs/prgs/01_steer.sbatch)
echo "01_steer: $job_id"

python - "$RUN_DIR" "$revision" "$job_id" <<'EOF'
import json, sys, datetime
run, rev, job_id = sys.argv[1:4]
json.dump({
    "submitted": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "study": "parametric_retrieval_steer_v1",
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "model_snapshot_revision": rev,
    "job_01_steer": job_id,
}, open(f"{run}/prgs_submit_manifest.json", "w"), indent=2)
EOF
echo "wrote $RUN_DIR/prgs_submit_manifest.json"
