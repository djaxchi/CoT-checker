#!/bin/bash
# Submit parametric_retrieval_component_v1 on TamIA (login node):
#   01 extract components (whole node, h100:4) -> merge
#   -> 02 experiment (afterok, whole node): baseline/val/select/test/analyze/capture
#
# Runs a login-node preflight and refuses to submit when a prerequisite from
# the completed access_v1 run is missing. Records job ids in
# <run_dir>/prgc_submit_manifest.json.
#
# Prerequisites (from the finished access_v1 run, already on the cluster):
#   runs/parametric_retrieval_access_v1/{metadata.parquet,grading.jsonl,
#     extraction_set.json,pairs.parquet,candidates.json}
#   runs/parametric_retrieval_access_v1/hidden_states_v1/hs_meta.parquet
#   Qwen/Qwen2.5-7B-Instruct in $HF_CACHE_ROOT

set -euo pipefail
cd "$HOME/CoT-checker"

RUN_DIR="${RUN_DIR:-runs/parametric_retrieval_access_v1}"
MODEL_DIR_NAME="models--Qwen--Qwen2.5-7B-Instruct"

if [[ -f slurm/s1_model_size/models.env ]]; then
    source slurm/s1_model_size/models.env
fi
HF_ROOT="${HF_CACHE_ROOT:-/project/aip-azouaq/$USER/hf_cache}"

echo "== PRGC component_v1 preflight (login node) =="
fail=0
for f in "$RUN_DIR/metadata.parquet" "$RUN_DIR/grading.jsonl" \
         "$RUN_DIR/extraction_set.json" "$RUN_DIR/pairs.parquet" \
         "$RUN_DIR/candidates.json" \
         "$RUN_DIR/hidden_states_v1/hs_meta.parquet" \
         scripts/parametric_retrieval/prgc_extract_components.py \
         scripts/parametric_retrieval/prgc_component.py; do
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

if [[ -e "$RUN_DIR/expE" && -z "${FORCE:-}" ]]; then
    echo "NOTE: $RUN_DIR/expE already exists (set FORCE=1 to overwrite)"
fi

[[ $fail -ne 0 ]] && { echo "preflight FAILED, nothing submitted"; exit 1; }
echo "preflight OK"

mkdir -p results/logs
ext_id=$(sbatch --parsable scripts/tamia/jobs/prgc/01_extract_components.sbatch)
echo "01_extract_components: $ext_id"
exp_id=$(sbatch --parsable --dependency=afterok:"$ext_id" \
    scripts/tamia/jobs/prgc/02_experiment.sbatch)
echo "02_experiment: $exp_id (afterok:$ext_id)"

python - "$RUN_DIR" "$revision" "$ext_id" "$exp_id" <<'EOF'
import json, sys, datetime
run, rev, ext_id, exp_id = sys.argv[1:5]
json.dump({
    "submitted": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "study": "parametric_retrieval_component_v1",
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "model_snapshot_revision": rev,
    "job_01_components": ext_id,
    "job_02_experiment": exp_id,
    "dependency": f"afterok:{ext_id}",
}, open(f"{run}/prgc_submit_manifest.json", "w"), indent=2)
EOF
echo "wrote $RUN_DIR/prgc_submit_manifest.json"
