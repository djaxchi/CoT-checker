#!/bin/bash
# Submit parametric_retrieval_minimal_v1 on TamIA (login node):
#   01 extract MLP neuron activations (whole node) -> merge
#   -> 02 experiment (afterok): coord + neuron + attribution + greedy, analyze
#
# Prerequisites (from the finished component_v1 / access_v1 runs, on cluster):
#   runs/parametric_retrieval_access_v1/expE/{selection.json,baseline.parquet}
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

echo "== PRGM minimal_v1 preflight (login node) =="
fail=0
for f in "$RUN_DIR/expE/selection.json" "$RUN_DIR/expE/baseline.parquet" \
         "$RUN_DIR/metadata.parquet" "$RUN_DIR/grading.jsonl" \
         "$RUN_DIR/extraction_set.json" "$RUN_DIR/pairs.parquet" \
         "$RUN_DIR/candidates.json" \
         "$RUN_DIR/hidden_states_v1/hs_meta.parquet" \
         scripts/parametric_retrieval/prgm_extract_neurons.py \
         scripts/parametric_retrieval/prgm_minimal.py; do
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

if [[ -e "$RUN_DIR/expF" && -z "${FORCE:-}" ]]; then
    echo "NOTE: $RUN_DIR/expF already exists (set FORCE=1 to overwrite)"
fi

[[ $fail -ne 0 ]] && { echo "preflight FAILED, nothing submitted"; exit 1; }
echo "preflight OK"

mkdir -p results/logs
ext_id=$(sbatch --parsable scripts/tamia/jobs/prgm/01_extract_neurons.sbatch)
echo "01_extract_neurons: $ext_id"
exp_id=$(sbatch --parsable --dependency=afterok:"$ext_id" \
    scripts/tamia/jobs/prgm/02_experiment.sbatch)
echo "02_experiment: $exp_id (afterok:$ext_id)"

python - "$RUN_DIR" "$revision" "$ext_id" "$exp_id" <<'EOF'
import json, sys, datetime
run, rev, ext_id, exp_id = sys.argv[1:5]
json.dump({
    "submitted": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "study": "parametric_retrieval_minimal_v1",
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "model_snapshot_revision": rev,
    "job_01_neurons": ext_id,
    "job_02_experiment": exp_id,
    "dependency": f"afterok:{ext_id}",
}, open(f"{run}/prgm_submit_manifest.json", "w"), indent=2)
EOF
echo "wrote $RUN_DIR/prgm_submit_manifest.json"
