#!/bin/bash
# Submit parametric_retrieval_sae_decomp on TamIA (login node): downloads the
# L28 andyrdt SAE (login node has internet), then a short GPU job that encodes
# the flip pairs' residuals and decomposes the difference in the SAE basis.
#
# Prerequisites (from access_v1, on cluster):
#   runs/parametric_retrieval_access_v1/hidden_states_v1/{hs_meta.parquet,
#     layer_28.safetensors}
#   runs/parametric_retrieval_access_v1/pairs.parquet

set -euo pipefail
cd "$HOME/CoT-checker"

RUN_DIR="${RUN_DIR:-runs/parametric_retrieval_access_v1}"
STORE="/project/aip-azouaq/$USER"
if [[ -f slurm/s1_model_size/models.env ]]; then
    source slurm/s1_model_size/models.env
fi
export HF_HOME="${HF_CACHE_ROOT:-$STORE/hf_cache}"
SAE_ROOT="${SAE_ROOT:-$HF_HOME/models/andyrdt/saes-qwen2.5-7b-instruct}"
TRAINER="${TRAINER:-1}"

echo "== PRGD sae_decomp preflight (login node) =="
fail=0
for f in "$RUN_DIR/hidden_states_v1/hs_meta.parquet" \
         "$RUN_DIR/hidden_states_v1/layer_28.safetensors" \
         "$RUN_DIR/pairs.parquet" \
         scripts/parametric_retrieval/prgd_sae_decomp.py; do
    if [[ ! -e "$f" ]]; then echo "MISSING: $f"; fail=1; fi
done
[[ $fail -ne 0 ]] && { echo "preflight FAILED, nothing submitted"; exit 1; }

# ---- download the L28 SAE (trainer_1) on the login node (needs internet) ----
ae_pt="$SAE_ROOT/resid_post_layer_27/trainer_${TRAINER}/ae.pt"
if [[ ! -f "$ae_pt" ]]; then
    echo "== downloading L28 SAE (resid_post_layer_27/trainer_${TRAINER}) =="
    module load StdEnv/2023 python/3.11 2>/dev/null || true
    source "$HOME/venvs/cot/bin/activate" 2>/dev/null || true
    if command -v hf >/dev/null 2>&1; then
        hf download andyrdt/saes-qwen2.5-7b-instruct \
            --include "resid_post_layer_27/trainer_${TRAINER}/*" \
            --local-dir "$SAE_ROOT"
    else
        huggingface-cli download andyrdt/saes-qwen2.5-7b-instruct \
            --include "resid_post_layer_27/trainer_${TRAINER}/*" \
            --local-dir "$SAE_ROOT"
    fi
fi
[[ -f "$ae_pt" ]] || { echo "SAE download failed: $ae_pt missing"; exit 1; }
echo "SAE present: $ae_pt"
echo "preflight OK"

mkdir -p results/logs
job_id=$(SAE_ROOT="$SAE_ROOT" TRAINER="$TRAINER" sbatch --parsable \
    --export=ALL,SAE_ROOT="$SAE_ROOT",TRAINER="$TRAINER" \
    scripts/tamia/jobs/prgd/01_decomp.sbatch)
echo "01_decomp: $job_id"

python - "$RUN_DIR" "$job_id" <<'EOF'
import json, sys, datetime
run, job_id = sys.argv[1:3]
json.dump({
    "submitted": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "study": "parametric_retrieval_sae_decomp",
    "sae": "andyrdt/saes-qwen2.5-7b-instruct resid_post_layer_27",
    "job_01_decomp": job_id,
}, open(f"{run}/prgd_submit_manifest.json", "w"), indent=2)
EOF
echo "wrote $RUN_DIR/prgd_submit_manifest.json"
