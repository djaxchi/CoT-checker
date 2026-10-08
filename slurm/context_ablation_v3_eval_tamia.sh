#!/bin/bash
#SBATCH --job-name=cav3_eval
#SBATCH --account=aip-azouaq
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=%x-%j.out

# Collect view predictions per source and evaluate the context ladder (CPU).
set -euo pipefail
: "${SNAP:?}" "${R:?}"
module purge
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0
virtualenv --no-download "$SLURM_TMPDIR/env" >/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index numpy scikit-learn 2>&1 | tail -1
cd "$SNAP"
P=/project/aip-azouaq/$USER/cot_mech
for src in v1human v2ds; do
  python scripts/collect_context_views.py --fits_root "$R/prod/fits" --views_meta "$R/views/q/meta" \
    --v1_manifest $P/bidirectional_token_probe_v1/manifest_v1 \
    --v2_manifest $P/bidirectional_token_probe_v2/manifest_v2 --source $src --out "$R/eval"
done
python scripts/eval_contextual_token_probe.py --fits_root "$R/eval/v1human/fits" --manifest "$R/eval/v1human" \
  --fit_prefix "" --out_dir "$R/results/v1human" --contrasts q-full,none-full,prev1-full,q-prev1 \
  --step_set rp_test=rp_test --step_set rp_test_post=rp_test:post --step_set rp_test_pre=rp_test:pre
python scripts/eval_contextual_token_probe.py --fits_root "$R/eval/v2ds/fits" --manifest "$R/eval/v2ds" \
  --fit_prefix "" --out_dir "$R/results/v2ds" --contrasts q-full,none-full,prev1-full,q-prev1 \
  --step_set rp_test_post=test:post --step_set rp_test_pre=test:pre --step_set prm_human_test=prm_human_test
echo "[$(date)] eval done"
