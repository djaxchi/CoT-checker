#!/bin/bash
# Pre-flight dependency check for SSAE.
#
# Verifies that the TamIA offline wheelhouse (pip --no-index) can satisfy the
# packages that the SSAE pipeline imports beyond what the easy-probe pipeline
# already needed (transformers, tokenizers).
#
# Run this once interactively on a TamIA login node BEFORE submitting
# slurm/train_ssae_methods_tamia.sh:
#
#   bash scripts/check_ssae_deps.sh
#
# Expected exit code 0 if everything resolves; otherwise the script prints the
# failing package and exits with code 2.

set -euo pipefail

module load StdEnv/2023 python/3.12

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

virtualenv --no-download "$TMPDIR/env"
source "$TMPDIR/env/bin/activate"

pip install --no-index --upgrade pip

# Packages the easy-probe pipeline already needs.
pip install --no-index torch numpy tqdm scikit-learn pyyaml

# SSAE-only additions.
for pkg in transformers tokenizers; do
  if ! pip install --no-index "$pkg" >/dev/null 2>&1; then
    echo "[check_ssae_deps] MISSING from offline wheelhouse: $pkg" >&2
    echo "[check_ssae_deps] Fix: ask TamIA support to add $pkg, or pre-stage" >&2
    echo "[check_ssae_deps] a wheel via 'pip download $pkg -d <wheel_dir>'"  >&2
    echo "[check_ssae_deps] from an internet-connected node and pass" >&2
    echo "[check_ssae_deps] --find-links <wheel_dir> in the SLURM script." >&2
    exit 2
  fi
done

# Verify imports actually resolve.
python - <<'PY'
import transformers, tokenizers
print(f"transformers={transformers.__version__} tokenizers={tokenizers.__version__}")
PY

echo "[check_ssae_deps] OK"
