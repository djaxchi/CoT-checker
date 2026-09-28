#!/bin/bash
# Run from an isolated, committed source snapshot on the TamIA login node.
set -euo pipefail
: "${PROJECT_ROOT:?Set an isolated source snapshot}"
: "${SOURCE_COMMIT:?Set the archived source revision}"
RUN_ROOT="${RUN_ROOT:-/project/aip-azouaq/dchikhi/cot_mech/instruct_leaderboard_v1}"
export RUN_ROOT PROJECT_ROOT SOURCE_COMMIT
module purge
module load StdEnv/2023 gcc arrow/24.0.0 python/3.11 cuda/12.2
source "$HOME/venvs/cot/bin/activate"
cd "$PROJECT_ROOT"
# Never submit a second DAG into the same output root.
mkdir -p "$(dirname "$RUN_ROOT")"
mkdir "$RUN_ROOT"
mkdir "$RUN_ROOT/cells" "$RUN_ROOT/logs"
python - <<'PY'
import json, os, shutil
from pathlib import Path
from scripts.validate_instruct_leaderboard import cell_tag, validate_grid
from src.repstore import split_fingerprint
root = Path(os.environ['RUN_ROOT'])
source = Path('/scratch/d/dchikhi/cot_mech/qwen3_8b_instruct_v1/runs/rep_grid_q3')
pair = ('step_tokens', 'transformer:d512,l2,f2048,h8')
refpath = source/cell_tag(*pair, 42)/'results.json'
ref = json.loads(refpath.read_text())
validate_grid(source, [pair], ref)
for key, digest in ref['inputs'].items():
    kind, stem = key.split('/')
    store = Path(ref['prm_store' if kind == 'prm' else 'pb_store'])/stem
    for specfile in store.glob('shard_*/spec.json'):
        spec = json.loads(specfile.read_text())
        assert spec['backbone'] in ('Qwen3-8B', 'Qwen/Qwen3-8B'), specfile
        assert spec['layer'] == 35 and spec['dim'] == 4096, specfile
        assert spec.get('prompt_style', 'verifier') == 'verifier', specfile
    assert list(store.glob('shard_*/spec.json')), store
    assert split_fingerprint(store) == digest, store
shutil.copy2(refpath, root/'reference.json')
for seed in (42, 43, 44):
    tag = cell_tag(*pair, seed)
    shutil.copytree(source/tag, root/'cells'/tag)
(root/'source_commit.txt').write_text(os.environ['SOURCE_COMMIT']+'\n')
print('Validated and archived all three completed d512 checkpoints and seven Instruct stores')
PY
common="ALL,PROJECT_ROOT=$PROJECT_ROOT,RUN_ROOT=$RUN_ROOT"
jvec=$(sbatch --parsable --time=08:00:00 --job-name=instruct_grid_vectors \
  --output="$RUN_ROOT/logs/%x-%j.out" --export="$common,GROUP=vectors" \
  slurm/instruct_leaderboard_tamia.sh)
printf 'vectors %s\n' "$jvec" > "$RUN_ROOT/submitted_jobs.txt"
jseq=$(sbatch --parsable --time=06:00:00 --job-name=instruct_grid_sequences \
  --output="$RUN_ROOT/logs/%x-%j.out" --export="$common,GROUP=sequences" \
  slurm/instruct_leaderboard_tamia.sh)
printf 'sequences %s\n' "$jseq" >> "$RUN_ROOT/submitted_jobs.txt"
jmerge=$(sbatch --parsable --dependency="afterok:$jvec:$jseq" \
  --output="$RUN_ROOT/logs/%x-%j.out" --export="$common" \
  slurm/merge_instruct_leaderboard_tamia.sh)
printf 'merge %s\n' "$jmerge" >> "$RUN_ROOT/submitted_jobs.txt"
cat "$RUN_ROOT/submitted_jobs.txt"
