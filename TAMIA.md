# Tamia (Compute Canada) Reference

Quick reference for running CoT-checker experiments on the Tamia H100 cluster.

---

## Connect

```bash
ssh $USER@tamia.calculquebec.ca
```

---

## Storage layout

| Variable | Path | Purpose |
|---|---|---|
| `$HOME` | `/home/d/$USER` | Code, venvs. Small quota — never store data here. |
| `$STORE` | `/project/aip-azouaq/$USER` | Persistent project data: checkpoints, probe data, results. Survives job end. |
| `$SCRATCH` | cluster-assigned per user | Large temporary files: HF cache, intermediate npz shards. Purged periodically. |

Key locations inside `$STORE`:

```
$STORE/
├── checkpoints/          # SSAE .pt files (gsm8k-385k_Qwen2.5-0.5b_spar-10.pt)
├── hf_cache/             # (legacy, now use $SCRATCH/hf_cache)
├── probe_data/           # Final .npz files: train_final.npz, eval_held_out.npz,
│                         #   processbench_gsm8k.npz, processbench_math.npz
└── results/              # Trained probe checkpoints and logs:
                          #   probe_seed{42..45}.pt, linear_probe_seed{42..45}.pt
```

---

## Session preamble

Every interactive session (and every SLURM script) needs this before anything else:

```bash
module purge
module load StdEnv/2023 gcc arrow/24.0.0 python/3.11 cuda/12.2
source $HOME/venvs/cot/bin/activate
export HF_HOME=$SCRATCH/hf_cache
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache
```

---

## Fetch latest code

```bash
cd ~/CoT-checker
git pull
```

If you need to push from the cluster (rare):

```bash
git add <files>
git commit -m "[scope] message"
git push
```

---

## Download a HuggingFace dataset (login node, once per dataset)

Run on the login node before submitting any job that needs the dataset.
Compute nodes have no internet access.

**Important:** there are two separate caches:
- `$STORE/hf_cache` — Qwen model weights (large, persistent). Set by `tamia_download_qwen.sh`.
- `$SCRATCH/hf_cache` — datasets (small, fast to re-download). Use for `load_dataset` calls.

To download a dataset to the right place:

```bash
export HF_HOME=$SCRATCH/hf_cache
python -c "from datasets import load_dataset; load_dataset('<owner>/<name>', split='<split>')"
```

Example — ProcessBench both splits:

```bash
export HF_HOME=$SCRATCH/hf_cache
python -c "
from datasets import load_dataset
load_dataset('Qwen/ProcessBench', split='gsm8k')
load_dataset('Qwen/ProcessBench', split='math')
print('cached.')
"
```

---

## Running experiments

All jobs are submitted with `sbatch`. The cluster requires a full H100 node (`--gpus-per-node=h100:4`).

| Script | What it does | Approx. time |
|---|---|---|
| `scripts/slurm/tamia_setup.sh` | First-time environment setup | 20 min |
| `scripts/slurm/tamia_download_qwen.sh` | Download Qwen model weights to `$STORE/hf_cache` | 30 min |
| `scripts/slurm/tamia_generate_data.sh` | Encode Math-Shepherd → train + eval .npz | 3–4 h |
| `scripts/slurm/tamia_train_probe.sh` | Train MLP probe, 4 seeds in parallel | 20 min |
| `scripts/slurm/tamia_baselines.sh` | Linear probe + LLM self-judge baselines | 2–4 h |
| `scripts/slurm/tamia_processbench.sh` | Encode + eval ProcessBench GSM8K & MATH | 30 min |

```bash
cd ~/CoT-checker
mkdir -p logs
sbatch scripts/slurm/tamia_<script>.sh
```

---

## Monitor a job

```bash
# See all your running/pending jobs
squeue -u $USER

# Watch live log output (replace JOBID)
tail -f logs/<jobname>_<JOBID>.out

# Full job history with exit codes
sacct -u $USER --format=JobID,JobName,State,Elapsed,ExitCode | tail -20

# Cancel a job
scancel <JOBID>
```

---

## Check results after a job finishes

```bash
# Probe training results (logs have a SUMMARY line)
grep SUMMARY ~/projects/aip-azouaq/$USER/results/probe_seed*.log

# ProcessBench output is in the main job log
cat logs/processbench_<JOBID>.out

# List all saved probe checkpoints
ls -lh ~/projects/aip-azouaq/$USER/results/

# List encoded latent files
ls -lh ~/projects/aip-azouaq/$USER/probe_data/
```

---

## Copy results back to your laptop

From your Mac:

```bash
# Single file
scp $USER@tamia.calculquebec.ca:~/projects/aip-azouaq/$USER/results/probe_seed42.log .

# Whole results folder
rsync -avz $USER@tamia.calculquebec.ca:~/projects/aip-azouaq/$USER/results/ ./results/tamia/

# Job log
scp "$USER@tamia.calculquebec.ca:~/CoT-checker/logs/processbench_<JOBID>.out" .
```

---

## Common issues

**`No module named 'pyarrow'`** — forgot to load modules before activating venv. Run the session preamble.

**`Not enough disk space`** — HF_HOME is pointing at `$HOME` or `$STORE`. Make sure `export HF_HOME=$SCRATCH/hf_cache` is set.

**`Requested node configuration is not available`** — used `--gpus=h100:N` instead of `--gpus-per-node=h100:4`. Tamia requires a full node.

**`HF_DATASETS_OFFLINE=1` errors** — dataset not cached yet. Run the download command on the login node first (compute nodes have no internet).

**Job stuck in `PD` (pending)** — normal during busy periods. Check `squeue -u $USER` for the reason column. `Resources` means waiting for a free node.
