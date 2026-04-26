# TamIA Skill

## When to use this skill

Use this skill whenever work involves the TamIA GPU cluster:
- Syncing local code or data to the cluster before a run
- Writing or editing a Slurm batch script for TamIA
- Submitting a job and checking its status
- Retrieving logs, checkpoints, or result files back to the laptop
- Debugging a failed TamIA job

Do NOT use this skill for local training runs, CI, or any cluster other than TamIA.

---

## TamIA constraints (hard rules — never violate)

| Constraint | Detail |
|---|---|
| No internet on compute nodes | All model weights and datasets must be downloaded on the login node before `sbatch`. |
| Full-node allocation required | H100 nodes: `--gpus-per-node=h100:4`. H200 nodes: `--gpus-per-node=h200:8`. Partial allocations are rejected. |
| Max walltime | 24 hours (`--time=24:00:00`). |
| Min walltime | 1 hour for production jobs; 5 minutes for test jobs. |
| Max queued+running jobs | 1 000 per user. |
| Slurm account | Always `--account=aip-azouaq`. |
| No crontab | Scheduled tasks must be submitted via `sbatch` or the automation node `robot.tamia.ecpia.ca`. |
| VSCode banned on login nodes | Use VSCode only on compute nodes if needed. |
| HOME is tiny | Never write checkpoints, HF cache, or datasets to `$HOME`. Use `$STORE` or `$SCRATCH`. |
| SCRATCH is purged | Anything in `$SCRATCH` that is not accessed regularly will be deleted. Copy final outputs to `$STORE`. |

---

## Storage layout

Three tiers — choose based on size and how long the file needs to survive:

| Variable | Canonical path | Quota | Survives purge? | Use for |
|---|---|---|---|---|
| `$HOME` | `/home/d/$USER` | ~50 GB | Yes (backed up) | Code checkout and venvs **only**. Never write data here. |
| `$STORE` | `/project/aip-azouaq/$USER` | ~1 TB (group) | Yes (backed up) | Persistent outputs: checkpoints, final training data, probe .npz, results. Everything you want to keep across jobs. |
| `$SCRATCH` | system-assigned | **Very large** (many TB) | **No** — files not accessed in ~60 days are auto-deleted | Large temporary files during a run: HF dataset cache, intermediate shards, raw data before preprocessing. The fast, big tier — use it for anything that can be regenerated. |

Rule of thumb: **generate in `$SCRATCH`, keep in `$STORE`.**

Key paths inside `$STORE` for this project:
```
$STORE/
├── data/            # Training/eval JSONL files (gsm8k_385K_train.json, gsm8k_385K_valid.json)
├── checkpoints/     # SSAE .pt files
├── hf_cache/        # Qwen model weights (large, persistent — set HF_HOME=$STORE/hf_cache for model downloads)
├── probe_data/      # .npz datasets: train_final, eval_held_out, processbench_*
└── results/
    └── checkpoints/ # Future-SSAE experiment outputs (best.pt, train_log.jsonl)
```

Key paths inside `$SCRATCH` for this project:
```
$SCRATCH/
└── hf_cache/        # HuggingFace dataset cache (set HF_HOME=$SCRATCH/hf_cache for dataset downloads)
```

---

## One-time setup

Run once per new cluster account. From the login node:

```bash
ssh $USER@tamia.alliancecan.ca
bash ~/CoT-checker/scripts/slurm/tamia_setup.sh
```

What `tamia_setup.sh` does:
1. Creates the venv at `$HOME/venvs/cot`
2. Installs repo dependencies
3. Creates `$STORE/{checkpoints,probe_data,results}` directories
4. Downloads Qwen model weights to `$STORE/hf_cache`

After setup, verify with:
```bash
source $HOME/venvs/cot/bin/activate
python -c "import torch; print(torch.__version__)"
```

---

## Session preamble

Every interactive session and every Slurm script must start with:

```bash
module purge
module load StdEnv/2023 gcc arrow/24.0.0 python/3.11 cuda/12.2
source $HOME/venvs/cot/bin/activate
export HF_HOME=$SCRATCH/hf_cache
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache
```

Add `export TRANSFORMERS_OFFLINE=1` and `export HF_DATASETS_OFFLINE=1` inside Slurm scripts to prevent accidental network calls from compute nodes.

---

## Sync to cluster

Use the helper script:

```bash
# From your laptop — dry run first
bash scripts/tamia/sync.sh --dry-run

# Real sync
bash scripts/tamia/sync.sh
```

What it syncs: `src/`, `scripts/`, `experiments/`, `tests/`, `pyproject.toml`

What it excludes: `results/`, `data/`, `*.pt`, `*.npz`, `__pycache__`, `.git`

Never rsync results or checkpoints from laptop to cluster — flow is always cluster → laptop for outputs.

Manual equivalent:
```bash
rsync -avz --exclude='results/' --exclude='*.pt' --exclude='*.npz' \
  --exclude='__pycache__' --exclude='.git' \
  ./ $USER@tamia.alliancecan.ca:~/CoT-checker/
```

---

## Download models/datasets to cluster (login node)

Always do this on the login node before submitting a job. Compute nodes have no internet.

```bash
ssh $USER@tamia.alliancecan.ca

# Qwen model weights → $STORE/hf_cache  (large, persistent)
export HF_HOME=/project/aip-azouaq/$USER/hf_cache
python -c "from transformers import AutoModel; AutoModel.from_pretrained('Qwen/Qwen2.5-0.5B')"

# HuggingFace dataset → $SCRATCH/hf_cache  (fast to re-download)
export HF_HOME=$SCRATCH/hf_cache
python -c "from datasets import load_dataset; load_dataset('Qwen/ProcessBench', split='gsm8k')"
```

---

## Submit a job

```bash
# From your laptop — syncs code then submits
bash scripts/tamia/submit.sh scripts/slurm/tamia_train_probe.sh

# Or from the cluster login node directly
cd ~/CoT-checker
mkdir -p logs
sbatch scripts/slurm/tamia_train_probe.sh
```

After `sbatch` you get back a job ID, e.g. `Submitted batch job 123456`.

Available Slurm scripts:

| Script | Purpose | Approx. time |
|---|---|---|
| `tamia_setup.sh` | First-time env + deps | 20 min |
| `tamia_download_qwen.sh` | Download Qwen weights | 30 min |
| `tamia_generate_data.sh` | Encode Math-Shepherd → .npz | 3–4 h |
| `tamia_train_probe.sh` | Train MLP probe, 4 seeds | 20 min |
| `tamia_baselines.sh` | Linear probe + LLM judge | 2–4 h |
| `tamia_processbench.sh` | Encode + eval ProcessBench | 30 min |

---

## Monitor a job

```bash
# From laptop
bash scripts/tamia/monitor.sh [JOBID]

# Or directly on the cluster
squeue -u $USER                                          # running/pending jobs
tail -f ~/CoT-checker/logs/<jobname>_<JOBID>.out        # live log
sacct -u $USER --format=JobID,JobName,State,Elapsed,ExitCode | tail -20  # history
scancel <JOBID>                                          # cancel
```

States: `R` = running, `PD` = pending, `CG` = completing, `F` = failed, `CD` = completed.

---

## Retrieve outputs

```bash
# From your laptop
bash scripts/tamia/retrieve.sh

# What it pulls:
#   $STORE/results/      → ./results/tamia/
#   $STORE/probe_data/   → ./data/tamia/
#   logs for a specific job: pass JOBID as argument
bash scripts/tamia/retrieve.sh 123456
```

Manual equivalent:
```bash
rsync -avz $USER@tamia.alliancecan.ca:/project/aip-azouaq/$USER/results/ ./results/tamia/
scp "$USER@tamia.alliancecan.ca:~/CoT-checker/logs/probe_123456.out" .
```

---

## Common failure modes

### Model download attempted on compute node
**Symptom:** `ConnectionError` or `requests.exceptions.` in job log.  
**Fix:** Run the download command on the login node (see "Download models/datasets" above). Add `TRANSFORMERS_OFFLINE=1` and `HF_DATASETS_OFFLINE=1` to your Slurm script to make the failure explicit at submit time rather than silently stalling.

### Missing HF cache
**Symptom:** `OSError: Can't load tokenizer for 'Qwen/...'` or `FileNotFoundError` for model config.  
**Fix:** Model weights → `HF_HOME=$STORE/hf_cache`. Re-run `tamia_download_qwen.sh` on the login node. Dataset cache → `HF_HOME=$SCRATCH/hf_cache`; re-download on the login node.

### Missing training data (`FileNotFoundError` for .json or .jsonl)
**Symptom:** `FileNotFoundError: .../gsm8k_385K_train.json` or similar at the start of a job.  
**Fix:** Training JSONL files live in `$STORE/data/`. They are not synced from the laptop — they must be generated on the cluster (or uploaded once). Check `ls $STORE/data/` on the login node. If the directory is empty, run the data-generation job first or upload the files with `scp`/`rsync`.

### Wrong Slurm account
**Symptom:** Job rejected immediately with `Invalid account or account/partition combination specified`.  
**Fix:** Add `#SBATCH --account=aip-azouaq` to the script header. Verify your account membership with `sacctmgr show user $USER withassoc`.

### Job requests not using full GPU node
**Symptom:** `Requested node configuration is not available` or job stuck in `PD` indefinitely.  
**Fix:** TamIA allocates full nodes only. Replace `--gpus=h100:N` with `--gpus-per-node=h100:4` (H100) or `--gpus-per-node=h200:8` (H200). Never request fewer than the full complement.

### Outputs written to HOME instead of project/scratch
**Symptom:** Disk-quota error or outputs silently missing from `$STORE`.  
**Fix:** In the Slurm script, set `STORE="/project/aip-azouaq/$USER"` explicitly and write all outputs there. Never use `~` or `$HOME` for data files. Check with `quota -s` that HOME is not full.

### rsync accidentally uploads results/checkpoints to cluster
**Symptom:** Large `.pt` or `.npz` files in HOME on the cluster, potentially hitting quota.  
**Fix:** `scripts/tamia/sync.sh` excludes `results/`, `*.pt`, and `*.npz`. If you ran a manual rsync, remove the accidentally uploaded files with `rm` on the cluster login node. Flow rule: **laptop → cluster for code only; cluster → laptop for outputs only.**
