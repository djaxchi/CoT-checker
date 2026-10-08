# bidirectional_token_probe_v1

Does direct probe attention to later reasoning tokens improve step-validity
prediction over a matched causal probe, on frozen Qwen3-8B (Instruct) states?
Plan: `docs/bidirectional_token_probe_v1_plan.md`. Resolved settings: `config.yaml`.

## Pipeline

| stage | entry point | where |
|---|---|---|
| manifests (align, dedup, PB overlap, splits, token audit) | `scripts/build_prm800k_trajectory_dataset.py` | login node, CPU, ~6 min |
| encode (Qwen3-8B, hidden_states[35], fp16, 32 shards) | `scripts/encode_trajectory_token_store.py` | inside every GPU job, node-local disk |
| fits (roster per job, one process per fit, all GPUs) | `slurm/bidirectional_token_probe_train_tamia.sh` + `*.fits` | whole node |
| hyperparameter selection | `scripts/select_contextual_probe_hparams.py` | CPU |
| metrics, paired bootstrap | `scripts/eval_contextual_token_probe.py` | CPU |
| perturbation diagnostics | `scripts/diagnose_contextual_token_probe.py` | GPU |
| structural baseline, figures, qualitative audit | `scripts/analysis/bidirectional_token_probe_report.py` | CPU |

Rosters: `smoke.fits`, `search.fits` (12 fits), `final.fits` (written by the selection script, 12 fits).

Code is shipped to the cluster as an immutable snapshot by `scripts/tamia/snapshot.sh`
(records git HEAD, a hash of the uncommitted diff, the untracked files shipped, and a
tree hash) instead of committing.

## Why the store is node-local

38.5M tokens x 4096 x 2 bytes = 293 GiB. On 2026-10-05 /scratch had 126 GB free
and /project 154 GB. Each GPU job therefore encodes the frozen manifest onto its
own `$SLURM_TMPDIR`; every shard's fingerprint is copied to `logs/<job>/encode_stats/`
so the copies can be compared across jobs.

## Labels and their limits

PRM800K phase-2 human labels stop at the first error; later steps of the original
trajectory are kept as input but carry no label. ProcessBench steps after the
annotated first error are unknown. The study assumes (accepted by the PI) that a
verifier trained on available labels generalizes to later errors; post-error
predictions are written out but never scored as correct or incorrect.
