# Bidirectional token probe on frozen Instruct representations

Date: 2026-10-05. Experiment ID: `bidirectional_token_probe_v1`.

**1. Objective and decisions already made.** Determine whether direct attention to subsequent reasoning tokens improves a probe's ability to assess the validity of a target step. Use only `Qwen/Qwen3-8B`, the Instruct checkpoint already used in this project. Freeze the backbone. Train the probe on PRM800K and evaluate on held-out PRM800K and all four ProcessBench subsets.

Retain full, prefix-conditioned token sequences. Let the trainable probe attend directly to token states across steps before producing one score per step. The model must score every step, including steps after an earlier error. Do not stop scoring at the first predicted error, propagate an error label to all later steps, impose monotonically decreasing validity, or train a first-error index classifier.

Djalil accepts the working assumption that a verifier trained on available correctness labels, including first errors, can generalize to errors later in a trace. Missing post-error labels do not block this experiment. Report that assumption and the limits of measured evaluation without adding an annotation campaign to the scope.

Use the dataset's validity convention where labels exist. Do not introduce a new inherited-error taxonomy or require a separate decision about post-error semantics to begin implementation.

**2. Research question and contrasts.** The primary question is whether adding actual future-token access improves predictions when the backbone, input states, probe architecture, supervision, and optimization protocol are matched.

Train these four conditions:

| ID | Visibility when scoring step i | Role |
|---|---|---|
| `local` | Question/template prefix, step i's tokens, and its pre-step boundary state | Current-step readout control |
| `causal` | Question/template prefix and all tokens through the end of step i | Main control |
| `future1` | Same prefix plus at most the complete next step | Short-lookahead treatment |
| `full` | All tokens in the complete recorded solution | Main treatment |

The main contrast is `full - causal`. `causal - local` measures the value of giving the probe explicit access to earlier steps. `future1 - causal` measures a short lookahead. The `local` condition still reads prefix-conditioned backbone states: it is not a text-without-history condition.

All conditions have the same trainable parameter count. Within each final seed, pair initial weights, training-example order, splits, and batch membership. Optimizer updates naturally diverge after training starts.

**3. Read the existing implementation before extending it.** Read `TAMIA.md` and the relevant report sections. Relevant entry points are:

- `REPORT.md`, sections 19.1, 21.10, 21.12, and 21.15.
- `scripts/encode_processbench_full_store.py`: complete causal passes, step boundaries, and full token stores.
- `scripts/build_prm800k_prestudy.py` and `scripts/build_prm800k_full.py`: raw-label handling and existing problem splits.
- `src/repstore/store.py` and `src/repstore/fingerprint.py`: packed storage and provenance.
- `src/harness/learners.py`, `src/harness/spanloader.py`, and `scripts/train_rep_learner_cell.py`: existing probe training components.
- `scripts/analysis/lookahead_ceiling.py`: historical pooled-lookahead experiment and evaluation details.
- `slurm/instruct_leaderboard_tamia.sh`, `slurm/train_rep_grid_7b_tamia.sh`, and `slurm/prm_backbone_tamia.sh`: working cluster patterns. The last file is an infrastructure reference only; do not run its PRM backbone.

Existing learner interfaces return one logit per independent example. Extend or add a small trajectory-specific module with explicit per-step outputs and masks rather than forcing an incompatible sequence into that interface. Reuse metrics, storage utilities, and checkpoint machinery where their contracts match.

**4. Construct a trajectory dataset with real continuations.** The existing materialized candidate-step files are insufficient by themselves for learning future access. Raw PRM800K phase 2 has `question.pre_generated_steps`, which contains the original full solution even when human labels stop at its first error. Missing later labels and missing later text are different cases.

Audit local and cluster data availability before generation or extraction. Use original phase-2 trajectories with matching labels and observed continuations as the primary training population. Do not launch the old continuation-generation pipeline. Do not assume the previous 513,810 candidate examples correspond to 513,810 usable trajectories.

For each raw record:

1. Read the original problem and full `pre_generated_steps`.
2. Align human-rated completions to the original generated step at the same position. Validate the preceding trajectory as well as the target text. Alternative completions and human repairs may refer to another branch.
3. Assign `y=1` to rating -1, `y=0` to ratings 0 and +1. Keep the original rating in metadata.
4. Assign `label_mask=false` wherever a label is unavailable, flagged, ambiguous, contradictory, or belongs to a different branch. Do not infer later labels from final-answer correctness, annotation termination, or an earlier error.
5. Keep later unlabeled steps as input context. Unknown labels never enter the supervised loss.
6. Exclude malformed trajectories and unresolved bad-problem records. Keep only trajectories with at least one trustworthy supervised target. Deduplicate repeated trajectories and annotations; exclude conflicting supervision unless the source supplies an explicit adjudication.

Do not attach a chosen branch's suffix to a rejected alternative. Do not truncate the input at the last labeled step. Never feed annotation masks, ratings, finish reasons, reference answers, ground-truth solutions, or existing verifier scores to the model. Store those fields separately where needed for auditing.

Phase-1 and alternative-candidate records can be counted in the audit but are outside the main trajectory comparison unless an authentic, aligned continuation is available. Do not mix in large numbers of suffix-free examples as an undocumented training change.

If the raw originals are absent locally, retrieve/cache the public raw data on the TamIA login node. If reconstruction fails, finish the audit and implementation tests and report the concrete data blocker. Training a nominal bidirectional arm with no future tokens is not an acceptable fallback.

**5. Freeze problem-disjoint splits.** Recover the existing train/validation/test problem assignment where possible. Canonicalize problem identity using original IDs and normalized problem-text hashes, since a record-specific ID can conceal duplicate problems. Keep all solutions and duplicates of one problem in one split.

Split existing validation problems deterministically into equally sized `dev` and `calibration` groups, using seed 1729 and stratifying by whether problems have erroneous trajectories where feasible. If the old membership cannot be recovered reliably, create and document a new 80/10/10 train/validation/test split by problem, then divide validation into dev and calibration. Never silently present this as the old split.

Audit exact and normalized-text overlap with ProcessBench. Remove PB-overlapping problems from PRM training, development, and calibration, and report exclusions. Identify any remaining evaluation overlap. This audit does not establish absence of pretraining contamination.

Use natural trajectory and labeled-step prevalence. Do not rebalance evaluation data. Freeze the manifests before any parallel encoding. Report counts of problems, traces, labeled correct/incorrect/neutral steps, unlabeled steps, targets with future context, and length distributions for every split and PB subset.

A manifest entry needs at least: stable problem/trace IDs, source record IDs, problem text, ordered step text, per-step labels and supervision mask, original ratings, first-error annotation if known, split, source hashes, and deduplication/alignment status. Keep public model inputs separate from evaluation metadata.

**6. Freeze the backbone representation.** Use `Qwen/Qwen3-8B`, `model.eval()`, no gradients, causal backbone attention, and `hidden_states[35]` as the primary extraction point. Verify the actual model configuration and indexing convention. Record the model and tokenizer revision, hidden dimension, precision, and extraction code hash.

Encode a complete trajectory once using the same deterministic full-solution format for PRM800K and PB:

```text
Problem:
{problem}

Solution:
{step_0}

{step_1}
...
```

Adapt the existing `tokenize_solution` helper and retain its exact token IDs and additive boundaries. Do not add a verification request, a chat-template variant, a generated critique, or thinking tokens. This experiment reads Instruct activations under a fixed solution-reading format. Historical per-candidate verifier-template scores are context, not matched baselines.

Retain every input token's state, including problem/template tokens, plus exact `[start,end)` step spans. Each target step's own states encode only its prefix because the backbone remains causal. A future state can already carry information about earlier steps; this is part of the representation being tested, not an independent encoding of future text.

Store one packed item per trajectory in sharded memory-mapped arrays, with per-step metadata and supervision masks. Reuse the repstore contract where possible, but do not mistake its single `y.npy` label per item for complete step supervision. Define a documented sidecar or small extension for per-step labels.

Store float16 states and cast as needed in the probe. Check conversion error on the extraction smoke test. Cache raw states, with no pooled summaries, truncation to 512 tokens per step, top-token selection, or PCA bottleneck. A trainable per-token feature projection is allowed and specified below.

Audit token lengths before setting the context cap. Start with 8,192 total tokens if the checkpoint and pilot support it; raise to 16,384 if necessary for coverage and feasible. Freeze the cap before production and report excluded traces and target coverage by dataset/label. All arms use identical retained traces. Do not choose the cap from evaluation performance or replace full tokens with pooled vectors to fit memory.

Do not launch a layer sweep in v1. Layer 35 is a declared scope choice, not a claim that it is optimal.

**7. Direct token-attention probe.** Implement a transformer over projected frozen token states, with one learned readout query per step. Preserve individual token positions until the step query produces its output.

Proposed fixed architecture:

| Component | Value |
|---|---|
| Input | All retained frozen token states and pre-step boundary states |
| Feature projection | Trainable linear map from hidden dimension to 256 |
| Transformer | 2 pre-norm layers, 4 heads, feed-forward width 1,024 |
| Dropout | 0.1 |
| Position encoding | Deterministic absolute token and step positions, independent of total trace length |
| Role encoding | Learned token/boundary/query role embeddings |
| Step readout | Shared learned query, separately instantiated for each step |
| Output | Shared linear scalar head and sigmoid, `p_incorrect`; expose `1-p_incorrect` as validity |
| Normalization | Per-token LayerNorm in the probe; no normalization over the full trajectory |

Insert a copy of each step's pre-step backbone state as a boundary token owned by that step, so every condition exposes the same explicit anchor. Give copied boundaries their original source token positions plus their role identity. Readout queries are probe-side objects; do not insert new tokens into the frozen backbone pass.

Use a query at each step to attend directly to permitted token/boundary states. Do not mean-pool, max-pool, or compress each step into a fixed number of summary vectors before cross-step attention. Token states may update through the probe's transformer layers. To keep the implementation auditable, token and query rows may read token/boundary keys, but query positions are never keys. Queries do not communicate labels or previous predicted scores.

The query output is a learned contextual representation of the target step, followed by a linear classifier. Interpret it as a trained verifier over frozen features; a gain alone does not establish that the frozen backbone already computed an explicit correctness judgment.

**8. Attention-mask contract and leakage prevention.** Assign each reasoning token and copied boundary an owner step. Treat question/template-prefix tokens as a separate prefix group. Exclude separators outside the prefix and step spans from probe inputs, or assign them deterministically without exposing later steps. Document the choice.

Apply these rules in every probe layer, to both token updates and query reads:

- Prefix rows read prefix keys only. Otherwise a prefix row could import the future and relay it to a causal query.
- `local`: a row owned by i reads prefix keys and keys owned by i.
- `causal`: a row owned by i reads prefix keys and keys owned by j <= i.
- `full`: reasoning/query rows read prefix keys and keys from all steps.
- Padding never supplies keys or contributes a scored query or loss.

Both causal and full arms permit attention to the entire current step. Use a step-based causal mask, not ordinary token-causal attention: the experimental change is access to subsequent steps, not access to the remainder of the current step.

For `future1`, score target i on a view physically capped at the end of step i+1. Within that view, rows owned by j may read through j+1, and prefix rows remain prefix-only. Score only target i from that view. Reusing the cached backbone prefix is valid because it is causal. A banded mask over a full trace without the physical target cap is invalid: multiple layers would expand the future horizon.

Do not feed total trace length or relative progress i/T to the probe. Absolute prefix/token positions are allowed. Future arms can still infer continuation length from visible tokens; measure that possible shortcut through the structural baseline below.

Use memory-efficient attention that supports the exact mask. Verify backend mask semantics, including Boolean-mask conventions. If a dense fallback is required, lower microbatch size, use gradient accumulation/checkpointing, and shard work. Do not relax mask correctness for speed. Log backend, effective precision, peak memory, and timing.

**9. Supervised objective and optimization.** Train independent validity logits for all labeled steps, without a sequence-wide softmax or an absorbing error state. Use unweighted BCE, averaged over valid labeled targets within a trajectory and then averaged over trajectories. Unknown labels remain masked.

For target-cropped computation, preserve that objective by weighting target contributions by the number of labeled targets in their source trajectory. Do not give long traces extra weight merely because they generate more target views. Accumulate a complete effective batch before an optimizer update.

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| Learning-rate grid | 1e-4, 3e-4, 1e-3 |
| Weight-decay grid | 0.01, 0.1 |
| Effective batch | 32 trajectories, with length bucketing and gradient accumulation |
| Maximum epochs | 30 |
| Early stopping | Patience 5 on development step F1 |
| Gradient clipping | Global norm 1.0 |
| Search seed | 0 |
| Final seeds | 42, 43, 44 |
| Data cap | None for final training |

At each development checkpoint, obtain the best incorrect-step F1 over an explicit deterministic threshold search on labeled dev steps. Use this development score for early stopping and hyperparameter selection; it is not a held-out result. Specify threshold tie handling, with the higher threshold winning ties.

Run six settings for each of `causal` and `full`, 12 search fits. Pick one shared learning-rate/weight-decay setting maximizing the average development score of those two arms. Apply it to all four final conditions. Train four conditions over three final seeds, 12 final fits. Total base budget: 24 fits plus diagnostic controls and smoke tests. Use all eligible training trajectories; any bounded pilot must be labeled as such and kept separate.

Save last and best checkpoints, optimizer/scaler/RNG/sampler state, selection decisions, and learning curves. Save at least once per epoch and periodically within long epochs. Resumed runs must preserve sample weighting and optimizer progress.

After choosing a checkpoint, freeze its threshold using labeled PRM calibration steps. Optimize incorrect-step F1 there, then apply that threshold unchanged to PRM test and PB. The PB metric may prefer another threshold; expose that only through the explicitly marked oracle result. Never tune architecture, checkpoints, or production thresholds on PB.

**10. Evaluation reflects the available labels.** Write one score per original step, with problem ID, trace ID, index, visible horizon, condition, seed, checkpoint hash, and annotation availability. Retain predictions after the first actual or predicted error.

On held-out PRM800K, report incorrect-step F1 at the calibration-selected threshold, oracle-threshold F1 as a ceiling, AUROC, precision, recall, prevalence, and the always-positive F1 baseline. Use only valid labels. Report rating-0 performance separately and provide results on labeled targets with at least one subsequent step as a prespecified slice.

On each PB subset, report:

- Step F1 and AUROC on known-label positions only: steps before the annotated first error are correct, the first error is incorrect, later positions are unknown. All steps in an annotated error-free trace are correct.
- Standard PB first-error F1, `Acc_error`, `Acc_correct`, and exact localization accuracy. Derive this benchmark prediction from the first threshold crossing in the saved complete score sequence; do not change how the model scores steps.
- Calibration-selected and oracle thresholds/results, and trivial always-no-error and always-flag-step-0 rules for the trace metric.
- Premature alarms before the annotated error, late/missed detections, and coverage/exclusions.

Average PB first-error F1 equally over the four subsets. Keep per-subset results visible. Do not compare raw step F1 across datasets with different prevalence or describe post-error PB positions as labeled correct/incorrect. Do not label every position except the first error as correct, as that would train a different target.

Post-error scores can appear in qualitative examples, but neither PB first-error accuracy nor PRM labels before termination validate every later prediction. State the generalization assumption once in the results and leave this experiment focused on testing the proposed representation/readout.

**11. Controls and interpretation.** Complete the main four-condition comparison even if the initial full-versus-causal result is null, provided the implementation and learning diagnostics are valid.

Run a cheap logistic structural baseline trained on the same labeled training targets using step index, target-token length, prefix-token length, number of later steps, and later-token count. Fit scaling on training data and select/calibrate through the same splits. This measures how much label prediction can come from position and continuation availability alone.

For a bounded diagnostic, select 256 known-label targets with at least two subsequent steps from evaluation data, stratified by dataset/subset and label using seed 1729, independently of model predictions. Use all eligible targets if fewer exist. Keep this diagnostic separate from headline test scores.

For those targets, evaluate the final `full` models on:

1. The unchanged trace.
2. The same prefix and target with the subsequent step order shuffled.
3. The same prefix and target with explicit final-answer text removed, where an auditable extraction rule identifies that text without changing the target.
4. The same prefix and target with the entire future hidden from the probe.

For textual modifications, re-encode the changed continuation. Shuffling cached vectors does not erase their original causal context. Validate that target/prefix tokenization and backbone states stay unchanged. Preserve labels only for the unchanged target; do not reuse labels for modified downstream text.

Report paired score changes, target F1/AUROC where both classes exist, and eligibility counts. Evaluation perturbations measure model reliance and can introduce distribution shift. Do not call a drop proof of causal necessity or semantic reasoning. A suffix-only frozen-state baseline is not text-independent, because suffix states already contain prefix information; do not make that interpretation.

Review a deterministic sample of causal/full disagreements, including gains, regressions, premature alarms, and post-error score patterns. Provide exact trace excerpts and scores. Label any human interpretation of post-error validity as qualitative.

**12. Statistical reporting and practical conclusions.** Report `full - causal` and `future1 - causal` on identical examples for each seed. Give the mean paired difference and seed spread. Compute 95% paired bootstrap intervals with 10,000 resamples clustered by problem, preserving sibling traces and recomputing metrics with fixed calibration-selected thresholds. Resample problem groups consistently across any duplicated evaluation membership. Explain that these intervals condition on the fitted checkpoints; three seeds provide a separate view of training variability.

Keep incorrect-step F1 primary for the labeled-step task and AUROC secondary. PB first-error F1 is the external benchmark projection of the all-step scores. A +0.02 absolute F1 gain is a useful descriptive practical yardstick, not a new significance test or a reason to suppress smaller results.

Interpret outcomes precisely:

- Gains on PRM and PB, stable across seeds, support a useful contextual readout within this setting.
- A PRM-only gain suggests limited transfer.
- Better AUROC with weaker fixed-threshold F1 suggests a calibration issue worth reporting.
- Gains largely reproduced by structural cues or accompanied by many premature alarms weaken a local-validity interpretation.
- A well-measured null constrains this layer, token representation, architecture, and training population. It does not show that future text contains no useful information.

**13. Tests before production.** Write meaningful pytest tests alongside implementation, using synthetic states and model/tokenizer stubs. Required tests:

- Raw trajectory alignment rejects an alternative completion's borrowed suffix and preserves authentic original continuations.
- Ratings map correctly; unknown/flagged labels contribute zero loss and zero label-derived input features.
- Problem grouping prevents split leakage, including duplicate problem text with different record IDs.
- The model emits scores for every step, including after an error, with no forced monotonicity.
- Perturbing a later step cannot change a causal prediction, including through prefix rows or intermediate token updates.
- Perturbing steps beyond i+1 cannot change target i under `future1` across two layers.
- Perturbing a next-step token can affect a full/future1 output in a constructed nondegenerate fixture, so the treatment is not accidentally masked off.
- Both causal and full masks expose the complete current step.
- Padding, batch composition, and other packed trajectories do not affect an example's outputs in evaluation mode.
- Boundary ownership, copied positions, query-key exclusion, and known-label metric masks are correct.
- Chunked/microbatched loss matches the intended per-trajectory objective, and checkpoint resume preserves training progress.
- PB aggregation matches existing reference metrics on synthetic cases, including error-free traces and missing post-error labels.

On the TamIA smoke run, compare prefix-only and complete causal-backbone extraction on shared token positions within documented numerical tolerance. Audit actual hidden-state indices and input fingerprints. Fit a tiny synthetic task in which only the next step determines the target label; the full model should learn it and the causal model should remain at its appropriate chance baseline. Also verify tiny-real-data overfitting and finite gradients. These are implementation checks, not research results.

**14. Suggested implementation artifacts.** Use these names unless a nearby existing abstraction makes a smaller change possible:

```text
experiments/bidirectional_token_probe_v1/config.yaml
experiments/bidirectional_token_probe_v1/README.md
src/data/prm_trajectories.py
src/probes/contextual_token_probe.py
scripts/build_prm800k_trajectory_dataset.py
scripts/encode_trajectory_token_store.py
scripts/train_contextual_token_probe.py
scripts/eval_contextual_token_probe.py
scripts/analysis/bidirectional_token_probe_report.py
tests/data/test_prm_trajectories.py
tests/probes/test_contextual_token_probe.py
tests/eval/test_contextual_token_probe_metrics.py
slurm/bidirectional_token_probe_encode_tamia.sh
slurm/bidirectional_token_probe_train_tamia.sh
slurm/submit_bidirectional_token_probe.sh
```

Keep the resolved config authoritative. Include every default, fingerprint, filtering rule, model revision, mask variant, label convention, hyperparameter choice, seed, threshold rule, and context cap. Existing results directories and baselines remain untouched.

**15. TamIA execution.** Follow the current local TamIA skill and inspect the working scripts before submission. `TAMIA.md` includes an obsolete external hostname; use `dchikhi@tamia.alliancecan.ca` or a verified existing SSH alias. Authentication may require the user to complete Duo. Do not attempt to bypass it.

Use account `aip-azouaq` and whole-node GPU requests: `--gpus-per-node=h100:4` or `--gpus-per-node=h200:8`. The maximum job walltime is 24 hours. Use at least one hour for production jobs and the short-test allowance for smoke jobs. Measure throughput first, then request tight walltimes with a modest margin. Split work and resume when a stage would exceed a job limit.

Use the prescribed module/venv setup from the current TamIA skill, or a working verified environment already used for these models. The repository has both older Python 3.11 and newer Python 3.12 job patterns; do not mix their environments without an import smoke test. Record the actual torch/transformers/CUDA versions and attention backend. Resolve `HF_CACHE_ROOT` to the existing complete checkpoint cache and pin local model paths/revisions. Enable `HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, and `TRANSFORMERS_OFFLINE=1`; use `local_files_only=True` for model/tokenizer loads. Cache missing dependencies and raw datasets on the login node before scheduling. Do not run GPU extraction or training on the login node.

Use unique source snapshots under the project storage and record the git hash plus a hash/archive of the dirty working tree used. Do not commit to obtain provenance. Preserve existing untracked files. Dry-run code sync first and verify `slurm/` and this plan are included: the current `scripts/tamia/sync.sh` include list omits them. Add an explicit safe transfer or extend the helper carefully. Do not blindly run its `--delete` behavior against a shared checkout used by other jobs. Do not sync checkpoints/results from the laptop to the cluster.

Use storage roots resolved from the real account environment:

```text
$STORE/cot_mech/bidirectional_token_probe_v1/       manifests, configs, source snapshots, checkpoints, logs, final results
$SCRATCH/cot_mech/bidirectional_token_probe_v1/     reproducible large activation shards
$SLURM_TMPDIR/bidirectional_token_probe_v1/        per-job staging and temporary caches
```

Do not put data, activations, or checkpoints in HOME. Check free space/quota before extraction. At hidden dimension 4096, float16 states cost 8192 bytes per token before metadata, so compute the projected cache size from audited token counts. Stream shards and use memory mapping instead of loading the whole corpus in every process. Budget CPU RAM across all concurrent workers. Preserve irreplaceable manifests, checkpoints, metrics, and logs in STORE before job exit; node-local storage disappears and scratch is purgeable.

Use all allocated GPUs through independent extraction shards or fit workers where memory permits. Account for mask and sequence length when choosing concurrency. Whole-node allocation is mandatory even if memory constraints limit active workers. Prefer resumable shards over repeated full encodes.

Execution order:

1. Audit sources and data, build manifests, and run local unit tests.
2. Sync an isolated source snapshot and run a small end-to-end TamIA smoke job on PRM and PB samples.
3. Record numerical checks, token throughput, fit throughput, GPU/RAM peaks, and projected storage/runtime. Fix failures before production.
4. Encode frozen manifests with disjoint shards, atomic completion markers, and manifest checksums. Validate complete coverage before training.
5. Run the 12 paired search fits and freeze the shared setting from PRM dev.
6. Run all 12 final fits, calibrate, and evaluate. Use dependency chains and nonzero exits for incomplete rosters.
7. Run the structural baseline, bounded diagnostics, bootstrap analysis, and qualitative audit.
8. Retrieve durable results and logs, generate the report and figures, and update `REPORT.md`.

Track job IDs and every stage in a run ledger. Check `squeue` and `sacct`, inspect failures, and resume recoverable work without changing the scientific comparison. A successful Slurm exit must also validate expected outputs, seeds, hashes, and prediction counts. Stop a bad job from this experiment when necessary; do not cancel unrelated jobs. Queueing alone is not experiment completion.

**16. Outputs and final report.** Retrieve concise artifacts locally under `results/bidirectional_token_probe_v1/` and keep complete provenance/checkpoints in STORE. Required outputs include:

- `data_audit.json` and a readable audit, plus frozen split/encoding manifests.
- `run_manifest.json` with config, source/model/tokenizer hashes, package versions, paths, jobs, timing, and coverage.
- Per-run learning curves, selected hyperparameters/checkpoints, calibration thresholds, and resumable checkpoints.
- Per-step predictions for all positions and explicit label-availability masks.
- `metrics.json`, `leaderboard.csv`, `paired_differences.json`, and diagnostic results.
- `summary.md` explaining the question, implementation, exact quantitative results, uncertainty, errors, and limits.
- Figures for paired condition comparisons, performance versus future horizon, and selected per-step score traces. Include labeled-step F1 and PB F1 in separate panels and label oracle results.

Use standard plotting tools and save all figure paths. Update the authoritative `REPORT.md` with the report skill after results exist. Preserve the historical measurements in section 19.1, but qualify its unsupported claim that the pooled-lookahead null established absence of future information. Explain that the new study tests direct token interactions under a different representation/readout and supervision population.

In the final response to Djalil, state what ran, the primary paired effects with intervals and seed variation, whether results transfer to PB, what the controls show, runtime/resource use, and any incomplete work. Link the plan, summary, metrics, relevant code, and cluster output location. List the exact local paths of every generated figure at the end. Never describe unmeasured post-error validity as established, and never report a hypothesis as confirmed without its supporting numbers or examples.

Completion means implementation tests pass, all declared core conditions/seeds have completed, outputs are retrieved and validated, the analysis is written, and `REPORT.md` is updated. If a failure remains unresolved or an external blocker prevents completion, mark the experiment incomplete and report the specific blocker, failure record, and exact resume command/state rather than claiming success.

**17. References and scope boundaries.** The literature precedent is [The Bidirectional Process Reward Model](https://arxiv.org/html/2508.01682v2). This study tests a frozen Instruct backbone with direct token-level probe attention, rather than reproducing its reverse-order reward-model training.

Dataset semantics: [PRM800K original data documentation](https://github.com/openai/prm800k) and [ProcessBench paper](https://aclanthology.org/2025.acl-long.50/).

Do not add a PRM backbone, native-format PRM-head comparison, backbone fine-tuning, new annotation campaign, continuation generation, layer/width scaling study, downstream search controller, or deployment task to this v1. Handle routine implementation and resource choices autonomously within the declared comparison. After code changes, run `graphify update .` as required by the project.
