# Bidirectional token probe v2: all-step DeepSeek-R1 labels on Qwen3-8B's own trajectories

Date: 2026-10-05. Experiment ID: `bidirectional_token_probe_v2`. Predecessor: `bidirectional_token_probe_v1` (REPORT §21.16, `results/bidirectional_token_probe_v1/summary.md`).

## 1. Why v2 exists

v1 found that direct probe attention to later steps lowered step F1 against a matched causal probe: full - causal was -0.026 PRM800K F1 [-0.031, -0.020] and -0.022 mean ProcessBench F1_PB [-0.033, -0.010]. Training curves rule out under-capacity: full reaches lower training loss than causal from epoch 4 and overfits after epochs 2 to 6.

v1's supervision is the leading explanation. PRM800K human labels stop at the first error. For every labeled causal target, "some error is visible in my context" therefore equals the label. A full-context query sees the error in every erroneous trace and must localize it instead.

v2 changes only the supervision population, to one where steps after an error are labeled, so that shortcut no longer equals the label:

* **Trajectories.** Qwen3-8B's own solutions to PRM800K problems, released by ReProbe (Ni et al., arXiv:2511.06209, v5).
* **Labels.** DeepSeek-R1 step annotations on those same trajectories. The annotator saw the problem, the ground-truth solution and every step, and labeled **every** step, not just up to the first error.

Measured on the release (2026-10-05): 54% of the labeled steps that follow a trace's first DeepSeek-labeled error are labeled **correct**. The whole suffix is marked incorrect in only 33% of error traces.

**Primary question.** With all-step labels, does `full - causal` become non-negative on held-out labeled steps, and does any change transfer to ProcessBench?

## 2. Decisions already made

* **Labels.** Primary: DeepSeek-R1, from `rediska0123/train_prm800k_Qwen3-8B_finished`. Qwen3-8B self-labels on the identical trajectories (`JingweiNi/train_prm800k_Qwen3-8B_finished_self_annotate`) are a secondary sensitivity check only (§8).
* **Backbone, layer, reading format, probe architecture, four conditions, HP grid, seeds, early stopping and threshold rule.** All identical to v1 (`experiments/bidirectional_token_probe_v1/config.yaml`). Only the data changes.
* **ProcessBench stays a full evaluation set.** Overlap is removed from **training**, not from evaluation. 752 ReProbe problems match ProcessBench math problems; 788 of the 1,000 ProcessBench math traces sit on them. Every trajectory on those problems is excluded from all ReProbe-derived splits. All 3,400 ProcessBench traces in the four subsets stay in evaluation, as in v1.
* **Supplementary transfer set.** PRM800K human labels (the v1 test split, first-error convention).
* **No new generation and no new annotation.** The run is a single encode-and-train job.

## 3. Source data and provenance

| item | value |
|---|---|
| DeepSeek labels | `rediska0123/train_prm800k_Qwen3-8B_finished`, 32,484 rows, public, ungated, no license field |
| self labels | `JingweiNi/train_prm800k_Qwen3-8B_finished_self_annotate`, same 32,484 replies |
| code | https://github.com/ReProbe/ReProbe (`synthetic_dataset_generation/`, `utils/step_fact_check.py`) |
| problems | 10,827 PRM800K phase-2 train problems, 3 Qwen3-8B samples each (T=1.0, top-k 50, top-p 0.95, length-capped) |
| fields | question, answer (ground-truth solution), input_ids (prompt + reply), reply ("- Step k: ..." one per line), claims (per-step sentence and token alignment), verified |

Label convention, verified against `step_fact_check.py`: `verified = 1` means the annotator called the step **incorrect**, 0 means correct, and NaN means unparsed or unlabeled.

Steps:

* Pin the dataset revisions (HF commit hashes) and the parquet sha256.
* Download on the login node into `$STORE/cot_mech/bidirectional_token_probe_v2/raw/`.
* Never redistribute the data. Record it as third-party research data in `data_audit.json`.

## 4. Build the trajectory dataset

`scripts/build_reprobe_trajectory_dataset.py` writes v1's manifest schema: `inputs/`, `meta/`, `encode_manifest.jsonl`, `data_audit.json`, `manifest_checksums.json`. Rules:

1. **Steps.** Use `claims[k].sentence`, in order. Strip the leading `- Step k:` marker with an auditable regex (`^\s*-\s*Step\s+\d+\s*:\s*`). This matches the ProcessBench step format and removes an explicit position cue.
   * Keep `<Answer>` lines as a step when they are claims.
   * Validate that the stripped steps, rejoined, reproduce the reply's step lines (case and whitespace normalized). Count failures; exclude the trace if any step cannot be aligned.
2. **Labels.**
   * `y = 1` if `verified == 1`, `y = 0` if `verified == 0`.
   * `label_mask = False` for NaN, for zero-length token alignments, and for traces whose label list length disagrees with the claim count.
   * Keep the original value in `meta`.
3. **Truncation.** About 47% of replies have no `<Answer>` line (length cap during generation).
   * Flag those traces `unfinished`.
   * Mask the label of the final step only if the reply does not end with a newline, i.e. the step was cut mid-sentence.
   * Keep every unmasked step. Report counts, and report the slices for unfinished and finished traces.
4. **Deduplication.** Collapse identical (normalized problem, step list) trajectories, keeping agreeing labels and masking conflicting positions, as in v1.
5. **Problem identity and splits.** Reuse v1's canonical `problem_key` and v1's frozen problem-to-split assignment (`manifest_v1/meta/*.jsonl`).
   * Measured: 8,053 ReProbe problems are in v1 train, 497 in dev, 506 in calib, 1,007 in test, and 764 are absent.
   * Absent problems that overlap ProcessBench (752, by exact and normalized text) go to `excluded_pb_overlap`.
   * The remaining absent problems (about 12) are assigned with v1's seeded rule (seed 1729, stratified by "has an erroneous trajectory") and reported.
   * This keeps v2's in-domain test problem-disjoint from both v1's and v2's training sets.
6. **ProcessBench and PRM800K human evaluation sets.** Copy them unchanged from `manifest_v1` (`pb_*`, and `test` renamed `prm_human_test`). ProcessBench and PRM800K problems may coincide across sources only through the already-excluded overlap; re-run the overlap audit against the final v2 training problems and fail the build if any remain.
7. **Lengths.** Audit tokens in the v1 reading format. Keep the 8,192 cap, which the input_ids maxima (952 with prompt) are far below.

**Model inputs.** Model inputs are the problem and the stripped steps in the v1 reading format: "Problem:\n{problem}\n\nSolution:\n" plus the steps joined by blank lines, via v1's `tokenize_solution`.

**Recorded deviation.** The trajectories were generated by Qwen3-8B under ReProbe's prompt, so encoding them in the reading format is not the exact generation context. The released `input_ids` allow a later on-policy-context arm. That arm is out of scope here and is listed in §12.

Report the following for every split, by the same rules as v1:

* problems, traces, labeled correct and incorrect steps, and unlabeled steps
* labeled steps after a first error, and the share of those that are correct
* suffix-all-incorrect share, finished and unfinished traces
* lengths

## 5. Representation

Reuse `scripts/encode_trajectory_token_store.py` unchanged:

* Qwen3-8B revision b968826, `hidden_states[35]`, bf16 forward, fp16 store.
* Node-local store in every GPU job, 32 shards, fingerprints copied to STORE.
* About 32.5k ReProbe traces plus 3,400 ProcessBench and 7,228 PRM800K-human traces. The projected store is well under 150 GB; compute it from the audit.

## 6. Training

Reuse `scripts/train_contextual_token_probe.py` and the slurm roster machinery unchanged.

* **Conditions.** `local`, `causal`, `future1` and `full`, with the identical probe (2,638,593 parameters).
* **Loss.** Per-trajectory mean BCE over labeled steps, then a mean over 32 trajectories.
* **Class weighting.** None, as in v1. ReProbe used class weighting. Keeping v1's objective preserves the matched comparison; class weighting is noted as a deviation from ReProbe.
* **Search.** Rerun the 12 search fits (causal and full x lr {1e-4, 3e-4, 1e-3} x wd {0.01, 0.1}, seed 0) on DeepSeek dev labels, because the label population changed. Select one shared setting by mean dev F1, with ties going to the lower lr and then the lower wd.
* **Finals.** 4 conditions x seeds 42, 43, 44. Pairing within a seed is preserved by seeded initialization and a seed-only sampler.
* **Thresholds.** Calibrated on DeepSeek-labeled calib steps (best F1, higher threshold on ties), then frozen for every evaluation set.

## 7. Evaluation

Primary and secondary sets, all scored from complete per-step score sequences:

1. **Held-out ReProbe test (DeepSeek labels, all labeled steps).** This is primary. Report:
   * F1 at the calib threshold, oracle F1, AUROC, precision, recall, prevalence, and the always-positive F1 baseline.
   * Prespecified slices:
     * (a) **post-first-error labeled steps**, where the v1 shortcut is broken, with its own prevalence
     * (b) pre-error steps plus the first error, which mirrors v1's label structure
     * (c) finished vs unfinished traces
     * (d) targets with at least one later step
2. **ProcessBench, all four subsets, intact.** Known-label step F1 and AUROC, F1_PB from the first threshold crossing, Acc_error, Acc_correct, exact localization, the oracle threshold, the trivial rules, and premature and late rates, exactly as in v1.
3. **PRM800K human test (v1 test split).** First-error labels, same metrics as v1, as a label-convention transfer check.

**Paired contrasts.** `full - causal` (main), `future1 - causal`, `causal - local` and `full - future1`.

* Each contrast is computed per seed and as a seed mean.
* Intervals: 10,000 problem-clustered paired bootstrap resamples at fixed thresholds.
* The main contrast is also reported on slice (a) and slice (b) of set 1. The gap between them is the direct test of the shortcut explanation.

**Prespecified reading.**

* The shortcut explanation predicts `full - causal` on slice (a) is greater than on slice (b).
* It also predicts that the overall `full - causal` moves up from v1's -0.026.
* A null or negative result on slice (a) with stable seeds weakens that explanation for this layer and readout.

Never compare raw F1 across sets with different prevalence. DeepSeek step prevalence is about 7.4%, against 12.5% for PRM800K human test.

## 8. Controls and diagnostics

* **Structural baseline.** As in v1: logistic on step index, target tokens, prefix tokens, later steps and later tokens. Train on DeepSeek train labels and calibrate on DeepSeek calib.
* **Label-structure baseline.** On the DeepSeek test labels, report P(incorrect | some earlier labeled step is incorrect) and P(incorrect | no earlier labeled error). This shows how much of the label is predictable from earlier labels alone. It is a property of the labels, not a model.
* **Annotator sensitivity (secondary, optional, last in the queue).** Train `causal` and `full` (seeds 42 to 44) on the self-labels with the selected HP, and score the same sets. Report self vs DeepSeek label agreement on the test split. Measured on all data: kappa 0.78, same first error in 92% of traces.
* **Perturbation diagnostic.** v1's diagnostic (`scripts/diagnose_contextual_token_probe.py`) on 256 stratified targets with at least 2 later steps. Draw them from the held-out ReProbe test and ProcessBench, with labels taken only from the unchanged target.
* **Qualitative audit.** As v1, plus examples where full and causal disagree on post-error labeled steps.

## 9. Tests (write before production)

* **Parsing.** The step-prefix strip, claim-to-reply alignment, and rejection of unalignable traces.
* **Labels.** The `verified` mapping (1 to y=1, 0 to y=0, NaN masked), masked labels contributing zero loss, and the truncated-final-step rule.
* **Splits.**
  * v1 split assignment is inherited exactly.
  * A ProcessBench-overlapping problem can never reach train, dev, calib or test.
  * Duplicate problem text with a different id lands in one split.
* **ProcessBench intact.** The ProcessBench evaluation sets are byte-identical to v1's manifest.
* **Slices.** Slice construction for post-first-error vs pre-error targets.
* **Reuse.** All v1 probe, trainer and metric tests still pass unchanged.

## 10. TamIA execution

1. **Login node.** Download both parquets, verify the sha256 values, build the manifest (CPU), and run the overlap audit. Record that the old v1 manifest is reused.
2. **Smoke job, H200:8, 1 h.** Encode about 60 traces per shard, run the smoke fits, run pytest on the node, and check a tiny-data overfit.
3. **Production job, one H200:8 node, 7 h walltime, 2 processes per GPU.** Run in this order:
   * encode
   * 12 search fits
   * selection
   * 12 final fits
   * the optional self-label fits
4. **Post-processing.** Separate jobs, so a post-processing failure cannot hide completed fits:
   * a GPU job (H200:8 with an H100 hedge) for diagnostics
   * a CPU job for eval and report
   * do not chain post-processing as an in-job extra step (v1's job 506725 lost its extra steps to a permission bit)
5. **Hygiene.** Set `TORCH_THREADS=4` and `OMP_NUM_THREADS=4` per process. v1 measured a 15x slowdown without this.

Log every stage in `results/bidirectional_token_probe_v2/run_ledger.md`.

## 11. Outputs

Under `results/bidirectional_token_probe_v2/`, mirroring v1:

* `data_audit.json`, `run_manifest.json`, `metrics.json`, `leaderboard.csv`
* `paired_differences.json` (with the slice (a) and (b) contrasts), `structural_baseline.json`, `label_structure.json`
* `diagnostics_summary.json`, `qualitative_audit.md`, `summary.md`
* `figures/` (paired conditions, horizon, post-error vs pre-error slice contrast, score traces, learning curves)

Then update `REPORT.md` with a new §21.17 that cross-references §21.16.

## 12. Out of scope for v2

* New generation or annotation.
* The on-policy generation-context encoding, i.e. the released `input_ids` with ReProbe's prompt.
* Any layer sweep, including all-layer features (ReProbe's paper says all layers; its released configs read only the last layer).
* Probe size scaling, ReProbe's attention-plus-logits features, and best-of-N or beam search.
* Each is a candidate v3 axis once v2 says whether label structure was the binding constraint.
