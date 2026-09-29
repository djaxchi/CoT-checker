# Instruct verifier signal experiment: launch record

**Current status:** the first submission was cancelled by its failed dependency. The user explicitly retained the 64-run leaderboard. The replacement runs independently with the 16 available diagnostic checkpoints. See the execution amendment below.

On 29 September 2026, the user requested execution of the prepared experiment on the leading Instruct verifiers. The local leaderboard gives calibrated ProcessBench F1 means of 0.6107 for d512, 0.6052 for attention-query, and 0.6030 for d256. These point estimates do not establish pairwise superiority. The diagnostic will compare response patterns across the frozen six-probe roster instead of choosing one winner.

**Scope and selection.** The experiment retains the 28 September dataset and roster: 912 examples, 72 mathematical families, six probes, and three training seeds. It includes the top three leaderboard entries, d128, and last-token and step-mean linear controls. It does not expand to all 22 cells after seeing the leaderboard. The two missing seed-44 results must finish before scoring. The full protocol appears in [the experiment plan](verifier_signal_v1_plan.md).

**Checks before submission.** Local validation reproduced the frozen data, and 29 tests passed. A read-only cluster check validated the identities, protocols, selected hyperparameters, input fingerprints, checkpoints, and four ProcessBench score files of all 64 available runs. Only attention-query seed 44 and d128 seed 44 were missing. The original training and test representation stores matched the reference fingerprints; shard specifications showed width 4096, hidden-state index 35, Qwen3-8B, and the verifier prompt. The cached tokenizer, model configuration, and all weight shards were available offline.

I reviewed the sample families in `review_samples.md`, including the negative-divisor inequality and the inherited-error conditions. This is an assistant review of examples, not independent human annotation. The manifest retains its human-review-pending status. Labels refer to the operational definitions in the plan, including local consistency with an intermediate equation that can itself be wrong.

**Environment correction.** Extraction job 487175 records Python 3.12, torch 2.14.0, transformers 5.14.1, numpy 2.5.3, and pyyaml 6.0.3. The old HOME environment contains torch 2.11.0 and transformers 5.6.2. The diagnostic batch script now creates a node-local environment and installs the recorded versions from the cluster's offline wheelhouse. Each run saves its full installed package list.

The cached Qwen revision is `b968826d9c46dd6066d109eabc6255188de91218`. The cache contains one snapshot; its main reference was last modified on 25 September at 17:32, before the representation-store creation at 17:41 local time. Extraction job 487175 names the same offline cache. This evidence supports using that snapshot, but historical metadata does not contain a model-weight checksum proving the match. The runner pins this revision for both tokenizer and model and checks the loaded model revision. Before diagnostic scoring, it requires each checkpoint to reproduce its recorded source-test AUROC within 0.001.

**Execution locations.**

- Source snapshot: `/home/d/dchikhi/CoT-checker-snapshots/verifier-signal-20260929`
- Leaderboard: `/project/aip-azouaq/dchikhi/cot_mech/instruct_leaderboard_v1`
- Original representation store: `/scratch/d/dchikhi/cot_mech/qwen3_8b_instruct_v1/repstore/step_spans`
- Diagnostic output: `/project/aip-azouaq/dchikhi/cot_mech/verifier_signal_v1/run_20260929`
- Scheduler logs: `/project/aip-azouaq/dchikhi/cot_mech/verifier_signal_v1/logs`

The isolated snapshot includes the uncommitted experiment files and records the parent Git revision. It leaves the training snapshot unchanged. No commit was requested or made. The diagnostic requests one whole four-H100 node for one hour. Queue and upstream training time are additional; one hour is an allocation limit, not a measured runtime prediction.

**How the result should guide use.**

| Diagnostic observation | Next use to test | Necessary downstream comparison |
|---|---|---|
| Context-dependent reversals and strong local contrasts even after an invalid prefix | Use the verifier to trigger resampling of the new step | Equal-budget local resampling versus random locations and unassisted continuation |
| Scores mainly reflect inherited errors | Revisit earlier steps when a later alarm fires | Equal-budget backtracking versus resampling only the flagged step |
| Scores mainly reflect final-conclusion validity | Rank completed candidates | Same candidate pool, answer grading, and budget as majority voting and existing reranking |
| Different heads show different response patterns on the same examples | Allocate extra checking when the heads disagree | Equal-budget uncertainty-based and random checking controls |
| Strong wording dependence or weak semantic contrasts | Restrict the interpretation and test natural-trace robustness | Blinded annotation of local error, inherited error, and answer correctness |

These are conditional next experiments. The synthetic diagnostic does not establish that any policy improves downstream performance. It also does not establish the ProcessBench-to-TTS correlation: that requires comparing probes on the same generated candidate pools across sampling budgets, with uncertainty at the problem level. We should interpret all prespecified domains and seeds before selecting a downstream intervention.

**Submission.** Slurm accepted diagnostic job **494672** on 29 September 2026 with `afterok:494595`. Merge job 494595 depends on training-completion job 494594. Submission receipt: `/project/aip-azouaq/dchikhi/cot_mech/verifier_signal_v1/submission_20260929.txt`. The uploaded source hashes are in `source_manifest_20260929.sha256` beside that receipt. Expected scheduler log: `/project/aip-azouaq/dchikhi/cot_mech/verifier_signal_v1/logs/verifier_signal_v1-494672.out`. No diagnostic scores existed at submission.

**Execution amendment after user cancellation.** The user cancelled upstream jobs 494594 and 494595. Slurm cancelled diagnostic job 494672 with `DependencyNeverSatisfied`, before allocation or scoring. I did not resubmit either upstream job. The replacement uses the new frozen bundle `experiments/verifier_signal_v1_available64`, whose examples are byte-identical to v1. Only the declared checkpoint availability changes: seeds 42/43 for attention-query and d128, and 42/43/44 for the other four probes. There are 16 checkpoints. Analysis reports the seeds for every probe and averages within each probe's available seeds. Different seed counts limit direct comparisons; the family bootstrap does not quantify training-seed uncertainty.

The validator still requires all 66 runs by default for the original leaderboard. Its diagnostic API now accepts an explicit seed roster and rejects undeclared missing seeds. Tests cover both the complete and mixed-seed analyses, preservation of default completeness, and invalid seed declarations. All 31 tests pass.

The replacement source snapshot is `/home/d/dchikhi/CoT-checker-snapshots/verifier-signal-20260929-available64`. Its fresh output directory is `/project/aip-azouaq/dchikhi/cot_mech/verifier_signal_v1/run_20260929_available64`. The runtime performs the same protocol, checkpoint, store, source-AUROC, and model-revision checks. It does not require the cancelled merge job. No diagnostic scores were inspected before this amendment.

**Replacement submission.** Cluster preflight validated all **16 trained checkpoints** under the explicit roster. Slurm accepted independent job **494679**, with no dependency. Its receipt is `/project/aip-azouaq/dchikhi/cot_mech/verifier_signal_v1/submission_20260929_available64.txt`; preflight output is `preflight_20260929_available64.log` in the same directory. The scheduler log is `/project/aip-azouaq/dchikhi/cot_mech/verifier_signal_v1/logs/verifier_signal_v1-494679.out`. Expected result: `/project/aip-azouaq/dchikhi/cot_mech/verifier_signal_v1/run_20260929_available64/summary.md`. The job has not produced diagnostic results at the time of this record.

**Completion and local replay.** Job 494679 completed successfully in 2 minutes 9 seconds. All activations and selected checkpoints are now on the laptop. Local CPU replay reproduced all 16 probes on all 912 examples with maximum absolute score difference 0.000002265. See [local execution and findings](verifier_signal_v1_local_results.md) for paths, commands, limitations, and interpretation.
