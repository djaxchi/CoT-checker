# Step value labels v1: downstream risk for every step from shared rollouts

Date: 2026-10-13. Experiment ID: `step_value_labels_v1`. Predecessors: `factorized_step_labels_v1` (§21.19, the risk definition and its 501-fork labels), `risk_probe_v0` (§21.20, the reason for more labels).

## 1. Why

risk_probe_v0 asked whether Qwen3-8B's L35 states carry downstream risk beyond validity, on 531 labeled forks:

* A linear readout learned risk beyond validity above a permutation null (residual Spearman 0.195 against a null 95th percentile of 0.055).
* Adding it to the validity score did not improve `harmful` detection: +0.019 AUROC, 95% CI [-0.008, 0.045].
* The interval is wide because only 98 forks are `harmful`, not because the labels are noisy.

More labels are needed, and the v1 protocol cannot supply them:

* **Supply.** A v1 fork needs every earlier step correct, so only first errors qualify. The ReProbe v2 train split holds 1,056 such incorrect candidates (440 early, 370 mid, 246 late), 3.4x the current 315.
* **Cost.** One v1 fork costs 8 sampled alternatives plus 72 continuations, for one labeled step.

## 2. Design

### 2.1 The identity

v1 defined, for step s of a trace:

```
risk_diff(s) = err(A) - mean_j err(C_j)
```

* A continues after the real step s.
* C_j samples a step s from the same prefix, then continues.

A C_j rollout is therefore a free rollout from the cut before step s, and an A rollout is a free rollout from the cut after it. With err(c) the error rate of rollouts from cut c:

```
risk(s) = err(s + 1) - err(s)        (in expectation)
```

Cut c is the reply truncated before step line c (c = 0..n-1), or before the answer line (c = n). err(0) is the policy's error rate on the problem.

One set of k rollouts per cut labels every step of a trace. Each cut serves two neighboring steps.

### 2.2 Sampler and grading

These are unchanged from v1, so the labels are the same quantity:

* Qwen3-8B at revision b968826d, bf16.
* ReProbe's released token ids under `reprobe_prompt`, never re-tokenized text.
* T = 1.0, top-p 0.95, top-k 50.
* Reply cap 4,096 tokens, counted from the reply start. A truncated reply is an error.
* The project grader through `extract_reprobe_answer`. All rollouts of one trace are graded together, one grade per answer identity.

### 2.3 Rollouts per cut

**k = 16 at every cut, fixed.**

Precision is comparable to v1. At an error rate of 0.5, v1's SE is sqrt(0.25/8 + 0.25/64) = 0.19; here it is sqrt(0.25/16 + 0.25/16) = 0.18.

Cost per labeled step is about 16 (n + 1) / n rollouts: 19 for a 6-step trace, against v1's 72 plus 8 alternatives.

**Adaptive allocation was checked and rejected.** `scripts/step_value_labels/adaptive_check.py` replayed the rule "draw 4 rollouts at both cuts and stop if all 8 agree" on the v1 full-run rollouts:

* It would stop 58% of forks: 51% of incorrect, 69% of correct.
* It would also zero 7 of the 94 forks the full-precision labels call `harmful`. Those are steps where the real step fails 4 of 4, the alternatives also fail 4 of 4, and yet the alternatives fail less often at full precision.

That loss falls on the signal this experiment exists to measure. The estimator is also biased: at a true error rate of 0.1 it returns 0.05 in expectation.

### 2.4 Throughput

v1 sampled 8 continuations per `generate` call, one fork at a time: about 1.45 rollouts per second per GPU, with GPU utilization near 55%.

The labeler instead batches the rollouts of several traces, up to 128 rows per call, left-padded. It samples in rounds of 512 new tokens, and rows that have not ended are carried into the next round with their tokens appended (`sample_rounds`).

Continuing generation on the same ids has the same distribution as an uninterrupted call. It keeps a rare runaway row from holding a whole batch to the 4,096-token cap. In v1, 2.3% of continuations exceeded 512 tokens and 0.55% exceeded 2,048.

## 3. Sets

Each set is frozen to JSON before any job (`scripts/step_value_labels/build_trace_set.py`). Each record carries the problem, gold answer, prompt and reply ids, step spans, cuts and cut offsets.

**Validation set (`trace_set_validate.json`).**

* The 501 v1 full-run forks, each with cuts [s, s + 1].
* 1,002 cuts, 16,032 rollouts.
* Built from `fork_split_full.json` (sha256 bd324961...). The set's sha256 is in `build_audit_validate.json`.

**Label set (`trace_set_label.json`).**

* Source: whole traces from the ReProbe v2 `train` split, parsed token-exactly as in v1. All step lines are labeled, not only first errors. Validity is the per-step DeepSeek label (the §21.17 all-step labels), with self labels carried alongside.
* Exclusions:
  * every problem of the 531 risk_probe_v0 forks (`fork_split_full.json`, `fork_split_v1.json`), which stay a held-out test set;
  * the ProcessBench-overlap problems, already outside v2 `train`.
* Groups, filled in sha1(trace_id) order, at most 2 traces per problem, the scarce group first:
  * `error`: at least one DeepSeek-incorrect step. All of them.
  * `hard_clean`: no DeepSeek error, on a problem where some trace has a labeled error or a wrong answer. As many as `error`.
  * `easy`: the rest. 10% of the two groups above, to anchor the low end.
* Traces that hit ReProbe's 256-token cap (`--include_unfinished`) join too. They are cut after their last complete step line (`parse_unfinished_row`).

**Amendment 2026-10-13, before any label job: the set includes capped traces.** The count pass showed why finished traces alone are too few:

* Finished traces only: 957 error traces and 1,856 incorrect steps after exclusions. The eval-fork exclusion removed 1,145 traces, mostly from hard problems.
* The v2 train split holds 11,147 retained traces that hit the cap, 3,058 of them with an error, carrying 5,595 incorrect steps.
* 97% of them parse: 10,845 ok, 302 decode mismatches. Their step lines are complete and end in a newline, and any cut-off fragment after the last line is dropped.
* A capped trace's prefixes are policy samples like any other, and its rollouts run to the usual 4,096-token cap.

The frozen set (`trace_set_label.json`, sha256 in `build_audit_label.json`):

| group | finished | capped | traces |
|---|---|---|---|
| error | 859 | 2,496 | 3,355 |
| hard_clean | 569 | 1,196 | 1,765 |
| easy | 469 | 285 | 754 |

* 5,874 traces from 3,304 problems, with no problem in common with the 531 test forks.
* 38,190 cuts, so 611k rollouts at k = 16.
* 6,124 DeepSeek-incorrect steps: 3,355 first errors and 2,769 post-error steps. 1,635 come from finished traces and 4,489 from capped ones.
* The held-out test forks are all finished first errors. Every step row carries `finished`, so risk_probe_v1 can train with and without capped traces.

## 4. Gates and outputs

**Validation gate.** This runs before any label-set job.

On the 501 forks, compare the labeler's err(s) with v1's mean err(C), and its err(s + 1) with v1's err(A). For both mean differences (fork bootstrap, 2,000 resamples):

* the 95% CI must include 0;
* the absolute mean must be below 0.03.

This tests the identity of §2.1 and the batched, left-padded, multi-round sampler in one step. Spearman(new risk, v1 risk_diff) is reported, with no threshold, since both carry an SE near 0.08 to 0.18.

If the gate fails, no label is produced. The first suspects are left-padding position ids and the round boundary.

**Label-set outputs** (`scripts/step_value_labels/summarize_values.py`):

* `step_values.jsonl`, one row per step. View id `<trace_id>#<claim index>` (the context_ablation_v3 view id), err before and after, risk, SE, `harmful`, DeepSeek and self validity, and phase: pre-error, first error or post-error.
* `summary.json`: rollout counts, truncation and ungradeable shares, err(0), and risk and harmful shares by validity, phase and group.

**Read-outs before training.** The label set should reproduce v1 where they overlap:

* risk near 0 on correct steps;
* about 44% of first-error steps `harmful` (v1: 44% of incorrect steps above 2 SE).

Post-error steps are expected to show compressed risk: with err(s) near 1 there is little left to lose.

## 5. Execution

1. `build_trace_set.py --from_fork_splits fork_split_full.json --tag validate`, locally. Done 2026-10-13.
2. `build_trace_set.py --ds_parquet ... --manifest_v2 ... --exclude_fork_splits fork_split_full.json fork_split_v1.json --tag label --local_files_only`:
   * a `--count_only` pass first;
   * inputs `bidirectional_token_probe_v2/raw/ds.parquet` and `manifest_v2` on TamIA.
3. Upload both sets and their audits to `$R/split`, where `R = /project/aip-azouaq/dchikhi/cot_mech/step_value_labels_v1`, then make a code snapshot (`scripts/tamia/snapshot.sh step_value_labels_v1`).
4. `slurm/submit_step_value_labels.sh <snap> smoke`: 2 traces per shard, 20 minutes.
5. `slurm/submit_step_value_labels.sh <snap> validate`: one H100:4 node. Check the gate with `summarize_values.py --track1_dir` against the v1 full-run track-1 shards.
6. `TIME=... slurm/submit_step_value_labels.sh <snap> full 4`, with the walltime set from step 5's measured rollouts per second.

## 6. Next experiment (not part of this plan)

risk_probe_v1 retrains the risk readouts on the label set and tests on the 531 held-out forks. Q1 to Q3 of risk_probe_v0 are reused unchanged, so the numbers compare directly.

It needs `full` views of the label-set steps at L35. The capped v2_train views cover part of them, and `build_context_views.py` covers the rest. Its design is written once this set's counts and gate are in.

## 7. Caveats

* Neighboring steps share a cut, so their risk estimates are negatively correlated through it. Evaluation resamples whole traces or problems, never single steps.
* Risk is defined relative to this policy and sampler. Validity is a property of the text.
* Finished traces come from problems ReProbe's Qwen3-8B finished within 256 tokens. Capped traces are on average longer and harder, so the label set's mix differs from the finished-only test forks.
* Post-error steps have a floor effect, so the first-error subset is reported separately and remains the primary target.
