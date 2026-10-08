# Context ablation v3: does a step judged with less context localize errors better out of domain?

Date: 2026-10-06. Experiment ID: `context_ablation_v3`. Predecessors: §21.16 (v1), §21.17 (v2).

## 1. Why

v1 and v2 found that more probe context did not help. On v2, the readout that sees only the current step:

* ranks post-error steps best (AUROC 0.741, against 0.697 for causal);
* transfers best to PRM800K human labels (F1 0.518, against 0.424).

But every v1 and v2 readout read backbone states computed over the whole prefix. "Local" withheld earlier steps from the probe, not from the backbone.

v3 removes context at the **backbone**. Each target step is re-encoded under a shrinking context, and the readout, positions, labels, training examples and order are all held fixed.

## 2. Conditions

Each target step i of each trace is turned into a short "view". Every view is encoded once by frozen Qwen3-8B at `hidden_states[35]`.

| ctx | backbone input | header (probe prefix rows) |
|---|---|---|
| `full` | "Problem: {p} Solution:" + steps 0..i | "Problem: {p}\n\nSolution:\n" |
| `prev1` | "Problem: {p} Solution:" + step i-1 + step i | same |
| `q` | "Problem: {p} Solution:" + step i | same |
| `none` | "Solution:" + step i | "Solution:\n" |

Steps are joined by blank lines (v1 format). `prev1` equals `q` at i = 0.

**Compact storage.** Each view stores only:

* the header rows;
* the pre-step boundary token, which is the token before step i in that view;
* step i's tokens.

Positions are renumbered contiguously, so every context gives the probe the same layout and the same positions. The views of `full`, `prev1` and `q` have identical stored lengths; `none` has a shorter header. The only thing that differs across contexts is how much the backbone saw when computing those states.

**Probe.** The v1 `local` arm (2,638,593 parameters): prefix rows plus the step's own rows. Earlier steps are never probe rows.

**Training.** Each view carries one labeled target, so the loss is the mean BCE over steps. This is a per-step objective, not v1's per-trajectory one, and it is the same for every context.

* Batches are bucketed by step length, which is identical across contexts. Within a seed, every context therefore sees the same views in the same batches and starts from the same initial weights.
* Settings: v1's selected lr 1e-4, wd 0.1, patience 5, 30 epochs max.
* Seeds 42, 43, 44. There is no new hyperparameter search; this is recorded as a choice.

## 3. Sources and evaluation (the "other datasets")

| source | train / dev / calib | evaluated on |
|---|---|---|
| `v1human` | v1 PRM800K phase-2 human labels (first-error) | ReProbe test (DeepSeek labels), PRM800K human test (in-domain), ProcessBench x4 |
| `v2ds` | v2 ReProbe Qwen3-8B trajectories, DeepSeek-R1 all-step labels | PRM800K human test, ReProbe test (in-domain), ProcessBench x4 |

**Split integrity.** Both sources inherit v1's problem split. Every training problem is therefore disjoint from both test sets and from ProcessBench, because the ProcessBench overlap was already removed in v1 and v2.

**Bounded sizes**, deterministic by sha1(trace_id), documented as a bounded study:

* Train: 8,000 trajectories per source (labeled steps only).
* Dev and calib: 1,500 trajectories each.
* PRM800K human test: 3,000 trajectories.
* ReProbe test: all 2,998 trajectories.
* ProcessBench: all 3,400 traces, every step, so first-error localization can be read from full score sequences.

**Thresholds.** Calibrated on the source's calib split (best F1, higher threshold on ties), then frozen.

## 4. Metrics and contrasts

**Primary: error localization out of domain.** ProcessBench F1_PB from the first threshold crossing, per subset and the mean over the four, plus exact localization accuracy.

**Secondary.**

* Known-label step F1 and AUROC on each evaluation set.
* The post-first-error and up-to-first-error slices on the ReProbe test.

**Paired contrasts.** `q - full` (main), `none - full` and `prev1 - full`, per source.

* Computed per seed and as a seed mean.
* 10,000 problem-clustered bootstrap resamples at fixed thresholds.

**Prespecified reading.**

* "Less context localizes better out of domain" predicts:
  * `q - full` greater than 0 on ProcessBench mean F1_PB for both sources;
  * a monotone ordering full ≤ prev1 ≤ q.
* `none` tests whether the problem itself matters.
* A positive in-domain but negative out-of-domain result is a transfer failure, not support.

**Control.** The v1 structural baseline for each source.

## 5. Execution

1. `scripts/build_context_views.py` builds one view manifest per context from manifest_v1 and manifest_v2, on a CPU job.
2. The encoder gains view headers and compact storage (`encode_trajectory_token_store.py`, backward compatible).
3. One GPU job per context runs in parallel on all available nodes: encode, then 6 fits (2 sources x 3 seeds), then scoring.
4. `scripts/collect_context_views.py` maps view predictions back to (trace, step) per source. `eval_contextual_token_probe.py` then runs with configurable arms, contrasts and step sets.
5. Results go to `results/context_ablation_v3/`, and REPORT gets §21.18.

## 6. Out of scope

* Probe-context arms other than `local`, a layer sweep, generation-time context, and new labels.
