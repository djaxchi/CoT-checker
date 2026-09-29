# What do the Instruct verifiers detect?

Prepared 28 September 2026. For the 29 September launch and environment corrections, see `docs/verifier_signal_v1_launch_2026-09-29.md`. Original preparation status: dataset and execution setup prepared locally; no backbone extraction or checkpoint scoring has run for this experiment. Another agent should execute it after the Instruct leaderboard finishes. The user's latest supplied job IDs were 493184 (d256), 493185 (attention-query and d128), and 493186 (dependent merge). Those IDs are a handoff reference, not a live status check.

**Current execution amendment, 29 September.** The user retained the 64-run leaderboard and cancelled completion of the two seed-44 runs. The active bundle is `experiments/verifier_signal_v1_available64`: identical example bytes and metrics, with two seeds for attention-query and d128 and three for the other four probes, totaling 16 checkpoints. Set `SIGNAL_BUNDLE` to that bundle. The batch script validates its exact roster without requiring a complete 66-run leaderboard or a merge dependency. The original three-seed design below remains the reference; the launch record supersedes its completeness and scheduling requirements.

**Question and scope.**

Does a frozen verifier respond to a newly invalid inference, to an earlier error in the reasoning prefix, to the correctness of the current conclusion relative to the original problem, or to surface wording? The experiment uses controlled text interventions to make these explanations predict different score patterns.

Earlier Qwen2.5 audits in `REPORT.md` §15 narrowed several alternatives: length/position residualization changed AUROC from 0.808 to 0.763; measured confidence controls changed another probe's AUROC from 0.699 to 0.697; matched minimal-edit forks retained 0.723 pair accuracy. Those observations establish remaining predictive information, not its semantic identity. They also do not identify what the newly trained Qwen3 Instruct heads use.

This pilot measures the behavior of the frozen verifier. It does not establish a circuit, prove that the generator uses the decoded information, or show that the score supplies an effective repair location. Positive results identify a narrower hypothesis for later activation interventions and natural-trace validation.

**Prepared files.**

| File | Purpose |
|---|---|
| `experiments/verifier_signal_v1/config.json` | Frozen data, model, seed, and analysis settings |
| `experiments/verifier_signal_v1/examples.jsonl` | 912 materialized examples with arithmetic witnesses and separate labels |
| `experiments/verifier_signal_v1/manifest.json` | Counts and SHA-256 hashes |
| `experiments/verifier_signal_v1/probes.cells` | Six representation/learner pairs, each at seeds 42/43/44 |
| `experiments/verifier_signal_v1/review_samples.md` | Readable complete families from every experiment/domain |
| `src/analysis/verifier_signal.py` | Deterministic builder, label checks, paired metrics, family bootstrap |
| `scripts/analysis/verifier_signal_experiment.py` | Build, validate, preflight, extract, score, analyze commands |
| `slurm/verifier_signal_tamia.sh` | Whole-node offline TamIA run; not submitted |
| `tests/analysis/test_verifier_signal.py` | Semantics, rejection cases, estimands, and real adapter smoke tests with model stubs |

The prepared data contain 72 families: 48 contextual-validity reversal families and 24 inherited-error families. Three candidate wording variants yield 864 target rows. An additional 48 rows score the two inherited-error prefixes before their next step. All variants of a family stay together. The fixed dev partition contains 18 families, and the test partition 54. These are numeric-instance partitions within the same templates, not held-out-template generalization. No training split exists and no probe will be fitted on this data.

The builder validates exact arithmetic and rendered text. No human has signed off on the full dataset; the manifest records this explicitly. Review the sample families before execution. If an ambiguity requires changing examples or analysis, create a new version and preserve v1. Do not inspect scores and then silently alter its tests.

**Test A: contextual validity reversal.**

Hold the problem instruction and candidate step fixed while changing the mathematical context that determines the candidate's validity. For example:

| Prefix | Candidate: `x = 4.` | Candidate: `x = 3.` |
|---|---|---|
| `The equation is 2*x + 3 = 11.` | Valid | Invalid |
| `The equation is 2*x + 3 = 9.` | Invalid | Valid |

The problem asks the reader to solve the equation established in the next line. Each candidate string occurs in both validity conditions, and each prefix occurs with both candidates. A fixed preference for one answer string cannot satisfy both paired comparisons. A prefix-only offset also cancels within each comparison.

Generate 12 families in each of four domains: affine equations, multiplication, inequalities with sign reversal, and function substitution. The inequality families explicitly test the reversal of the inequality when dividing by a negative number. The examples use small exact integers and avoid approximate numeric grading.

Within each family, let `s[p,c]` denote the probability of incorrectness for prefix variant p and candidate variant c. Diagonal cells are locally valid. Compute:

```text
d0 = s[0,1] - s[0,0]
d1 = s[1,0] - s[1,1]
local_contrast_mean = (d0 + d1) / 2
both_preferences_correct = 1 if d0 > 0 and d1 > 0, else 0
```

The primary endpoint is the mean local contrast on plain-wording test families, reported by domain. Report both-preferences-correct and each component contrast alongside it. Record ties explicitly; do not treat them as successful reversals or assume a universal chance rate for the joint endpoint. This balanced contrast removes additive prefix/string preferences, but it does not exclude more complex shortcuts.

**Test B: inherited error versus newly invalid inference.**

Keep the original problem fixed. Cross whether the preceding equation is valid with whether the next step follows from that equation. For example, original problem `Solve 2*x + 3 = 11 for real x.`:

| Prefix | Candidate | Prefix invalid | Local inference invalid | Conclusion invalid relative to problem | Trace invalid |
|---|---|---:|---:|---:|---:|
| `Subtracting 3 from both sides gives 2*x = 8.` | `x = 4.` | 0 | 0 | 0 | 0 |
| Same prefix | `x = 5.` | 0 | 1 | 1 | 1 |
| `Subtracting 3 from both sides gives 2*x = 10.` | `x = 5.` | 1 | 0 | 1 | 1 |
| Same invalid prefix | `x = 4.` | 1 | 1 | 0 | 1 |

Here, local validity means consistency with the immediately preceding equation. A correct final conclusion following an invalid prefix does not establish deliberate self-correction: the last row contains an unannounced inconsistent transition. The input is inconsistent as a whole, so ordinary classical entailment is not the labeling rule. These operational labels are explicit and do not pretend to be human PRM800K annotations.

The contrast between local and global validity necessarily changes what a label means. Preserve all four labels, and do not collapse them into one headline accuracy. `label` in the representation store is only a transport field equal to `local_invalid`; the scoring path never fits or evaluates ProcessBench metrics against it.

For the coded variants, candidate 0 is the solution to the original problem, candidate 1 is the alternative, and prefix 1 introduces the earlier error. Local validity again lies on the diagonal. Compute the same d0 and d1 as Test A. Also compute:

```text
prefix_error_contrast
  = [s[1,1] + s[1,0] - s[0,0] - s[0,1]] / 2

global_conclusion_contrast
  = [s[0,1] + s[1,1] - s[0,0] - s[1,0]] / 2
```

These are balanced factorial contrasts, not additive assumptions about the underlying mechanism. Both a local response and an inherited response can coexist. Examine the individual cell scores when interactions are large. `local_contrast_other_prefix` refers to the invalid prefix in Test B, but to the second valid context in Test A.

Score each preceding equation separately as a candidate under the original problem and an empty reasoning prefix. `before_error_contrast` compares those two scores. `prefix_contrast_after_minus_before` subtracts it from the inherited-prefix contrast after the next step. This is descriptive: the model reads different candidate spans at the two times, so the difference is not a causal mediation estimate or a conserved quantity.

**Wording controls.**

For each mathematical candidate use three wrappers: the bare assertion, `Therefore, ...`, and `I am certain that ...`. Prefixes stay fixed. Both valid and invalid examples receive every wrapper. Primary results use the plain version; the others measure sensitivity to discourse and expressed certainty.

Report signed and absolute score changes relative to plain wording and the change in the local contrast. Do not divide by a near-zero semantic contrast. These wrappers change token count and style together, so any effect is wrapper sensitivity, not a clean estimate of confidence or length alone. No claim of invariance follows from a nonsignificant difference in this small pilot.

**Frozen models and input contract.**

Use the six pairs listed in `probes.cells`: last-token linear, mean-pooling linear, learned-query pooling, and the exact d128, d256, and d512 Transformer specifications from the Instruct grid. Use all three seeds, for 18 checkpoints. These choices cover distinct readouts and capacities; do not select them based on diagnostic scores. The lengthfree and SAE extensions are outside v1 because their preprocessing requires separate validation.

Use Qwen/Qwen3-8B, `hidden_states[35]`, width 4096, bfloat16 forward passes and float16 storage. This is the Instruct model ID, not Qwen3-8B-Base. Use the existing verifier prompt function and its separate prefix/candidate tokenization. Do not apply a chat template or switch to generation-context states. Save the pre-step boundary and candidate token states using the existing span-store encoder. Each example encodes independently, with no future step supplied.

All chosen checkpoints use `rescale=none` and sequence cap 512. The extractor rejects candidates longer than the cap and rejects whole inputs beyond 2048 tokens rather than truncating. It saves token lengths so score sensitivity can be checked against tokenization differences. It records the resolved cached model revision, but historical training metadata may not pin that revision. The executing agent must verify that the cached backbone/tokenizer snapshot is the one used for Instruct extraction; matching a model name alone is not sufficient.

**Integrity and uncertainty.**

The batch script first validates the complete 66-run leaderboard, then the 18-checkpoint diagnostic roster. It checks training protocol, seed identities, checkpoint availability, input fingerprints, training-store layer/dimension/backbone, and bundle hashes. Missing cells fail the run rather than changing the comparison set.

Before scoring diagnostic examples, the scorer evaluates every checkpoint on the original `test_2k` representation store. It requires the recorded source-test fingerprint and AUROC reproduction within absolute 0.001. If it fails, inspect numerical precision and preprocessing; do not widen the tolerance after inspecting diagnostic outcomes. AUROC reproduction is a smoke check, not proof that every score matches bit for bit.

Checkpoint and result-file SHA-256 hashes, extraction provenance, dataset hash, source-code hashes, and store fingerprints accompany the outputs. The extractor refuses to overwrite an existing shard; scoring refuses to overwrite a scores directory. Use a fresh output root for a rerun. Analysis requires every frozen checkpoint and every example exactly once.

Use 2,000 bootstrap resamples of entire families within domain and partition. Average each family's metric across training seeds before the main bootstrap. Also report every seed separately. The seed-averaged joint-preference endpoint averages per-seed decisions; it is not the performance of a score ensemble. Do not count 912 rows or three training seeds as 2,736 independent mathematical cases. Intervals describe sensitivity to the sampled numeric families within these templates; they do not establish generalization to natural reasoning or unseen templates. Report all prespecified domains and endpoints without selecting favorable ones. Intervals are descriptive and do not correct for multiple comparisons.

No F1 threshold is selected on this diagnostic data. Do not compare synthetic-set F1 to ProcessBench F1 or natural-data F1. This is a paired sensitivity study with explicit labels, not a replacement benchmark.

**Local preparation and validation commands.**

From the project root, with its Python environment active:

```bash
python scripts/analysis/verifier_signal_experiment.py validate
python -m pytest tests/analysis/test_verifier_signal.py -q
```

The examples already exist. The `build` command deliberately refuses to overwrite them. To reproduce in a temporary directory, copy `config.json` and `probes.cells` there and pass that directory as `--bundle` to `build`, then compare `examples.jsonl` hashes.

**Execution handoff after training.**

Read the TamIA skill and `TAMIA.md` before cluster work. Confirm the merge job succeeded and the full leaderboard validator passes. The user supplied job 493186 as the merge job; check whether a newer replacement exists. No cluster status was queried while preparing this setup.

Copy this code and the complete experiment bundle into an isolated source snapshot that also contains the trainer and scorer dependencies. The prepared JSONL lives under `experiments/`, not the ignored root `data/` directory. Do not replace the source tree of a running training job. Verify the snapshot includes the new uncommitted files; a plain archive of HEAD will omit them until they are committed. Commits require user authorization and must follow repository conventions.

Set paths on TamIA, adjusting them only to the validated final training location:

```bash
export PROJECT_ROOT=/path/to/isolated/CoT-checker-snapshot
export GRID_ROOT=/project/aip-azouaq/dchikhi/cot_mech/instruct_leaderboard_v1
export PRM_STORE=/scratch/d/dchikhi/cot_mech/qwen3_8b_instruct_v1/repstore/step_spans
export SIGNAL_OUT=/project/aip-azouaq/dchikhi/cot_mech/verifier_signal_v1/run_01
export HF_CACHE_ROOT=/project/aip-azouaq/dchikhi/hf_cache
export SIGNAL_MODEL_REVISION=b968826d9c46dd6066d109eabc6255188de91218
```

The wrapper uses Python 3.12 and a node-local environment with offline installs pinned to torch 2.14.0, transformers 5.14.1, numpy 2.5.3, and pyyaml 6.0.3. These versions match the original Instruct extraction log. It records the full installed environment and requires a pinned cached model revision. Do not download models on compute nodes.

After those checks, the executing agent may submit:

```bash
sbatch "$PROJECT_ROOT/slurm/verifier_signal_tamia.sh"
```

If submitting while the confirmed merge job is still pending, use `--dependency=afterok:<actual_merge_job_id>` instead. The explicit full-grid validator remains required even with a successful Slurm dependency. The requested one-hour whole-H100-node allocation is a conservative initial limit for this small extraction plus head scoring, not a measured runtime estimate. Four GPU workers extract disjoint shards; a single GPU then scores the 18 small heads sequentially.

For debugging, the CLI also exposes the stages separately:

```bash
python scripts/analysis/verifier_signal_experiment.py preflight \
  --cells-root "$GRID_ROOT/cells" --reference "$GRID_ROOT/reference.json" --prm-store "$PRM_STORE"

# Extraction normally runs all four shard indices through the batch script.
CUDA_VISIBLE_DEVICES=0 python scripts/analysis/verifier_signal_experiment.py extract \
  --out "$SIGNAL_OUT" --shard-idx 0 --num-shards 4

CUDA_VISIBLE_DEVICES=0 python scripts/analysis/verifier_signal_experiment.py score \
  --out "$SIGNAL_OUT" --cells-root "$GRID_ROOT/cells" \
  --reference "$GRID_ROOT/reference.json" --prm-store "$PRM_STORE" --num-shards 4 --batch-size 128

python scripts/analysis/verifier_signal_experiment.py analyze --out "$SIGNAL_OUT"
```

Run these from the isolated project root. Running only the shard-0 example does not complete extraction, and scoring will reject that incomplete store.

**Required outputs and interpretation.**

The run creates `store/diagnostic/shard_00` through `shard_03`, an extraction manifest per shard, 18 files in `scores/`, `analysis.json` containing per-cell and seed-averaged family metrics and intervals, and a readable `summary.md` of the plain-wording test endpoints. It also preserves source hashes and run logs. Probability-scale effect sizes depend on calibration; compare response patterns within checkpoints and report preference reversals rather than ranking heads by raw contrast magnitude. No plotting is implemented or required for this pilot. Keep all outputs on persistent project storage and report exact paths.

Interpret possible outcomes as follows:

| Observation | Supported reading | Still unresolved |
|---|---|---|
| Preference reverses with context across domains | Sensitivity to the specified contextual relations | Mechanism, natural-data importance, and richer shortcuts |
| Scores rise mainly with invalid prefixes, with weak local contrasts | Greater sensitivity to inherited trouble in this test | Whether this explains natural false positives or helps selection |
| Strong local contrasts under both valid and invalid prefixes | Sensitivity to the new transition's local consistency | Whether flagged transitions are useful restart locations |
| Strong conclusion contrast, weaker local contrast | Greater sensitivity to agreement with the original problem's answer | Whether the head computes the answer or uses correlated evidence |
| Large wrapper shifts or changed preference reversals | Dependence on the tested wording | Which part of wording, length, or certainty caused it |
| Weak or mixed results | This test does not identify a stable response pattern | Domain shift, power, calibration, and heterogeneous mechanisms |

Do not label the last row of Test B as demonstrated repair, interpret null steering as noncausality, or rename the whole residual signal “correctness” after excluding these alternatives. Local, inherited, and conclusion sensitivity can coexist.

After inspecting these results, validate the most specific surviving interpretation on independently reviewed natural Instruct traces, with scores hidden during initial annotation. Natural-trace annotation, recovery rollouts, and activation patching are follow-ups, not prepared data or completed work in v1. Choose patching targets only after identifying a reproducible behavioral contrast; measure effects on the verifier separately from effects on the generator.

Relevant methodological context: [Belinkov, Probing Classifiers](https://aclanthology.org/2022.cl-1.7/) distinguishes the promises and limitations of probe-based interpretations; [Zhang and Nanda, Activation Patching](https://arxiv.org/abs/2309.16042) shows why intervention design and metrics affect mechanistic conclusions. This pilot does not claim that counterfactual evaluation itself is novel.
