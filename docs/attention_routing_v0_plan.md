# attention_routing_v0: attention routing as a signal of step correctness

Descriptive, mechanistic first pass on the research brief: do matched correct
and incorrect CoT steps route attention differently toward the question,
previous steps, and themselves? No probe is trained in v0 (brief section 11).
The decision gate:

> Is there a robust, controlled difference in inter-step attention routing
> between matched correct and incorrect candidate steps?

## Setting

- Model: Qwen2.5-7B base, teacher-forced, eager attention, batch 1.
- Data: the 4000 matched PRM800K forks from the S4 contrib-cluster arm
  (`runs/contrib_cluster/forks.jsonl`; fields fork_id, question,
  prefix_steps, step_index, correct, wrong). Text assembly matches
  `s4_contrib_extract_forks.py` exactly:
  `"\n".join([question, *prefix_steps, candidate])`, so results are
  comparable with the hidden-state arm.
- Pairwise design throughout: every statistic is a paired delta
  f(correct) - f(wrong) on the identical prefix, which controls problem,
  prefix content and length, depth, topic, and position by construction.

## What v0 measures (hypothesis coverage)

Per layer x query head x read position (first / last / mean candidate
token), 10 features of the candidate-row attention:

- question_mass, prev_all_mass, prev1_mass, older_mass, self_mass,
  other_mass (region partition; H1, H3, H4 partially)
- entropy, top5_mass (H5), mean_distance (H6)
- sink_mass: attention to token 0. Qwen concentrates an attention sink
  there and, with no BOS, token 0 is question text, so raw question_mass
  overlaps the sink. The analysis reports question_nosink_mass and uses it
  in the grounding ratio. Confirmed in the 0.5B smoke run: the apparent
  question_mass delta (+0.0027) was mostly sink (+0.0031).

Derived at analysis time: question_nosink = question - sink,
external_semantic = prev_all + question_nosink (the grounding-ratio
denominator, saved because the ratio alone is unstable when both masses are
small), prefix_mass = 1 - self_mass (external reliance vs intra-candidate
reliance), grounding_ratio = prev_all / external_semantic (H1),
recency_ratio = prev1 / prev_all (H3). Head-level tables cover H7, layer
curves cover H8, the three read positions cover H9. Sink mass is reported
independently in every table: a sink-only difference is an architectural
allocation signal, not semantic grounding (outcome E of the review).

The three reads are not equivalent tests. First token = pure routing from
the shared prefix (cleanest test of H1, the primary confirmatory readout);
mean = computation while constructing the step; last = routing after most
of the candidate is already in context.

Deferred to a later version: H2/stage-5 relevant-step retrieval (needs the
per-source-step attention matrix), H10 value-weighted contributions,
natural-trace external validation (stage 6), within-trace transition
analysis, and any classifier (stage 7).

## Region assignment

Char-offset based: a token belongs to the segment containing its first
character; separator newlines merge into the preceding token under the Qwen
tokenizer and therefore attach to the preceding segment. Verified by unit
test and by the stage-1 heatmaps (region strip + boundaries).

## Controls recorded per row

candidate/prefix/question/prev1/older token counts, candidate char length,
step_index, n_prefix_steps, number and operator counts, teacher-forced mean
log-prob and mean predictive entropy of the candidate. Length control in the
analysis: Spearman(delta_feature, delta_candidate_length), a rerun on the
length-matched subset (relative length difference <= 10%), and a paired
regression of each first-token delta on the surface deltas (length,
log-prob, predictive entropy, number count) with CR1 problem-clustered SEs;
beta0 is the routing difference after surface controls.

## Inference protocol (frozen before the 7B run)

- All tests are cluster-aware at the problem level (question hash):
  bootstrap resamples problems; sign test and Wilcoxon run on problem-mean
  deltas, never on fork-level deltas.
- Confirmatory metrics, fixed in advance, evaluated on the FULL data at the
  first candidate token: grounding_ratio, prev_all_mass,
  external_semantic_mass, recency_ratio, entropy, mean_distance.
- Head-level analysis is exploratory and split: problems are divided 50/50
  (seeded permutation); heads are selected on the discovery half (BH q<0.05
  within feature, ranked |dz|, max 20 per feature) and re-tested on the
  confirmation half; a head "replicates" only if the confirmation CI
  excludes zero with the discovery sign. Heatmaps show discovery data only.
- Backend parity gate: eager vs sdpa compared on 100 forks (candidate
  log-probs, NLL, greedy agreement, final hidden-state cosine) before the
  extraction shards run, so attention results stay comparable with the
  sdpa-based S4 hidden-state arm.

## Stage-1 validation (built into extraction)

Asserted on the first pass and recorded in the manifest: attention tensors
are per QUERY head under GQA (28 query vs 4 KV heads on 7B), rows normalized
(max err 2.4e-7 in smoke), zero future mass (causal masking), candidate span
contiguous. Head-mean attention rows for the first 8 forks are saved and
rendered by `ar_inspect_heatmaps.py` for manual alignment checking.

## Pipeline

1. `scripts/attention_routing/ar_extract_forks.py` (GPU, 4-way in-node
   sharding): metadata.parquet + features.npy
   (n_rows, 28, 28, 10 features, 3 reads) float16 + inspect npz + manifest.
2. `scripts/analysis/ar_fork_analysis.py` (CPU): table_global.csv (cluster
   bootstrap CI, sign test, Wilcoxon, P(f+ > f-), dz),
   table_global_lengthmatched.csv, table_layer.csv, table_head_top.csv,
   head_effects.npz (dz / P(f+ > f-) / BH-corrected Wilcoxon mask), plots
   (ratio distributions, delta histograms, layer curves, layer x head
   heatmaps), summary.md.
3. `scripts/analysis/ar_inspect_heatmaps.py`: stage-1 token-level heatmaps.
4. `scripts/tamia/jobs/ar/01_extract.sbatch`: whole H100 node, 4 shards via
   CUDA_VISIBLE_DEVICES, merge, analysis in-job.

Core feature logic is pure numpy in `src/analysis/attention_routing.py`,
unit-tested in `tests/analysis/test_attention_routing.py` (21 tests).
Smoke-tested end to end locally on Qwen2.5-0.5B (6 forks, CPU).

## Interpretation guardrails

Effect sizes and P(f+ > f-) over p-values (n=4000 makes tiny effects
significant); head-level findings need BH correction and later a
discovery/validation split before any head is called "special"; a positive
v0 result supports only the weak conclusion (different average routing
statistics), not retrieval failure or causality.
