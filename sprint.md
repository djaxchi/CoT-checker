# Sprint Log

One row per sprint: what we hypothesized, what we tested, what we got. Keep it terse. Newest on top.

| Sprint | Dates | Hypothesis | Verdict |
|---|---|---|---|
| S3 | 2026-06-16 → (open) | **Pivot:** stop optimizing the ProcessBench score; explain it. The ~41% hidden-state signal is a *mixture of failure-mode-specific signals*, not one unified correctness concept. Decompose via failure-type annotation, controlled contrasts, layer-wise probing, geometry, transfer, and causal intervention. | in progress |
| S2 | 2026-06-05 → 06-09 | Fork-based preference supervision (ranking / triplet) shapes a better latent than reconstruction-only; + DenseLinear model-size scaling (1.5B–32B) | ~ partial: forks improve transfer not oracle; scale raises oracle ceiling (0.476 @ 32B) but not deployable calibration (closed) |
| S1 | 2026-05-21 → 05-26 | SAE/SSAE latents localize the first wrong CoT step better than raw dense hidden states | ✗ not supported |

---

## S3 — What do hidden states encode about CoT correctness? (2026-06-16 → open)

**Pivot.** S1/S2 asked whether a learned representation can *beat the ProcessBench score*; answer was
no (raw dense holds the strongest oracle signal; scale lifts the ceiling to 0.476 @ 32B but not
deployable calibration). S3 changes the question from optimizing the number to **explaining it**.
The empirical anchor is fixed: hidden-state probes reach ~41% on ProcessBench using only internal
activations — too weak for general correctness, too strong to be noise. ProcessBench becomes the
*starting point and a downstream diagnostic*, not the objective.

**Central hypothesis.** Correctness is not an atomic internal concept. The binary correct/incorrect
label collapses mechanistically distinct failure modes (arithmetic, algebraic, variable binding,
unit mismatch, unsupported premise, goal drift, constraint violation, logical inference, post-hoc).
The ~41% is a mixture of failure-mode-specific signals; a binary probe wins when a benchmark's
failures align with the encoded signatures and loses otherwise.

**Plan (6 stages).** (1) profile the ProcessBench signal by failure mode; (2) build controlled
correct/corrupted contrasts per mode; (3) extract dense last-token all-layer hidden states (SAE/SSAE
deferred to where the dense signal appears); (4) per (mode × layer) linear probes → F1 heatmap vs
surface baselines; (5) geometry + transfer (controlled → natural → ProcessBench); (6) causality via
activation patching, steering, sparse-feature ablation.

**Metrics.** macro F1 for detection; ProcessBench first-error score as diagnostic only; geometry
(direction alignment, class distances, seed/size stability); causal (continuation correctness,
final-answer likelihood, dose-response).

**Full governing brief:** `docs/research_direction_s3.md` (sourced from the research proposal).

### Stage 1 findings — failure-mode profiling and geometry (2026-06-18)

Profiled the 7B DenseLinear run (oracle macro F1_PB=0.413) at the **last layer, last token of
each step**. Built a loader joining hidden state + probe score + gold first-error + text
(`src/data/processbench_probe_data.py`), then ran four analyses. Headline: the central hypothesis
(that the signal decomposes into a discrete taxonomy of failure modes) is **not supported by the
representation geometry** at this layer/token.

1. **No discrete failure-mode taxonomy in the signal.** Sampled 200 first-error steps (stratified
   50/50 detected/missed x 4 subsets), labeled by Claude Opus subagents into the 10-mode taxonomy
   (Haiku rejected: only 44% agreement with Opus, systematic over-tagging of arithmetic), plus two
   hand-corrections. Per-mode detection rate spread is narrow and not significant: best-caught
   `unsupported_premise` 0.61 (n=28) vs hardest `logical_inference_error` 0.45 (n=60), two-proportion
   z=1.4, p=0.17. Failure mode is not linearly decodable from the hidden state: 5-fold balanced
   logistic regression scores at or below the majority floor on PCA-50 (0.29 vs 0.30), and only 0.32
   on the full 3,584 dims (decorrelated), 2 points over floor at n=200, not significant. The earlier
   "probe is bad at arithmetic" reading was a Haiku labeling artifact and disappears with clean labels.

2. **No clusters of first-error steps.** Clustering the 200 tagged steps on the full decorrelated
   hidden state (KMeans, cosine) gives a best silhouette of 0.14, i.e. a continuous cloud, not
   separable types. Only soft regions appear; a tentative high-confidence/low-detection
   "computational" region vs low-confidence/high-detection "logical" region is suggestive but
   n-limited.

3. **First-error steps are not globally distinguishable from correct steps.** Projected all 25,697
   step encodings (full 3,584 dims, UMAP cosine, HDBSCAN): 7 clusters fall out, but first-error rate
   is flat across them (0.026 to 0.108 vs 0.086 base rate). Error steps are sprinkled through every
   cluster, never pooled. Incorrectness is a diffuse property, not a region of representation space.

4. **The one robust axis is step locality.** The clusters that separate cleanly are defined by
   **position in the trace** (final-step clusters at step_frac ~0.99 split off as their own islands);
   topic/subset is secondary (cluster subset-purity 0.29 to 0.50). Neither failure mode nor
   correctness organizes the space; step position does.

**Caveat and next step.** All of the above uses a single readout: last layer, last token of the step.
The next move is to ablate this choice over **layers and token positions** before concluding that the
representation lacks failure-mode structure, since mid-layer or step-internal tokens may carry it.

Artifacts: `scripts/analysis/s3_failure_mode_scatter.py`, `s3_cluster_failures.py`, `s3_project_all.py`
(+ sampling, labeling, and correction scripts); outputs under `results/s3_first_error/` (gitignored).

---

## S2 — Fork-based representation shaping + model-size scaling (2026-06-05 → 06-09)

**Motivation.** S1 showed the so-called "contrastive" methods are recon + L1 + an auxiliary BCE head that is *discarded* (not contrastive learning), and no learned representation beat raw dense. PRM800K forks (same prefix, multiple rated next steps) give matched positive/negative continuations. We use them to shape the *representation*, not to train a better classifier; eval stays the locked pipeline (freeze representation → fresh probe → ProcessBench).

**Hypotheses.**
- H1 fork preference supervision > reconstruction-only.
- H2 sparse reps benefit more from preference supervision than dense AE reps.
- H3 reasoning validity is local preference structure (ranking helps).
- H4 reasoning validity is local geometry around a prefix (triplet helps).

**Design.** 2×2 matrix = {AE, SAE} × {Ranking, Triplet} → ae_rank, sae_rank, ae_triplet, sae_triplet. Baselines carried from S1 + dense AE (recon-only) + SAE-mixed. Backbone Qwen2.5-1.5B, last-token hidden states. Data: full-scale PRM800K forks (~40k train / 5k val, problem-disjoint, surplus held out). Ranking = scalar head, score(pos) > score(neg); Triplet = anchor(prefix) close to pos, far from neg in latent space. Heads discarded; only `representation.pt` saved. Eval: 4 ProcessBench subsets, macro F1_PB, val-selected + oracle thresholds.

**Results.** Full study in `mpb_results.md` §3.11. Fork dataset: 54,737 valid forks → 40k train / 5k val (problem-disjoint, one pair/fork, 120k train + 15k val encoded anchor/pos/neg rows). The preference objective is genuinely learned (pair accuracy ≈ 0.85–0.88 across rank/triplet at w=1). Recon-only controls keep high oracle separability (AE Recon oracle 0.360) but collapse at the deployable threshold (val-selected 0.046). Fork objectives mainly improve val-selected transfer: AE Rank w=100 reaches macro val-selected F1_PB ≈ 0.235 (vs DenseLinear 0.1855), but its oracle macro is only ≈ 0.290, below DenseLinear's 0.3773 and SAE-mixed's 0.3688. Raising obj_weight improves val-selected F1 while pair accuracy slightly drops (0.879 → 0.857 for rank), so the gain is calibration / score-distribution reshaping, not better fork ordering. Triplet is less stable (inflates recon MSE, no overall win).

**Verdict.** ~ Partially supported. Fork preference objectives improve deployable (val-selected) threshold transfer over reconstruction-only, but do **not** raise the oracle ceiling; dense hidden states still hold the strongest oracle-readable signal. Study closed.

### Conclusion of contrastive / preference-objective audit

- The earlier `sae_contrastive` label was **inaccurate**. That method was `MSE reconstruction + L1 sparsity + auxiliary BCE label head` with the head discarded before the fresh probe; correctly described as `SAE + auxiliary BCE shaping`, not contrastive learning.
- An **implementation bug was ruled out** during the Sprint 2 forensic audit: the rank/triplet objectives are graph-connected, gradients reach the encoder, obj_weight is applied, and the optimizer minimizes the combined loss. The suspiciously fast runtimes are kernel-launch overhead, not skipped computation.
- The **pairwise objectives are optimized correctly**, reaching ≈ 0.85–0.88 pair accuracy on the fork signal.
- **Recon-only controls were added** (`ae_recon`, `sae_recon`) so any F1 delta is attributable to the objective term, and the **obj_weight sweep** (1, 10, 100) was run for all four objective cells.
- **Classifier-level follow-up** (`mpb_results.md` §3.12): dense pairwise scorers (`dense_rank`, `dense_anchor_rank`, `dense_rank_absmargin`) train *only* a linear scorer on frozen hidden states (no AE/SAE), swapping BCE for a ranking/margin loss. `dense_rank` slightly beats DenseLinear's val-selected macro (0.2087 vs 0.1855) but its oracle macro collapses to 0.2328 (vs 0.3773). Since no representation is trained, this **isolates the ceiling/calibration trade-off to the pairwise objective itself**, not representation damage. `dense_anchor_rank` is the worst (val-selected 0.1228). A second forensic audit (3 agents) confirmed no bug: the fast runtime is a single linear layer (~512× less compute than the SAE methods), and the numbers are internally self-consistent.
- **Final conclusion:** preference objectives help val-selected threshold transfer but do not beat dense hidden states in oracle separability. They reshape calibration, not the maximum recoverable correctness signal. DenseLinear BCE remains the strongest oracle method.
- **Next research direction:** move to model size / a stronger backbone (and/or explicit threshold calibration and score-distribution analysis) rather than continuing this objective family.

**Next.** Larger-backbone direction is now done (model-size scaling below). Remaining S2 follow-up: a held-out / PB-calibrated threshold rule to make the oracle gains deployable.

See `docs/sprint2_dataset_design.md` for the dataset + objective design note and exact commands.

### Model-size scaling (DenseLinear, `mpb_results.md` §3.13)

**Setup.** Run the §3.2 DenseLinear pipeline unchanged across Qwen2.5 {1.5B, 3B, 7B, 14B, 32B}, varying only the backbone (probe input dim inferred from `model.config.hidden_size`). Same frozen PRM800K 40K/1K split, ProcessBench 4 subsets, F1_PB, val-selected + 0.005 oracle, seed 42, float16 (on transformers 5.9 / torch 2.12). Hard constraint enforced: **no context truncation** (question + all previous steps + current step; per-model `max_position_embeddings`, 32768 for 3B being the strict case; `num_truncated_examples = 0` verified for every size). The 1.5B row reproduces §3.2 exactly (val 0.18554, oracle 0.37729). Code: `slurm/s1_model_size/run_sweep_one_job.sh`; artifacts: `runs/s1_model_size_dense/`.

**Results.** Oracle macro F1_PB scales monotonically: 1.5B 0.377 · 3B 0.382 · 7B 0.413 · 14B 0.449 · 32B **0.476**. val-selected macro is erratic and does not scale: 1.5B 0.186 · 3B 0.038 · 7B 0.237 · 14B 0.238 · 32B 0.061 (3B and 32B collapse at their PRM800K-val threshold t_val=0.40, hitting F1_PB=0.0 on olympiadbench).

**Verdict.** Scale raises the oracle ceiling (+0.10 macro 1.5B→32B: the final-layer representation holds more first-error signal as the backbone grows) but does **not** fix deployable calibration: PRM800K-val→ProcessBench threshold transfer stays unreliable and non-monotonic in scale (a larger model can score worse at the deployable threshold despite a higher ceiling). The bottleneck for a deployable step-checker is calibration/readout, not representation capacity. Motivates a PB-held-out threshold rule.

---

## S1 — Full ProcessBench evaluation (2026-05-21 → 05-26)

**Hypothesis.** SAE / SSAE latents carry a step-level correctness signal that beats (or matches) the raw final-layer hidden state for first-error localization on full ProcessBench.

**Tested.** Backbone Qwen2.5-1.5B, PRM800K supervision. Methods: DenseLinear, SAE-{positive,mixed,contrastive}, SSAE-{positive,mixed,contrastive}, + audit of the original-paper SSAE ckpt (Qwen2.5-0.5B). Eval: 4 ProcessBench subsets (gsm8k/math/olympiadbench/omnimath), macro F1_PB, val-selected + oracle (0.005-grid) thresholds.

**Results (oracle macro F1_PB).** DenseLinear 0.377 · SAE-mixed 0.369 · SAE-contrastive 0.335 · SAE-positive 0.315 · paper SSAE 0.5B 0.295 · in-house SSAE all ≈0.18 · random 0.167.

**Verdict.** ✗ No learned representation beats raw dense at oracle. In-house 1.5B SSAE collapses to ≈random; the paper's 0.5B SSAE ckpt beats them → suspect is SSAE *training*, not architecture. SSAE threshold transfer (val→PB) is broken (val-selected ≈0).

**Next.** Diagnose why in-house SSAE underperforms the 0.5B ckpt; consider contrastive/ranking objectives over PRM800K sibling forks.

---

<!-- TEMPLATE — copy above the previous entry

## S<N> — <name> (<start> → <end>)

**Hypothesis.** <one sentence>

**Tested.** <models / data / methods / eval metric>

**Results.** <key numbers>

**Verdict.** <✓/✗ + one-line reason>

**Next.** <one line>

Also add a row to the index table at the top.
-->
