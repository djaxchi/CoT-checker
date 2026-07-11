# Sprint Log

One row per sprint: what we hypothesized, what we tested, what we got. Keep it terse. Newest on top.

| Sprint | Dates | Hypothesis | Verdict |
|---|---|---|---|
| S5 | 2026-07-03 → 07-10 | **Knowledge boundary.** Does Qwen2.5-7B-Instruct internally represent, before it answers, whether it knows a fact, and is that signal causal? New dataset (WikiProfile), 4 behavioral retrieval classes (direct_retrieval / reasoning_unlocked / unstable / non_retrieved). | retrieved-vs-non is linearly readable pre-answer (~0.79 AUROC, 0.85 answer-side, vs 0.70 confound baseline) and carried by ~10 sparse SAE "answer-commitment" features (one at 0.76 alone, not popularity/topic). But the 4-way split does not separate and reasoning_unlocked is geometrically invisible (bal-acc ≤0.41). Causally: single-feature steering is NULL (gauge = matched random control, both necessity and sufficiency); the recall signal is absent at the prompt and builds through reasoning (within-instance 0.49→0.64); activation patching finds a small, deep-layer, fact-specific transplantable core (matched 17-19% vs mismatched 7-9%, verified real knowledge flips) but far below in-context CoT (84%). **Net: recall is mostly dynamic; the prompt-time signal is a gauge, not a lever.** |
| S4 | 2026-06-27 → 07-02 | Step-representation geometry in PRM800K CoT: does the per-step *contribution* vector (h_i − h_{i-1}) cluster into interpretable reasoning-move types, and do correct/incorrect steps diverge along the trajectory? | descriptive / infrastructure: built 3 step reprs (state / question-residual / contribution), tag-enrichment clustering, and an interactive explorer (fork pairs, click-to-trace trajectories, dynamics + identity). No clean discrete step-type taxonomy (consistent with the S3 stage-1 null); the geometry + explorer tooling was reused in S5. |
| S3 | 2026-06-16 → (open) | **Pivot:** stop optimizing the ProcessBench score; explain it. The ~41% hidden-state signal is a *mixture of failure-mode-specific signals*, not one unified correctness concept. Decompose via failure-type annotation, controlled contrasts, layer-wise probing, geometry, transfer, and causal intervention. | in progress |
| S2 | 2026-06-05 → 06-09 | Fork-based preference supervision (ranking / triplet) shapes a better latent than reconstruction-only; + DenseLinear model-size scaling (1.5B–32B) | ~ partial: forks improve transfer not oracle; scale raises oracle ceiling (0.476 @ 32B) but not deployable calibration (closed) |
| S1 | 2026-05-21 → 05-26 | SAE/SSAE latents localize the first wrong CoT step better than raw dense hidden states | ✗ not supported |

---

## S5 — Knowledge boundary: does the model know that it knows, and is it causal? (2026-07-03 → 07-10)

**Question.** A new line, distinct from the S1-S4 ProcessBench/CoT-correctness thread. Following
"Empty Shelves or Lost Keys? Recall Is the Bottleneck for Parametric Factuality," split factual
failure into *missing knowledge* vs *knowledge present but not accessed*, and ask (1) whether these
regimes correspond to distinguishable internal states before the model answers, and (2) whether any
internal "I know this" signal is a *cause* of retrieval or merely a *readout*.

**Setup.** WikiProfile (subject-object facts grounded in Wikipedia), 800 facts × 4 closed-book
question forms = 3,200 instances. Model Qwen2.5-7B-Instruct on 4× H100. Four behavioral classes
assigned from the model's own graded behavior (direct greedy, 4 sampled direct, CoT greedy, 4
sampled CoT): non_retrieved 1,647 (51%), direct_retrieval 893 (28%), unstable 409 (13%),
reasoning_unlocked 251 (8%). Hidden states read at layers 20/24, mainly the final prompt token.

**Findings.**
1. **Static geometry.** Retrieved-vs-non is linearly readable *before generation*: ~0.79 AUROC at
   the final prompt token (0.85 answer-side), vs a confound baseline (popularity, length, family) of
   0.70. The cot-minus-direct prompt-state delta reaches 0.81. But the 4-way split is not separable
   (bal-acc ≤ 0.41, chance 0.25); classes lie on one graded confidence axis and reasoning_unlocked is
   geometrically invisible, sitting in the overlap.
2. **Interpretable feature (public SAE, ~131k latents).** The signal survives sparsification: full
   latents 0.82 (L24 prompt) / 0.86 (L20 answer); an L1 probe with ~10 features reaches 0.78-0.81; a
   single feature (58264) reaches 0.76 alone. Not popularity/topic (within-bin 0.749, within-category
   0.730, ρ_gbc 0.13). Token-wise it peaks at the *answer-commitment* position (the "Answer:" token in
   2,913/3,200), not the subject entity: an "am I ready to answer" gauge.
3. **Causal test 1 — single-feature steering (necessity + sufficiency vs matched random).** Clamp the
   feature down on known facts: breaks 2.0% (58264) / 8.7% (88965) of correct answers, but a
   matched-norm random edit breaks 3.3% / 6.7% — never more than random. Boost it on
   reasoning_unlocked (fact accessible): rescues ≤ 4.0%, never beats random. **Gauge, not lever.**
4. **Causal test 2 — within-instance rollout dynamics.** On 524 mixed-outcome questions (~4,200
   rollouts, 44% success), predict which rollout succeeds from the *instance-demeaned* state (fact
   erased, only path remains): AUROC rises from 0.495 at the prompt and 0.489 at gen-0 (chance),
   through 0.531 / 0.587 mid-reasoning, to 0.606 pre-answer and 0.640 at answer onset. **The recall
   signal is absent at the prompt and built during reasoning**, not set by lucky early sampling.
5. **Causal test 3 — activation patching (strong test).** Transplant each reasoning_unlocked
   question's own post-CoT answer-onset residual into its direct (no-reasoning) decision point. Direct
   baseline 0.7% → matched 16.7-19.3% (~25×), but a mismatched-donor control also lifts to 7.3-8.7%
   with near-zero answer-copying (0.7-4.0%), so ~half is a non-specific "commit to answering" effect.
   The fact-specific component (matched − mismatched) is +9 to +11 pts at block 23 (z≈2.5, p≈0.01),
   ~0 at block 19 α=0.5. Hand-verified as real knowledge flips (Vigenère cipher "de Vigenère" →
   correct "Bellaso"; Cynthia Cooper "Enron" → "WorldCom"; Siskel&Ebert "Ebert" → "Siskel"). But
   ≪ in-context CoT ceiling (84%).

**Net.** The model carries a real, distributed pre-answer retrieval-access signal, concentrated at
the answer-commitment token and captured by a few SAE features, but it is a *gauge, not a lever*.
The reasoning_unlocked regime is not a property of the prompt state at all; recall is computed
*through* reasoning (dynamic), with only a small, deep-layer, fact-specific fraction that is
transplantable after the fact.

**Caveats.** One model, 1-2 layers, single patch position; transplant recovers a minority; labels are
budget-dependent (4 samples / 256 CoT tokens); grading is deterministic string-match (a few percent of
transplant "successes" are containment-rule fragments).

**Artifacts.** `runs/parametric_retrieval_geometry_v0/` (geometry, sae/, rollouts/, explorer_payload/,
figures under sae/plots/); ~15 scripts in `scripts/parametric_retrieval/`; 7 TamIA jobs in
`scripts/tamia/jobs/prg/`; manifest-driven `explorer.html`.

---

## S4 — Step-representation geometry (contribution clusters) in PRM800K CoT (2026-06-27 → 07-02)

**Question.** Extends S3's geometry lens to the *trajectory*: represent each reasoning step by its
*contribution* vector (h_i − h_{i-1}, the closed form of the corrected residual-stream recursion;
distinct from a naive diff), alongside state and question-residual reprs, and ask whether steps
cluster into interpretable reasoning-move types and whether correct vs incorrect steps diverge as the
chain unfolds.

**What was built / found.** Three step reprs (state / qres / contribution) at L20/L28; tag-enrichment
clustering; an interactive explorer with joint UMAP, correct-vs-incorrect fork pairs, click-to-trace
trajectory paths, and cross-chain displacement/identity metrics. Result is descriptive: no clean
discrete step-type taxonomy emerges (consistent with the S3 stage-1 null on failure-mode
clustering); the value was the reusable geometry + explorer tooling, which S5 adopted for its own
retrieval-regime explorer.

**Artifacts.** `runs/contrib_cluster/` (reprs, clusters, tags, trajectories, `explorer.html`);
`scripts/s4_contrib_*.py`, `scripts/analysis/s4_*.py`.

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

### Stage 2 findings: layer × token ablation and the geometry of the signal (2026-06-19)

Pre-paid the Stage-1 caveat: one 4-GPU pass encoded the natural PRM800K held-out test (6,000 steps,
balanced 3,000 correct / 3,000 incorrect by rating ±1) capturing **first and last token at layers
11, 17, 20, 22, 25, 28** of Qwen2.5-7B (4D tensor 6,000×6×2×3,584). Decodability = 5-fold balanced
logistic regression, floor 0.50. Headline: sweeping every layer and both tokens **does not surface
correctness clusters**; it pins down *where* the signal sits and *why* it never appears as separation.

1. **Last token >> first, at every depth.** Last-token decodability peaks at **L20 = 0.742** (0.71
   depth) and plateaus ~0.73 to the final layer; first-token is flat ~0.68 across all layers. Only the
   last token has attended over the whole step. Actionable: the readout should be last-token (or
   mean-pool), never first. This is a real fix, not a tuning knob.

2. **L20 slightly beats the final layer.** L20/last (0.742) > L28/last (0.731) on accuracy and AUC and
   after every control; mid-late layers hold the most abstract step-validity before the final layer
   re-specializes toward next-token prediction.

3. **The signal is real, mostly not a confound.** Incorrect steps are longer (median 246 vs 193
   tokens) and later (step_idx 5.2 vs 4.4), but length+position alone decode only 0.583; after
   regressing them out of every dim, decodability is still 0.711 (~87% of the lift survives). It is
   carried by **direction, not magnitude**: row-normalizing (cosine) raises it to 0.766 and the raw
   norm is near-useless (AUC 0.575).

4. **Why no clusters: a 0.01%-variance linear margin.** The correctness direction holds **0.4 of 3,584
   total variance (0.01%)** with class means only **d'=1.19 SD** apart (AUC 0.808). The top PC carries
   none of it (0.517); removing the top-1 PC leaves 0.741; you must strip ~100-200 PCs to collapse it.
   UMAP/HDBSCAN spend their dimensions on the 99.99% of variance that is topic/problem/length and never
   surface it. A supervised projection onto the probe axis shows the shift plainly; an unsupervised map
   cannot. There is no contradiction between strong linear decodability and an unstructured 2D map.

5. **The minimal separating set exists but still does not cluster.** An L1 sparse probe keeps
   **235/3,584 dims (6.6%)** at full accuracy (0.747; generalizes to 0.733 train→test); ~93% of
   activations are statistically common to both classes and droppable. Yet mapping only the
   discriminative 235 dims gives 2D-separability **0.660 vs a shuffled-label null of 0.654** and
   HDBSCAN purity 0.61 vs 0.56. Correctness is a distributed linear direction, not a localizable
   cluster of "error neurons."

**Conclusion.** Stage 1's "incorrectness is diffuse" now holds across all layers and both tokens, with
a mechanism: a genuine but very low-variance, low-effect-size linear direction, linearly decodable yet
invisible to any variance- or neighborhood-based embedding. The natural balanced ceiling is ~0.74
(optimistic: rating ±1 extremes vs ProcessBench's natural mix). This is the strongest motivation yet
for the deferred SAE stage. Only a learned sparse basis can give this direction its own coordinate
and make it discrete, interpretable, and clusterable.

**Representation analysis & figures (REPORT §15).** The geometry is the payload of this stage; the
plots make the "decodable-but-invisible" paradox concrete. Each is in `results/prm800k_layers/`
(gitignored), cross-referenced to the REPORT subsection that reads it.

- `layer_decodability.png` (§15.2) — last vs first token decodability across L11–L28. Shows last-token
  peak at **L20 = 0.742** plateauing to the final layer, first-token flat ~0.68. This is *why* the
  readout switched to last-token (never first) and to L20 over L28.
- `probe_anatomy/anatomy_L20_last.png` (+ L20_first, L28_last, L28_first) (§15.3–15.4) — the confound
  audit and variance breakdown. Visualizes that the margin is **direction not magnitude** (cosine 0.766
  > full 0.742 > raw-norm 0.556), survives length+position residualization (0.711, ~87% of lift is
  content), and that the probe direction holds only **0.4/3,584 variance (0.01%)** with class means
  **d'=1.19** apart — heavy-overlap shift, not two blobs.
- `minimal_subspace/minimal_subspace.png` (§15.5) — the L1 sparse probe keeps **235/3,584 dims (6.6%)**
  at full accuracy (0.747, generalizes 0.733 train→test), yet those discriminative dims map in 2D at
  0.660 vs a shuffled null of 0.654. The "minimal separating set exists but still won't cluster" panel:
  a distributed linear direction, not localizable "error neurons."
- `supervised_view_L20_last.png` (§15.4) — the resolution of the paradox: a *supervised* projection
  onto the probe axis shows the class shift plainly, while UMAP/PCA spend both axes on the 99.99% of
  variance (topic/problem/length) orthogonal to correctness and show nothing.

**Interpretation (why this matters for the program).** Correctness at this readout is a genuine but
low-variance (0.01%), low-effect (d'≈1.2) **linear direction** — linearly decodable to ~0.74 yet
invisible to every variance- or neighborhood-based embedding, for *any* (layer, token). That is the
sharpest motivation yet for the deferred SAE stage: only a learned sparse basis can allocate a
dedicated coordinate to a 0.01%-variance direction and turn this margin into discrete, interpretable,
clusterable features. Two readout fixes are now locked in for all downstream S3 work: last-token (or
mean-pool) over first, and L20 over the final layer. Caveats: residualization removes only *linear*
confound, and the 6k held-out set is the artificial rating ±1 balanced extremes (optimistic ceiling).

Artifacts: `scripts/analysis/s3_prm800k_layer_projection.py`, `s3_prm800k_probe_anatomy.py`,
`s3_prm800k_minimal_subspace.py`, `s3_prm800k_supervised_view.py`; encode
`scripts/encode_prm800k_multitoken_multilayer.py` (+ merge); outputs under `results/prm800k_layers/`
(gitignored). Full writeup: `REPORT.md` §15.

### Stage 3 findings: model-size scaling on the full natural PRM800K test (2026-06-19)

Took the deployed dense readout (last layer L28 / last token) to the **full OpenAI PRM800K test set**
(`phase2_test`, 24,244 candidate steps over 2,724 problems, problem-disjoint from train/val, natural
**25.1% incorrect** = 18,169 correct / 6,075 incorrect) so the numbers are literature-comparable rather
than on our balanced 6k. Encoded the set with each backbone (1.5B, 3B, 7B, 14B, 32B) in one 4-GPU node
and evaluated each size's own probe in its own hidden space at its val-selected threshold.

| size | AUC | F1 (inc) | bal-acc | oracle bal-acc | val→test gap |
|---|---|---|---|---|---|
| 1.5B | 0.776 | 0.543 | 0.703 | 0.706 | -0.018 |
| 3B | 0.795 | 0.555 | 0.720 | 0.724 | -0.024 |
| 7B | 0.806 | 0.579 | 0.730 | 0.734 | -0.036 |
| 14B | 0.825 | 0.589 | 0.747 | 0.748 | -0.016 |
| 32B | 0.828 | 0.588 | 0.749 | 0.753 | -0.019 |

1. **Monotonic scaling, diminishing past 14B.** AUC rises 0.776 → 0.828 and balanced accuracy 0.703 →
   0.749 across a 20x param range. The largest single step is 7B → 14B (AUC +0.019, bal-acc +0.017);
   14B → 32B is nearly flat (AUC +0.003, F1 ties at ~0.589). 14B captures essentially all the available
   linear correctness signal.

2. **Thresholds transfer cleanly.** Deployed-threshold balanced accuracy is within ~0.003-0.005 of the
   oracle ceiling at every size, and val→test gaps are small and negative (-0.016 to -0.036). The
   val-selected thresholds hold on the full natural distribution with no overfit on unseen problems.

3. **Low F1 is a base-rate artifact, not weak discrimination.** F1 on the incorrect class is 0.54-0.59
   only because incorrect steps are the 25% minority. Backing out the 7B operating point: recall ≈ 0.70,
   specificity ≈ 0.76, but precision ≈ 0.50 because correct steps outnumber incorrect 3:1, so a 0.76
   specificity still leaks roughly as many false positives as there are true positives. The same probe
   at the same threshold scores F1 0.720 on the balanced 6k set. AUC and balanced accuracy are
   prevalence-invariant; F1 is not, so cross-paper comparison must lead with AUROC (~0.83 at scale) and
   quote F1 only alongside the 25% base rate.

**Conclusion.** As a step-correctness classifier on the full natural PRM800K test the dense probe reaches
**AUROC ~0.83 / balanced accuracy ~0.75 at 14B-32B**, scaling smoothly with backbone size and
generalizing cleanly to unseen problems. This is the literature-comparable headline; it reinforces that
the ProcessBench gap is distribution shift, not a broken probe. The L20 best-correctness-layer eval
(Stage 2: L20/last > L28/last) is still open here because it needs an L20-trained probe per size.

Artifacts: `scripts/eval_prm800k_heldout_probe.py`, `scripts/aggregate_heldout_eval.py`; build
`scripts/build_prm800k_heldout_test.py --full`; slurm `encode_prm800k_heldout_allsizes_tamia.sh` (+
CPU-only re-eval `eval_prm800k_fulltest_allsizes_tamia.sh`); table `results/prm800k_test_full_eval/table.csv`.

### Stage 4 findings: what the L28 signal actually encodes — confound elimination (2026-06-25)

Stage 2 located the margin (a 0.01%-variance linear direction); Stage 4 asks *what it is*. Eliminated
candidate explanations in order, each with an A/B control, on matched PRM800K forks (1000/size,
anchor/pos/neg sharing a prefix) and then on the model's own generations.

1. **Matched-fork audit.** The supervised margin `w·(h_neg−h_pos)` is real, non-surface, and **scales**:
   P(neg>pos) 0.686→0.782 (1.5B→32B), survives length+lexical residualization (7B 0.749→0.776) and the
   surface-matched minimal-edit subset (0.723); surface explains only ~13–20% of margin variance, almost
   all `length_diff`. Falsified the stronger claim that matched differencing *unsupervisedly* isolates
   the direction (cos(μ_Δ, w)≈0.07). **L20 multilayer re-encode = NO-GO** (signal already healthy +
   surface-robust at L28; geometry never isolates correctness at any layer).

2. **Per-step drivers.** Pooled score AUC (0.665→0.736) is **not** length (removing it nudges AUC up),
   numeric, position, or answer-presence (removing all four drops 7B AUC only 0.699→0.694).

3. **Confidence/perplexity battery — GATE A (PASSED).** Surprise is ruled out harder than length: each
   of {nll_mean, nll_max, entropy, logit_gap} alone predicts the label at AUC ~0.47–0.56 vs probe 0.699;
   removing all four moves probe AUC 0.699→0.697 (fraction of lift removed 0.01–0.02 at every size). With
   scale, raw surprise *drops* (nll_pos 0.860→0.672) while probe AUC *rises* — opposite trends, not the
   same signal.

4. **On-policy control — GATE B (PASSED).** The forks use human-written negatives, so the probe could be
   reading "off-distribution human text." Decisive test: the model's **own** generations (uniformly
   low-perplexity). 7B, 1200 generated trajectories (53.2% incorrect by final-answer match), 11679 steps.
   On-policy step NLL 0.617 vs fork-negative 0.804 (drop +0.187 nats), yet the probe still separates:
   **trajectory AUC 0.720 (95% CI 0.691–0.748)**, step AUC 0.615.

**Honest F1 caveat (headline metric is F1 at a threshold).** On this on-policy set F1 is unflattering by
construction: 53% incorrect makes the trivial predict-all-incorrect F1 = 0.694 (traj) / 0.747 (step).
Trajectory **oracle** F1 0.728 clears trivial, but the deployable **val-selected** F1 0.686 dips just
under it, and step F1 = trivial exactly. A 0.72-AUC signal on a near-balanced positive-heavy set cannot
push max-F1 past the all-positive corner — F1's prevalence-dependence works against the probe here. On
the natural 25%-incorrect PRM800K test the same probe scores F1 0.58 vs a 0.40 trivial bar (clear win),
so the deployable F1/reliability headline belongs on the natural distribution, not this balanced set.

**Verdict.** After ruling out length, numeric, position, answer-presence, perplexity/surprise (GATE A)
**and** off-distribution-ness (GATE B), the residual is **correctness** — genuine, distributed,
low-variance, scaling. Next: name the direction `w` (logit-lens through `W_U`, per-token DLA, optional
steering) then the deferred SAE stage; compute the deployable F1/reliability statement on the natural
held-out test. Writeup REPORT §15.7; artifacts `runs/fork_rep_audit/<tag>/` (gitignored); stage plan
`~/.claude/plans/piped-wishing-platypus.md`.

### Stage 5 findings: is `w` causal? Additive steering during generation (2026-06-26)

Tested the plan's causal question on 7B: add `+/- alpha * s_layer * w_hat` to the residual at the
L20 and L28 decoder blocks DURING the model's own generation, grade the final answer, vs matched-norm
controls. `w_hat` per-layer (L20 trained in-space, L28 deployed), oriented toward correct so
`alpha>0` should repair / `alpha<0` corrupt. `alpha` is a fraction of median residual norm
`s_layer~100`. Decode-only steering (prompt untouched), paired to the alpha=0 baseline.

1. **Scale cliff.** `alpha>=1` (vector >= residual norm) destroys generation for every direction:
   P(correct)->0, gradeable collapses, probe logit hits +/-835. Usable regime is small fractions.
2. **Readout moves (Tier-0).** Teacher-forced fork margin rises with toward-correct steering (probe
   0.074->0.441 at alpha+2); the probe logit shifts as expected. The direction is movable.
3. **Behavior does NOT move (Tier-1).** n=400/cell (100 problems x4, paired), gradeable=1.000,
   baseline P(correct)=0.375, dP SE~0.03. No antisymmetric repair/corrupt: the faint trend is
   BACKWARDS (toward-incorrect mildly helps) and <1.5 SE. **probe ≈ random** at every matched alpha
   (dP within 0.04); at +0.2 probe corrupts MORE (0.267 vs 0.213) and repairs LESS (0.084 vs 0.116)
   than a random push. Only real effect is direction-agnostic degradation with `|alpha|`.

**Verdict.** ✗ `w` is a **diagnostic readout, not a causal lever** (additive steering, L20/L28, 7B):
the readout shifts but on-policy correctness does not, and `w` is no better than a random direction.
Exactly the disambiguation the plan set up. Caveat: falsifies *additive* steering only, NOT activation
patching (the stronger test; could still flip it). Next: either activation-patching confirmation, or
pivot to naming `w` (logit-lens/`W_U`, per-token DLA) + SAE stage. Writeup REPORT §15.8; code
`scripts/{build_steering_directions,s1ms_steer_generate,analyze_steer_causality}.py`,
`s1ms_steer_forks.py --directions_npz`, `slurm/s1_model_size/run_steer_causality_7b.sh`; artifacts
`runs/fork_rep_audit/qwen2_5_7b/steer_causality/` (gitignored).

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
