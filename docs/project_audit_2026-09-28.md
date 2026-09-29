**Project audit, 28 September 2026**

**Snapshot note.** During this audit, a separate workspace edit added grader fixes, regraded pools, and REPORT.md §21.10. The main analysis below records the original pools, whose hashes and sensitivity estimates are preserved. I inspected the new artifacts before delivery: the regraded d512 N=2 lift is now **6.63 points Base versus 0.54 Instruct on GSM8K**, and **6.30 versus 1.67 on MATH**. Instruct pass@1 in `regrade_summary.json` is **93.53% GSM8K and 84.06% MATH**. These replace the original absolute figures for current reporting. The main conclusions below survive. The separate update does not repair the conflicting-label answer groups or numeric vote fragmentation: I rechecked both on the regraded pools. The new §21.10 also says tie-breaking remains at or above every weighted vote, although its own MATH N=10 table gives **90.44% weighted vote versus 89.84% tie-break**. That is a point-estimate contradiction, not proof of a significant difference. The targeted grader tests pass, **37 passed**, after that separate edit. I did not make those grader or REPORT.md changes.

Your strongest current result concerns when a verifier improves a decision. The evidence supports useful hidden-state verification, substantial benefits from reading multiple tokens, and a dependence of downstream gains on the policy, scorer, and aggregation rule. It does not support a universal preference for tie-breaking, a general absence of causal correctness representations, or priority over all small-budget verification work.

The immediate publication blockers are evaluation consistency, checkpoint and split provenance, and compute accounting. Several large effects survive the checks below. The project needs a smaller set of defensible claims and a reproducible evaluation release before it needs another architecture sweep.

**Scope and evidence**

I reviewed the research trajectory in REPORT.md, recent commits and uncommitted analysis changes, the Instruct plan and results, the representation harness, causal-study descriptions, scoring and aggregation implementations, local trajectories and score manifests, and primary literature. I ran the local non-slow test suite and independent CPU analyses of the saved Base and Instruct pools. This is a research and implementation audit of the main CoT-Checker project, not a line-by-line certification of every script or the independent Paper-Scrapper application. I did not rerun GPU experiments or inspect cluster-only tensors and checkpoints. In particular, the Instruct activation audit's compare.md and full three-seed retrain outputs were not available among the local artifacts inspected; their numerical summaries remain reported results rather than independently reproduced measurements.

The audit keeps three evidence levels separate: numbers recomputed from local trajectories; numbers read from saved result files; and numbers reported in the research log or plan. The distinction matters because the initial REPORT.md snapshot stopped short of several completed September 25-26 experiments. The concurrently added §21.10 now covers the Instruct results, while the older PRM and rule-selection conclusions still require reconciliation.

The local command `uv run python -m pytest -q -m 'not slow'` completed with **1,120 passed, 2 skipped**. The README's `uv run pytest` form failed collection on three causal-graph smoke tests because it could not import `scripts`. This is an invocation and packaging issue, not 1,120 failing scientific tests. Passing unit tests also does not validate answer normalization, training-set independence, or the interpretation of null results.

**1. The PRM result changes the central conclusion**

REPORT.md §21 recommends that “nobody should rerank,” and §21.8 argues that giving a verifier authority beyond ties loses accuracy. The newer files `results/tts_roster_v1/frontier_v4.json` and `rule_contrast_prm.json` contradict that generalization.

At N=10 on the saved Base pool:

| Selection rule | GSM8K accuracy | MATH-500 accuracy |
|---|---:|---:|
| Majority vote | 92.47% | 61.42% |
| On-policy d256 probe, template-state rerank | 89.23% | 55.02% |
| On-policy d256 probe, template-state tie-break | 92.95% | 61.48% |
| Qwen2.5-Math-PRM-7B tie-break | 93.33% | 63.09% |
| Qwen2.5-Math-PRM-7B rerank | 93.57% | 63.64% |
| Qwen2.5-Math-PRM-7B weighted vote | **94.47%** | **65.70%** |

The saved paired contrasts put weighted vote above PRM tie-breaking by **1.14 percentage points [0.45, 1.90]** on GSM8K and **2.61 [0.60, 4.62]** on MATH-500. These are pointwise bootstrap intervals on the existing evaluation, not multiplicity-adjusted confirmatory results. The MATH N=10 rows contain 498 questions because the current analysis excludes two questions with an unparseable sample.

**Regraded-pool follow-up:** I reran `tts_rule_contrast.py` on `cot-checker-results/tts_regraded_v1/base`, N=10, 32 draw orders, using the current grader normalization and saved regraded labels. PRM weighted vote still beats PRM tie-breaking: **+1.44 [0.68, 2.20] points on GSM8K** and **+2.41 [0.40, 4.42] on MATH**. Against majority, its gains are **+2.14 [1.35, 3.00]** and **+4.40 [2.26, 6.62]**. Thus the main rule-selection correction survives the concurrent regrading. These intervals still inherit the unresolved answer-group and evaluation-selection issues identified below. Results: `runs/project_audit_20260928/regraded_prm_contrast.json`.

The appropriate interpretation is that restricting the tested lightweight probe to ties protects the vote from that probe's mistakes. The stronger PRM can profit from changing the answer even when there is a unique plurality. The PRM result strengthens a study of verifier reliability and decision rules, while overturning a universal tie-break recommendation.

One smaller correction: the sequence 0.9247, 0.9257, 0.9143, 0.9083, 0.9007, 0.8939 in §21.8 is not monotonically decreasing. It has an initial increase. Also, failure to find a significant win over tie-breaking does not establish equivalence or optimality.

**2. The Instruct experiment exposes two distinct causes of diminishing returns**

The September 26 results in `docs/instruct_arm_v1_plan.md` report matched three-seed PRM800K AUROC of **0.8946 Base versus 0.9128 Instruct**, ProcessBench val-selected F1 of **0.469 versus 0.560**, and oracle F1 of **0.607 versus 0.655**. The saved downstream contrasts nevertheless show the d512 tie-break lift at N=2 falling from **6.66 to 0.37 points on GSM8K**, and from **6.08 to 1.67 points on MATH-500**.

That combination is a valuable result. Better off-policy detection does not guarantee larger decision gains. However, Base versus Instruct downstream evaluation changes the policy, prompt, generated trace distribution, and token cap as well as the verifier backbone. It identifies a difference between two deployment configurations, not a pure causal effect of instruction tuning.

I independently enumerated all 90 ordered pairs of the ten saved candidates per problem, removing the Monte Carlo error from the original 32 draw orders at N=2. I excluded questions whose answer groups contain contradictory saved labels. The analysis retains unparseable attempts in the sampling population and preserves the selector's rule that they cannot receive votes. It uses 4,000 paired question-bootstrap draws, seed 915. These are sensitivity checks on the existing data, not a replacement for corrected grading or a new untouched test set.

| Policy and scorer | Questions | Exact-pair lift over majority | Lift over mean-token confidence |
|---|---:|---:|---:|
| Base, d256 generation-state probe, GSM8K | 1,319 | +5.48 [5.05, 5.92] | +1.25 [0.83, 1.67] |
| Base, d256 generation-state probe, MATH, excluding historical fit/validation overlap | 473 | +5.42 [4.68, 6.19] | +3.02 [2.26, 3.82] |
| Instruct, d512 template-state probe, GSM8K | 1,319 | +0.43 [0.23, 0.64] | +0.04 [-0.20, 0.29] |
| Instruct, d512 template-state probe, MATH | 497 | +1.61 [1.09, 2.11] | +0.59 [0.06, 1.12] |

All entries are percentage points. The first two rows and last two rows use different probe architectures and training sets; compare each with its own baselines. For the architecture-matched d512 analysis, the fraction of N=2 oracle headroom recovered falls from **60.8% to 16.5% on GSM8K** and **49.8% to 29.3% on MATH**, using the consistent-label questions. Thus reduced headroom does not explain the entire loss: the Instruct verifier also recovers a smaller fraction of the available headroom.

The existing `conf` split gives the Instruct d512 probe **+0.59 [0.32, 0.86]** over majority on GSM8K and **+2.48 [1.76, 3.24]** on MATH. Against mean-token confidence the corresponding differences are **+0.01 [-0.31, 0.32]** and **+1.05 [0.21, 1.90]**. The MATH signal survives this split; the GSM8K advantage over a free signal does not. Calling these newly confirmatory would require evidence that prior development never consulted that split.

At N=2, with consistent answer grouping and parseable outputs, tie-breaking and reranking choose the same answer. Agreement makes selection irrelevant; disagreement produces a 1-1 tie. Consequently N=2 establishes pairwise ranking utility, but cannot establish the superiority of restricting a verifier to ties. Use N=3 and N=4 to test that restriction.

A useful exact decomposition is:

`N=2 lift = P(exactly one candidate is correct) × (P(verifier selects it | mixed pair) - 0.5)`.

This separates available opportunities from ranking skill. At larger N, decompose gains into tie frequency, whether a correct answer belongs to the tied bloc, conditional tie-selection accuracy, and the net gains or losses from overriding a unique plurality. This would turn the current descriptive curves into an explanation.

**3. Grading and voting disagree about answer identity**

The voting code groups candidates with `normalize_answer`, while grading adds numeric and symbolic equivalence. Equivalent answers can therefore split their votes. The local pools contain multiple distinct normalized strings among saved-correct answers for **32 Base GSM8K questions, 48 Instruct GSM8K questions, 10 Base MATH questions, and 6 Instruct MATH questions**. These counts describe grading/voting inconsistency; they do not certify every saved label as mathematically correct.

Examples include GSM8K question 24 with `26` and `26.00`, and MATH question 388 with `5.5` and `\frac{11}{2}`. In general, splitting a correct answer can weaken majority voting, create artificial ties, and distort the claimed source of a tie-break improvement.

There is a more direct invariant violation. For `math500_00255`, Base trajectory g1 predicts `E` and is labelled false, while g5 predicts `\text{E}` and is labelled true. Both normalize to `E`. The current grader reproduces the discrepancy against gold `\text{(E)}`. Instruct also has conflicting labels within normalized answer groups on questions 99 and 420, involving equivalent matrix formatting. `majority_expected` and `tie_break` take the correctness of the first candidate in a group, so candidate order can affect the measured correctness of an identical voted answer.

Fix answer equivalence before interpreting tenths of a percentage point. Give grading and voting one shared, gold-independent answer canonicalization policy, audit matrix and multiple-choice formats, and assert that every answer group within a problem has one correctness label. Do not group candidates using the gold answer. Regrade the complete saved pools and publish the number and types of changed labels.

The current absolute numeric tolerance also deserves attention for small answers: the Instruct pool accepts both `0.0000671` and `0.0000672` for question 176. Whether both should count requires inspecting the problem and its intended precision. It illustrates why a fixed absolute tolerance is not a universal mathematical equivalence rule. The concurrent grader fix also treats bare comma lists as unordered solution sets; that is a task-dependent interpretation and needs an explicit rule for ordered answers without brackets.

**4. Historical question overlap reaches the newer TTS pool**

I rejoined the archived GPT-OSS labels and reconstructed the historical split with seed 0 and validation fraction 0.15. It reproduces **4,657 training trajectories and 43,837 training steps**, the training size named for the d256 TTS scorer. Comparing whitespace-canonicalized question text with MATH-500 gives **23 training overlaps and 3 validation overlaps**, with 26 distinct affected questions. GSM8K has zero overlap with that archived pool.

The generation-state score manifests identify a checkpoint under `reprobe_v1/cells/step_tokens__transformer_d256_l2_f1024_h4__seed42`, and the Instruct plan identifies the §21 scorer as the historical 43,837-step model. This is strong evidence that the headline MATH result requires an unseen-question analysis. A checkpoint checksum and immutable training manifest should settle whether any same-path retrain replaced the historical model. A corrected splitting script alone does not retroactively clean an older checkpoint.

The effect survives the available sensitivity check: excluding those 26 questions and the one inconsistent-label question leaves **473 questions**, on which the d256 generation-state probe gains **5.42 points [4.68, 6.19]** over majority at N=2. The overlap is therefore a real reporting defect, not an explanation that makes this entire result disappear.

Do the same question-level cross-corpus audit for the PRM800K-trained d512 cells, their model-selection data, ProcessBench, and TTS sets. I did not establish that overlap from the locally available training manifests. Treat it as an unresolved requirement, not an allegation that all those evaluations leak. For public PRMs, distinguish known verifier-training exposure from unknown backbone pretraining contamination.

**5. The token frontier currently prices a hypothetical stopping implementation**

The sampler calls `model.generate`, then truncates the decoded Base output at a delimiter or answer line. It records both `n_gen_tokens_raw` and retained `n_gen_tokens`. The frontier charges the latter.

| Pool | Retained tokens charged | Raw generated tokens recorded | Raw / retained |
|---|---:|---:|---:|
| Base GSM8K | 1,614,955 | 2,689,985 | 1.67x |
| Base MATH | 1,041,360 | 1,761,121 | 1.69x |
| Instruct GSM8K | 4,029,150 | 4,042,355 | 1.003x |
| Instruct MATH | 4,425,359 | 4,430,255 | 1.001x |

Across the Base pool, the frontier excludes about **40.3% of recorded raw tokens**. Prefix causality makes an early-stop implementation plausible, but post-generation truncation did not save those tokens in these runs. Report retained completion length as an ideal-stop estimate, raw generation as the measured token count, and measured latency separately. Even raw counts are not a complete FLOP or batch-padding accounting.

The matched-compute script has another approximation: `reads × mean_input_tokens × parameter_ratio`. For lazy verification, the correct expected charge is the mean of the **sum of lengths of the candidates actually selected for scoring**. It is not generally the mean number of reads multiplied by the unconditional mean trace length. Hard questions can generate longer traces and more ties, producing covariance the current formula omits.

Also include generator prompt prefill, actual prompt caching assumptions, verifier prefix rereads, attention costs where material, and the head's measured overhead. Parameter-ratio FLOPs are a rough model; they are not an upper bound on real latency. Batched PRM prefill and autoregressive generation use hardware differently. Linear interpolation of average accuracy against average cost describes randomized mixtures of policies at an expected budget, not a demonstrated algorithm satisfying a hard per-question budget. Bootstrap the full matched-budget comparison, including the interpolation, before attaching a significance claim.

**6. Reconstructed generation states are a useful advance, with a deployment test still missing**

The d256 `__gen` scorer has a clear improvement in provenance: its script reconstructs the generation prompt and scores the full sequence once. Its manifests separate backbone and head timing. At Base N=2, the saved frontier gives generation-state accuracy **84.21% GSM8K and 50.45% MATH**, close to template-state **84.51% and 50.91%**. The independent pair enumeration also finds gains over token confidence.

This supports transfer from verifier-template training to the generation context. It does not yet demonstrate an integrated zero-reread serving path. The local tests check slicing and truncation, but do not compare reconstructed hidden states with states captured during cached decoding. Verify identical token IDs, special tokens, truncation boundaries, the final generated token's state, layer indexing, and scores under prefill versus cached decoding. Storing post-token states can require processing the last emitted token, even after a stopping decision.

The PRM800K-trained Instruct d512 TTS comparison still uses verifier-template states. Do not attach the d256 generation-state cost argument to it. First score or retrain the matched Instruct model on generation-context states and measure the actual head-only path.

**7. The Instruct artifact audit tests useful alternatives, but cannot certify content-only reasoning**

The audit measures norm concentration, attention destinations, probe occlusion, and length/position residualization. These are worthwhile controls. They support the statement that the tested artifacts do not explain the observed gain, conditional on the reported results.

The implementation of `residual_auroc` fits its score-on-covariate regression on the full evaluation split, then evaluates the residual there. Its covariate-only classifier uses random step folds instead of problem-grouped folds. The former is descriptive residualization, not an out-of-sample nuisance-removal test; the latter allows related steps into both fit and evaluation folds. Use grouped cross-fitting for both and bootstrap by problem.

A probe need not read a template token directly to use information from it. Contextual step states already contain information from preceding tokens. A lack of norm outliers or concentrated occlusion is also compatible with distributed formatting cues. Likewise, large-norm coordinates can carry useful content. Avoid equating concentration with artifact or its absence with semantic validity.

The stronger experiment is to preserve mathematical content while varying formatting, step boundaries, wrappers, or paraphrase. Measure prediction stability, especially on disagreement cases where the verifier changes the selected answer. Include nonlinear length controls and matched-length comparisons if the scientific claim extends beyond ruling out a linear length trend.

**8. Interpret the earlier program with narrower claims**

The dense-probe scaling result, reported AUROC **0.776 at 1.5B to 0.828 at 32B**, establishes increasing decodability in the tested setup. It does not establish explicit self-knowledge or a mechanism the generator uses. Model size also changes representational dimension, capability, and potentially the meaning of a fixed absolute layer index; report those alongside scaling.

The SAE negative result is useful as a matched practical comparison. It supports saying that the tested sparse representations did not improve verification over their dense baselines. It does not show that correctness is absent from sparse features, that reconstruction objectives cannot encode it, or that SAEs have no interpretive utility. Kantamneni et al. already found that strong non-SAE baselines remove much of the apparent advantage of sparse probing, including multi-token settings. Position your result as a reasoning-step-specific extension with matched controls. [Are Sparse Autoencoders Useful?](https://arxiv.org/abs/2502.16681)

The representation harness reports **0.394 to 0.485 ProcessBench F1** from last-token to step-statistics with a linear learner, and **0.500 to 0.522** from attention pooling to a transformer. That is strong evidence for the two contrasts. The statement that representation matters four times more than learner is not a general effect-size law: the former also increases feature dimension fivefold, and the latter compares two already capable learners on a different representation. A factorial analysis, matched parameter budgets, and repeated seeds can support a broader claim. Keep calib-20, source-val-selected, and oracle thresholds visibly separate. Calib-20 uses target-domain labels, so comparisons with uncalibrated published systems are not equal-adaptation-budget comparisons.

Low variance does not imply high intrinsic dimension. A linear probe is itself a one-dimensional score, even if its weight vector is dense in the model's coordinate basis. REPORT.md also reports that **235 of 3,584 coordinates** retain substantial accuracy. Distinguish sparsity in a chosen basis, sufficient predictive dimension, and the dimension required for a causal intervention. Failure of PCA or UMAP to reveal clusters establishes neither the absence of nonlinear structure nor the impossibility of a useful sparse basis. UMAP is not a variance-maximization method.

The causal work deserves more visibility and less overstatement. REPORT.md §18.1 reports **0.88 teacher-forced margin recovery** and **0.35 solve-gap recovery** from whole-step span interchange, while the learned subspace gives behavioral recovery **0.09, p=0.53, n=160**. Whole-span transfer has a behavioral effect in the tested setup. The learned edit lacks demonstrated behavioral benefit. These results do not prove that correctness is never causal or that no compact causal subspace exists. The learned subspace optimizes a teacher-forced margin, not solve probability, and the report itself acknowledges limited power. Null steering can also reflect location, timing, redundancy, or an ineffective intervention family.

Similarly, the WikiProfile study reports failed-prompt gold hits@1 **37.5% among 32 candidates**, same-fact patch rescue **44.9%**, and learned versus random steering exact match **8.7% versus 9.0%**. That supports accessible answer-related information and fact-specific transfer. It does not establish the nonexistence of a fact-independent access mechanism. First-token candidate ranking requires controls for token collisions, answer frequency, and full-answer likelihood. The proposed unembedding or learned decoder remains a sensible next test. Keep this side study separate from the main verification paper unless it predicts a CoT-checker failure mode that you then measure.

The transition-operator result is also narrower than some of its prose: within-fork correctness **0.655-0.670**, versus **0.671 for the raw post-state**, with operation retrieval near chance, supports a failed v0 objective relative to pooling. It does not exhaust effect-predictive representations. The roughly 14% skipped training batches in A/AB require reporting which examples were skipped; the zero-skip B arm helps, but it is not an identical clean rerun of A/AB.

**9. Literature positioning needs correction and expansion**

| Literature | Relationship to your results | Consequence for the paper |
|---|---|---|
| [ReProbe](https://arxiv.org/html/2511.06209v5) | Already uses small transformer probes over internal token features for step verification and downstream scaling. Its Table 3 reports GSM8K majority 97.6 and strongest listed probe/PRM 97.8. | Lightweight hidden-state step verification is established prior work. Your matched representation comparisons and decision analysis must carry the contribution. Do not treat a different policy's +0.2 as a universal ceiling. |
| [HSRM](https://arxiv.org/html/2608.30841v1) | Encodes sequences of step-boundary states, trains a within-problem candidate-ranking objective, and includes representation/capacity ablations. | The steps-versus-tokens distinction is real, but architecture size and use of hidden states are not sufficient novelty. Compare supervision and selection objectives as well as inputs. |
| [Lessons of Developing PRMs](https://arxiv.org/html/2501.07301v2) | Distinguishes process correctness from outcome success, documents biased BoN evaluation, and studies different score aggregations. | It anticipates part of the benchmark-to-utility mismatch. Your contribution can quantify the mismatch under matched representations and policies. Keep process and outcome metrics together. |
| [Deep Think with Confidence](https://arxiv.org/html/2508.15260v1) | Supplies free confidence signals, weighted voting, and filtering. | The 32-token window is an adaptation of its much longer windows. Call it a DeepConf-derived short-trace baseline and record its tuning split. Include the published-window variants. |
| [Let's Verify Step by Step](https://arxiv.org/html/2305.20050v1) | Reports no noticeable benefit from adding RM-weighted voting in that setup, while PRM selection beats majority. | It does not support a blanket claim that reranking or weighting never works. Your new PRM weighted-vote result is a useful contrasting regime. |
| [Entropy-Gated Branching](https://aclanthology.org/2026.eacl-long.235.pdf) | Figure 3 evaluates budgets 2, 4, 8, 16, 32, with self-consistency and verifier-assisted search. For self-consistency, budget means sampled solutions; for search it means width times expansions. | The broad claim that the literature never studies budgets 2-4 is false. A narrower claim about your exact paired cheap-probe/tie-break protocol may remain, subject to a more specific search. |
| [RISC](https://arxiv.org/html/2606.05054v1) | Learns answer selection from frequency, semantic centrality, and trace-consistency features. | A learned combination of vote and verifier evidence is an existing direction. Include a simple calibrated fusion baseline before declaring a hand-designed rule optimal. |
| [Compute-optimal test-time scaling](https://arxiv.org/abs/2408.03314) and [LATTS](https://arxiv.org/abs/2509.20368) | Study difficulty-dependent compute allocation and local verification actions. | Online rejection belongs in this lineage. The contribution must be its measured efficiency or its controlled diagnosis. |
| [Inference-time alignment with imperfect rewards](https://arxiv.org/abs/2503.21878) | Explains how large-N selection can exploit reward-model errors under stated assumptions. | Your probe's degrading rerank curve is compatible with this account. A curve alone does not prove reward hacking; inspect the selected false positives. |

The original SSAE paper's current title is [Step-Level Sparse Autoencoder for Reasoning Process Interpretation](https://arxiv.org/abs/2603.03031). Update the README's title and pin the version you reproduced. A local commit date can document your implementation chronology, but without establishing its public availability it should not carry a priority claim. HSRM's August 31 submission supports describing nearby work as concurrent with your August implementation, with appropriate qualification.

The broad “small-N regime is empty” claim should be removed. It is unnecessary for a strong paper. The more useful question is whether a controlled evaluation can predict when a weak verifier should defer to consensus and when a stronger one can safely override it.

**10. Statistical and software gaps to close**

The project makes good use of paired trajectories, question bootstrap, frozen input pools, and saved per-step scores. Preserve those choices. The next corrections are specific:

1. Make confirmatory comparisons use a locked set of scorers, thresholds, aggregators, and budgets. `tts_rule_contrast.py` currently has no split filter and pools all questions, despite the exploratory/confirmatory split in the trajectories. Repeatedly inspecting 93 rules and many budgets requires an exploratory label or family-level uncertainty control. An existing `conf` field is not proof that its results remained untouched.
2. Enumerate subsets for small pools where practical. All N=2 unordered pairs number only 45; N=4 subsets number 210. The original 32-order simulation adds avoidable noise precisely where some claimed effects are small. Exact enumeration still samples only ten generated candidates per question; it does not replace an independent generation pool.
3. Keep all attempts and their costs. `tts_build_frontier.py` drops `gradeable=False` candidates before drawing. As a result MATH N=10 evaluates 498 questions while smaller N evaluates 500. Count unparseable outputs as failed attempts and keep denominators fixed.
4. Validate score completeness, finite ranges, unique trajectory IDs, matching model/checkpoint hashes, and shard coverage before creating a curve. Missing scores becoming negative infinity is an explicit failure policy, not a neutral repair. The inspected main scorers cover all gradeable trajectories; the concern is that the generic pipeline does not enforce this invariant.
5. Report per-rule scoring cost. The frontier stores lazy tied-bloc `scored_mean` even on rerank and weighted-vote rows. The matched-compute script repairs this for its three named rules, but a generic consumer can silently use the wrong cost. Also, the documented beta=0 equivalence to majority fails with missing qualities: a reproduced two-answer example returns 0.0 instead of majority's 0.5.
6. Use a disjoint calibration set and a complete plain baseline for online rejection. The reported strongest comparison uses the 120 calibration problems, while rejection has 300. The q=0.65 result follows a threshold sweep on those outcomes. Its **+11.7 [6.7, 17.5]** points against blind retries is promising exploratory evidence, not a validated compute-optimal operating policy. A nonsignificant blind-control lift does not establish that blind retries have zero benefit.
7. Distinguish process-error F1, step AUROC, within-problem candidate AUROC, and final-answer accuracy. They answer different questions. AUROC is prevalence-invariant at fixed class-conditional score distributions, not domain-invariant, and global step AUROC is not the quantity a selector directly optimizes. Report tail ranking on disagreement cases and calibration for weighted voting.
8. Preserve model and artifact identity. Cell directory names omit training dataset and prompt context. A frontier currently records time, N, and draw count, but not a complete source-hash chain. Record source trajectory hashes, scoring manifests, checkpoint hash, training split hash, label source, rescaling, prompt, dependency versions, code revision, and dirty-tree status.
9. Declare direct dependencies. Core evaluation imports scikit-learn and analysis uses SciPy, but pyproject.toml does not declare them. The local environment passing tests does not establish that a clean documented install reproduces the analyses. Pin the model-loading environment that the PRM requires and test the README invocation.

**11. Recommended research sequence**

**First, repair and freeze the evaluation without spending GPU time.** Unify answer equivalence, regrade all saved outputs, retain failed parses, reconstruct train/selection/evaluation question identities for every scorer, and regenerate split-specific paired comparisons. Report measured raw-token and ideal-stop curves separately. Consolidate September 26 results into the authoritative research record and mark earlier conclusions as superseded. Keep a single manifest-backed table of which checkpoint produced each headline number.

**Second, explain the rule crossover on the existing pool.** For each scorer and N, measure rescues and harms on unique-plurality cases, tie-selection accuracy, length dependence, and extreme-score false positives. Compare min, mean, last, and a validation-fitted length-adjusted aggregation. A worst-step score can penalize a long correct trace simply because it offers more opportunities for a false positive. Test whether removing the final-answer step changes ranking: otherwise a nominal process verifier may function mainly as an answer detector. Fit at most one small vote-plus-score combiner on the exploratory split and compare it on held-out questions, with a confidence-only counterpart.

**Third, perform one clean deployment replication.** Use a checksum-identified probe trained on generation-context states, compare it with the same confidence baselines and a PRM, and measure capture/head latency under the actual sampler. For online rejection, select the threshold on separate questions and run plain, blind, and learned-rejection arms on the same held-out problems with repeated seeds and matched measured compute. Use a harder task or a second model family if the intended claim extends beyond these two Qwen policy configurations.

**Defer further SAE objectives and broad steering searches.** Neither addresses the current blockers. If mechanistic interpretation remains the main scientific goal, choose one failure-mode-specific intervention with adequate power and a behavioral objective. Otherwise, keep the causal studies as carefully scoped negative and positive controls supporting the verifier-evaluation story.

A defensible paper could ask: **when should a hidden-state verifier overrule consensus?** Its central evidence would be the matched representation study, the loss of predictive value when moving from off-policy detection to actual decisions, the Base/Instruct headroom decomposition, and the weak-probe/strong-PRM rule crossover. That is a coherent contribution after evaluation repair. “A new lightweight verifier,” “SAEs cannot encode correctness,” and “verification only helps at N=2” are not supported central claims.

**Audit artifacts**

All generated audit evidence lives in `runs/project_audit_20260928/`:

- `evidence.json`: initial pool inventory, fragmentation examples, and first exact-pair diagnostic. Its all-question MATH calculations precede exclusion of conflicting-label groups; use the next file for reported sensitivity estimates.
- `exact_n2_consistent.json`: exact ordered-pair estimates, split-specific paired intervals, and per-question aggregates after excluding contradictory answer groups. Contains separate historical-overlap exclusions.
- `historical_tts_overlap.json`: reconstructed historical split counts and all 26 affected MATH-500 IDs.
- `cost_accounting.json`: retained versus recorded raw token totals.
- `source_sha256.json`: hashes of the trajectory, score, and confidence files used for exact-pair analyses.
- `regraded_prm_contrast.json`: follow-up N=10 paired comparisons on the separately regraded Base pool.

I made no edits to the existing code, report, experiments, or baseline artifacts. Separate concurrent edits changed the grader and REPORT.md during the audit, as recorded above. I launched no GPU jobs and made no commits. No plots were generated by this audit.
