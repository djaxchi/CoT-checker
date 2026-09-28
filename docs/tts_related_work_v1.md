**Literature review and research implications, 28 September 2026**

Qwen3-8B Instruct, non-thinking, is now the primary policy. The most defensible research question is: **which information in a frozen model's reasoning states improves verification, and under which decision and compute constraints does that information improve answers?** The current results support a connection between representation, supervision and downstream selection. They do not establish that a particular aggregation rule is universally best or that a correctness representation has no causal role.

This review supersedes the September 22 positioning in this file. It covers primary papers available through September 28, including August work. It is a focused critical review, not a systematic census. Numerical comparisons below preserve the source's task and protocol; same model size alone does not establish comparability. REPORT.md records experiments; `docs/project_context.md` summarizes the current evidence and `docs/tts_reading_list_v1.md` gives a reading order.

**1. Distinguish the scientific targets before comparing methods**

A step classifier, a trajectory ranker and a search controller predict different quantities. A human can mark a locally invalid inference even when later steps repair it or the final answer happens to be right. An outcome verifier can prefer that trajectory. A controller can accept a locally unhelpful step if the continuation remains recoverable. For a fixed continuation policy, the probability of eventual success is a value function, not a definition of mathematical validity.

This distinction explains why training a stronger step detector need not improve best-of-N. Averaging step AUROC across questions also credits separation between easy and hard problems, whereas selection compares candidates for the same question. A monotone transformation leaves AUROC unchanged while changing probability-weighted voting. Ranking, calibration and decision value therefore require separate evaluation.

The current project crosses all three targets: PRM800K step labels, ProcessBench first-error identification, and final-answer selection on model-generated pools. Preserve those distinctions in tables. Report source-validation-selected F1 and its trivial baseline for binary step detection, ProcessBench's own trace metric for first-error identification, and final-answer accuracy for selection. Keep target-calibrated and oracle thresholds separate. AUROC avoids prevalence-driven changes in F1, but remains sensitive to the conditional populations being compared.

**2. Sparse features and dense representations**

Yang et al.'s [Step-Level Sparse Autoencoder for Reasoning Process Interpretation](https://arxiv.org/abs/2603.03031) is the project's origin. Use its published title and distinguish reproducing an encoder from reproducing an evaluation: replacing reconstruction-derived targets with external correctness labels changes the estimand. The historical 77.50% versus 78.58% comparison is not an equal-protocol accuracy contest.

Kantamneni et al.'s [Are Sparse Autoencoders Useful?](https://arxiv.org/abs/2502.16681) tests sparse probing under scarce data, imbalance, label noise and distribution shift. Their non-SAE baselines remove several apparent advantages, including in multi-token probing. This is close prior art for the project's dense-versus-sparse controls. Our inference should concern the tested SAE checkpoints and tasks. Inferiority in predictive accuracy neither proves absence of correctness information nor settles whether individual features help explain a model.

Orgad et al.'s [LLMs Know More Than They Show](https://arxiv.org/abs/2410.02707v4) links token position, error type and transfer: their detectors exploit localized information but fail to generalize across datasets, and internal answer information can disagree with emitted answers. For this project, token pooling and failure-mode decomposition address substantive questions already present in the literature. Token localization is task-dependent; diffuse signals in mathematical steps would qualify, rather than contradict, their result. The WikiProfile study belongs here, provided answer decoding tests whole answers and controls candidate priors.

A useful next representation experiment crosses last token, simple token statistics and learned pooling with matched training data, parameter budgets and adaptation budgets. The recorded +9.1 F1 versus +2.2 F1 contrasts motivate this experiment; their ratio is not a general law that representations matter four times as much as learners. Reading five concatenated statistics changes feature dimension and the linear hypothesis class. A factorial design should make that change visible.

**3. Internal-state verifiers are established prior art**

[ReProbe](https://arxiv.org/html/2511.06209v5) already trains a small transformer over token features, pools within a reasoning step, and uses the score in offline selection and online search. It studies hidden states as well as attention/logit features. A token transformer or a sub-10M parameter count cannot anchor a novelty claim here. The meaningful comparison concerns feature choice, fixed representation/learner controls, supervision and decision-level evaluation.

[HSRM](https://arxiv.org/html/2608.30841v1) processes a sequence of step-boundary hidden states and trains a candidate-level ranker with outcome-derived pairwise supervision. Our d512 verifier processes tokens within a step and trains on human step labels. That difference matters: HSRM's objective matches within-question candidate selection more directly. It motivates a controlled outcome-ranking ablation on the same states, rather than an argument about coincident transformer dimensions. Its main selection results do not supply the matched self-consistency comparison needed for our decision claim. Its publication date alone cannot establish our public priority. Appendix A trains on the first 150 MATH-500 questions and tests on the remaining 350; Appendix D labels the 85.3 scaling result as an earlier split. Appendix B adds LLM relabeling. This benchmark adaptation and grading differ from our full MATH-500 transfer evaluation.

Zhang et al.'s [Reasoning Models Know When They're Right](https://arxiv.org/abs/2504.05419) probes intermediate answer correctness and uses the probe to stop reasoning, reporting 24% fewer inference tokens without lower performance. It also detects future answer correctness before the answer is complete. This suggests testing whether our probe reflects local invalidity, impending failure, or recognition of an answer already formed. These require labels and interventions at different positions. A step-correctness classifier is not automatically an early-exit policy.

The literature already bridges probing and downstream control. The stronger contribution available here is a controlled account of why better decoding of correctness sometimes fails to translate into better decisions, including the role of supervision, candidate disagreement and the cost of obtaining the states.

**4. The ReProbe comparison needs a protocol correction**

ReProbe Table 3 gives Qwen3-8B GSM8K pass@1 95.6, majority@10 97.6 and its strongest listed PRM 97.8. These provide context for small incremental gains. They do not establish a field-wide maximum of 0.2 points. More consequentially, §4.2 uses temperature 1.0, and grades non-GSM8K tasks with DeepSeek-R1. Appendix C.2 describes sampled test subsets. Its MATH 92.4 is not established as MATH-500 under our symbolic grader. [Source: §4.2, Table 3 and Appendix C.2](https://arxiv.org/html/2511.06209v5)

Our regraded Instruct pool has exact all-sample accuracy 93.53% on GSM8K and 84.06% on MATH-500. A comparison with the published figures must first align question IDs, prompt, stopping rules, thinking mode and grading. Lowering temperature can alter accuracy, diversity and length together. Even an accuracy increase would not isolate the cause of the cross-paper gap. Treat a recommended-settings rerun as a policy sensitivity study, with the existing pool as its control.

Keep a comparability ledger:

| Comparison | What currently matches | What remains unmatched |
|---|---|---|
| Base versus Instruct d512 detection | Frozen source splits, head architecture, rescale setting, three seeds | Backbone representations; full artifact outputs need archiving |
| Base versus Instruct TTS | Benchmark questions, ten candidates, nominal sampling settings | Four-shot versus chat prompt, 2,048 versus 4,096 token cap, sampled errors |
| Our Instruct versus ReProbe | Backbone family, non-thinking setting, N=10 | Exact subset, prompt, grading, training labels and features |
| Our verifier versus HSRM | Hidden-state inputs, small learned readout | Token/step axis, process/outcome target, benchmark adaptation, grading, sampling, budget |
| Instruct probe versus external PRM | Can share the saved candidate pool | Instruct PRM scores and measured costs are still required |

**5. Process supervision and decision supervision can disagree**

Lightman et al.'s [Let's Verify Step by Step](https://arxiv.org/html/2305.20050v1) establishes strong process-supervision results and introduces PRM800K. Their reported lack of improvement from reward-weighted voting is specific to their experiments. Their successful PRM reranking also prevents interpreting the paper as a general case against verifier authority. Cite the result with its setting rather than extending it to every scorer.

[The Lessons of Developing Process Reward Models](https://arxiv.org/html/2501.07301v2) studies mismatches between rollout-derived labels, process correctness and best-of-N evaluation. In Table 6, majority@8 averages 66.2 while Qwen2.5-Math-PRM-7B reaches 67.6; several other PRMs fall below majority. Appendix aggregation comparisons show that last-step, product and minimum scores behave differently. These results motivate both a strong PRM comparator and an explicit aggregation ablation. Keep Qwen2.5-Math-7B-PRM800K distinct from Qwen2.5-Math-PRM-7B.

Our saved Base PRM scores already reject the broad claim that weighting never pays: the regraded N=10 analysis gives weighted-PRM gains over majority of 2.14 points on GSM8K and 4.40 on MATH. This is historical evidence on Base, with grading and exposure limitations, not an Instruct result. The Instruct weighted d512 vote also exceeds tie-breaking in the MATH N=10 point estimate, 90.44% versus 89.84%, although each contrast with majority has an interval spanning zero.

Setlur et al.'s [Rewarding Progress](https://arxiv.org/abs/2410.08146) defines process advantage through the change in future success probability under a prover policy. Its relevance to step deltas is conceptual: `h_i - h_{i-1}` is a representation difference, not an advantage estimate. Test its association with controlled changes in continuation success before treating it as progress. This is a promising bridge between the S4 contribution study and a controller, but the bridge needs supervision from continuations.

**6. Voting, confidence and the small-budget claim**

[Self-consistency](https://arxiv.org/abs/2203.11171) established answer aggregation across sampled reasoning paths. Its benefit depends on the answer distribution. A more accurate policy can leave fewer mixed correct/incorrect pairs, while systematic errors can dominate every vote. Accuracy and diversity should therefore accompany every selection curve.

[Deep Think with Confidence](https://arxiv.org/html/2508.15260v1) defines overlapping token-group confidence, low-confidence group summaries, filtering and confidence-weighted voting. Its examples use windows of 1,024 or 2,048 tokens. Our short-window adaptation for short non-thinking traces is useful, but should carry its actual window and aggregation settings. It is not a complete reproduction of every offline and online DeepConf configuration. Comparisons should include plain mean-token confidence and preserve score orientation and filtering rules.

[Boosting Self-Consistency with Ranking](https://arxiv.org/html/2606.05054v1) combines answer frequency, semantic centrality and reasoning consistency with a small learned ranker. It is direct prior art for learning when a score should overrule counts. For this project, a held-out fusion of vote margin, probe score gap and confidence would test complementarity. It must outperform its individual inputs on unseen questions and justify any extra state extraction cost.

[Entropy-Gated Branching](https://aclanthology.org/2026.eacl-long.235/) explicitly evaluates budgets 2, 4, 8, 16 and 32 against self-consistency and search methods. Figure 3 distinguishes sampled-solution count from beam-width-times-expansions. This refutes the broad claim that the small-budget literature is empty. It does not make its branching budget identical to our offline N. A narrower contribution is a paired decomposition of small-pool hidden-state selection, with explicit scorer authority and measured costs.

At N=2, consistently graded parseable candidates make tie-breaking and reranking equivalent in correctness. Agreement leaves no decision; disagreement creates a 1-1 tie. Thus the N=2 gain establishes ranking utility, not the superiority of restricting a verifier to ties. N=3 and N=4 distinguish those rules.

**7. What the new Instruct decomposition establishes**

For two candidates drawn without replacement, let M mean that exactly one is correct, and q be the verifier's conditional probability of selecting it, averaging exact score ties uniformly. Then:

`lift over majority = P(M) × (q - 0.5)`

`oracle headroom = P(M) / 2`

These identities require consistent answer labels and parseable candidates. They average over the empirical pool, not an assumed independent Bernoulli model of correctness. In particular, do not substitute `2p(1-p)` using dataset-average pass@1; question difficulty and repeated-sample dependence matter.

The new script enumerates all 45 pairs in each ten-candidate pool. On the shared eligible questions:

| Dataset | Questions | Policy | Mixed-pair rate | Correct choice given mixed pair | Fraction of headroom recovered |
|---|---:|---|---:|---:|---:|
| GSM8K | 1,319 | Base | 21.32% | 81.37% | 62.73% |
| GSM8K | 1,319 | Instruct | 4.53% | 62.73% | 25.46% |
| MATH-500 | 495 | Base | 24.48% | 75.33% | 50.67% |
| MATH-500 | 495 | Instruct | 10.77% | 65.07% | 30.14% |

Both opportunity and conditional selection performance fall. That observation supports examining the remaining Instruct error types and score calibration. It does not isolate a causal effect of instruction tuning: prompts, caps and candidate populations differ. On each arm's eligible subset, Instruct gains are 0.577 points [0.378, 0.779] on GSM8K and 1.636 [1.120, 2.133] on MATH. These are pointwise question-cluster bootstrap intervals, conditional on the saved pool and exclusions. They differ from the report's 32-order estimates by design.

Numeric answer fragmentation still matters at larger N, where it changes vote counts. At N=2, splitting two equally correct answers can create an apparent disagreement but cannot create a correctness gain under this identity. Distinguish answer disagreement from a mixed-correctness opportunity.

**8. Causal interpretation needs intervention-specific claims**

Belinkov's [Probing Classifiers](https://aclanthology.org/2022.cl-1.7/) reviews the gap between information a classifier can extract and information a model uses. This project should use three separate claims: decodability, predictive utility and causal influence on generation. An accurate nonlinear head supplies the first; an improvement in answer selection supplies the second; a controlled change in the model's continuation supplies the third.

Zhang and Nanda's [Activation Patching: Metrics and Methods](https://arxiv.org/abs/2309.16042) shows that corruption methods and outcome metrics alter localization conclusions. Heimersheim and Nanda's [How to Use and Interpret Activation Patching](https://arxiv.org/abs/2404.15255) develops related interpretation guidance. For our matched forks, report donor selection, intervention site, norm, dose, random controls, teacher-forced readout and free-generation outcome separately. Whole-span patching can move many content variables at once. Steering a probe direction can fail despite a causal role for other representations of the same information.

The project's §18.1 records whole-span solve-gap recovery 0.35, p=0.02, while a learned subspace's free-generation recovery is 0.09, p=0.53. Those outcomes support intervention-specific conclusions. They do not support declaring correctness a readout that the model never uses. Likewise, failure of a top-PC visualization cannot prove high intrinsic dimensionality; even a dense linear classifier uses a scalar projection.

[VerifySteer](https://arxiv.org/html/2605.20745v1) intervenes on verifier strictness. Its target is the verification decision, which differs from steering the generator toward a valid next mathematical step. Treat it as evidence that a verifier's decision can have controllable internal structure, not as a direct counterexample to a generator-steering null result.

**9. Compute and search theory guide experiments, not explanations by citation**

Snell et al.'s [Scaling LLM Test-Time Compute Optimally](https://arxiv.org/abs/2408.03314) makes allocation depend on task difficulty and evaluates compute tradeoffs. [LATTS](https://arxiv.org/abs/2509.20368) applies local verification to continuation decisions including resampling and backtracking. Both motivate an adaptive controller. Neither allows pricing our current template-state scorer as a head operating on cached generation states before that path exists and passes an equivalence test.

Huang et al.'s [Is Best-of-N the Best of Them?](https://arxiv.org/abs/2503.21878) analyzes coverage, imperfect rewards and finite-budget optimality. Our declining rerank curve is compatible with selection exploiting scoring errors. It does not identify reward hacking as the mechanism. The decisive local analysis records how often reranking rescues a wrong plurality and how often it replaces a correct one, then relates the changes to score extremes, length and error type.

Charge raw generation tokens, prompt processing, actual verifier reads and measured latency. A count of small-head parameters is insufficient if computing its input requires an additional backbone pass. For lazy verification, sum the lengths of the candidates actually read; multiplying average reads by unconditional average length can miss their covariance. A frontier interpolated in expected compute does not establish a hard per-question budget guarantee.

**10. Research priorities after consolidation**

The immediate scientific return comes from repairing the measurement contract and explaining the Instruct decisions. First unify gold-independent answer identity for grading and voting, resolve conflicting groups, retain failed samples in the full-pool metric and archive checkpoint/split hashes. Then score the identical Instruct candidates with the external PRM and compare majority, confidence, tie-break, rerank and one preregistered weighted rule. Select calibration and aggregation on development questions only.

Next separate step validity from outcome prediction. Audit Instruct mixed pairs by first invalid step, recoverable error, correct answer with invalid reasoning, answer-format failure and persistent misconception. Blind the annotation to the selector where feasible. Test whether a within-question outcome-ranking objective improves these decisions while preserving the original process-supervised head as a control.

For the representation result, begin with last token, token statistics and attention pooling before repeating all 19 cells. Apply grouped cross-fitting to length/position controls and content-preserving format changes. A strong matched representation effect would strengthen the mechanistic thread more than another unrestricted score sweep.

Finally establish generation-state equivalence and measured cost before an Instruct rejection trial. Compare rejection against plain generation, a rejection-rate-matched random controller and confidence-based rejection at the same budget. Keep Base as a historical control, and keep WikiProfile and transition-operator findings separate until they test the same causal hypothesis with comparable outcomes.
