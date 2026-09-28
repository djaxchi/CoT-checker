**CoT-Checker: current research context**

*Updated 2026-09-28. Experimental record: [REPORT.md](../REPORT.md), especially §21.10 and §21.11. Critical literature review: [tts_related_work_v1.md](tts_related_work_v1.md).*

The project investigates which model-internal signals reveal invalid reasoning steps and when those signals improve a decision. It began with step-level sparse autoencoders, developed matched dense representation probes, and now tests the connection between detection, causal interventions and answer selection. Qwen3-8B Instruct in non-thinking mode is the primary policy. Base results remain historical controls. Replacing the policy does not retroactively reproduce the earlier representation grid, generation-state scorer or online rejection experiments.

**Current model and evidence**

The matched verifier reads the residual states of all tokens in each step at `hidden_states[35]`, after block 34 of the 36-block Qwen3-8B. A projection and two-layer transformer, width 512, eight heads, feed-forward width 2,048, produce a pooled binary step score. The head has 8,665,089 parameters. Training uses 513,810 PRM800K steps, frozen source splits and `rescale=none`, with seeds 42, 43 and 44. Step suspicion increases with predicted incorrectness; the TTS `worst` aggregation takes the maximum suspicion over steps.

The research log reports the following three-seed averages. The full retraining and activation-audit outputs must accompany a release; this consolidation reads their summaries rather than reproducing GPU training.

| Detection measure | Base | Instruct | Interpretation |
|---|---:|---:|---|
| PRM800K test AUROC | 0.8946 | 0.9128 | Better discrimination on the same source evaluation protocol |
| ProcessBench F1_PB, source-val threshold | 0.469 | 0.560 | Better transfer under the stated source threshold selection |
| ProcessBench F1_PB, oracle threshold | 0.607 | 0.655 | A ceiling using target labels, not deployment performance |

Calib-20 consumes target labels and belongs in a separate adaptation comparison. Do not mix the older 0.566 Base grid result with the matched retrain's metrics without establishing identical threshold and averaging protocols. ProcessBench F1_PB is not ordinary binary step F1.

The activation audit reports similar norm concentration and attention destinations in the two arms, and a gain after linear length/position residualization. These controls narrow possible explanations. They do not certify a content-only signal: contextual step states can contain template information, and full-evaluation residualization is descriptive. Grouped cross-fitting and content-preserving format changes remain useful tests.

**What the saved Instruct pool says**

The policy generated ten candidates for each of 1,319 GSM8K questions and 500 MATH questions. It uses the non-thinking chat template, temperature 1.0, top-p 0.95, top-k 50 and a 4,096-token cap. The Base control uses four-shot prompting and a 2,048-token cap, so this comparison does not isolate instruction tuning alone.

Exact all-candidate accuracy after the September 28 regrade is 93.53% on GSM8K and 84.06% on MATH-500. The saved 32-draw-order paired analysis reports:

| Instruct tie-break minus majority, points | N=2 | N=4 | N=10 |
|---|---:|---:|---:|
| GSM8K | +0.54 [0.33, 0.74] | +0.22 [0.09, 0.36] | +0.08 [-0.19, 0.38] |
| MATH-500 | +1.67 [1.12, 2.24] | +0.65 [0.27, 1.07] | -0.17 [-0.70, 0.41] |

These are pointwise 95% question-cluster bootstrap intervals. They describe the current exploratory evaluation, not an untouched confirmatory test. The d512 scores use verifier-template states, so these rows do not establish a head-only generation-state deployment cost.

The exact-pair sensitivity analysis adds an explanation. On the same 495 eligible MATH questions, mixed correct/incorrect pairs occur 24.48% of the time for Base and 10.77% for Instruct. The matched d512 heads select the correct candidate within those pairs 75.33% and 65.07% of the time. Smaller headroom and lower conditional ranking performance both contribute. At N=2, tie-breaking and reranking coincide in correctness under consistent labels. Use larger N to test whether restricting score authority helps.

At Instruct MATH N=10, weighted voting scores 90.44%, tie-breaking 89.84%, and majority 90.01%. Neither weighted voting nor tie-breaking has a resolved gain over majority. Historical Base PRM scores also show that weighted voting can win. The appropriate conclusion concerns each policy, scorer and rule combination.

**Interpretation of the earlier program**

The dense scaling study reports AUROC 0.776 at 1.5B and 0.828 at 32B. This establishes increasing decodability under its protocol. It does not by itself establish self-awareness, a universal correctness direction or causal use by the generator.

The tested sparse representations did not improve verification over strong matched dense baselines. Preserve this negative result with its checkpoint and task scope. It does not close all SAE-based interpretation questions.

The representation harness gives concrete evidence for using information across tokens: a linear last-token to token-statistics contrast moves ProcessBench F1 from 0.394 to 0.485, while attention pooling to transformer moves 0.500 to 0.522 in the cited setting. These contrasts use different representations and capacities; a fourfold ratio is not a general factor-effect estimate. The complete grid has not yet been repeated on Instruct.

The causal work contains both positive and null outcomes. Whole-span patching produces reported solve-gap recovery 0.35, p=0.02. A learned subspace's free-generation recovery is 0.09, p=0.53. Distinguish moving a teacher-forced readout from improving a generated solution. A failed direction intervention cannot establish that the generator never uses correctness-related information. Keep WikiProfile retrieval and transition-operator experiments as separate evidence until a common intervention and outcome connect them.

**Measurement gaps to close**

The current regraded pools still assign different correctness labels to identical normalized answer groups on MATH question 255 in both policies, and 99 and 420 in Instruct. Equivalent numeric forms also divide votes. Establish one gold-independent answer identity policy, check each answer group's label consistency and retain unparseable samples as failures in the full evaluation. The exact-pair analysis excludes problematic questions and records them; it does not repair the underlying grading.

Historical on-policy d256 training data contain 23 training and 3 validation question overlaps with MATH-500. That finding does not establish overlap in the PRM800K-trained d512 arm. Archive training question hashes, checkpoint hashes and cross-corpus overlap checks for each scorer, including model-selection data. Do not infer that a new splitting script changed an older checkpoint.

Historical Base frontiers charge retained text after truncation, excluding about 40.3% of recorded raw generation tokens. The Instruct discrepancy is small, but verifier re-encoding and actual read lengths still matter. Measure prompt processing, raw generation, verifier reads and latency separately. Test equivalence between captured generation states and reconstructed states before asserting zero backbone rereads.

**Consolidated research direction**

Repair evaluation identity and provenance first, then compare the external PRM and the small verifier on the identical Instruct candidates. Explain rescues and harmful overrides by error type, answer disagreement, score margin and trace length. Keep process detection distinct from within-question outcome ranking.

A compact Instruct representation experiment should precede the full grid. Last token, token statistics and attention pooling can establish whether the earlier representation advantage survives. A measured generation-state path should precede an Instruct online rejection claim, with plain generation, random rejection and confidence rejection as budget-matched controls.

The literature already contains token-level hidden-state verifiers, small-budget search and learned combinations of scores with votes. The opportunity here is a controlled explanation of the relationship between decodable correctness and useful decisions, supported by reproducible artifacts and narrowly stated causal conclusions.
