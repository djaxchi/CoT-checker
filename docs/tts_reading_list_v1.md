**Reading order for the Instruct consolidation**

*Updated 2026-09-28. This replaces the earlier July cutoff. Read with [the critical review](tts_related_work_v1.md), which distinguishes source findings from project interpretations.*

| Order | Primary source | Read closely | Question to bring back to this project |
|---:|---|---|---|
| 1 | [ReProbe, v5](https://arxiv.org/html/2511.06209v5) | §3, §4.2, Table 3, Appendix C.2 and prompts | Which parts of our token verifier are already established, and which evaluation protocols actually match? |
| 2 | [The Lessons of Developing PRMs](https://arxiv.org/html/2501.07301v2) | Label construction, process-versus-outcome analysis, Table 6, aggregation appendix | Does the score predict local validity or successful final answers, and does the aggregation fit that target? |
| 3 | [HSRM](https://arxiv.org/html/2608.30841v1) | Step-boundary extraction, pairwise loss, input ablations | Can a matched outcome-ranking objective explain a gap between detection and selection? |
| 4 | [Let's Verify Step by Step](https://arxiv.org/html/2305.20050v1) | Supervision comparison, scaling curves, weighted-voting discussion | What supports process supervision, and how narrow is the aggregation result? |
| 5 | [Deep Think with Confidence](https://arxiv.org/html/2508.15260v1) | Equations 2, 4, 5 and 8, online stopping | Which exact confidence statistic, window and filtering rule are we comparing? |
| 6 | [Boosting Self-Consistency with Ranking](https://arxiv.org/html/2606.05054v1) | Candidate-relative features and ablations | Does learned fusion add information beyond answer counts and a single score? |
| 7 | [Entropy-Gated Branching](https://aclanthology.org/2026.eacl-long.235/) | Figure 3 and its budget definition | How should offline N=2 to 4 compare with small-budget sequential search? |
| 8 | [Rewarding Progress](https://arxiv.org/abs/2410.08146) | Process advantage and prover-policy dependence | Which continuation-success target could give a step delta a behavioral meaning? |
| 9 | [Reasoning Models Know When They're Right](https://arxiv.org/abs/2504.05419) | Intermediate answers, lookahead probing, early exit | Does our signal identify an invalid step or anticipate eventual failure? |
| 10 | [LLMs Know More Than They Show](https://arxiv.org/abs/2410.02707v4) | Token localization, transfer and error types | Which Instruct failure modes carry distinct signals? |
| 11 | [Are Sparse Autoencoders Useful?](https://arxiv.org/abs/2502.16681) | Matched baselines and multi-token experiments | What does our sparse negative result add under controlled supervision? |
| 12 | [Probing Classifiers](https://aclanthology.org/2022.cl-1.7/) and [Activation Patching: Metrics and Methods](https://arxiv.org/abs/2309.16042) | Probe limitations, corruption choices and behavioral metrics | What evidence distinguishes extracted information from a causal mechanism? |
| 13 | [Scaling Test-Time Compute Optimally](https://arxiv.org/abs/2408.03314) and [LATTS](https://arxiv.org/abs/2509.20368) | Difficulty-dependent allocation and local control | Which controller merits testing after measuring the scorer's real cost? |
| 14 | [Is Best-of-N the Best of Them?](https://arxiv.org/abs/2503.21878) | Assumptions behind finite-budget optimality | Which empirical observations would identify selection on scoring errors? |

For the project's origin, cite [Step-Level Sparse Autoencoder for Reasoning Process Interpretation](https://arxiv.org/abs/2603.03031). For intervention interpretation, add [How to Use and Interpret Activation Patching](https://arxiv.org/abs/2404.15255). [VerifySteer](https://arxiv.org/html/2605.20745v1) concerns intervention on verifier strictness, a different target from improving the generator's next reasoning step.

Use each paper to specify an experiment or constrain a claim. The main unresolved decisions are evaluation identity, the Instruct PRM comparison, process versus outcome supervision, representation controls and the cost of obtaining deployable states. A longer bibliography cannot resolve those decisions without the corresponding data.
