# Research direction feedback, 28 September 2026

Review of [COT_Research_Paper](https://github.com/djaxchi/COT_Research_Paper) at commit `3ef34db`, including the experimental sections and appendices, alongside the newer local Instruct results. This note records feedback and proposed experiments. It does not replace `REPORT.md` or establish new experimental findings.

**Recommendation: organize the paper around when better error identification improves reasoning, and explain the cases where it does not.** Keep the matched Instruct leaderboard as the controlled experiment that supports this question.

You currently pursue three goals: explaining internal representations, building an effective verifier, and understanding test-time selection. Each requires different evidence. A representation can predict errors without providing an effective steering direction. A verifier can identify errors without improving answer selection. These outcomes are compatible, but the draft sometimes treats them as contradictions to resolve.

The Instruct transition gives you a reason to revisit the argument as well as the numbers.

**Preserve the controlled representation results.**

The last-token versus mean-pooling comparison is one of your clearest results: ProcessBench F1 rises from **0.419 to 0.469**, with the same **4,097 trainable parameters**. The manuscript reports a paired interval of **[0.032, 0.067]** for the improvement. This isolates a meaningful representation choice.

The learned-query result also matters: **8,193 parameters reach F1 0.558**, compared with **0.566** for the large sequence Transformer. The reported difference interval, **[−0.003, 0.018]**, supports a restrained conclusion: a small readout accesses much of the available performance in this setting. It does not establish that capacity never matters.

These results give you a controlled family of verifiers. You can ask whether an improvement from changing the representation survives downstream while holding much of the training protocol fixed.

Lightweight verification from hidden states already appears in [ReProbe](https://reprobe.github.io/) and [HSRM](https://arxiv.org/abs/2608.30841). Your strongest contribution would explain which improvements transfer, under which conditions, and why.

**Narrow the causal claim.**

The introduction calls the signal “diagnostic, not causal.” Yet the manuscript reports **88% recovery of the answer-preference margin through whole-span patching**, and **35% recovery of the free-generation accuracy gap** in the reported experiment. Additive steering along the probe direction did not reliably outperform random directions.

Together, those observations support this statement:

> The tested probe direction predicts correctness but does not provide an effective steering intervention; broader activation replacements can affect the answer.

They do not establish that correctness-related information is noncausal, or that the relevant causal information must occupy the entire span. Whole-span patching transfers many things simultaneously, including mathematical content.

[Yuan et al.](https://arxiv.org/abs/2605.09502) already make “diagnostic, not causal” their central claim. Your positive patching results could help refine that claim. Repeating it would obscure the difference.

**Reconsider the claim that overriding the vote always hurts.**

The newer Instruct results show a more complicated pattern. With the external Qwen2.5-Math PRM, worst-step aggregation, and ten candidates:

| Selection rule | GSM8K accuracy | MATH-500 accuracy |
|---|---:|---:|
| Majority vote | 95.00% | 90.04% |
| PRM breaks vote ties | 95.30% | 90.04% |
| PRM reranks all candidates | 96.13% | 88.15% |

Source: [`frontier_instruct_regraded.json`](../results/instruct_arm_v1/frontier_instruct_regraded.json), created `2026-09-28T17:05:32.320462+00:00`, `split=all`, `n=10`, 64 sampling orders. PRM rows use `prm_qwen25_math_7b::worst`. The results directory is local and may be absent from a clean clone.

These are point estimates, not a claim of significant paired differences. The grading issues identified in the audit still need resolution before publication. The pattern nevertheless challenges a universal recommendation to restrict verifier authority: the same verifier and rule behave differently across tasks. Understanding that interaction is a stronger research target.

At two candidates, tie-breaking and reranking coincide under consistent answer grouping and correctness labels. Gains at that budget cannot establish the superiority of restricting a verifier to ties.

**Replace the “5–10×” transfer claim with comparable quantities.**

An F1-point improvement and an answer-accuracy-point improvement measure different things. Their ratio does not measure transfer efficiency. Downstream improvement also depends on the available correct-answer selection headroom.

Report downstream gain, oracle headroom, and the fraction of headroom recovered. Keep their interpretation separate from the detection metric.

**Study the decisions where the verifier matters.**

ProcessBench measures error identification across its evaluation population. A selector must distinguish candidates for the same question, particularly when its preference conflicts with the vote. A verifier can improve on many benchmark examples while remaining weak on this consequential subset.

The exact two-candidate analysis on shared eligible GSM8K questions already separates two effects:

| Quantity | Base | Instruct |
|---|---:|---:|
| Fraction of pairs containing one correct and one incorrect candidate | 21.32% | 4.53% |
| Probability the evaluated verifier selects the correct candidate within those pairs | 81.37% | 62.73% |

Sources: `runs/project_audit_20260928/regraded_pair_base.json` and `runs/project_audit_20260928/regraded_pair_instruct.json`, using the shared eligible question set. These are local audit artifacts.

Both the opportunity to help and the conditional ranking performance decreased. This comparison does not isolate instruction tuning because the generation protocols also differ. It shows why ProcessBench F1 alone cannot explain downstream gain.

For two candidates, consistent answer grouping and grading, and random tie-breaking as the baseline:

```text
accuracy gain
  = P(one correct, one incorrect)
    × [P(select correct | mixed pair) − 1/2]
```

This provides an explanatory structure: available opportunity multiplied by the ability to exploit it.

At larger budgets, report how often the verifier rescues a wrong majority decision and how often it replaces a correct majority decision with a wrong one. Net accuracy hides the distinction.

The Instruct leaderboard should answer three questions:

1. Does ProcessBench rank predict downstream rank?
2. Does it predict performance on consequential disagreements?
3. Does it add predictive information beyond token confidence and vote strength?

The second and third questions would make the study more informative than a correlation plot alone. Use correctness labels to define retrospective evaluation subsets, not to route candidates at deployment.

**Prioritize one experiment on error recoverability.**

Investigate the distinction between an incorrect step and an error that prevents eventual success. One trace may contain an arithmetic mistake that the model later repairs. Another may contain a plausible but invalid assumption that determines the final answer. Both contain process errors, but they have different implications for selecting, continuing, or repairing the trace.

[Lessons of Developing Process Reward Models](https://arxiv.org/abs/2501.07301) discusses the mismatch between process validity and final-answer evaluation. [Rewarding Progress](https://arxiv.org/abs/2410.08146) motivates rewards based on changes in future success probability. Your opportunity is to investigate which properties your internal signals capture.

Start with a small, manually checked collection of matched reasoning prefixes:

- A valid prefix.
- A minimally edited version containing one verified mathematical error.
- Multiple continuations from each prefix under the same generation policy.

Read the verifier score before generating those continuations. Measure local validity, eventual success, and whether the model explicitly repairs the error.

The discriminating question is whether the verifier distinguishes errors the model can recover from. If it detects recoverable and persistent errors equally well, while selection depends mainly on persistent errors, that would support an explanation for strong detection and weaker downstream performance. If the pattern is absent, reject that explanation and investigate calibration, aggregation, or distribution shift.

This experiment would connect the mechanistic and downstream work without another broad architecture sweep.

A related hypothesis is that the readout signals an error without specifying its repair. Knowing that a derivation is wrong does not identify the replacement computation. This could explain why scalar steering fails while replacing substantive reasoning states succeeds. It remains a hypothesis. It suggests testing whether the score helps decide when to check or resample. Hidden-state pruning already exists in [STEP](https://aclanthology.org/2026.findings-acl.1336/), so the contribution would require the explanation and controlled test, rather than the controller alone.

**Next steps, in order.**

1. **Finish the matched Instruct comparison and stabilize evaluation.** Resolve answer-equivalence inconsistencies, document checkpoint and threshold provenance, and distinguish generation-state scoring from scoring that requires another forward pass. The draft currently describes both under its cost argument.
2. **Analyze disagreements before training additional models.** Use the same candidate pools across verifiers. Measure rescue, harm, mixed-pair ranking, and remaining oracle headroom. Bootstrap by question, jointly across methods. Keep seed variation separate. Inspect representative failures with scores hidden during initial annotation.
3. **Run one targeted recoverability experiment.** Use the disagreement analysis to choose the error types. Specify the observation that would support the hypothesis and the observation that would lead you to reject it.

Defer additional model scales, generic probe architectures, and a broad SAE expansion until the analysis identifies a specific question they can answer. SAE analysis could help distinguish repairable arithmetic slips from persistent invalid assumptions, for example. Sparsity alone would not explain the phenomenon.

The paper could then follow one argument: representation choices improve error detection; those improvements transfer unevenly to selection; decision-level analysis identifies the mismatch; a targeted experiment tests its explanation.

A working title is **“When Does Better Error Detection Improve Reasoning?”** A positive correlation would establish predictive value within the tested setting. A weak correlation would become informative if you explain it and show that the experiment had enough headroom and precision to detect a meaningful relationship.

Keep the Instruct leaderboard, and make explaining its successes and failures the main intellectual task. Use that question to decide which existing results belong in the paper and which next experiment could change your interpretation.
