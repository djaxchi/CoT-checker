# What the Instruct verifiers respond to: deeper audit

29 September 2026. I investigated the frozen probes locally using the 912 cached examples, inspected their implementations and training-data construction, read relevant literature, and ran controlled interventions on their inputs. This audit extends the original diagnostic. It is exploratory: I chose the interventions after seeing the first results. No new backbone extraction, training, or GPU job was necessary.

My current interpretation is that these probes read contextual information about whether a candidate fits the immediately preceding reasoning. The strongest evidence concerns the direction of late candidate-token states. The resulting score also depends on wording and domain. It should not be interpreted as a universal, calibrated probability that the entire reasoning trace is correct.

**What I ran.**

I reproduced the unmodified outputs of all 16 checkpoints before interpreting interventions. The two leading architectures, attention-query at seeds 42/43 and d512 at seeds 42/43/44, received additional tests: retain only the last token state; hide the last state; retain only digit-token states; hide digit-token states; reverse state order; and, for attention-query, replace its weights with uniform pooling. Masks retain the surviving tokens' original position indices.

I also formed 144 same-candidate context pairs per checkpoint: two candidates for each of 72 mathematical families. The candidate text and its tokenization are identical in each pair, but the preceding context changes its local validity. I swapped first-token, last-token, digit-token, non-digit-token, or all states in both directions. Two further interventions exchange token norms or directions. The five checkpoints therefore received 10,080 hybrid-input evaluations. Those evaluations are dependent observations, not 10,080 independent mathematical tests.

Headline results use the existing test families and plain wording unless stated otherwise. The bootstrap averages candidates and seeds within each family, then resamples whole families within each domain. There are 18 held-out inheritance families and nine held-out families per reversal domain. Intervals are descriptive and do not capture training-seed uncertainty or correct for multiple comparisons.

**1. The verifier distinguishes a new invalid transition from a wrong answer inherited from the prefix.**

The earlier example remains useful. For `Solve 4*x + 6 = 50`, d512 gives `4*x = 48` an error score of 0.981 when that subtraction is the current step. It then gives the continuation `x = 12` a score of 0.036. In contrast, the unexplained jump from `4*x = 48` to the correct answer `x = 11` receives 0.9998. These are three-seed means.

Across the held-out inheritance families, d512's balanced local contrast is 0.879, compared with 0.026 for inherited-prefix error and 0.043 for conclusion error. Attention-query has the same ordering. This supports local-consistency sensitivity under the stated operational labels. It does not establish that all natural errors produce this behavior, or that explicit repair would be rejected. An explained correction and an unsupported jump are different inputs; the existing data contain only the latter.

A later low score should therefore not erase an earlier alarm. Conversely, permanently rejecting every trace with an earlier alarm could reject successful repair. We need a separate repair test before choosing either policy.

**2. The context-dependent distinction reaches the probe through late token states.**

For the inheritance arm, swapping states between locally valid and invalid contexts produces these symmetric logit shifts. Positive values move the readout toward the donor's validity condition. The table averages both patch directions, both candidates, and training seeds within each family.

| States transferred | d512 shift | Attention-query shift |
|---|---:|---:|
| All candidate states | 12.016 | 10.870 |
| First candidate state | 0.001 | approximately 0 |
| Digit-token states | 8.159 | 7.178 |
| Final period state | 3.861 | 3.704 |

The candidate text is unchanged in these swaps. This rules out a fixed preference for the candidate string as the sole explanation of the context effect. The input-state intervention moves the frozen verifier's output, so it goes beyond observing a correlation between an attention plot and a score.

However, the states already contain contextual information from Qwen. The period's state can encode the entire preceding candidate. A period-state effect is not evidence that punctuation itself causes the mathematical judgment. These swaps identify inputs that influence the small verifier; they do not locate the Qwen circuit that generated the information. Hybrid states can also fall outside the normal activation distribution.

Digit and non-digit patches are complementary. Because I average both directions, their symmetric effects sum to the full gap by construction. That identity is not evidence that the nonlinear model consists of independent additive mechanisms. First- and last-state interventions provide different, narrower comparisons.

**3. Attention-query changes mainly because token-state values change.**

The attention-query head has an exact decomposition:

```text
logit = bias + sum_t attention_weight[t] * (readout_vector dot hidden_state[t])
```

For a valid/invalid context pair with weights `a0, a1` and values `v0, v1`, I split the logit change symmetrically:

```text
value contribution   = sum_t ((a0 + a1)/2) * (v1 - v0)
routing contribution = sum_t ((v0 + v1)/2) * (a1 - a0)
```

Their sum equals the observed logit difference. This is an exact accounting of this head, not a claim that the two quantities are independent causal mechanisms inside Qwen.

| Held-out domain | Full logit gap | Value contribution | Routing contribution |
|---|---:|---:|---:|
| Inheritance, affine | 10.870 | 10.212 | 0.658 |
| Reversal, affine | 6.238 | 5.896 | 0.343 |
| Reversal, inequality | 7.677 | 6.646 | 1.031 |
| Reversal, multiplication | 11.170 | 10.606 | 0.565 |
| Reversal, substitution | 1.314 | 1.424 | −0.110 |

In the inheritance arm, about 94% of the mean logit difference falls in the value term. Uniform attention also preserves every paired preference on the held-out affine, inequality, multiplication, and inheritance cases. Learned pooling is not necessary for those preference signs on this small diagnostic. It can still matter for calibration, harder examples, and the natural benchmark where it was trained and ranked.

This is why I would study the readout direction and contextual token states before interpreting the attention weights as the explanation. Jain and Wallace's [Attention is not Explanation](https://arxiv.org/abs/1902.10186) motivates that distinction; our exact decomposition and interventions provide model-specific evidence beyond the general warning.

**4. Activation magnitude does not explain the observed paired effect.**

For each token, I transferred the donor's L2 norm while retaining the recipient's direction. I also performed the complementary intervention: transfer the donor direction while preserving the recipient norm.

| Inheritance intervention | d512 logit shift, 95% family interval | Attention-query shift, 95% interval |
|---|---|---|
| Full context-state swap | 12.016 [10.170, 13.907] | 10.870 [9.465, 12.160] |
| Norms only | −0.577 [−0.724, −0.430] | −0.187 [−0.245, −0.132] |
| Directions only | 12.594 [10.668, 14.553] | 11.058 [9.610, 12.385] |

The corresponding norm-only shifts remain small relative to full gaps across the four reversal domains. This weighs against the hypothesis that a scalar activation-size cue explains these paired distinctions. It does not rule out every geometric shortcut, identify an interpretable feature, or establish robustness to arbitrary rescaling. Norm and direction interventions are complementary hybrid inputs, so their symmetric sum is also an accounting identity.

**5. The small Transformer relies little on the order of these already-contextualized states.**

Reversing d512's candidate-state order changes its logits by an average absolute 0.0054, with maximum 0.0317 over held-out target examples, all wording styles, and its three seeds. The aggregate plain-wording joint-preference rates remain unchanged. Attention-query is permutation-invariant by architecture; its approximately 0.0000007 mean change is numerical noise and acts as a sanity check.

This suggests that d512 behaves largely as a reader of an already informative collection of states on these short inputs. It does not show that token order is irrelevant to Qwen, that d512 is redundant on natural traces, or that its architecture never uses order. The backbone states already encode causal position and context before I reverse them.

**6. Wording alters the absolute score even when discrimination survives.**

| Held-out inheritance candidates | d512 mean error score | Attention-query mean error score |
|---|---:|---:|
| Locally valid, plain wording | 0.109 | 0.103 |
| Locally valid, `I am certain that` | 0.574 | 0.594 |
| Locally invalid, plain wording | 0.988 | 0.990 |
| Locally invalid, `I am certain that` | 0.994 | 0.977 |

Both heads still get both paired preferences right in every inheritance family and seed under the confident wrapper. The wrapper harms the absolute level without destroying that ordering. For d512, the valid-score means have family intervals [0.069, 0.155] for plain wording and [0.542, 0.606] for the confident wrapper.

The logit gap also narrows, so this is not solely sigmoid saturation. The wrapper raises the mean valid d512 logit from −3.054 to 0.335 and lowers the mean invalid logit from 8.962 to 6.037. Yet its effect differs by domain: adding the same wrapper changes valid affine-reversal scores little, from 0.539 to 0.531. There is no evidence for one universal rule such as trusting or distrusting verbal certainty.

These comparisons mix wording, token count, token identity, and contextual changes to later states. They do not isolate confidence as a psychological or semantic variable. They do explain how good paired ranking and poor threshold transfer can coexist. I use “error score” throughout because sigmoid output alone does not establish probabilistic calibration on this synthetic distribution.

**7. Substitution is a mixture of indiscriminate rejection and weak contextual discrimination.**

| Plain substitution cases | d512 | Attention-query |
|---|---:|---:|
| Mean score, mathematically valid | 0.957 | 0.997 |
| Mean score, mathematically invalid | 0.995 | 1.000 |
| Both contextual preferences correct | 0.333 | 0.500 |

The valid-score intervals are [0.907, 0.990] for d512 and [0.993, 1.000] for attention-query. The heads often assign near-maximal error scores to correct evaluations of `f(x) = x^2 + b`. Calling this only a lack of confidence would be misleading.

There is still a positive average context-dependent logit gap: 1.506 for d512 and 1.314 for attention-query. Sigmoid saturation hides part of that difference. However, the low joint-preference rates show that a simple domain-specific threshold shift would not fix every family: threshold changes cannot repair a reversed ordering. We need both better discrimination and better score calibration on this format.

Retaining only d512's final token state raises its substitution joint-preference fraction from 0.333 to 0.519, while hurting affine reversal from 1.000 to 0.852. This exploratory tradeoff is a clue for follow-up, not an optimized replacement probe or a statistically established improvement. It argues against treating probe capacity as a universal ordering of usefulness.

**What the evidence supports now.**

A working description is: a trained readout of contextual candidate compatibility, dominated in these examples by late-state directions, with substantial domain and wording dependence. The score carries useful relational information that a fixed answer-string preference cannot explain. It does not consistently track whether the final answer is correct, whether earlier errors remain unresolved, or whether a complete trace deserves acceptance.

Several alternatives remain live. The states may encode mathematical inconsistency, predictive unexpectedness, familiarity with particular derivation patterns, or a mixture. We have not measured candidate token likelihoods here, so we cannot separate semantic verification from a richer conditional-surprise signal. We also have not localized a backbone circuit, tested explicit repair, or established that these effects explain the natural ProcessBench leaderboard. Belinkov's [review of probing classifiers](https://aclanthology.org/2022.cl-1.7/) is relevant to the distinction between extracting information and explaining how the original model uses it.

**How this relates to your leaderboard and the literature.**

[ProcessBench](https://arxiv.org/abs/2412.06559) asks for the earliest erroneous step, or a judgment that all steps are correct. My inference is that success on that task does not require identifying whether a later step repairs an earlier error. That is a task gap we should measure before using the leaderboard to choose a repair policy.

[Yang et al., Beyond the First Error](https://arxiv.org/abs/2505.14391), already studies error propagation and error cessation in reflective reasoning. It develops annotation rules and trains a PRM for those cases. We should cite that work rather than claim that distinguishing propagation and recovery is itself novel. Your possible contribution lies in explaining what frozen hidden-state probes encode, how the readout transforms that information, and when those properties predict downstream utility.

[Zhang and Nanda](https://arxiv.org/abs/2309.16042) show that patching outcomes depend on intervention and metric choices. Here I use matched candidate strings, both patch directions, full-swap checks, and logits alongside scores. These controls strengthen the input-level result without converting it into a discovered backbone circuit.

The training builder at `scripts/build_prm800k_prestudy.py` advances the prefix with the human completion when available, otherwise the chosen completion. It does not enforce a prefix-correctness check in that branch. I therefore do not claim that all training prefixes were correct or that a measured absence of inherited-error training caused the behavior. Establishing that causal training explanation would require auditing the actual source records and comparing training variants.

**The next decisive experiment.**

I would now freeze a new set of natural and controlled examples before extracting any more activations. After the same wrong prefix, compare: continuing the error; an unexplained correct-answer jump; a correct explanation that repairs the error; and repair language followed by incorrect arithmetic. Include corresponding continuations after a correct prefix so that “I made a mistake” is not itself a label cue. Score both the repair and a subsequent step. Matched correct and incorrect repairs must share the same discourse framing.

For the substitution failure, keep the arithmetic fixed while varying function notation versus an explicit arithmetic expansion. Cross correct and incorrect results in every format. Simultaneously record candidate log likelihoods and token entropies from the backbone. Those measurements would test whether probe contrasts survive comparisons with matched surprise; correlation alone would not prove a mechanism.

Then annotate a natural-trace sample with scores hidden, distinguishing new error, propagated error, legitimate correction, and globally correct conclusion. Only after that should we test local resampling or backtracking at equal generation budgets. Maintain previous alarms until a separately validated repair criterion clears them. This is a proposed policy experiment, not a demonstrated improvement.

**Reproduction and artifacts.**

```bash
.venv/bin/python scripts/analysis/verifier_signal_deep.py \
  --out results/verifier_signal_v1/deep_audit_repeat
.venv/bin/python scripts/analysis/summarize_verifier_signal_deep.py \
  --run results/verifier_signal_v1/deep_audit_repeat
```

The completed run is `results/verifier_signal_v1/deep_audit_v3/`:

- `readouts.json`: raw logits and per-seed summaries for every readout intervention.
- `attention.json`: exact per-token weights and readout values for attention-query.
- `patches.json`: every paired state intervention and symmetric contribution.
- `patch_summary.json`: family-bootstrap summaries of patching and the exact attention decomposition.
- `behavior_summary.json`: family-bootstrap summaries of conditional score levels and intervention behavior.
- `provenance.json`: hashes, environment, and analysis scope.

The first exploratory implementation omitted the training adapter's float16 roundtrip for pooled vectors. Its baseline check caught a maximum discrepancy of 0.00144 and stopped execution. I corrected that adapter mismatch and added a regression test before interpreting results. The complete v3 audit passes its unmodified-score, exact-decomposition, and full-swap checks; 37 related tests pass. The failed preliminary directory contains no completed analysis and is not a result source. No plots or new cluster jobs were generated, and no files were committed.
