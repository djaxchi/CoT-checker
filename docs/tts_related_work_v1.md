# Where our curves sit in the field

*2026-09-22. Literature pass for sprint 8, supporting `docs/tts_sota_v1_plan.md`.
Every number below is quoted from the cited paper, not recomputed.*

## 0. The comparability problem, stated first

Our policy is **Qwen3-8B-Base**, prompted 4-shot, sampled at temperature 1.0.
Almost every published TTS result uses an instruct or reasoning model. The
absolute accuracies therefore do not line up and should not be put side by side:

| setting | GSM8K | MATH-500 | source |
|---|---|---|---|
| Qwen3-8B-Base, greedy / few-shot | 81.9 | 67.2 to 68.6 | Qwen3 tech report and third-party evals |
| Qwen3-8B instruct, non-thinking, pass@1 | 95.6 | 92.4 | ReProbe Table 3 |
| **ours, Base, 4-shot, T=1.0, pass@1** | **78.8** | **45.0** | REPORT.md §21 |

Our GSM8K pass@1 of 78.8 against a reported 81.9 is the expected cost of
sampling at temperature 1.0 instead of greedy. Our MATH-500 pass@1 of 45.0
against a reported 67 to 69 is a much larger gap and is not explained by
temperature alone. Flag it rather than hide it: it means our math500 curves
start from a weaker policy than anyone else's, which makes the headroom larger
and the voting baseline weaker.

Three things **are** comparable across backbones and are what we should report:

1. **lift over self-consistency at matched N**, since both arms use the same policy;
2. **verifier AUROC**, which is threshold-free and policy-relative;
3. **cost**, in generated tokens and in verifier parameters.

## 1. Internal-state verifiers: our direct competitors

| work | probe | params | signal read | policy | benchmarks | N | self-consistency baseline? |
|---|---|---|---|---|---|---|---|
| **ReProbe / UHeads**, arXiv 2511.06209, ACL 2026 | transformer | 9.8M | attention to the 1 to 3 preceding tokens plus top-K logits | Qwen3-8B, Phi-4 | MATH, GSM8K, ProofNet, 3 planning, 2 QA | 10 offline, 5 online | **partial**: 3 of 8 datasets, absent on MATH |
| **HSRM**, arXiv 2608.30841 | 2-layer transformer, d=256, 4 heads, FF 4x256 | 2.12M | last-layer hidden state at step boundaries | Qwen3 1.7B/4B/8B/14B, Llama-3.x | GSM8K, MATH-500, AIME, OlympiadBench | 8, 16, 32, 64 | **none** |
| **STEP**, arXiv 2601.09093 | 2-layer MLP, 512 hidden | ~1.3M | last-layer hidden at `\n\n` boundaries | Qwen3-4B-Thinking, R1-Qwen3-8B, Phi-4-reasoning | AIME-25, HMMT-24/25, GPQA-D | 64 | **yes** |
| **ELHSR**, arXiv 2505.12225, KDD 2026 | linear per-token head | <0.005% of baseline RM | hidden states, or logits alone | multiple | BoN math | - | reward models only |
| **PHSV**, arXiv 2504.05419 | MLP per reasoning chunk | small | hidden state per chunk | reasoning models | math | - | early-exit, 24% token saving |
| **VerifySteer**, arXiv 2605.20745 | correctness probe plus latent steering | small | paragraph-boundary hidden states | Qwen3-1.7B | ProcessBench, Hard2Verify | - | **yes**, +3.7 F1 at 4x less compute |

**HSRM is concurrent work, and it is not the same verifier.**

On priority: HSRM v1 was submitted to arXiv on **31 August 2026**. Our
representation-by-learner grid, which contains `transformer_d256_l2_f1024_h4`,
was committed to this repository on **24 August 2026 at 13:15:36 -0400**, one
week earlier, in a public git history. By any normal convention that is
independent concurrent work, cited as concurrent, not a priority claim against
us.

On substance, the encoder hyperparameters coincide and the model does not. The
generic block (2 layers, width 256, 4 heads, feed-forward 1024) is a standard
small-encoder size, and what matters is what it encodes:

| | HSRM | ours |
|---|---|---|
| sequence axis | the **steps** of a trace, S <= 100 step vectors | the **tokens** inside one step |
| output | one rank per candidate solution | one suspicion score per step |
| read layer | final layer L | block 34 `resid_post` of 36 |
| labels | outcome labels propagated onto self-generated trajectories | PRM800K human step labels |
| pooling | encoder over step summaries | masked mean over step-token states |

Theirs is a trace-level ranker built from step summaries. Ours is a step-level
verifier built from token states, which is why it can drive the online rejection
loop in §21 and a trace-level ranker cannot. Sharing an encoder size is the same
kind of coincidence as two papers both using a 3x3 convolution.

Its Qwen3-8B numbers: Best-of-8 GSM8K 93.8 at AUROC 0.682, MATH-500 85.3 at
AUROC 0.577, scaling to 95.0 at N=64 on GSM8K. Generators run non-thinking,
float16, temperature 0.7, top-p 0.9, trained on 64 self-generated candidates per
problem with outcome labels. Our AUROC of 0.831 to 0.895 is well above their
0.577 to 0.682 on the same backbone size, and the two are **not** directly
comparable: theirs ranks whole candidates, ours discriminates steps. Say that
rather than quote the gap.

**The gap none of them fill.** HSRM reports no self-consistency baseline at all.
ReProbe reports majority voting on GSM8K, StrQA and SciQA only, and leaves the
MATH cell empty. Where ReProbe does report it, the comparison is brutal:

| Qwen3-8B, N=10 | GSM8K |
|---|---|
| pass@1 | 95.6 |
| **majority voting** | **97.6** |
| Qwen2.5-Math-PRM-7B (7B params) | 97.8 |
| ReProbe, Attn+Logit, DeepSeek-anno | 97.8 |
| pass@10 ceiling | 99.2 |

A 7B process reward model and a 9.8M probe both buy **+0.2 points** over free
majority voting. That is the honest state of the art on GSM8K at N=10, and it is
the number our +0.48 at N=10 should be read against.

## 2. Process reward models: the verifiers the field deploys

Best-of-8 with Qwen2.5-Math-7B-Instruct as the policy, from *The Lessons of
Developing Process Reward Models in Mathematical Reasoning*, arXiv 2501.07301:

| verifier | GSM8K | MATH | Minerva | GaoKao | Olympiad | College | MMLU STEM | **Avg** |
|---|---|---|---|---|---|---|---|---|
| pass@8 ceiling | 98.1 | 92.0 | 49.3 | 80.5 | 59.6 | 52.6 | 90.5 | 74.7 |
| **maj@8 (free)** | 96.7 | 87.1 | 41.2 | 72.5 | 44.4 | 47.8 | 73.8 | **66.2** |
| Qwen2.5-Math-PRM-7B | 97.1 | 88.0 | 42.6 | 74.5 | 47.6 | 48.7 | 74.5 | **67.6** |
| Math-Shepherd-PRM-7B | 97.3 | 85.4 | 37.9 | 70.6 | 40.4 | 47.2 | 70.5 | 64.2 |
| RLHFlow-PRM-Mistral-8B | 97.0 | 86.1 | 37.1 | 70.6 | 41.2 | 47.6 | 69.5 | 64.2 |
| RLHFlow-PRM-Deepseek-8B | 97.3 | 86.3 | 40.8 | 70.9 | 42.2 | 47.2 | 69.3 | 64.9 |
| Skywork-PRM-7B | 97.3 | 87.3 | 38.2 | 71.9 | 43.7 | 47.8 | 67.7 | 64.8 |
| EurusPRM-Stage1 | 95.6 | 83.0 | 35.7 | 66.2 | 38.2 | 46.2 | 66.6 | 61.6 |
| EurusPRM-Stage2 | 95.4 | 83.4 | 34.9 | 67.3 | 39.1 | 46.3 | 67.3 | 62.0 |

**Six of the seven published PRMs lose to majority voting on average.** Only
Qwen2.5-Math-PRM-7B beats it, by 1.4 points. This is the single most useful
slide in the whole literature for us, because it establishes that beating
self-consistency at all is the bar, not a formality.

Step-level error identification is a different task and the ordering there is
not the same. ProcessBench mean F1: Qwen2.5-Math-PRM-72B 78.3,
Qwen2.5-Math-PRM-7B 73.5, Math-Shepherd-PRM-7B 31.5. QwQ-32B as a prompted
critic outperforms every PRM including the 72B.

The PRM we have downloaded, **Qwen2.5-Math-PRM-7B**, is the right choice: it is
the only one in the table that clears majority voting, and it is the one both
ReProbe and HSRM use as their strong baseline.

`Rethinking Reward Models for Multi-Domain Test-Time Scaling` (arXiv 2510.00492,
TMLR 2026) adds a caution worth one line in the deck: across 14 domains,
discriminative ORMs perform on par with discriminative PRMs, and generative ORMs
are the most robust of the four. Step-level supervision is not self-evidently
better outside math.

## 3. Aggregation rules: the family §21.8 measured

| rule | origin | headline number |
|---|---|---|
| self-consistency | Wang et al., arXiv 2203.11171, ICLR 2023 | +17.9 points on GSM8K with PaLM-540B |
| verifier-weighted vote | Li et al. DIVERSE, arXiv 2206.02336, ACL 2023 | GSM8K 74.4 to 83.2 on code-davinci-002 |
| best-of-N with a PRM | Lightman et al., arXiv 2305.20050, ICLR 2024 | MATH 78.2 at best-of-1860, ORM 72.4 |
| self-certainty plus Borda | Kang et al., arXiv 2502.18581 | MATH N=64: SC 63.40, Borda 64.10 |
| DeepConf filter then weighted vote | arXiv 2508.15260 | AIME-25 99.9 at N=512, 84.7% fewer tokens |
| Consilience | arXiv 2608.09898 | confidence-based selection collapses on hard tasks |
| adaptive-consistency | Aggarwal et al., arXiv 2305.11860, EMNLP 2023 | 7.9x fewer samples, <0.1% accuracy drop |
| GenSelect | arXiv 2507.17797 | LLM reasons over N candidates and picks |

**Lightman et al. corroborate §21.8 directly.** Their own text: they experimented
with RM-weighted voting to combine the PRM with majority voting, and it *did not
noticeably improve performance*. We found the same thing three years later with
a different verifier and a paired interval on it. That is a citation, not a
coincidence, and it should be on the slide.

The counterweight, which we must state ourselves before anyone else does:
Lightman's PRM **beats** majority voting at every N and the gap **widens** with
N. Our rerank does the opposite, degrading on math500 past N=6. The difference
is verifier quality, and that is exactly what the PRM run will price.

Two numbers for calibrating how large a real lift is at large N:

- self-certainty Borda over self-consistency, Llama-3.1-8B-Instruct, N=64: MATH
  +0.70 points, GSM8K +0.08 points;
- STEP over self-consistency at N=64: +0.4 to +5.0 points depending on benchmark.

## 4. Search and sequential scaling: the axis we are not on

Relevant because the director will ask, and because one of them is our §21
rejection loop under another name.

| work | method | result |
|---|---|---|
| Snell et al., arXiv 2408.03314 | compute-optimal allocation between revision and search | 4x more efficient than best-of-N; beats a 14x larger model FLOPs-matched |
| Liu et al., arXiv 2502.06703 | compute-optimal TTS across policies and PRMs | a 1B policy surpasses a 405B on MATH-500 |
| Beeching et al., HF search-and-learn | beam search and DVTS over a PRM | the reference open implementation |
| Puri et al., arXiv 2502.01618 | particle filtering over a PRM | 4 to 16x better scaling rate than deterministic search |
| **Zhang et al. LATTS, arXiv 2509.20368** | **per-step verifier acceptance: resample, backtrack, restart or stop** | **beats beam search by ~15% at 1e5 tokens; 10x fewer tokens at fixed accuracy; ~20 verifier calls per problem against beam search's ~80** |
| Muennighoff et al. s1, arXiv 2501.19393, EMNLP 2025 | budget forcing | AIME24 50 to 57 by appending "Wait" |
| Brown et al., arXiv 2407.21787 | repeated sampling | coverage is log-linear in N; **selection methods plateau past a few hundred samples** |

**LATTS is our §21 online rejection arm, generalised and published.** Same
primitive: score a step, and if the verifier objects, spend more compute on that
step. Theirs adds backtrack and restart actions and an adaptive acceptance
threshold. Their verifier is a 7B PRM and their policy is Llama-3.2-1B, so the
verifier is larger than the policy, which is the opposite of our regime. Our §21
plain-versus-rejection comparison needs to cite them and say what differs: we
price the retry loop with a blind control, which they do not, and our verifier
is 1,700x smaller than theirs.

Brown et al.'s plateau finding is the honest frame for our own ceiling trace:
coverage keeps rising with N, selection does not keep up, and that gap is the
open problem for everyone, not a weakness of our rule.

## 5. Where this leaves us

**What is genuinely ours.**

Every paper above reports at N greater than or equal to 8, and most report at
N=64 or beyond: HSRM N=8, ReProbe N=10, the Lessons paper N=8, self-certainty
N=8 to 64, STEP N=64, DeepConf N=512, Lightman N up to 1860. **Nobody reports
N=2 to 4.** Our §21 result lives there, and the lift there is an order of
magnitude larger than anything published at large N:

| our lift over self-consistency | gsm8k | math500 |
|---|---|---|
| N=2 | **+5.66** points | **+6.02** points |
| N=4 | +1.37 | +3.18 |
| N=10 | +0.48 | +0.07 (interval spans zero) |

against a published band of +0.08 to +0.9 points at N=64 for confidence rules,
+0.2 for a 7B PRM and for ReProbe at N=10, and +1.4 average for the best PRM at
N=8. The small-N regime is not a limitation of our study, it is the only place
in this literature where a verifier buys something large, and it is the regime a
practitioner on a budget actually occupies.

Second, the rule-versus-scorer separation. The field reports one rule per paper
and treats it as the method. §21.8 sweeps the rule family with the scorer held
fixed and finds the rule matters more than the scorer at small N. No paper in
this list does that.

**What is not ours and must be said.** The weighted vote is Lightman's and
DIVERSE's. Per-step rejection is LATTS's. The confidence statistics are
DeepConf's. The small-encoder block is standard, and HSRM (31 August 2026)
reached a trace-level version of it independently, one week after our grid
landed here; cite it as concurrent work and state the difference in what the
encoder reads.

**What the comparison still needs**, in order:

1. **Qwen2.5-Math-PRM-7B on our traces.** It is the only PRM in the literature
   that beats majority voting, both competitor probe papers use it as their
   strong baseline, and until it runs our verifier has never met a real one.
2. **The N=2 to 4 regime reported for a PRM**, which no paper has done. If our
   small-N lift survives with a 7B PRM in the same plot, that is the sprint 8
   result.
3. **A self-consistency baseline on the competitor benchmarks**, since HSRM
   omits it entirely and ReProbe omits it on MATH. Re-reporting HSRM's setup
   with the baseline it skipped is cheap and is a contribution on its own.
4. **LATTS as the named comparator for §21's online arm**, replacing the
   "nobody has done this" framing, which is now false.

Deferred: beam search, DVTS and particle filtering all need regeneration under a
different sampler and belong to a later sprint.
