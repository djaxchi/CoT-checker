# Reading map: hidden-state verifiers and the downstream use of them

*2026-09-22. Assembled for the literature-review update. Cutoff is **July 2026**:
nothing submitted after 31 July 2026 appears below.*

## 0. How to read this

The review has two threads, and the single most useful structural observation is
that **for most of their history the two threads did not cite each other.**

- **Thread A**, probing a frozen model's internal states, grew out of
  hallucination and truthfulness detection. Its evaluation is AUROC on a
  detection task. Almost none of it before 2025 mentions best-of-N, voting, or
  any downstream decision.
- **Thread B**, test-time scaling, grew out of math verifiers. Its evaluation is
  downstream accuracy at a sample budget. Almost none of it reads hidden states;
  the verifier is a second language model.

The bridge papers are recent and few: ReProbe, PHSV, ELHSR, STEP. That junction
is where our work sits, and saying so explicitly is a stronger framing for the
review than treating either thread as the background.

A second observation worth stating early: **Thread B's own limits literature is
the most underread part of it** and is the most useful to us, because it
explains why every verifier in Thread B buys so little.

Tiers: **[1]** read closely, **[2]** read the results section, **[3]** cite and
move on.

---

# Thread A: trained verifiers on hidden states

## A1. The lineage, which the review needs for provenance

These establish that a linear or shallow probe on frozen activations recovers
correctness-like information. None of them does test-time scaling. Cite them for
the premise, not for the method.

| # | work | id | what it established |
|---|---|---|---|
| **[2]** | Azaria & Mitchell, SAPLMA | 2304.13734 | a classifier on hidden states predicts statement truth; the first clean version of the claim |
| **[3]** | Burns et al., CCS | 2212.03827 | latent truth direction found without labels; the unsupervised counterpoint |
| **[2]** | Chen et al., INSIDE / EigenScore | 2402.03744, ICLR 2024 | eigenvalues of the response covariance in embedding space as a hallucination score; sampling-based, not a trained probe |
| **[1]** | Orgad et al., LLMs Know More Than They Show | 2410.02707, ICLR 2025 | truthfulness is concentrated in **specific tokens**, error detectors **do not generalise across datasets**, and models often encode the right answer while emitting the wrong one |
| **[2]** | Kossen et al., Semantic Entropy Probes | 2406.15927 | a probe on hidden states approximates semantic entropy without sampling; the cheap-proxy pattern our work reuses |
| **[3]** | Farquhar et al., semantic entropy | Nature, Jun 2024 | the sampling-based uncertainty measure the probes approximate |

Orgad is the one to read closely. Its two findings, token-localisation and
cross-dataset generalisation failure, are exactly the confounds a step-level
verifier has to answer for, and its framing is the cleanest in the thread.

## A2. Internal-state verifiers used for a downstream decision

This is the direct competitor set. Every one of these is 2025 or later.

| # | work | id | probe | signal | downstream use | self-consistency baseline |
|---|---|---|---|---|---|---|
| **[1]** | ReProbe / UHeads | 2511.06209, ACL 2026 | transformer, 9.8M | attention to 1-3 prior tokens plus top-K logits | best-of-N and beam search | partial: 3 of 8 datasets, absent on MATH |
| **[1]** | Zhang et al., PHSV, *Reasoning Models Know When They're Right* | 2504.05419 | MLP per reasoning chunk | hidden state per chunk | early exit, 24% token saving | no |
| **[2]** | Guo et al., ELHSR | 2505.12225, KDD 2026 | linear per-token head | hidden states, or logits alone | best-of-N | reward models only |
| **[2]** | STEP | 2601.09093 | 2-layer MLP, 512 wide | last-layer state at `\n\n` | trace pruning under memory pressure | **yes**, +0.4 to +5.0 at N=64 |
| **[2]** | VerifySteer | 2605.20745 | correctness probe plus latent steering | paragraph-boundary states | routing and strictness control | **yes**, +3.7 F1 at 4x less compute |

Read ReProbe and PHSV closely. ReProbe is the paper our leaderboard is measured
against; PHSV is the one that shows the hidden state carries **look-ahead**
information, correctness predictable before the answer is articulated, which is
the strongest version of the premise and directly relevant to the lookahead
representation thread.

## A3. What Thread A does not do

Worth one paragraph in the review, because it is the honest gap:

1. **It does not report self-consistency.** Only STEP and VerifySteer do.
   A probe that beats a PRM but not counting has not been shown to be useful.
2. **It reports at one budget.** N=8, N=10 or N=64, never a sweep from N=1.
3. **It conflates the scorer with the rule.** Each paper picks one selection
   rule and reports it as the method, so a scorer's contribution cannot be
   separated from the rule's.
4. **Generalisation is known to be weak** (Orgad) and is rarely tested by the
   verifier papers.

---

# Thread B: the downstream use, test-time scaling

## B1. The verifier lineage, ORM to PRM

| # | work | id | why |
|---|---|---|---|
| **[1]** | Cobbe et al., GSM8K verifiers | 2110.14168 | introduces the dataset and the outcome verifier; the origin of best-of-N in this literature |
| **[2]** | Uesato et al. | 2211.14275 | first careful process-versus-outcome supervision comparison |
| **[1]** | Lightman et al., *Let's Verify Step by Step* | 2305.20050, ICLR 2024 | PRM800K; PRM beats ORM and majority voting with a gap that **widens** in N; and their own note that reward-model-weighted voting **did not noticeably improve performance** |
| **[2]** | Wang et al., Math-Shepherd | 2312.08935 | automatic step labels from completion rollouts, removing the human annotator |
| **[3]** | Luo et al., OmegaPRM | 2406.06592 | MCTS-based automatic annotation, 1.5M labels |
| **[2]** | Zhang et al., GenRM | 2408.15240 | verification as next-token prediction; GSM8K best-of-N 73 to 93.4 |
| **[2]** | Setlur et al., PAV, *Rewarding Progress* | 2410.08146, ICLR 2025 | reward **progress** under a prover policy rather than absolute correctness; beam search over PAVs is >8% better and 1.5-5x more compute efficient than re-ranking against an ORM |
| **[1]** | Zhang et al., *The Lessons of Developing PRMs* | 2501.07301 | Qwen2.5-Math-PRM-7B, and the best-of-8 table where **six of seven published PRMs lose to maj@8** |
| **[2]** | Lee et al., *Rethinking Reward Models for Multi-Domain TTS* | 2510.00492, TMLR 2026 | across 14 domains, discriminative ORM ties discriminative PRM and generative ORM is most robust |
| **[3]** | PRM survey | 2510.08049 | map of the space if you want one reference instead of eight |

Lightman and the Lessons paper are the two to read closely. Lightman is the
method; the Lessons paper is the audit that shows how little it generalises.

## B2. Aggregation: what to do with N candidates

| # | work | id | rule |
|---|---|---|---|
| **[1]** | Wang et al., self-consistency | 2203.11171, ICLR 2023 | majority vote; the baseline everything is measured against |
| **[2]** | Li et al., DIVERSE | 2206.02336, ACL 2023 | verifier-weighted voting plus step-aware verification |
| **[2]** | Kang et al., self-certainty plus Borda | 2502.18581 | rank-weighted voting on a KL-to-uniform confidence; MATH N=64, 63.40 to 64.10 |
| **[1]** | Zhao et al., DeepConf | 2508.15260 | group confidence, filter the worst fraction, then confidence-weighted vote; AIME-25 99.9 at N=512 with 84.7% fewer tokens |
| **[2]** | Moshkov et al., GenSelect | 2507.17797 | the model reasons over all N candidates and picks, instead of scoring each |
| **[2]** | Li et al., ESC | 2401.10480, ICLR 2024 | stop sampling when a window agrees; GSM8K -80.1% samples |
| **[2]** | Aggarwal et al., adaptive-consistency | 2305.11860, EMNLP 2023 | the same idea with a stopping criterion; 7.9x fewer samples at <0.1% drop |

ESC and adaptive-consistency matter more than their citation counts suggest:
they are the only papers that treat **N as a per-problem decision**, which is
the natural extension of a small-N result.

## B3. Search and sequential scaling

| # | work | id | mechanism |
|---|---|---|---|
| **[2]** | Wu et al., inference scaling laws, REBASE | 2408.00724, ICLR 2025 | first compute-optimal formulation; REBASE tree search is Pareto-optimal at all budgets, 7B usually the optimal size |
| **[1]** | Snell et al. | 2408.03314 | compute-optimal allocation between revision and search; 4x over best-of-N; beats a 14x larger model FLOPs-matched |
| **[2]** | Liu et al., *Can 1B Surpass 405B?* | 2502.06703 | the same sweep across many policies and PRMs; the strategy's effectiveness is tied to the specific policy-PRM pair |
| **[2]** | Guan et al., rStar-Math | 2501.04519 | MCTS with a process preference model; small models reach o1 on MATH |
| **[3]** | Zhang et al., ReST-MCTS* | 2406.03816 | process-reward-guided tree search for self-training |
| **[2]** | Puri et al., particle filtering | 2502.01618 | inference scaling as probabilistic inference; 4-16x better scaling rate than deterministic search |
| **[1]** | LATTS | 2509.20368 | **per-step verifier acceptance: resample, backtrack, restart or stop.** This is our online rejection arm generalised, and the closest prior art to it |
| **[2]** | Muennighoff et al., s1 | 2501.19393, EMNLP 2025 | budget forcing; sequential scaling with no verifier at all |
| **[2]** | *Reject, Resample, Repeat* | 2603.07887 | SMC framing that connects rejection sampling, particle filtering and best-of-N under one analysis, with conditions on PRM accuracy |
| **[3]** | step-level verifier-guided hybrid TTS | 2507.15512 | combines step verification with sampling on MATH500 and GSM8K |

LATTS and the SMC paper are the two that matter for the online arm. The SMC
paper in particular gives the theory for when per-step rejection should work,
stated as bounds on the verifier's accuracy, which is a better framing for our
rejection results than the empirical one we have.

## B4. The limits, and why this section matters most

Thread B's negative results explain Thread B's small effect sizes. The review is
much stronger if this section is present, because it reframes a modest lift from
a weakness into the expected outcome.

| # | work | id | finding |
|---|---|---|---|
| **[1]** | Brown et al., *Large Language Monkeys* | 2407.21787 | coverage is log-linear in N across four orders of magnitude, but **majority voting and reward models plateau past a few hundred samples**; the verifier, not the sampler, is the bottleneck |
| **[1]** | Stroebl et al., *Inference Scaling fLaws* | 2411.17501 | with an imperfect verifier, resampling cannot drive the false-positive rate down, so there is a **hard accuracy ceiling independent of budget**; also bounds rejection-sampling data curation |
| **[2]** | Huang et al., *Is Best-of-N the Best of Them?* | 2503.21878, ICML 2025 | best-of-N at large N **provably suffers reward hacking**; optimal only at a well-chosen N; InferenceTimePessimism fixes it |
| **[2]** | Huang et al., *LLMs Cannot Self-Correct Reasoning Yet* | 2310.01798, ICLR 2024 | intrinsic self-correction without external feedback does not improve reasoning; the reason a verifier is needed at all |

The Huang ICML paper is the theoretical statement of our own empirical result:
rerank degrades as N grows. Citing it turns our finding from an anomaly into an
instance.

## B5. Benchmarks and evaluation instruments

| # | work | id | what it measures |
|---|---|---|---|
| **[1]** | ProcessBench | 2412.06559 | first-error localisation; Qwen2.5-Math-PRM-72B 78.3 F1, -7B 73.5, Math-Shepherd-7B 31.5, and a prompted QwQ-32B beats every PRM |
| **[2]** | PRMBench | 2501.03124, ACL 2025 | 6,216 problems, 83,456 step labels across simplicity, soundness, sensitivity; 25 models evaluated |
| **[3]** | TTS survey | 2503.24235 | what/how/where/how-well taxonomy; use it to check nothing is missing, not as a source |

---

## Reading order, if time is short

1. Lightman 2305.20050, for the method and for the weighted-voting note.
2. The Lessons paper 2501.07301, for the table where PRMs lose to counting.
3. Brown 2407.21787 and Stroebl 2411.17501, for why they lose.
4. ReProbe 2511.06209 and PHSV 2504.05419, for the competitor probes.
5. Orgad 2410.02707, for what a probe on activations actually generalises to.
6. LATTS 2509.20368 and *Reject, Resample, Repeat* 2603.07887, for the online arm.
7. Huang 2503.21878, for the theory behind rerank degrading in N.

## The cutoff

Excluded by the July 2026 line, listed so the decision is deliberate rather than
an omission:

- **HSRM**, 2608.30841, submitted 31 August 2026, EMNLP 2026. A trace-level
  hidden-state ranker on the Qwen3 family. Concurrent with our own work by any
  normal convention; our grid landed 24 August 2026.
- **Consilience for Verifier-Free TTS**, 2608.09898, August 2026.
- **When Self-Consistency Backfires**, 2608.11403, August 2026. Majority voting
  reduces accuracy on most hard GPQA-Diamond problems for small models.
