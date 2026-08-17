# Related Work — Representations for Step-Level Correctness

Literature map for the unified representation study. Each entry is tagged by how
it relates to our leaderboard:

- **[COMPARABLE]** — same task, data, and eval (ProcessBench first-error F1); its
  reported score goes on the leaderboard as an external reference.
- **[REPRODUCE]** — same task but different data/eval or a design that is a
  rep x learner inside our framework; we can reproduce it under our protocol.
- **[CONCEPTUAL]** — motivates the direction; no comparable leaderboard number.
- **[BENCHMARK / DATA]** — a dataset or benchmark we build on.

Our positioning: move the focus from *detector design* to *representation
design*. Before adding classifier capacity, determine which learned
representation most clearly exposes step correctness, under one frozen protocol
(see `data_setup.md`).

---

## 1. Directly related: hidden-state representations for step correctness

### ReProbe (Ni et al., 2025) — arXiv:2511.06209, ACL 2026  **[REPRODUCE]**
Predicts step correctness from frozen internal states with a small
(<10M-param) **transformer probe** over **all tokens of a step** using **all
layers** (or attention+logits), projected down, encoded, then mean-pooled over
the step and classified. Trained on ~32k LLM-generated trajectories seeded from
10,800 PRM800K problems, with **LLM-judge (DeepSeek-R1) step labels** (not the
human PRM800K labels we use). Eval is in-domain MATH/GSM8K/ProofNet and OOD
planning/QA, **PR-AUC** — better OOD than in-domain, but all under the same
LLM-judge annotation, *not* human-labeled ProcessBench.
- **Why not COMPARABLE:** different labels (LLM-judge vs human), different metric
  (PR-AUC vs F1_PB), different eval sets (not ProcessBench).
- **Most direct competitor.** Its design confounds representation with a powerful
  transformer detector. In our framework it is exactly `token_store` (all last-
  layer tokens) x `transformer` learner. We reproduce it under our protocol to
  isolate representation from detector capacity. (Its all-layers variant needs a
  multi-layer store; last-layer all-tokens is a direct subset we already store.)

### Hidden States as Early Signals (Liang et al., 2026) — arXiv:2601.09093  **[REPRODUCE]**
Final-layer hidden state at each step boundary + a 2-layer MLP; scores accumulate
along a trace to prune trajectories at test time (45-70% latency cut, better
final accuracy on AIME/HMMT/GPQA).
- **Why not COMPARABLE:** labels are **trace-level correctness propagated to every
  step** (an early-correct step in a failing trace is labeled incorrect), and the
  paper reports **no step-classification metrics** (no F1/AUROC), only downstream
  pruning/latency. It is a trace-success predictor at checkpoints, not a step
  classifier.
- In our framework it is `last_token` x `mlp` learner. Trivial to include as an
  MLP-learner row on our proper step labels.

### Hidden Error Awareness: Diagnostic, not Causal (Yuan et al., 2026) — arXiv:2605.09502, ICML 2026 MI workshop  **[CONCEPTUAL]**
Correctness is linearly decodable from hidden states, but causal interventions on
the detected signal do not reliably improve reasoning.
- Strong conceptual anchor: **decodability != causal usefulness**. Directly echoes
  our own S3 Stage-5 result (additive steering of the probe direction is null; the
  direction is a diagnostic readout, not a lever). No comparable leaderboard number.

---

## 2. Adjacent: step correctness without a hidden-state representation

### ThinkPRM (Khalifa et al., 2025) — arXiv:2504.16828  **[COMPARABLE, harder subsets]**
Generative verifier that writes an explicit verification chain before judging,
using only ~1% of PRM800K process labels. SOTA-class on ProcessBench.
- Grounded ProcessBench F1: **ThinkPRM-14B OlympiadBench 87.3, OmniMath 85.7**;
  ThinkPRM-1.5B 76.3 / 75.7. (GSM8K/MATH per-subset not isolated in the paper's
  tables we retrieved.) Comparable metric, but on the *harder* subsets we have not
  yet encoded; use as the SOTA ceiling reference.

### Let's Verify Step by Step / PRM800K (Lightman et al., 2023) — arXiv:2305.20050, ICLR 2024  **[DATA]**
Introduces process supervision and releases PRM800K human step labels — our
in-domain training source.

### ProcessBench (Zheng et al., 2024) — arXiv:2412.06559, ACL 2025  **[BENCHMARK]**
Human-annotated earliest-error benchmark. Our OOD eval and the axis on which the
external PRM references below are directly comparable.

### Circuit-based Reasoning Verification / CRV (Zhao et al., 2025) — arXiv:2510.09312, ICLR 2026 oral  **[CONCEPTUAL]**
Represents a trace as an attribution graph / circuit and classifies correctness
from its structure. Same problem, structured (not raw hidden-state)
representation. Noted as very heavy to reproduce; not a near-term leaderboard row.

### PRMBench (Song et al., 2025) — arXiv:2501.03124, ACL 2025  **[BENCHMARK]**
Fine-grained PRM benchmark across error types; a future probe of whether a
representation captures genuine correctness vs dataset-specific patterns.

### One-Token Verification / OTV (Zhuang et al., 2026) — arXiv:2603.01025  **[REPRODUCE, out-of-store]**
A learnable verification token (LoRA, base frozen) attends over the KV cache and
emits a correctness score in one forward pass. Representation is *implicit*
(whatever the token retrieves), not an explicit hidden-state embedding.
- Reports only downstream metrics (Best-of-N, pass@k, up to 90% token savings),
  **no step-classification F1/AUROC** — so it is impossible to know if success
  comes from a discriminative representation or a sufficient ranking signal.
- Reproducible but outside the frozen-probe store (it trains a LoRA token).

---

## 3. Representation learning and mechanistic interpretability

### Reasoning Models Know When They're Right (Zhang et al., 2025) — arXiv:2504.05419  **[CONCEPTUAL]**
Linear probe on the final-layer hidden state at the *intermediate-answer*
position; ROC-AUC >0.7 (some >0.9), predictable before the answer is finished,
enables early exit (up to 24% token savings). Task is intermediate-**answer**
correctness, not step logical correctness. Motivated our long-CoT Qwen choice.

### CLUE (Liang et al., 2025) — arXiv:2510.01591  **[REPRODUCE]**
Non-parametric: represents a whole trace by the **activation delta**
`h(end) - h(start)` across all layers, then nearest-centroid (correct vs
incorrect). No trainable parameters; strong reranking on AIME/GPQA.
- Trace-level, not step-level, so not directly comparable. Very close to our
  **step_delta / transition** representation: in our framework it is the `step_delta` rep x
  a nearest-centroid learner, localized to steps. Reproduce to test whether the
  centroid geometry survives at step granularity.

### Step-level Sparse Autoencoder / SSAE (Yang et al., 2026) — arXiv:2603.03031, ICML 2026  **[REPRODUCE, in registry]**
Compresses a step into a sparse latent + classifier. Already reproduced by us:
their reported labels measure reconstruction fidelity, not correctness; with
Math-Shepherd labels we still found a strong linearly decodable signal. It is a
`sparse` representation in our registry.

### How Does Chain of Thought Think? (Chen et al., 2025) — arXiv:2507.22928  **[CONCEPTUAL]**
SAEs comparing activations with/without CoT; some sparse features are causally
transferable. Central mechanistic reference for the sparse direction.

### Towards Monosemanticity (Bricken et al., 2023) — Transformer Circuits  **[CONCEPTUAL]**
Dictionary learning / SAE foundation for interpretable decomposition of
polysemantic activations.

### Iteration Head (Cabannes et al., 2024) — arXiv:2406.02128  **[CONCEPTUAL]**
Mechanistic evidence that CoT reorganizes internal computation (attention +
MLP retrieve/update intermediate state), not just adds output text.

### Method families we may draw on
- **Contrastive / metric learning** (SimCLR, Chen et al. 2020, ICML; plus
  supervised-contrastive / deep-metric variants): shape the space so
  correct/incorrect continuations from the same context separate. Maps to our
  fork-based `*_rank` / `*_triplet` registry methods.
- **Self-supervised** on unlabeled traces (next-state prediction, masked
  transition reconstruction, real-vs-corrupted transition). Relevant but needs a
  concrete objective; our Future-SSAE next-step objective is one instance.
- **Information Bottleneck** (Tishby et al., 2000): theoretical framing for a
  compact transform that keeps correctness and drops nuisance (wording, length,
  position). Motivation, not a baseline unless we implement a (variational)
  bottleneck.
- **Learned autoencoders** (non-sparse): compress activations for predictive
  utility/invariance rather than interpretability; an alternative to SAEs when the
  objective is performance.

---

## 4. Background courses (mechanistic interpretability)
- Causal Mechanistic Interpretability — Atticus Geiger (Stanford, lecture 1)
- Computational Motifs — Jack Merullo (Stanford, lecture 2)
- An Introduction to Mechanistic Interpretability — Neel Nanda (IASEAI 2025)
- We Can Monitor AI's Thoughts... For Now — Neel Nanda (Google DeepMind)
- The Dark Matter of AI [Mechanistic Interpretability]

---

## 5. What lands on the leaderboard

**External reference scores (ProcessBench F1, directly comparable metric+task+data).**
Same first-error harmonic-mean F1 on the same human-labeled ProcessBench solutions
as our F1_PB. They differ from us only in being fine-tuned 7B PRMs (vs our frozen-
state probe) and in threshold selection. See `leaderboard.md`.

| system | GSM8K | MATH | avg (4 subsets) | source |
|---|---|---|---|---|
| Math-Shepherd-PRM-7B | 47.9 | 29.5 | 31.5 | ProcessBench paper |
| Skywork-PRM-7B | 70.8 | 53.6 | 42.1 | ProcessBench paper |
| Qwen2.5-Math-7B-PRM800K | 68.2 | 62.6 | 56.5 | ProcessBench paper |
| ThinkPRM-14B | — | — | Olympiad 87.3 / Omni 85.7 | ThinkPRM paper |

**Reproduce inside this framework (same protocol, one rep x learner each).**

| paper | our realization | status |
|---|---|---|
| ReProbe | `step_tokens` x transformer learner (all last-layer step tokens) | DONE (top single row on the leaderboard) |
| CLUE | `step_delta` rep x nearest-centroid learner (step-localized) | `step_delta` derived; centroid learner next |
| SSAE | `sparse` rep x linear/MLP | in registry (reproduced) |
| Hidden States as Early Signals | `last_token` x MLP learner | trivial (MLP learner) |
| OTV | LoRA verification token | out of frozen-store scope |

**Caveats.** All four ProcessBench subsets are now encoded, so the 4-subset average
in `leaderboard.md` is directly comparable to the reference column above.
The reference PRMs are fully fine-tuned models; our rows are frozen-state probes.

Sources: [ProcessBench (alphaXiv)](https://www.alphaxiv.org/overview/2412.06559v4),
[ThinkPRM](https://arxiv.org/abs/2504.16828).
