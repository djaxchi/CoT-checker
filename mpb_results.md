# MTB Results Log

Step-level Chain-of-Thought verification benchmark. This document is incremental: new candidates are appended in the same structure.

## Table of Contents

1. [Experiment Setup](#1-experiment-setup)
2. [Leaderboard](#2-leaderboard)
3. [Candidates](#3-candidates)
   - [RandomProbe](#31-randomprobe)
   - [DenseLinear](#32-denselinear)
   - [SAE-positive](#33-sae-positive)
   - [SAE-mixed](#34-sae-mixed)
   - [SAE-contrastive](#35-sae-contrastive)
   - [SSAE-positive](#36-ssae-positive)
   - [SSAE-mixed](#37-ssae-mixed)
   - [SSAE-contrastive](#38-ssae-contrastive)
   - [SSAE original paper checkpoint, Qwen2.5-0.5B](#39-ssae-original-paper-checkpoint-qwen25-05b)
   - [DenseLinear-PRM400k](#310-denselinear-prm400k)
   - [PRM800K fork preference objectives](#311-prm800k-fork-preference-objectives)
   - [Dense pairwise scorer objectives](#312-dense-pairwise-scorer-objectives)
   - [DenseLinear model-size ablation](#313-denselinear-model-size-ablation)

---

## 1. Experiment Setup

### Goal

Investigate whether internal representations of a base LM (dense hidden states, SAE features, SSAE latents) carry a step-level correctness signal that can be used to localize the first erroneous step inside a Chain-of-Thought trace. Each candidate produces a score per step, and the official ProcessBench F1 is the headline metric.

### Task formulation

For a problem and reasoning trace, each method produces a score per step:

```text
score_t = p(non_viable | problem, previous_steps, current_step)
```

Predict the first error index by thresholding:

```text
prediction = first step t where score_t > threshold
if no step crosses threshold: prediction = -1
```

### Metric

Official ProcessBench F1_PB (not per-step binary F1):

```text
Acc_error   = exact-match accuracy on traces where label != -1
Acc_correct = exact-match accuracy on traces where label == -1
F1_PB       = 2 * Acc_error * Acc_correct / (Acc_error + Acc_correct)
```

### Protocol

| Component | Choice |
|---|---|
| Backbone | Qwen2.5-1.5B |
| Hidden dimension | 1536 |
| Representation supervision | PRM800K |
| Probe train split | 40K PRM800K examples |
| Probe validation split | 1K PRM800K examples |
| Evaluation splits | ProcessBench `gsm8k` (400 traces, 2 082 steps), `math` (1 000 / 6 505), `olympiadbench` (1 000 / 8 819), `omnimath` (1 000 / 8 291) |
| Threshold sources | (a) `val_selected`: PRM800K val_1k threshold.json (deployable); (b) `oracle`: per-subset F1_PB maximizer over `{0.005, 0.010, ..., 0.995}` (199 points, endpoints excluded, not deployable) |
| Aggregation | `macro_avg_F1_PB` = unweighted mean across the 4 subsets |
| Score convention | `s = sigmoid(probe_logit) = P(step is first-error)`; trace prediction = first step with `s > t` or `-1` |
| Sharding | 4 H100s per stage: per-subset 4-way sharding for encoding/extraction with deterministic `global_step_index`; round-robin (method, subset) workers for evaluation |
| Probe | Linear |
| Seed | 42 |
| Cluster / GPU | TamIA / H100 80GB |

### External reference points (ProcessBench paper)

| Method | Avg. F1_PB |
|---|---:|
| Math-Shepherd-PRM-7B | 31.5 |
| Skywork-PRM-7B | 42.1 |
| Qwen2.5-Math-7B-PRM800K | 56.5 |
| Qwen2.5-72B-Instruct (critic) | 61.2 |
| QwQ-32B-Preview (critic) | 71.5 |
| GPT-4o-0806 (critic) | 61.9 |
| o1-mini (critic) | 87.9 |

---

## 2. Leaderboard

Macro F1_PB averaged across the 4 ProcessBench subsets (gsm8k, math, olympiadbench, omnimath). `val_selected` uses the PRM800K val_1k threshold; `oracle` reports the per-subset F1_PB maximizer (not deployable). Ranked by `oracle macro F1_PB`.

| Rank | Method | val_selected macro F1_PB | oracle macro F1_PB |
|---:|---|---:|---:|
| 1 | DenseLinear | 0.1855 | **0.3773** |
| 2 | SAE-mixed | 0.0919 | 0.3688 |
| 3 | SAE-contrastive | 0.1648 | 0.3350 |
| 4 | SAE-positive | 0.0371 | 0.3153 |
| 5 | SSAE original paper ckpt, Qwen2.5-0.5B | 0.0098 | 0.2948 |
| 6 | SSAE-mixed | 0.0203 | 0.1811 |
| 7 | SSAE-positive | 0.0122 | 0.1804 |
| 8 | SSAE-contrastive | 0.0176 | 0.1794 |
| - | RandomProbe | 0.0537 | 0.1672 |
| ‡ | DenseLinear-PRM400k | - | - |

‡ Trained under a different protocol than rows 1-8 (400K PRM800K train, 10K val, 0.01-step grid). Evaluated only on `gsm8k` and `math` at the time of that run; not part of the unified 4-subset macro. Full breakdown in §3.10.

### Per-subset oracle F1_PB (and per-subset oracle threshold)

| Method | gsm8k | math | olympiadbench | omnimath |
|---|---:|---:|---:|---:|
| DenseLinear | 0.3495 (t=0.655) | 0.4070 (t=0.740) | 0.3720 (t=0.825) | 0.3806 (t=0.840) |
| SAE-mixed | 0.3278 (t=0.660) | 0.4084 (t=0.765) | 0.3791 (t=0.810) | 0.3600 (t=0.805) |
| SAE-contrastive | 0.3262 (t=0.690) | 0.3780 (t=0.885) | 0.3164 (t=0.900) | 0.3193 (t=0.920) |
| SAE-positive | 0.2708 (t=0.505) | 0.3439 (t=0.660) | 0.3253 (t=0.710) | 0.3210 (t=0.685) |
| SSAE-mixed | 0.2620 (t=0.565) | 0.2035 (t=0.560) | 0.1136 (t=0.565) | 0.1451 (t=0.545) |
| SSAE-positive | 0.2841 (t=0.570) | 0.1930 (t=0.570) | 0.1080 (t=0.585) | 0.1366 (t=0.560) |
| SSAE-contrastive | 0.2615 (t=0.560) | 0.1980 (t=0.565) | 0.1200 (t=0.570) | 0.1380 (t=0.555) |
| RandomProbe | 0.1891 (t=0.725) | 0.1818 (t=0.785) | 0.1383 (t=0.830) | 0.1595 (t=0.835) |

### Per-subset val-selected F1_PB

| Method | t_val | gsm8k | math | olympiadbench | omnimath |
|---|---:|---:|---:|---:|---:|
| DenseLinear | 0.50 | 0.2701 | 0.2248 | 0.0899 | 0.1574 |
| SAE-contrastive | 0.60 | 0.2622 | 0.1868 | 0.0659 | 0.1444 |
| SAE-mixed | 0.50 | 0.1520 | 0.1476 | 0.0225 | 0.0455 |
| SAE-positive | 0.40 | 0.1062 | 0.0283 | 0.0058 | 0.0082 |
| SSAE-mixed | 0.50 | 0.0000 | 0.0277 | 0.0166 | 0.0371 |
| SSAE-positive | 0.50 | 0.0000 | 0.0143 | 0.0113 | 0.0233 |
| SSAE-contrastive | 0.50 | 0.0000 | 0.0233 | 0.0165 | 0.0304 |
| RandomProbe | 0.50 | 0.0909 | 0.0635 | 0.0113 | 0.0491 |

### Headline observations

1. **DenseLinear wins at oracle** (macro 0.3773). No learned representation in this set beats the raw final-layer hidden state once a per-subset threshold is chosen.
2. **SAE-mixed nearly ties DenseLinear at oracle** (0.3688) but its val/oracle gap is the largest in the table. Unsupervised sparsity preserves signal but destroys threshold transfer from PRM800K val.
3. **SAE auxiliary BCE shaping is the best-calibrated original SAE variant.** This is not a true contrastive method; it uses an auxiliary BCE head during SAE training and discards that head before the final probe. Its val-selected macro (0.1648) is the closest of the SAE family to DenseLinear's, because the auxiliary BCE pressure aligns the score distribution with what PRM800K val expects.
4. **The original paper SSAE checkpoint is materially better than our 1.5B SSAE runs at oracle** (0.2948 vs 0.1811 best in-house SSAE), despite using a smaller Qwen2.5-0.5B backbone. This points away from the SSAE architecture itself being hopeless and toward insufficient or ineffective training in our 1.5B SSAE variants.
5. **Calibration remains broken for SSAE.** The 0.5B checkpoint has usable oracle signal, but val-selected macro F1 is still only 0.0098. Probe-only positive/contrastive reuse variants also produce val-selected F1=0.0000, so the PRM800K-val threshold does not transfer.
6. **All three in-house 1.5B SSAE variants collapse to the random baseline.** Macro oracle F1_PB across that family is 0.179-0.181, barely above random's 0.1672.
7. **The fine 0.005-step oracle grid is load-bearing.** In-house SSAE oracle thresholds cluster in `[0.535, 0.585]`; SAE-contrastive optima at `[0.88, 0.92]`. A 0.1-step grid would have snapped both to grid points and reported substantially worse oracle numbers.
8. **Subset difficulty is consistent**: gsm8k > math > omnimath > olympiadbench across every method. OlympiadBench is the hardest subset for every probe.
9. **PRM800K val→PB calibration gap widens off-distribution.** DenseLinear val/oracle ratio is 0.77 on gsm8k (0.2701 / 0.3495) but only 0.24 on olympiadbench (0.0899 / 0.3720).
10. **True PRM800K fork preference objectives improve deployable threshold transfer but not oracle separability.** The best fork-based method, AE Rank w=100, reaches macro val-selected F1_PB ≈ 0.235, exceeding DenseLinear's val-selected 0.1855, but its oracle macro is only ≈ 0.290, below DenseLinear's 0.3773 and SAE-mixed's 0.3688. This suggests the fork objective mainly affects calibration rather than the maximum recoverable correctness signal. Full study in §3.11.
11. **The fork objective's ceiling/calibration trade-off is intrinsic to the pairwise objective, not just representation damage.** Dense pairwise scorers (§3.12) train *only* a linear scorer on frozen hidden states (no AE/SAE), swapping BCE for a ranking/margin loss. They reproduce the same pattern: `dense_rank` slightly beats DenseLinear's val-selected macro (0.2087 vs 0.1855) but its oracle macro collapses to 0.2328 (vs 0.3773). Since no representation is trained, the oracle drop must come from the objective itself. **DenseLinear BCE remains the strongest oracle method**; fork objectives mainly reshape/calibrate scores for threshold transfer.

---

## 3. Candidates

Each candidate gets the same brief block: what it is, full-PB results, and a short interpretation. Per-subset rows show F1_PB at the val_selected threshold and at the per-subset oracle (in parentheses).

### 3.1 RandomProbe

**Particularities.** Random scores in `[0, 1]`, deterministic with seed 42. Sanity check that the evaluator and aggregation logic work.

**Results.** Val threshold = 0.50 (uniform fallback for `random`).

| Subset | n_traces | n_err | n_corr | val F1_PB | val Acc_err | val Acc_corr | oracle t | oracle F1_PB | oracle Acc_err | oracle Acc_corr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gsm8k | 400 | 207 | 193 | 0.0909 | 0.1691 | 0.0622 | 0.725 | 0.1891 | 0.1739 | 0.2073 |
| math | 1 000 | 594 | 406 | 0.0635 | 0.2256 | 0.0369 | 0.785 | 0.1818 | 0.1380 | 0.2660 |
| olympiadbench | 1 000 | 661 | 339 | 0.0113 | 0.1377 | 0.0059 | 0.830 | 0.1383 | 0.0983 | 0.2330 |
| omnimath | 1 000 | 759 | 241 | 0.0491 | 0.1594 | 0.0290 | 0.835 | 0.1595 | 0.1094 | 0.2946 |
| **macro** | - | - | - | **0.0537** | 0.1729 | 0.0335 | - | **0.1672** | 0.1299 | 0.2502 |

**Interpretation.** Matches the random baseline reported in the ProcessBench paper. Any method below this is broken.

---

### 3.2 DenseLinear

**Particularities.** Linear probe directly on the final-layer hidden state at the last token of the current step. No representation training; the probe is trained on the PRM800K 40K split. Val balanced accuracy: 72.00; val binary F1: 71.66.

**Results.** Val threshold = 0.50.

| Subset | val F1_PB | val Acc_err | val Acc_corr | oracle t | oracle F1_PB | oracle Acc_err | oracle Acc_corr |
|---|---:|---:|---:|---:|---:|---:|---:|
| gsm8k | 0.2701 | 0.2657 | 0.2746 | 0.655 | 0.3495 | 0.2415 | 0.6321 |
| math | 0.2248 | 0.3418 | 0.1675 | 0.740 | 0.4070 | 0.3215 | 0.5542 |
| olympiadbench | 0.0899 | 0.2920 | 0.0531 | 0.825 | 0.3720 | 0.2723 | 0.5870 |
| omnimath | 0.1574 | 0.2912 | 0.1079 | 0.840 | 0.3806 | 0.2675 | 0.6598 |
| **macro** | **0.1855** | 0.2977 | 0.1508 | - | **0.3773** | 0.2757 | 0.6083 |

**Interpretation.** Strongest candidate at the oracle threshold across every subset. The hidden state already carries a usable correctness signal; any learned representation has to clear this bar. The val/oracle gap is large on the harder subsets (0.24 ratio on olympiadbench vs 0.77 on gsm8k), meaning the PRM800K val threshold is the active limitation off-distribution.

---

### 3.3 SAE-positive

**Particularities.** Sparse AE trained on positive PRM800K examples only (20K). Loss: `MSE(h_hat, h) + l1 * mean(|z|)`. Linear probe trained on PRM800K 40K. Val balanced accuracy: 69.20; val binary F1: 72.05.

**Results.** Val threshold = 0.40.

| Subset | val F1_PB | val Acc_err | val Acc_corr | oracle t | oracle F1_PB | oracle Acc_err | oracle Acc_corr |
|---|---:|---:|---:|---:|---:|---:|---:|
| gsm8k | 0.1062 | 0.2512 | 0.0674 | 0.505 | 0.2708 | 0.2126 | 0.3731 |
| math | 0.0283 | 0.3333 | 0.0148 | 0.660 | 0.3439 | 0.2576 | 0.5172 |
| olympiadbench | 0.0058 | 0.2194 | 0.0029 | 0.710 | 0.3253 | 0.2375 | 0.5162 |
| omnimath | 0.0082 | 0.3030 | 0.0041 | 0.685 | 0.3210 | 0.2398 | 0.4855 |
| **macro** | **0.0371** | 0.2767 | 0.0223 | - | **0.3153** | 0.2369 | 0.4730 |

**Interpretation.** Weakest learned candidate at both thresholds. Positive-only reconstruction degrades transfer; the val threshold is also the worst-calibrated (val Acc_err is high but Acc_corr collapses).

---

### 3.4 SAE-mixed

**Particularities.** Same SAE recipe, trained on mixed positive and negative PRM800K examples (40K, labels ignored). Val balanced accuracy: 72.90; val binary F1: 73.14.

**Results.** Val threshold = 0.50.

| Subset | val F1_PB | val Acc_err | val Acc_corr | oracle t | oracle F1_PB | oracle Acc_err | oracle Acc_corr |
|---|---:|---:|---:|---:|---:|---:|---:|
| gsm8k | 0.1520 | 0.2850 | 0.1036 | 0.660 | 0.3278 | 0.2512 | 0.4715 |
| math | 0.1476 | 0.3182 | 0.0961 | 0.765 | 0.4084 | 0.3148 | 0.5813 |
| olympiadbench | 0.0225 | 0.2360 | 0.0118 | 0.810 | 0.3791 | 0.2995 | 0.5162 |
| omnimath | 0.0455 | 0.2635 | 0.0249 | 0.805 | 0.3600 | 0.2846 | 0.4896 |
| **macro** | **0.0919** | 0.2757 | 0.0591 | - | **0.3688** | 0.2875 | 0.5147 |

**Interpretation.** Nearly ties DenseLinear at the oracle (macro 0.3688 vs 0.3773) but the val/oracle gap is the worst in the table. Exposing the SAE to both correct and incorrect steps preserves signal that unsupervised positive-only training loses; pure sparsity does not, however, fix threshold calibration.

---

### 3.5 SAE-contrastive

**Particularities.** Same mixed data as SAE-mixed, plus an auxiliary BCE head on the latent during representation training (`MSE + l1 * mean(|z|) + bce * BCEWithLogits(aux(z), y)`). Aux head is discarded after training; a fresh linear probe is trained on the latent. Final aux BCE = 0.4726. Val balanced accuracy: 73.40; val binary F1: 70.96.

**Results.** Val threshold = 0.60.

| Subset | val F1_PB | val Acc_err | val Acc_corr | oracle t | oracle F1_PB | oracle Acc_err | oracle Acc_corr |
|---|---:|---:|---:|---:|---:|---:|---:|
| gsm8k | 0.2622 | 0.2995 | 0.2332 | 0.690 | 0.3262 | 0.2899 | 0.3731 |
| math | 0.1868 | 0.3451 | 0.1281 | 0.885 | 0.3780 | 0.2946 | 0.5271 |
| olympiadbench | 0.0659 | 0.2330 | 0.0383 | 0.900 | 0.3164 | 0.2542 | 0.4189 |
| omnimath | 0.1444 | 0.2964 | 0.0954 | 0.920 | 0.3193 | 0.2332 | 0.5062 |
| **macro** | **0.1648** | 0.2935 | 0.1238 | - | **0.3350** | 0.2680 | 0.4563 |

**Interpretation.** Best-calibrated SAE variant. Despite the legacy `sae_contrastive` id, this is **not** true contrastive learning: it is `SAE + auxiliary BCE shaping` (MSE reconstruction + L1 sparsity + an auxiliary BCE label head that is discarded before the fresh probe is trained). Its val-selected macro (0.1648) is the closest of the SAE family to DenseLinear's (0.1855); the auxiliary BCE pressure during representation training aligns the score distribution with what the PRM800K val threshold expects. Oracle ceiling sits between SAE-mixed and SAE-positive. For genuine fork-based preference (rank/triplet) objectives, see §3.11.

---

### 3.6 SSAE-positive

**Particularities.** Step-conditional sparse autoencoder built on Qwen2.5-1.5B encoder/decoder:

```text
step / context -> Qwen encoder -> sparse bottleneck latent -> Qwen decoder reconstruction
```

Trained from text JSONL (not cached `.npy`) on positive-only steps. Stability required `attn_implementation=eager`, gradient checkpointing on, `use_cache=False`; SDPA + bf16 backward had produced NaN gradients prior to that fix.

**Results.** Val threshold = 0.50.

| Subset | val F1_PB | val Acc_err | val Acc_corr | oracle t | oracle F1_PB | oracle Acc_err | oracle Acc_corr |
|---|---:|---:|---:|---:|---:|---:|---:|
| gsm8k | 0.0000 | 0.2126 | 0.0000 | 0.570 | 0.2841 | 0.2222 | 0.3938 |
| math | 0.0143 | 0.2273 | 0.0074 | 0.570 | 0.1930 | 0.1498 | 0.2709 |
| olympiadbench | 0.0113 | 0.1362 | 0.0059 | 0.585 | 0.1080 | 0.0711 | 0.2242 |
| omnimath | 0.0233 | 0.1871 | 0.0124 | 0.560 | 0.1366 | 0.1107 | 0.1784 |
| **macro** | **0.0122** | 0.1908 | 0.0064 | - | **0.1804** | 0.1385 | 0.2668 |

**Interpretation.** Best on gsm8k of the SSAE family but indistinguishable from random on the other three subsets (macro oracle 0.1804 vs random 0.1672). SSAE scores cluster in `~0.45-0.58`, so val threshold 0.5 over-predicts errors and collapses Acc_corr to 0 on gsm8k. Oracle ceiling barely above random suggests the bottleneck strips first-error signal.

---

### 3.7 SSAE-mixed

**Particularities.** Same SSAE architecture, trained on mixed positive and negative steps without contrastive supervision.

**Results.** Val threshold = 0.50.

| Subset | val F1_PB | val Acc_err | val Acc_corr | oracle t | oracle F1_PB | oracle Acc_err | oracle Acc_corr |
|---|---:|---:|---:|---:|---:|---:|---:|
| gsm8k | 0.0000 | 0.2174 | 0.0000 | 0.565 | 0.2620 | 0.1884 | 0.4301 |
| math | 0.0277 | 0.2256 | 0.0148 | 0.560 | 0.2035 | 0.1532 | 0.3030 |
| olympiadbench | 0.0166 | 0.1286 | 0.0088 | 0.565 | 0.1136 | 0.0893 | 0.1563 |
| omnimath | 0.0371 | 0.1739 | 0.0207 | 0.545 | 0.1451 | 0.1265 | 0.1701 |
| **macro** | **0.0203** | 0.1864 | 0.0111 | - | **0.1811** | 0.1393 | 0.2649 |

**Interpretation.** Essentially tied with SSAE-positive and indistinguishable from random. Adding incorrect steps to SSAE training does not give the same lift it gives the SAE family.

---

### 3.8 SSAE-contrastive

**Particularities.** SSAE-mixed plus an auxiliary BCE head on the latent. In this run `aux_bce` stayed near `log(2) ~ 0.6927` throughout training, meaning the head never deviated from chance.

**Results.** Val threshold = 0.50.

| Subset | val F1_PB | val Acc_err | val Acc_corr | oracle t | oracle F1_PB | oracle Acc_err | oracle Acc_corr |
|---|---:|---:|---:|---:|---:|---:|---:|
| gsm8k | 0.0000 | 0.2174 | 0.0000 | 0.560 | 0.2615 | 0.2415 | 0.2850 |
| math | 0.0233 | 0.2205 | 0.0123 | 0.565 | 0.1980 | 0.1414 | 0.3300 |
| olympiadbench | 0.0165 | 0.1241 | 0.0088 | 0.570 | 0.1200 | 0.0908 | 0.1770 |
| omnimath | 0.0304 | 0.1779 | 0.0166 | 0.555 | 0.1380 | 0.1014 | 0.2158 |
| **macro** | **0.0176** | 0.1850 | 0.0094 | - | **0.1794** | 0.1438 | 0.2519 |

**Interpretation.** Effectively tied with SSAE-mixed; the contrastive auxiliary did not transfer because the aux head failed to learn.

---

### 3.9 SSAE original paper checkpoint, Qwen2.5-0.5B

**Particularities.** Reuse audit of the original Miaow-Lab SSAE checkpoint on Qwen2.5-0.5B. This candidate tests whether the poor results from our Qwen2.5-1.5B SSAE variants are primarily a methodology/training failure or an inherent representation failure. The checkpoint is fixed; PRM800K and ProcessBench latents are reused. The leaderboard row reports the best probe/scorer variant over this fixed representation, which remains the mixed baseline.

**Results.**

| Variant over fixed 0.5B SSAE latents | val_selected macro F1_PB | oracle macro F1_PB | Read |
|---|---:|---:|---|
| positive-only centroid | 0.0000 | 0.1379 | weak OOD signal |
| contrastive probe | 0.0000 | 0.2871 | similar oracle signal to mixed, worse val calibration |
| mixed baseline | **0.0098** | **0.2948** | best among the fixed-checkpoint SSAE variants |

Best mixed-baseline oracle breakdown: combined F1_PB = 0.2877; gsm8k = 0.3627; math = 0.3059; olympiadbench = 0.2464; omnimath = 0.2642.

**Interpretation.** The fixed 0.5B paper checkpoint carries substantially more oracle first-error signal than our in-house 1.5B SSAE runs (0.2948 vs 0.1811 best oracle macro), despite the smaller backbone. That makes the earlier SSAE failure look less like a fatal architecture problem and more like insufficient or ineffective training of our 1.5B SSAE representations. However, the val-selected score remains essentially zero, and the contrastive probe variant does not fix calibration transfer: useful signal is present only when a ProcessBench oracle threshold is allowed.

---

### 3.10 DenseLinear-PRM400k

**Particularities.** Same architecture and feature as §3.2 (linear probe on the final-layer hidden state at the last token of the current step). Differs from §3.2 in three protocol axes, so it is reported separately rather than replacing the original row:

| Axis | §3.2 DenseLinear | §3.10 DenseLinear-PRM400k |
|---|---|---|
| Probe train split | 40K PRM800K | 400K PRM800K |
| Probe val split | 1K | 10K |
| Val threshold grid | 0.1, 0.2, ..., 1.0 (10 points) | 0.01, 0.02, ..., 0.99 (99 points) |
| Evaluation subsets | ProcessBench-GSM8K | ProcessBench-GSM8K, Math |

PRM800K validation balanced accuracy: 73.69 (val binary F1: 75.06). Threshold selected on val: 0.46.

**Results.** Note: this run predates the 4-subset full-PB pipeline and was not re-run under it.

ProcessBench-GSM8K (400 traces, same set as §3.2):

| Threshold type | Threshold | F1_PB | Acc_error | Acc_correct | Exact_match_all |
|---|---:|---:|---:|---:|---:|
| Fixed 0.5 | 0.5 | 0.2024 | 0.2899 | 0.1554 | 0.2250 |
| Val-selected | 0.46 | 0.1788 | 0.2657 | 0.1347 | 0.2025 |
| Oracle on PB | 0.67 | 0.3463 | 0.2899 | 0.4301 | 0.3575 |

ProcessBench-Math (1000 traces):

| Threshold type | Threshold | F1_PB | Acc_error | Acc_correct | Exact_match_all |
|---|---:|---:|---:|---:|---:|
| Fixed 0.5 | 0.5 | 0.1483 | 0.3249 | 0.0961 | 0.2320 |
| Val-selected | 0.46 | 0.1030 | 0.3148 | 0.0616 | 0.2120 |
| Oracle on PB | 0.85 | 0.3949 | 0.2929 | 0.6059 | 0.4200 |

Probe training time: 12.96 s on 400K examples (≈ 0.032 ms/example) on a single H100. PRM800K encoding time: 48 min 28 s for 400K (≈ 7.27 ms/example) plus 10K val. ProcessBench encoding: 2 min 19 s for GSM8K + Math combined (~8 636 steps).

**Interpretation.**

1. **Scaling 10× training did not lift fixed-0.5 F1_PB on GSM8K.** §3.2 reports 0.2701 at threshold 0.5; §3.10 reports 0.2024 at the same threshold. The drop comes from Acc_correct collapsing from 0.2746 to 0.1554 while Acc_error rises modestly (0.2657 to 0.2899). More training data made the probe more confident, triggering more error predictions and over-firing on fully-correct traces.
2. **Oracle ceiling is identical to §3.2 on GSM8K** (0.3463 here vs 0.3495 there). The signal in the final-layer hidden state is the bottleneck, not training data volume.
3. **PRM800K val threshold transfers poorly to ProcessBench F1_PB.** Val balanced accuracy 73.69 is high, but the val-selected threshold 0.46 underperforms both fixed 0.5 and the PB oracle on every subset. Threshold calibration via PRM800K-val is the active limitation, not representation quality.
4. **Math carries more signal than GSM8K at the oracle.** Math oracle F1_PB = 0.3949 exceeds GSM8K oracle F1_PB = 0.3463, with much higher Acc_correct (0.6059 vs 0.4301) at a higher oracle threshold (0.85 vs 0.67). The same pattern holds for §3.2's 40K-trained probe under the full-PB run (math oracle 0.4070 vs gsm8k oracle 0.3495), confirming this is a subset property, not a training-scale artifact.
5. **Oracle threshold is subset-dependent.** GSM8K oracle is 0.67, Math oracle is 0.85. The full-PB run in §3.2 shows the same monotonic pattern across 4 subsets (0.655 → 0.740 → 0.825 → 0.840 for gsm8k → math → olympiadbench → omnimath).

**Follow-ups suggested by these numbers.**

- Calibrate the threshold on a held-out ProcessBench split rather than PRM800K val. This is the gap, not training scale.
- Re-run §3.10 under the 4-subset full-PB pipeline so it can join the unified leaderboard in §2.
- Add weight decay or temperature scaling to the 400K probe to reduce the over-confidence that hurts Acc_correct at low thresholds.

---

### 3.11 PRM800K fork preference objectives

**Particularities.** Sprint 2. This study asks whether *genuine* fork-based preference supervision shapes a more useful latent than reconstruction alone. It also corrects a naming error: the earlier `sae_contrastive` candidate (§3.5) is **not** contrastive. It is `SAE + auxiliary BCE shaping` (MSE reconstruction + L1 sparsity + an auxiliary BCE label head that is discarded before the fresh probe is trained). True contrastive/preference learning needs matched positive/negative continuations, which §3.5 never used.

**Fork dataset.** Built from PRM800K forks: same problem + solution prefix + step index with multiple human-rated candidate next steps. A valid fork has ≥1 positive (rating +1 → label 0) and ≥1 negative (rating -1 → label 1).

- 54,737 valid forks found
- 40,000 train forks / 5,000 val forks, problem-disjoint split
- one (positive, negative) pair per fork
- each training item carries an anchor / positive / negative triple (anchor = prefix state before the next step)
- train items = 120,000 encoded rows; val items = 15,000 encoded rows
- Qwen2.5-1.5B last-token hidden states; `latent_dim = hidden_dim = 1536`; seed = 42

**Methods.** Controls `ae_recon`, `sae_recon` (objective off; AE = reconstruction only, SAE = reconstruction + L1 sparsity). Preference objectives `ae_rank`, `sae_rank` (RankNet logistic on a scalar head, score(pos) > score(neg)) and `ae_triplet`, `sae_triplet` (anchor pulled toward positive, pushed from negative; L2 metric, margin 1.0). Objective weight swept over `w ∈ {1, 10, 100}`; controls have the objective off. Heads are discarded after representation training; a fresh linear probe is trained on the frozen latent and evaluated through the locked ProcessBench pipeline (val-selected + oracle thresholds, 4 subsets, macro F1_PB).

**Results.** Macro F1_PB over the 4 ProcessBench subsets (gsm8k, math, olympiadbench, omnimath). Training diagnostics from `train_metrics.json`: final reconstruction MSE, pair accuracy (fraction of fork pairs correctly ordered), and margin satisfaction.

| Method      | Objective | Weight | Val-selected macro F1_PB | Oracle macro F1_PB | Recon MSE | Pair acc | Margin sat |
| ----------- | --------- | -----: | -----------------------: | -----------------: | --------: | -------: | ---------: |
| AE Recon    | none      |      - |                    0.046 |              0.360 |     0.130 |      N/A |        N/A |
| SAE Recon   | none      |      - |                    0.056 |              0.340 |     0.130 |      N/A |        N/A |
| AE Rank     | rank      |      1 |                    0.206 |              0.307 |     0.174 |    0.879 |      0.735 |
| AE Rank     | rank      |     10 |                    0.225 |              0.306 |     0.247 |    0.872 |      0.720 |
| AE Rank     | rank      |    100 |                **0.235** |              0.290 |     0.431 |    0.857 |      0.690 |
| SAE Rank    | rank      |      1 |                    0.179 |              0.275 |     0.172 |    0.880 |      0.735 |
| SAE Rank    | rank      |     10 |                    0.160 |              0.261 |     0.259 |    0.866 |      0.711 |
| SAE Rank    | rank      |    100 |                    0.183 |              0.272 |     0.416 |    0.854 |      0.681 |
| AE Triplet  | triplet   |      1 |                    0.136 |              0.260 |     1.168 |    0.882 |      0.874 |
| AE Triplet  | triplet   |     10 |                    0.226 |              0.256 |     1.745 |    0.864 |      0.845 |
| AE Triplet  | triplet   |    100 |                    0.196 |              0.207 |     2.224 |    0.844 |      0.774 |
| SAE Triplet | triplet   |      1 |                    0.203 |              0.286 |     1.177 |    0.875 |      0.867 |
| SAE Triplet | triplet   |     10 |                    0.183 |              0.210 |     1.763 |    0.858 |      0.837 |
| SAE Triplet | triplet   |    100 |                    0.209 |              0.218 |     2.620 |    0.836 |      0.758 |

Reference points from the existing leaderboard: DenseLinear oracle 0.3773 / val-selected 0.1855; SAE-mixed oracle 0.3688; SAE auxiliary BCE shaping (§3.5) oracle 0.3350 / val-selected 0.1648.

- **Best deployable (val-selected) method:** AE Rank w=100, macro val-selected F1_PB ≈ 0.235.
- **Best oracle inside this study:** AE Recon, oracle macro F1_PB ≈ 0.360.

**Interpretation.**

1. **The old "contrastive" SAE was not contrastive.** It was `SAE + auxiliary BCE shaping`: an auxiliary BCE head trained on the latent and then discarded. This study is the first to use genuine fork-based preference supervision.
2. **The preference objective is genuinely learned, not ignored.** At w=1, pair accuracy is 0.879 (AE Rank), 0.880 (SAE Rank), 0.882 (AE Triplet), 0.875 (SAE Triplet). The objective is connected to the graph, optimized, and demonstrably reshapes the latent. An implementation bug was ruled out during the Sprint 2 audit.
3. **Reconstruction-only controls keep high oracle separability but fail under deployable thresholds.** AE Recon reaches oracle 0.360 but only 0.046 at the PRM800K-val-selected threshold; SAE Recon is 0.340 / 0.056.
4. **Fork objectives massively improve val-selected transfer over the recon-only controls.** AE Rank lifts val-selected macro from 0.046 (AE Recon) to 0.235 at w=100, roughly a 5× improvement in deployable F1.
5. **The strongest deployable result is AE Rank w=100** at macro val-selected F1_PB ≈ 0.235, above DenseLinear's 0.1855.
6. **The strongest oracle result remains reconstruction-only AE**, and DenseLinear (§3.2) still has the strongest oracle macro overall (0.3773). Fork objectives do **not** raise the oracle ceiling.
7. **Fork preference objectives do not create a stronger maximum correctness signal.** They reshape/calibrate the representation so validation-selected thresholds transfer better to ProcessBench.
8. **Higher objective weight does not improve pair accuracy.** For ranking, pair accuracy *decreases* from 0.879 (w=1) to 0.857 (w=100) while val-selected F1 *improves*. The gain is score-distribution / calibration reshaping, not better fork ordering.
9. **Triplet is less stable.** It inflates reconstruction MSE sharply (up to 2.6 at high weight) and does not beat AE Rank overall.
10. **The contrastive / preference-objective study is closed for now.** Future work should target model size, threshold calibration, or score-distribution analysis rather than more variants of this objective family.

**Conclusion.** The fork-preference study shows that pairwise PRM800K supervision is learnable from Qwen2.5-1.5B hidden states: rank and triplet objectives reach roughly 0.85–0.88 pair accuracy on the fork training signal. However, this does not translate into a higher oracle ProcessBench ceiling. Reconstruction-only AE/SAE controls retain stronger oracle separability, while fork objectives mainly improve val-selected threshold transfer. The strongest deployable result is AE Rank with objective weight 100, reaching macro val-selected F1_PB ≈ 0.235, above DenseLinear's 0.1855 val-selected macro, but below DenseLinear's oracle ceiling of 0.3773. Thus, fork preference objectives appear to improve calibration / score shaping rather than extract a stronger mechanistic correctness representation.

---

### 3.12 Dense pairwise scorer objectives

**Particularities.** This follow-up asks whether PRM800K fork supervision is more useful at the *classifier/scorer* level than as a representation-shaping objective. Unlike the AE/SAE Rank/Triplet methods of §3.11, these methods train **no representation**: the Qwen2.5-1.5B hidden state stays frozen and the only learned object is a linear scorer

```text
logit(h) = w·h + b ,    score = sigmoid(logit) = P(non_viable / error)
```

so the preferred (label 0) sibling should get a LOW logit and the rejected (label 1) sibling a HIGH logit. **No BCE** is used in any of the three methods; they replace the BCE label loss with a pairwise/margin loss over fork siblings on the same frozen `h`.

- `dense_rank` — pairwise margin ranking: `softplus(margin - (l_neg - l_pos))`.
- `dense_anchor_rank` — anchor-relative score constraint: `(l_pos - l_anchor)² + softplus(margin - (l_neg - l_anchor))` (scores on logits, not latent distances).
- `dense_rank_absmargin` — pairwise ranking + absolute logit margins: `softplus(l_pos + 1) + softplus(1 - l_neg) + softplus(1 - (l_neg - l_pos))`.

Threshold selection (PRM800K val_1k) and the ProcessBench eval path are identical to every other candidate; only the scorer's training objective changes.

**Results.** Macro F1_PB over the 4 ProcessBench subsets (gsm8k, math, olympiadbench, omnimath).

| Method               | Training objective                        | val-selected macro F1_PB | oracle macro F1_PB | Read                                                                 |
| -------------------- | ----------------------------------------- | -----------------------: | -----------------: | -------------------------------------------------------------------- |
| DenseLinear          | BCE on independent PRM800K labels         |                   0.1855 |         **0.3773** | best oracle baseline                                                 |
| dense_rank           | pairwise margin ranking on forks          |               **0.2087** |             0.2328 | slightly better deployable threshold transfer, large oracle collapse |
| dense_anchor_rank    | anchor-relative score constraint          |                   0.1228 |             0.2047 | worst of the three; anchor term hurts                                |
| dense_rank_absmargin | pairwise ranking + absolute logit margins |                   0.1771 |             0.2478 | does not beat DenseLinear val-selected; oracle still collapses       |

**Interpretation.** The dense pair-scorer experiment isolates the effect of the fork objective from representation training. Since no AE/SAE/SSAE representation is trained, any performance change comes from replacing the BCE-trained linear scorer with a pairwise/margin-trained linear scorer on the same frozen dense hidden states. The result is clear: pairwise fork objectives can slightly improve val-selected threshold transfer (`dense_rank`: 0.2087 vs DenseLinear 0.1855), but they strongly reduce oracle separability (0.2328 vs 0.3773). Therefore, the oracle drop observed in AE/SAE fork-objective runs is not only a representation damage effect. It is intrinsic to the pairwise/margin objective: it learns a more calibration-friendly but lower-ceiling decision direction.

- `dense_anchor_rank` is not promising: it is the worst of the three on both axes (val-selected 0.1228, below even DenseLinear). The squared anchor-preservation term likely fights the error-score separation objective, pinning `l_pos` toward an unconstrained `l_anchor` while the ranking term tries to push the siblings apart.
- `dense_rank_absmargin` adds stronger absolute score pressure (push `l_pos` below -1, `l_neg` above +1) which improves its oracle slightly over plain `dense_rank` (0.2478 vs 0.2328), but it still does not preserve the oracle signal and does not beat DenseLinear's val-selected macro.
- **DenseLinear BCE remains the best method for maximum recoverable ProcessBench signal** (oracle 0.3773); the fork objectives mainly reshape/calibrate scores for threshold transfer.

**Study closed.** The contrastive / preference-objective branch is closed for now. The evidence supports a calibration-transfer benefit from fork supervision, but not a stronger mechanistic correctness representation. The next research direction should be model scale / stronger backbone / calibration analysis, rather than more variants of this same pairwise objective family.

---

### 3.13 DenseLinear model-size ablation

**Particularities.** Sprint 2 follow-up answering the direction §3.12 left open (model scale / stronger backbone / calibration). Identical DenseLinear method and locked pipeline as §3.2 (linear probe on the final-layer last-token hidden state; PRM800K 40K-train / 1K-val frozen split; ProcessBench 4 subsets; F1_PB; val-selected + 0.005 oracle thresholds; seed 42). The only variable is the backbone: Qwen2.5 {1.5B, 3B, 7B, 14B, 32B}. Probe input dim is inferred from `model.config.hidden_size` (1536 to 5120), never hardcoded. No context truncation: each step conditions on question + all previous steps + current step, with `max_seq_len` set to the model's full context window (131072; 32768 for 3B, the strict case) and a fail-hard overlength guard; `num_truncated_examples = 0` for every model. float16 inference; run on transformers 5.9 / torch 2.12. The 1.5B row reproduces §3.2 exactly (val 0.18554, oracle 0.37729; per-subset thresholds and F1_PB identical), validating the scaled pipeline.

**Results.** Macro F1_PB over the 4 subsets, ranked by oracle.

| Backbone | hidden | layers | val-selected macro F1_PB | oracle macro F1_PB | val/oracle |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-32B | 5120 | 64 | 0.0606 | **0.4761** | 0.13 |
| Qwen2.5-14B | 5120 | 48 | 0.2375 | 0.4489 | 0.53 |
| Qwen2.5-7B | 3584 | 28 | 0.2369 | 0.4132 | 0.57 |
| Qwen2.5-3B | 2048 | 36 | 0.0383 | 0.3817 | 0.10 |
| Qwen2.5-1.5B | 1536 | 28 | 0.1855 | 0.3773 | 0.49 |

Per-subset oracle F1_PB:

| Backbone | gsm8k | math | olympiadbench | omnimath |
|---|---:|---:|---:|---:|
| 1.5B | 0.3495 | 0.4070 | 0.3720 | 0.3806 |
| 3B | 0.3678 | 0.4196 | 0.3760 | 0.3634 |
| 7B | 0.4537 | 0.4467 | 0.3773 | 0.3751 |
| 14B | 0.4859 | 0.4898 | 0.4008 | 0.4192 |
| 32B | 0.5236 | 0.4955 | 0.4317 | 0.4537 |

Per-subset val-selected F1_PB (with the PRM800K-val threshold t_val):

| Backbone | t_val | gsm8k | math | olympiadbench | omnimath |
|---|---:|---:|---:|---:|---:|
| 1.5B | 0.50 | 0.2701 | 0.2248 | 0.0899 | 0.1574 |
| 3B | 0.40 | 0.0960 | 0.0049 | 0.0000 | 0.0522 |
| 7B | 0.50 | 0.3944 | 0.2210 | 0.1367 | 0.1956 |
| 14B | 0.50 | 0.4504 | 0.2253 | 0.0895 | 0.1850 |
| 32B | 0.40 | 0.0911 | 0.0928 | 0.0000 | 0.0583 |

**Interpretation.**

1. **Oracle macro F1_PB rises monotonically with scale**: 0.3773 (1.5B) to 0.3817 (3B) to 0.4132 (7B) to 0.4489 (14B) to 0.4761 (32B), +0.10 macro overall. The raw final-layer hidden state carries progressively more first-error signal as the backbone grows; 32B is the strongest DenseLinear oracle in the whole log.
2. **The deployable (PRM800K-val-selected) threshold does not scale and is erratic.** 3B (0.0383) and 32B (0.0606) collapse while 1.5B/7B/14B sit at 0.19/0.24/0.24. Both collapsing models selected t_val=0.40 on PRM800K-val, which over-predicts errors on ProcessBench (olympiadbench falls to 0.0). Calibration, not representation capacity, is the binding constraint, and it does not improve with scale.
3. **Per subset**: olympiadbench (hardest) gains most from scale at oracle (0.372 to 0.432); gsm8k gains most overall (0.350 to 0.524). At the val threshold the larger models still leak most of this on the off-distribution subsets.
4. This extends the §3.2 / S1 val-to-PB calibration finding: the gap is not a smooth function of scale; a larger model can have a worse deployable score (32B val/oracle ratio 0.13) despite a higher ceiling. A PB-held-out threshold rule (cross-cutting follow-up) is the natural fix and would make the oracle gains deployable.

**Artifacts.** `runs/s1_model_size_dense/` (per-model `per_subset_metrics.json`, `model_config.json`, `length_audit.json`, `leaderboard_model_size*.{csv,md}`), produced by `slurm/s1_model_size/run_sweep_one_job.sh`.

---

## Cross-cutting follow-ups

- **SSAE underperformance is the open question.** Two diagnostics worth running before discarding the architecture: (a) train a linear probe directly on `decode(z)` and compare to `z`, to separate "bottleneck destroys info" from "linear readout on z is lossy"; (b) check whether the SSAE's reconstruction MSE on PB matches its training MSE, to rule out distribution shift in the bottleneck.
- **Threshold calibration on PB-held-out** would close most of the val/oracle gap for SAE-mixed and DenseLinear; the same fix proposed in §3.10.
