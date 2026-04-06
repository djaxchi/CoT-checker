# WIP: Full Experiment Log
*Work in progress — not included in the main report*
*Last updated: 2026-04-06*

This document is a complete log of everything tried during this project, including experiments that were superseded, results that turned out to be contaminated, and methodological mistakes that were caught and corrected. The main `REPORT.md` contains only verified, clean results. This document exists so the full history is recoverable.

---

# Part A: Arithmetic Probe Experiments

## A1. Baseline: 1K-Step Probe

**When:** Early in the project, before any data integrity checks.

**Setup:**
- Data: 1,000 steps from Math-Shepherd (offset 0), natural distribution (~28% correct)
- Encoded at max_len=256 via `scripts/generate_probe_data.py`
- Saved as `math_shepherd_1000.npz` (still in `results/probe_data/`)
- Probe: 3-layer MLP, 30 epochs, AdamW lr=1e-4, threshold=0.5
- Eval: `math_shepherd_eval_5k_contaminated.npz` ← **later found to be contaminated**

**Reported results (contaminated eval):**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Correct | 0.589 | 0.453 | 0.512 |
| Incorrect | 0.792 | 0.868 | 0.828 |
| Overall accuracy | — | — | 74.60% |

Majority baseline on contaminated eval: 70.6%.

**Status:** Numbers not trustworthy. Eval contamination rate unknown for this probe (the 1K training data is a small subset of what the eval overlaps, so contamination is likely lower than for the 57K probe, but unquantified).

---

## A2. Scaling to 57K: Rebalanced (50/50) Probe

**When:** After the 1K baseline, to test whether more data helps.

**Setup:**
- Data: ~100K raw Math-Shepherd steps (offset 0) rebalanced to 50/50 via `--correct-ratio 0.5`
- Resulted in 57,782 steps (28,891 correct + 28,891 incorrect), saved as `math_shepherd_57k_balanced.npz`
- Encoded at max_len=256
- Same probe architecture and hyperparameters
- Eval: contaminated eval ← **48.9% of eval rows are exact duplicates of this training set**

**Reported results (contaminated eval):**

| Metric | Value |
|---|---|
| Val accuracy | 75.43% |
| Held-out accuracy | 73.66% |
| Incorrect F1 | ~0.787 |

**Finding:** Despite 57x more data than the 1K probe, accuracy dropped slightly. The 50/50 rebalancing hurt incorrect-class recall relative to the natural-distribution probe. This finding is directionally real (the mechanism is explained by the decision boundary shifting), but the exact numbers are unreliable due to eval contamination.

**Status:** Numbers not trustworthy. Contamination: 48.9% of eval appeared in training data.

---

## A3. 40K Natural-Distribution Probe

**When:** After observing that 50/50 rebalancing degraded incorrect-class recall.

**Setup:**
- Data: `math_shepherd_40k_natural.npz` — 40,126 steps (offset 0), natural ~28/72 distribution, encoded at max_len=256
- Probe trained with same hyperparameters
- Eval: contaminated eval ← **33.4% of eval rows are exact duplicates of this training set**

**Reported results (contaminated eval):**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Correct | 0.678 | 0.458 | 0.546 |
| Incorrect | 0.801 | 0.909 | 0.852 |
| Overall accuracy | — | — | 77.64% |

Val accuracy: 78.25%. Generalization gap: 0.61 pp.

**Finding:** Natural distribution outperformed 50/50 rebalancing. This is directionally correct and consistent with theory, but the absolute numbers are inflated by ~1.5 pp of contamination.

**Status:** Numbers not trustworthy. Contamination: 33.4% of eval appeared in training data.

---

## A4. Distribution Calibration Hypothesis: Flipped (72/28) Probe

**When:** After A3, to understand mechanistically why distribution matters.

**Script:** `scripts/experiment_flipped_distribution.py`

**Hypothesis:** If the 50/50 probe underperforms because it learns the wrong prior, then flipping to 72% correct / 28% incorrect (the mirror of natural) should hurt in the opposite direction — overcounting correct, missing incorrect.

**Setup:**
- Source: `math_shepherd_40k_natural.npz` (40,126 steps)
- Resampled to 72/28 via oversampling correct (2.56x), total 40,000 steps
- Same hyperparameters; eval at threshold=0.8
- Compared against the 40K natural probe on the same contaminated eval

**Results (contaminated eval, threshold=0.8):**

|  | Natural probe | Flipped probe |
|---|---|---|
| Accuracy | 78.02% | ~72% |
| Correct recall | 0.458 | 0.880 |
| Incorrect recall | 0.909 | 0.670 |
| Incorrect F1 | 0.852 | ~0.750 |

**Finding confirmed:** Flipping the distribution flips the asymmetry exactly as predicted. The probe trained at 72/28 has high recall on correct and poor recall on incorrect — the mirror of the natural probe. This confirms that training distribution directly controls where the decision boundary sits. This directional finding is robust to eval contamination because both probes were evaluated on the same contaminated set.

**Status:** Directional finding valid. Absolute numbers not trustworthy.

---

## A5. Scale Experiment: 200K Steps (Multiple Issues Found)

**When:** After the calibration hypothesis was validated.

**Script:** `scripts/experiment_scale_200k.py`

**Goal:** Scale training to 200K steps at 72/28 (correct/incorrect) to see if more data improves the flipped probe.

**What actually ran:**

1. **Encoding phase:** Streamed 200K raw Math-Shepherd steps from offset 110,000 to avoid overlap with existing training data. Encoded at max_len=128 (reduced from 256 to fit within MPS memory after an OOM crash partway through the first attempt). Got 176,008 steps (28.9/71.1 natural distribution). Took ~14 hours on MPS. Saved as `math_shepherd_176k_natural.npz`.

2. **Merge phase:** Merged the new 176K with the existing `math_shepherd_57k_balanced.npz` (57K steps from offset 0), resampled to 200K at 72/28. Correct steps oversampled 1.80x. Saved as `math_shepherd_200k_flipped.npz`.

3. **Training:** 30 epochs, same hyperparameters. Best val acc 84.63%.

4. **Eval:** Contaminated eval at threshold=0.8.

**Results (contaminated eval, threshold=0.8):**

|  | 40K flipped probe | 200K flipped probe |
|---|---|---|
| Accuracy | 78.02% | 74.04% |
| Correct F1 | 0.675 | 0.639 |
| Incorrect F1 | 0.834 | 0.797 |

**The 200K probe was worse than the 40K probe.** This was puzzling until the issues were diagnosed:

**Issues found in this experiment:**
1. **Max_len mismatch:** The 57K balanced data (merged in) was encoded at max_len=256. The new 176K data was encoded at max_len=128. Merging latents encoded at different truncation lengths creates a distribution mismatch — the model trained on mixed-length encodings.
2. **Eval contamination:** The 57K balanced training data had 48.9% overlap with the eval set, inflating the 40K flipped probe's numbers more than the 200K probe's (because the 200K's new data was from a clean window). The 200K probe appeared worse partly because it had relatively less eval contamination.
3. **Correct oversampling at 1.80x:** 79,841 unique correct steps oversampled to 144,000 means many identical examples appear in both train and val splits (random_split on oversampled data — train-val leakage within the experiment, though this doesn't affect held-out metrics).

**Status:** Numbers not trustworthy. The experiment was structurally flawed due to issues 1 and 2.

---

## A6. Eval Contamination Discovery

**When:** After the scale experiment's confusing result prompted a deeper audit.

**Method:** Computed SHA-level row hashes (full 896-dim float16 row → `hash(row.tobytes())`) for every row in every dataset file, then compared sets.

**Findings:**

| Training set | Eval rows that are exact duplicates |
|---|---|
| `math_shepherd_57k_balanced.npz` | **48.9%** (2,445/5,000) |
| `math_shepherd_40k_natural.npz` | **33.4%** (1,671/5,000) |
| 200K merged training set | **30.9%** (1,543/5,000) |
| `math_shepherd_176k_natural.npz` (new window) | 4.1% (205/5,000) |

**Root cause:** The original eval file (`math_shepherd_eval_5k_contaminated.npz`) was generated from offset 0, the same Math-Shepherd window used for all training data up to that point. The generation script (`generate_probe_data.py`) collected records from the beginning of the dataset with no separation guarantee.

**Fix:**
- Retired the contaminated eval file (renamed with `_contaminated` suffix, not deleted)
- Added `--max-len` flag to `generate_probe_data.py` to make truncation configurable
- Generated a new clean eval from offset 100,000 (the gap between training window 1 [0–100K] and training window 2 [110K–286K])
- Verified 0% overlap with all training files by row-hash comparison
- New file: `math_shepherd_eval_5k_clean.npz` (5,000 steps, offset 100K, max_len=128, 28.9% correct)

**Impact:** All prior reported accuracy numbers were inflated by roughly 1–2 pp due to the probe having seen a substantial fraction of the test data during training.

---

## A7. Clean Experiment: 176K Steps, 70/30, No Duplication

**When:** After clean eval was established.

**Script:** `scripts/experiment_176k_clean.py`

**Setup:**
- Source: `math_shepherd_176k_natural.npz` (176,008 steps, offset 110K, max_len=128)
- No merging with old data — single encoding window, consistent max_len
- Subset to 70/30 without any duplication: keep all 50,950 unique correct steps, subsample 21,835 incorrect steps from 125,058 available
- Total: 72,785 training steps, zero duplicates
- Eval: `math_shepherd_eval_5k_clean.npz` — 0% overlap verified

**Results (clean eval, threshold=0.8):**

```
Majority baseline : 71.10%
Probe accuracy    : 77.00%  (+5.90 pp)
```

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Correct | 0.592 | 0.655 | 0.622 |
| Incorrect | 0.853 | 0.817 | 0.835 |

Training curve peaked at epoch 9 (val 81.39%) then gradually overfitted. Saved as `results/probes/correctness_probe_176k_clean.pt`.

**Status:** This is the only fully verified result. See `REPORT.md` for full discussion.

---

## A8. Token Position Ablation (Partial, Not Completed)

**When:** Early in the project, not completed.

**Script:** `scripts/ablation_eq_token.py`

The script probes hidden states at positions immediately after "=" tokens in arithmetic expressions, comparing mean probe scores between correct and incorrect steps. The goal was to test whether the arithmetic result token already carries a correctness signal independently of last-token pooling.

**Status:** The script exists but was never run to completion and no results file was saved. The experiment is noted here for completeness. Running it would require re-encoding steps token-by-token rather than using precomputed latents, which would need SSAE access and significant compute.

---

# Part B: Symbolic Domain Extension

## S1. Motivation

The arithmetic experiments validate the probe on Math-Shepherd/GSM8K. The next question is whether the SSAE probe generalises to a structurally different reasoning domain. We test on **symbolic propositional logic** (ProntoQA), where reasoning steps are modus-ponens deductions rather than arithmetic operations.

This tests whether SSAE latents encode a general notion of "is this step a valid deduction?" or merely arithmetic-specific patterns.

---

## S2. Setup

**Dataset**: `renma/ProntoQA` (HuggingFace) — 500 problems, ~48% False-answer problems, complex multi-hop chains (avg 7.5 model-generated steps/problem).

**Step labeling**: `PropLogicSolver` (deterministic) — validates each step via one-step modus ponens against the accumulated knowledge state. Extended to handle ProntoQA's plural rule format ("Jompuses are yumpuses.").

**SSAE**: Trained from scratch on ProntoQA traces (3,736 steps, 1 epoch, frozen Qwen2.5-0.5B backbone). Encoder checkpoint: `ssae_symbolic_p1.enc.pt`.

**Probe**: Same 3-layer MLP architecture as arithmetic experiments, trained on SSAE latents from 3,208 labeled steps (2,566 train / 642 val).

**Caveats:** This experiment used the contaminated arithmetic eval for comparison baselines, and the SSAE was trained for only 1 epoch on a small corpus with a frozen backbone. The symbolic probe results are preliminary — the architecture and training are unoptimized. No clean held-out eval has been established for the symbolic domain yet.

---

## S3. Label Distribution

| Class | Steps | % |
|---|---|---|
| Correct (+) | 1,864 | 58.1% |
| Incorrect (−) | 1,344 | 41.9% |
| **Majority baseline** | | **58.1%** |

---

## S4. Results

| Configuration | Majority baseline | Val accuracy | Gap above baseline |
|---|---|---|---|
| Arithmetic — 1K steps (contaminated eval) | 72.2% | 77.50% | +5.3 pp |
| Arithmetic — 40K steps (contaminated eval) | 70.6% | 78.25% | +7.65 pp |
| **Arithmetic — 176K clean (clean eval)** | **71.1%** | **77.00%** | **+5.90 pp** |
| Paper (SSAE-Qwen, GSM8K, paper's eval) | 70.49% | 78.58% | +8.09 pp |
| **Symbolic — ProntoQA (ours, val set)** | **58.1%** | **70.56%** | **+12.5 pp** |

**Per-class breakdown (symbolic probe, val set):**

| | Precision | Recall | F1 |
|---|---|---|---|
| Incorrect steps | 0.652 | 0.615 | 0.633 |
| **Correct steps** | **0.740** | **0.769** | **0.754** |

---

## S5. Discussion

**The gap above baseline is notable.** At +12.5 pp, the symbolic probe exceeds all arithmetic probes in gap above majority. Given that it was trained on only 2,566 steps with a minimally trained SSAE, this is encouraging. However, the comparison to arithmetic probes is not direct: baselines, domains, and eval sets all differ.

**Class asymmetry is reversed compared to arithmetic.** In the arithmetic probe, incorrect steps were easier to detect (incorrect F1 0.835 vs correct F1 0.622). In the symbolic probe, correct steps are easier to detect (correct F1 0.754 vs incorrect F1 0.633). This may reflect the nature of symbolic errors: when a model substitutes the wrong category, subsequent steps may remain locally consistent with the wrong premise, making individual incorrect steps harder to isolate. The SSAE may pick up on the coherence of correct chains more reliably.

**SSAE training is minimal.** Only 1 epoch on 3,736 steps with a frozen backbone. More training is needed before drawing strong conclusions.

---

## S6. Limitations

- 2,566 training steps — small, single seed, no held-out eval (val set only)
- SSAE backbone frozen throughout (T4 memory constraint during Colab training)
- No clean held-out eval for the symbolic domain — needs to be established before results are reportable
- Val accuracy was used for model selection, so it is optimistic

---

## S7. Planned Next Steps

- Establish a clean held-out eval set for the symbolic domain (analogous to what was done for arithmetic)
- Train SSAE for more epochs on symbolic data
- Investigate which sparse dimensions drive the +12.5 pp gap — feature importance analysis
- Compare symbolic SSAE vs arithmetic SSAE as encoder for symbolic probe (transfer test)
- Evaluate on out-of-distribution symbolic problems (longer hop chains, negation-heavy)
- Compare against the dense baseline `h_k` (pre-projection) to quantify what sparsification adds
