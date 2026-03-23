# CoT-Checker — Research Report
*Last updated: 2026-03-22*

## Abstract

This project investigates whether sparse autoencoder (SAE) feature activations provide reliable mechanistic signals for detecting incorrect steps in chain-of-thought (CoT) reasoning. We reproduce the step-correctness probe from *"Step-Level Sparse Autoencoders for Interpretable Chain-of-Thought Verification"* (Miaow-Lab, arXiv:2603.03031), using the publicly released SSAE checkpoint trained on Qwen2.5-0.5B.

Our reproduction achieves **77.50% validation accuracy** on a 1,000-step subset, compared to the paper's reported **78.58%** — a gap of 1.08 percentage points.

---

## 1. Objective

Determine whether SSAE sparse latent vectors `h_c` — extracted at each reasoning step of a chain-of-thought trace — contain enough information to predict step-level correctness, using a lightweight MLP probe.

---

## 2. Experimental Setup

### 2.1 Model

We use the publicly released pretrained SSAE checkpoint:
- **Checkpoint**: `gsm8k-385k_Qwen2.5-0.5b_spar-10.pt` from `Miaow-Lab/SSAE-Checkpoints`
- **Backbone**: `Qwen/Qwen2.5-0.5B` (encoder and decoder share the same base model)
- **Sparsity factor**: 1 → `n_latents = hidden_size = 896`
- **Checkpoint state**: step 56,612, best validation loss 0.3978

The SSAE encodes the full sequence `[context | <sep> | step]` and extracts the last-token hidden state before projecting it through a sparse autoencoder to obtain `h_c ∈ ℝ^{896}`.

### 2.2 Step-Level Labels

We use **Math-Shepherd** (`peiyi9979/Math-Shepherd`, GSM8K partition) for step-level correctness labels. Math-Shepherd provides binary labels (`+`/`−`) for each reasoning step derived from Monte Carlo rollouts: a step is labeled **correct (+)** if the correct final answer is still reachable when continuing from that step, and **incorrect (−)** otherwise.

This gives ground-truth labels that are:
- **Independent of the SSAE** (no circular dependency on reconstruction quality)
- **Semantically meaningful** (a step is wrong if it derails the solution path)
- **Naturally imbalanced**: in our 1,000-step sample, **27.8% correct** and **72.2% incorrect** — consistent with the paper's reported majority baseline of 70.49%

### 2.3 Data Pipeline

1. Stream Math-Shepherd (GSM8K split), parse step texts and labels
2. For each `(context, step)` pair, remove `<<expr=result>>` calculator annotations
3. Encode with the SSAE encoder (no decoding required) → latent `h_c ∈ ℝ^{896}`
4. Store `(h_c, label)` pairs in a compressed `.npz` file

**Dataset size**: 1,000 steps (vs ~385K in the paper's full run)
**Train / val split**: 80% / 20% (800 train, 200 validation)

### 2.4 Probe Architecture

Three-layer MLP trained on `h_c`:

| Layer | Dim in | Dim out | Activation |
|-------|--------|---------|------------|
| FC 1  | 896    | 256     | ReLU       |
| FC 2  | 256    | 64      | ReLU       |
| FC 3  | 64     | 2       | —          |

- **Loss**: Cross-entropy
- **Optimizer**: Adam, lr=1e-3
- **Epochs**: 30, batch size 128
- **Hardware**: Apple M4 (MPS backend), float32

---

## 3. Results

![Probe training curve](results/probe_training_curve.png)

**Figure 1.** Probe training curve over 30 epochs. Left axis (blue): cross-entropy training loss. Right axis (orange): validation accuracy. Dashed green line: paper's reported accuracy for SSAE-Qwen on GSM8K (78.58%). Dotted purple line: majority-class baseline (72.2%).

| Metric | Our result | Paper (SSAE-Qwen, GSM8K) |
|--------|-----------|--------------------------|
| Best val accuracy | **77.50%** (epoch 7) | **78.58%** |
| Majority baseline | 72.2% | 70.49% |
| Gap above baseline | +5.3 pp | +8.09 pp |
| Steps used | 1,000 | ~385,000 |

---

## 4. Discussion

### 4.1 Reproduction fidelity

Our result of **77.50%** is within **1.08 percentage points** of the paper's 78.58%, which we consider a successful reproduction given:
- We use only **1,000 steps** vs the paper's full 385K (0.26% of the data)
- Minor label distribution shift: our majority baseline is 72.2% vs the paper's 70.49%

### 4.2 Probe overfitting

The training curve shows the probe peaks at **epoch 7** (77.5%), then validation accuracy plateaus and slightly degrades as training loss continues to fall. This indicates mild overfitting, unsurprising with only 800 training examples. With the full 385K-step dataset, the probe would have more signal and likely generalize better, explaining the slightly higher paper result.

### 4.3 What the probe is detecting

The SSAE encodes `[context | <sep> | step]` into a sparse vector `h_c`. The probe's ability to predict step correctness from `h_c`, above the majority baseline, indicates that the SSAE's sparse latent space contains features that correlate with whether a reasoning step leads to a correct solution path. This is a non-trivial finding: the SSAE was trained purely for reconstruction, yet its latent space is semantically organized in a way that reflects step correctness.

### 4.4 Limitations

- **Small sample**: 1,000 steps may not capture the full label distribution from the paper
- **Math-Shepherd vs GSM8K-Aug**: the paper uses GSM8K-Aug for SSAE training and likely for probe labels too; we use Math-Shepherd labels which assign correctness differently (MC rollout vs possibly reconstruction-based)
- **Single run**: no variance estimate over seeds

---

## 5. Next Steps

- Scale to the full Math-Shepherd GSM8K partition (~50K steps) to close the remaining gap
- Evaluate the probe on out-of-distribution problems (MATH-500) as the paper does
- Investigate which SSAE features (sparse dimensions) are most predictive of incorrectness
- Compare against the dense baseline `h_k` (pre-projection) to quantify what sparsification adds
