# CoT-Checker: Research Report
*Last updated: 2026-07-16*

---

## 1. Hypothesis

The long-term research question driving this project is:

> **Can sparse autoencoder (SAE) feature activations serve as reliable mechanistic signals for detecting incorrect steps in chain-of-thought reasoning?**

If the latent space of an SAE trained on model activations encodes step-level correctness in a learnable way, then a lightweight probe on those latents could provide interpretable, model-internal verification of reasoning traces. This would be meaningfully different from black-box verifiers: rather than re-running the problem or calling a judge model, we read the model's own internal state.

The hypothesis is investigated in two steps:

1. **Does the signal exist?** Can a classifier trained on SAE latents predict step correctness above chance? If yes, the latent vector carries information about correctness — the signal is there, even if we do not yet understand it.
2. **What is the signal?** Which specific dimensions drive the predictions? Is it linearly decodable? Does it correspond to interpretable reasoning features?

This report covers step 1 and the first probe of step 2. We establish that an MLP probe trained on SSAE latents can detect step correctness on an independent oracle (Math-Shepherd), and we further show that the correctness signal is linearly decodable from the same latents — a linear probe achieves nearly identical accuracy, which is a stronger geometric claim about the latent space.

---

## 2. The Paper We Started From

We chose to reproduce **"Step-level Sparse Autoencoders for Interpretable CoT Verification"** (Miaow-Lab, arXiv:2603.03031). The paper proposes an SSAE (Step-level Sparse Autoencoder) architecture trained on GSM8K-Aug with a Qwen2.5-0.5B backbone. The paper's core claim is that a linear probe on the SSAE latent vector `h_c` can predict step correctness at close to 78.58% accuracy on GSM8K.

The paper releases its checkpoint publicly (`Miaow-Lab/SSAE-Checkpoints`), making it an ideal reproduction target: we can use the exact same encoder and evaluate probe performance on an independent task.

**What made this interesting as a starting point:** the SSAE is trained for reconstruction, not classification. Any correctness signal in `h_c` would be an emergent property of the latent space, not a supervised artifact. That is exactly the kind of mechanistic signal the hypothesis requires.

---

## 3. Why the SSAE Architecture and How We Diverge From the Paper

### 3.1 What Attracted Us to the SSAE Architecture

The SSAE has a specific geometric property that motivated this work. The sparse bottleneck forces the encoder to compress each step into a latent vector that captures only the additional information contained in that step, beyond what the backbone already represents from the context. The reconstruction objective requires the latent to be sufficient to recover the step text given the context, while the sparsity constraint prevents it from simply copying the backbone's contextual representation. The result is a step-specific vector that is, by construction, informative about the content of that step and not the sequence as a whole.

If the step-specific latent encodes what this step contributes to the reasoning, then a correctness signal, if it exists anywhere in the model's internal representation of a step, should be recoverable from `h_c`. We do not need to understand which features in `h_c` carry the signal; the question is simply whether it is there.

### 3.2 Our Question

Can the SSAE latent space, trained purely for reconstruction, encode step-level correctness in a way that a lightweight probe can extract?

### 3.3 How We Differ From the Paper

The paper trains and evaluates using labels derived from the SSAE's own reconstruction quality: a step is labeled "correct" if the SSAE faithfully decodes its final number. This creates a circular dependency between the encoder and the probe. Furthermore, the training corpus (GSM8K-Aug) contains only arithmetically correct steps, so the probe never observes a genuinely incorrect step during training.

We decouple the encoder from the labels entirely. We use the paper's SSAE checkpoint as a fixed encoder and replace the supervision signal with an external oracle (Math-Shepherd), which assigns correctness via Monte Carlo rollouts on model-generated solutions. The probe must find the correctness signal from first principles, with no access to anything the SSAE was trained on.

This changes the question from "does the probe predict SSAE reconstruction quality?" to "does the probe predict step correctness as judged by an independent source?" The encoding architecture is identical to the paper; only the labels change.

---

## 4. Our Methodological Shift

We kept the SSAE encoder as-is (checkpoint `gsm8k-385k_Qwen2.5-0.5b_spar-10.pt`, step 56,612, best val loss 0.3978) and replaced only the labels.

**Our labels come from Math-Shepherd** (`peiyi9979/Math-Shepherd`, GSM8K partition). Math-Shepherd assigns binary correctness labels to reasoning steps via Monte Carlo rollouts: a step is labeled correct (+) if the final answer is still reachable when the solution continues from that step, and incorrect (−) otherwise. The labels are:

- **Independent of the SSAE**: no circular dependency on reconstruction quality
- **Semantically meaningful**: a step is wrong if it derails the solution path, regardless of fluency
- **Naturally imbalanced**: roughly 28% correct, 72% incorrect in the wild

The question becomes: does the SSAE latent space, trained purely for reconstruction on correct steps, still encode enough information to predict step correctness as defined by an external oracle?

**A known limitation of this label source:** Math-Shepherd's MC rollout labels measure whether a step is on a viable solution path, not whether the step is mathematically correct in isolation. A computationally correct step can be labeled incorrect if it pursues the wrong sub-goal; a step with a small arithmetic error can be labeled correct if subsequent steps happen to recover the right answer. Meanwhile, the SSAE was trained to encode mathematically correct steps from GSM8K-Aug, a different dataset with different solution styles than Math-Shepherd's model-generated traces. There is therefore a mismatch between what the SSAE learned to represent (arithmetic correctness on human solutions) and what the labels measure (path viability on model-generated solutions). This report treats this as an acceptable approximation for a first-pass experiment, but a tighter setup would use labels directly aligned with the SSAE's training objective, or train a new SSAE on data with MC rollout correctness labels from the start.

---

## 5. Data Pipeline

### 5.1 Raw Input: What Math-Shepherd Looks Like

Each Math-Shepherd entry contains a `label` field with every reasoning step annotated. Here is an actual entry from the dataset:

```
Janet pays $40/hour for 3 hours per week of clarinet lessons and $28/hour
for 5 hours a week of piano lessons. How much more does she spend on piano
lessons than clarinet lessons in a year?
Step 1: Janet spends 3 hours + 5 hours = <<3+5=8>>8 hours per week on music lessons. +
Step 2: She spends 40 * 3 = <<40*3=120>>120 on clarinet lessons per week. +
Step 3: She spends 28 * 5 = <<28*5=140>>140 on piano lessons per week. +
Step 4: Janet spends 120 + 140 = <<260>>260 on music lessons per week. +
Step 5: She spends 260 * 52 = <<260*52=13520>>13520 on music lessons in a year. The answer is: 1 -
```

Step 5 is labeled `−` because it computes total music spending instead of the difference between piano and clarinet — it cannot lead to the correct final answer.

### 5.2 Parsing: From Raw Entry to Records

`scripts/generate_probe_data.py` parses each entry into one record per step:

1. Extract the question (everything before "Step 1:")
2. Split on step boundaries using the regex `(Step \d+:.*?)\s*([+\-])\s*(?=Step \d+:|$)`
3. Strip `<<expr=result>>` calculator annotations (e.g. `<<3+5=8>>` → removed)
4. Build context = question + all prior cleaned steps

After parsing, Step 5 above becomes:

```python
{
  "context": "Janet pays $40/hour for 3 hours per week of clarinet lessons ... "
             "Step 1: Janet spends 3 hours + 5 hours = 8 hours per week on music lessons. "
             "Step 2: She spends 40 * 3 = 120 on clarinet lessons per week. "
             "Step 3: She spends 28 * 5 = 140 on piano lessons per week. "
             "Step 4: Janet spends 120 + 140 = 260 on music lessons per week.",
  "text":    "Step 5: She spends 260 * 52 = 13520 on music lessons in a year. The answer is: 1",
  "label":   0  # incorrect
}
```

### 5.3 Encoding: From Text to Latent

Each record is encoded as a single sequence fed to the SSAE:

```
[context tokens] + [<sep>] + [step tokens] + [<eos>]
```

The context and step are tokenized without per-part length limits. If the combined sequence exceeds `max_seq_len=2048`, the oldest context tokens are dropped from the left — always preserving the full step text. In practice no sequence in this dataset reached 2048 tokens, so truncation was never triggered. This corrects the previous encoding, which truncated context and step independently at 128 tokens each, silently dropping the most recent prior steps when context was long.

The SSAE encoder (Qwen2.5-0.5B backbone + sparse projector) runs a forward pass and returns the hidden state at the **last token position**, projected through the sparse bottleneck. This yields a 896-dimensional latent vector `h_c`.

The encoder applies L2 normalization to `h_c` so all vectors lie on the unit sphere.

**What a latent actually looks like** (first 20 of 896 dimensions):

```
Correct step:   [0.0024, 0.1553, 0.0, 0.0, 0.0, 0.1688, 0.0, 0.0, 0.0, 0.0029, ...]
                Active dims: 205/896

Incorrect step: [0.0993, 0.1904, 0.0, 0.0, 0.0, 0.0902, 0.0, 0.0, 0.0, 0.0583, ...]
                Active dims: 223/896
```

Across the training pool, incorrect steps activate on average **9.7 more dimensions** than correct steps (209.1 vs 199.4). This is the learnable signal the probe uses.

### 5.4 Data Provenance and Integrity

We encode from the Math-Shepherd GSM8K partition (172,283 problems, ~648K total steps). The dataset has a natural distribution of ~28.9% correct / 71.1% incorrect within offsets 0–375K. Beyond offset 375K the distribution degrades toward 100% correct-only steps (an artifact of dataset construction); we exclude this tail entirely.

Data is separated at the encoding stage by offset:

| File | Steps | GSM8K step offset | Distribution |
|---|---|---|---|
| `eval_held_out.npz` | 50,000 | 0 – 90K | **50.0% correct** (balanced, carved from raw shard) |
| `train_final.npz` | 450,000 | 90K – 450K | 33.3% correct (natural) |

**Eval was carved first.** Offset 0–90K (~90K natural steps, ~26K correct + ~64K incorrect) was encoded, then 25,000 correct and 25,000 incorrect steps were selected uniformly at random (seed=42) to form the balanced held-out set. The remaining steps from that shard were discarded.

**Training covers only offsets 90K–450K.** Four shards of 90K steps each (offsets 90K, 180K, 270K, 360K) were encoded in parallel on 4 H100 GPUs and merged. The shard at offset 360K is near the dataset tail and has a slightly elevated correct rate (53.1%), but remains within acceptable range for a 70/30-subsampled training regime.

There is no row overlap between eval and training by construction: they cover non-overlapping offset windows.

---

## 6. Probe Architecture and Training

**Architecture** (`src/probes/classifier.py`):

```
Linear(896 → 1024) → LayerNorm(1024) → ReLU → Dropout(0.1)
Linear(1024 → 512) → LayerNorm(512)  → ReLU → Dropout(0.1)
Linear(512 → 1)    [raw logit; sigmoid gives P(correct)]
```

**Training** (`scripts/experiment_full_clean.py`):
- Loss: binary cross-entropy on the single logit output
- Optimizer: AdamW, lr=1e-4, weight_decay=0.01
- Scheduler: cosine annealing over 50 epochs
- Batch size: 512
- 90/10 train/internal-val split from the training subset; best internal-val-accuracy checkpoint saved

**Training subset construction:** the full 450K training pool contains 149,769 correct (33.3%) and 300,231 incorrect (66.7%) steps. We subsample to 70% correct / 30% incorrect without duplication — keeping all 149,769 correct steps and drawing 64,186 incorrect steps — for 213,955 total. This replicates the distribution that produced the best result in prior local runs.

Note: L2 normalization means every h_c vector has Euclidean length exactly 1.0 — all latents lie on the surface of a 896-dimensional unit sphere. The magnitude of activations carries no information; only the *direction* of h_c matters. Whether correct and incorrect steps are geometrically separated on this sphere has not been tested. That is part of the step-2 feature analysis.

---

## 7. Results

### 7.1 Dataset for This Experiment

| Split | Steps | Correct | Incorrect |
|---|---|---|---|
| Training pool (encoded) | 450,000 | 149,769 (33.3%) | 300,231 (66.7%) |
| Training subset (70/30, no duplication) | 213,955 | 149,769 (70.0%) | 64,186 (30.0%) |
| Internal val (10% of subset) | 21,396 | — | — |
| Train (90% of subset) | 192,559 | — | — |
| **Held-out eval (balanced, never seen)** | **50,000** | **25,000 (50.0%)** | **25,000 (50.0%)** |

The eval set is balanced 50/50 and drawn from offset 0–90K, a window that was encoded before any training data was assembled. A balanced eval makes accuracy directly interpretable as discrimination ability — a majority classifier scores 50%.

### 7.2 Training Curve

Across 4 seeds the probe reaches best internal validation accuracy between epochs 10 and 24, then plateaus. Loss continues to fall through epoch 50 while val accuracy stays flat — 50 epochs is sufficient.

| Seed | Best internal val acc | At epoch |
|---|---|---|
| 42 | 76.91% | 20 |
| 43 | 77.07% | 24 |
| 44 | 76.78% | 11 |
| 45 | 76.11% | 10 |

### 7.3 Held-Out Evaluation

Evaluated on `eval_held_out.npz` (50,000 steps, 25,000 correct / 25,000 incorrect). Results reported at threshold=0.5 (default) and threshold=0.7 (macro-F1-optimal across all seeds).

**At threshold = 0.5:**

| Seed | Accuracy | Gain vs majority | F1 correct | F1 incorrect | Macro F1 |
|---|---|---|---|---|---|
| 42 | 70.24% | +20.24 pp | 0.760 | 0.609 | 0.684 |
| 43 | 69.91% | +19.91 pp | 0.758 | 0.603 | 0.680 |
| 44 | 67.54% | +17.54 pp | 0.747 | 0.547 | 0.647 |
| 45 | 69.13% | +19.13 pp | 0.757 | 0.577 | 0.667 |
| **Mean** | **69.21%** | **+19.21 pp** | **0.756** | **0.584** | **0.670** |
| **Std** | **1.12 pp** | **1.12 pp** | **0.006** | **0.026** | **0.016** |

At threshold=0.5, correct recall is very high (0.94–0.96) but incorrect recall is low (0.39–0.46): the probe over-predicts "correct" and misses most incorrect steps.

**At threshold = 0.7 (macro-F1 optimal):**

| Seed | Accuracy | Gain vs majority | F1 correct | F1 incorrect | Macro F1 |
|---|---|---|---|---|---|
| 42 | 74.52% | +24.52 pp | 0.750 | 0.740 | 0.745 |
| 43 | 74.23% | +24.23 pp | 0.757 | 0.726 | 0.741 |
| 44 | 74.86% | +24.86 pp | 0.760 | 0.735 | 0.748 |
| 45 | 74.88% | +24.88 pp | 0.760 | 0.736 | 0.748 |
| **Mean** | **74.62%** | **+24.62 pp** | **0.757** | **0.734** | **0.746** |
| **Std** | **0.31 pp** | **0.31 pp** | **0.005** | **0.006** | **0.003** |

Threshold 0.7 is optimal for macro F1 on all four seeds and substantially reduces the class imbalance in predictions, making both F1 scores nearly equal.

**Full threshold sweep (mean across 4 seeds):**

| Threshold | Accuracy | F1 correct | F1 incorrect | Macro F1 |
|---|---|---|---|---|
| 0.3 | 59.65% | 0.712 | 0.331 | 0.521 |
| 0.4 | 64.47% | 0.734 | 0.464 | 0.599 |
| 0.5 | 69.21% | 0.756 | 0.584 | 0.670 |
| 0.6 | 72.76% | 0.768 | 0.677 | 0.723 |
| **0.7** | **74.62%** | **0.757** | **0.734** | **0.746** |
| 0.8 | 72.89% | 0.699 | 0.754 | 0.726 |

### 7.4 Inference Latency (H100 GPU)

Measured with `cuda.synchronize()` fences, 1,000 single-step trials per seed.

| Seed | Mean latency | Std | p99 | Batch throughput (batch=512) |
|---|---|---|---|---|
| 42 | 0.107 ms | 0.002 ms | 0.112 ms | 5,242,825 steps/s |
| 43 | 0.109 ms | 0.004 ms | 0.124 ms | 5,171,185 steps/s |
| 44 | 0.113 ms | 0.006 ms | 0.137 ms | 5,039,260 steps/s |
| 45 | 0.109 ms | 0.042 ms | 0.114 ms | 5,217,241 steps/s |
| **Mean** | **0.110 ms** | — | **0.122 ms** | **~5.2M steps/s** |

Training time per seed: ~57s on a single H100.

### 7.5 Baselines

We ran two additional baselines on the exact same 50K held-out eval set to contextualise the MLP probe result.

**Linear probe** (`scripts/experiment_linear_probe.py`): a single linear layer `Linear(896 → 1)` with 897 parameters, trained on the same 213,955-step subset (70/30 imbalance) for 50 epochs. Identical eval procedure and threshold sweep.

| Seed | Accuracy (t=0.5) | Macro F1 (t=0.5) | Accuracy (t=0.7) | Macro F1 (t=0.7) |
|---|---|---|---|---|
| 42 | 69.69% | 0.693 | 74.30% | 0.742 |
| 43 | 69.81% | 0.695 | 74.29% | 0.742 |
| 44 | 69.87% | 0.695 | 74.36% | 0.743 |
| 45 | 69.76% | 0.694 | 74.29% | 0.742 |
| **Mean** | **69.78%** | **0.694** | **74.31%** | **0.742** |
| **Std** | **0.08 pp** | **0.001** | **0.03 pp** | **0.001** |

**LLM self-judge** (`scripts/llm_self_judge.py`): we pass the question, prior steps, and current step to the language model with a path-viability prompt ("Will continuing from this next step lead to the correct final answer? Yes/No") and score P(Yes) via softmax over the logits of the first assistant token. Two models, evaluated in 4 parallel shards across 4 H100 GPUs.

Results at the best threshold per model (threshold that maximises macro F1):

| Model | Best threshold | Accuracy | F1 correct | F1 incorrect | Macro F1 |
|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 0.3 | 44.35% | 0.529 | 0.225 | 0.377 |
| Qwen2.5-Math-7B-Instruct | 0.6 | 66.29% | 0.686 | 0.639 | 0.662 |

**Summary: MLP probe vs baselines (threshold=0.7, 50K eval)**

| Method | Params | Accuracy | Macro F1 |
|---|---|---|---|
| Majority classifier | 0 | 50.00% | 0.333 |
| Qwen2.5-0.5B self-judge | 494M | 44.35% | 0.377 |
| Qwen2.5-Math-7B self-judge | 7.6B | 66.29% | 0.662 |
| **Linear probe (SSAE latents)** | **897** | **74.31% ± 0.03** | **0.742 ± 0.001** |
| **MLP probe (SSAE latents)** | **1.4M** | **74.62% ± 0.31** | **0.746 ± 0.003** |

Three findings:

1. **The correctness signal is linearly decodable.** The 897-parameter linear probe (74.31%) nearly matches the 1.4M-parameter MLP probe (74.62%). The gap is 0.31 pp, within noise. This means the signal is not hidden in some curved manifold: correct and incorrect steps form nearly linearly separable regions on the 896-dimensional unit sphere.

2. **The SSAE latents outperform a 7B judge by 8 pp.** Qwen2.5-Math-7B-Instruct, a model fine-tuned for mathematical reasoning, scores 66.29% macro F1 at best threshold. The linear probe on 896-dim latents of the tiny 0.5B base model beats it by 8 pp — with 8 million times fewer parameters and without any language modeling.

3. **The 0.5B model cannot self-judge at threshold=0.5.** Qwen2.5-0.5B-Instruct scores below majority classifier (44.35%), indicating it is strongly biased toward predicting "Yes" on nearly everything. Its best result (44.35% at threshold=0.3) reflects this: even at the lowest threshold most examples are classified as correct.

---

## 8. Mechanistic Analysis of the Latent Space

The following analysis is based on the actual experiment-7 encodings: the 50,000-step balanced held-out set (`eval_held_out.npz`) and four trained linear probe checkpoints (`linear_probe_seed{42..45}.pt`). All statistics below supersede the earlier estimates from smaller-scale runs.

### 8.1 Sparsity: Incorrect Steps Use More Dimensions, but the Overlap is Large

Across the 50,000-step held-out eval, incorrect steps activate on average **6.8 more dimensions** than correct steps (204.1 vs 197.3 out of 896, std ≈ 29–32). This is consistent with the SSAE producing more compact representations for in-distribution steps (correct reasoning it was trained on) and more diffuse representations for out-of-distribution ones (incorrect steps it never saw during training). The same direction holds in the training pool (5.4 dim delta over 50,000 sampled steps).

However, the distributions overlap almost entirely — both are roughly bell-shaped with peaks separated by less than one standard deviation. Active dimension count alone cannot reliably discriminate correct from incorrect steps at the individual level. It is a signal, but not the signal the probe uses.

### 8.2 The Probe Has Found a Clean Directional Separation

Projecting every h_c vector onto the learned probe direction (w^T h_c) reveals two clearly separated distributions: correct steps cluster at positive values (~0.05–0.15), incorrect steps cluster at negative-to-zero values (~−0.10–0.05), with substantial but not complete overlap in the transition region. The correct distribution is notably tighter than the incorrect one.

This directional separation is what produces 74.31% accuracy from 897 parameters. The probe is reading a well-defined direction in the 896-dimensional latent space — not a noisy average over many weak signals, but a coherent geometric axis along which the two classes are meaningfully offset.

### 8.3 The Correctness Signal Is Not in the Dominant Variance Directions

PCA on the eval latents reveals that the first two components explain 18.1% and 14.0% of variance respectively. In the PC1–PC2 plane, correct and incorrect steps are completely intermixed — neither class forms a visible cluster or region. The probe weight direction projects only weakly onto PC1 (−0.027) and moderately onto PC2 (+0.183), with the remainder distributed across higher components.

This means the correctness signal is a secondary structure in the latent space. The dominant variance captured by PC1 and PC2 represents something else: likely mathematical content, step length, or problem type — dimensions along which correct and incorrect steps vary similarly. Correctness is encoded in a direction that contributes little to total variance but carries high discriminative information.

### 8.4 The Probe Is Not Reading Mean Activation Differences

The most mechanistically surprising result is the low correlation between probe weights and per-feature activation deltas: Pearson r = 0.237. The activation delta for any individual feature (mean_correct[i] − mean_incorrect[i]) is very small across all 896 dimensions — the largest is 0.056, and most are below 0.02. Yet the probe assigns weights ranging from −2.42 to +2.78. Large probe weights attach to dimensions with nearly zero mean activation difference between classes.

This rules out a simple explanation for why the probe works. The probe is not learning "dimension i is more active for correct steps, therefore weight it positively." It has learned a direction that is nearly orthogonal to the mean-difference direction, exploiting the covariance structure of the latent space rather than marginal activation levels. Put differently: the correctness information is not in any single dimension, but in the joint pattern of activations across many dimensions simultaneously.

This is consistent with the LDA (Linear Discriminant Analysis) picture: the optimal linear classifier need not align with the direction of largest class-mean difference — it aligns with the direction of maximum between-class variance relative to within-class variance. The probe has learned the LDA direction, which in a 896-dimensional space can be almost orthogonal to what univariate statistics would suggest.

### 8.5 The Class Geometry on the Unit Sphere

Mean cosine similarities confirm that both classes cluster mildly on the unit sphere. Within-class similarity is 0.605 (correct) and 0.614 (incorrect); between-class similarity is 0.575. The gap of ~0.03 is small but consistent on both sides — incorrect steps are slightly more internally coherent than correct steps, which aligns with the sparsity observation that incorrect steps have a narrower distribution of active dimensions.

This mild but consistent geometric separation is what the linear probe exploits. The probe's decision hyperplane bisects the sphere between the two class clusters.

### 8.6 Summary: What the SSAE Latent Space Encodes

The mechanistic picture that emerges is the following. The SSAE was trained to reconstruct correct GSM8K steps from a sparse latent. When applied to incorrect steps (out-of-distribution for the encoder), it produces latents that are geometrically shifted: slightly more diffuse (more active dimensions), slightly more internally coherent as a class (higher within-class cosine similarity), and offset along a specific direction in the latent space that is not aligned with the main axes of variation.

That direction — the probe direction — carries 74.3% accuracy for distinguishing correct from incorrect. It is linearly decodable (the MLP adds nothing over the linear probe), it is stable across four independent training seeds (std ≈ 0.03 pp), and it is not captured by any individual feature's marginal statistics. The signal is in the geometry of the joint activation pattern, distributed across nearly all 896 dimensions.

The open question remains why this direction exists. The reconstruction objective gives no explicit supervision about correctness. The most plausible account is that the SSAE learned a compressed representation of "what a valid arithmetic step looks like," and incorrect steps, which were never seen during training, fall systematically off this manifold in a consistent direction — one that a linear probe can recover.

---

## 9. Out-of-Distribution Evaluation: ProcessBench

The Math-Shepherd results (§7) were encouraging: a linear probe on SSAE latents detects step-level incorrectness at 74.3% accuracy, outperforming a 7B self-judge by 8 pp. This suggested the latent space does encode a meaningful correctness signal. The natural next question was how this compares to dedicated process reward models (PRMs) designed specifically for step-level verification. ProcessBench is the standard benchmark used to compare PRM methods, so we applied our methodology there to situate our results within the existing literature.

### 9.1 What ProcessBench Measures

ProcessBench (arXiv:2412.06559) tests a strictly harder task than the step-level binary classification used in §7: **first-error localization**. Each solution in the benchmark is annotated with the index of the first incorrect step, or -1 if all steps are correct. A method must scan the solution left-to-right and identify, at the solution level, exactly where reasoning first goes wrong.

The benchmark metric, **PB-F1**, is the harmonic mean of two solution-level accuracies:
- **Acc(correct)**: fraction of all-correct solutions correctly predicted as having no error
- **Acc(error)**: fraction of incorrect solutions where the predicted first-error position matches the ground truth exactly

PB-F1 = 2 × Acc(correct) × Acc(error) / (Acc(correct) + Acc(error))

This makes PB-F1 much harder to score on than per-step accuracy: the probe must not only detect that something is wrong but pinpoint the exact step where correctness first fails. Our per-step binary probe is converted to a first-error prediction by scanning left-to-right and flagging the first step where P(correct) falls below a threshold.

### 9.2 Dataset Statistics

| Split | Solutions | Correct (-1) | Incorrect | Steps encoded | Steps correct | Steps incorrect |
|---|---|---|---|---|---|---|
| GSM8K | 400 | 193 (48.2%) | 207 (51.8%) | 1,568 | 1,361 (86.8%) | 207 (13.2%) |
| MATH | 1,000 | 406 (40.6%) | 594 (59.4%) | 4,366 | 3,772 (86.4%) | 594 (13.6%) |

The step-level class distribution is severely imbalanced: only the first-error step (and only one per incorrect solution) is labeled incorrect. Everything before it -- correct steps in correct solutions, and all pre-error steps in incorrect solutions -- is labeled correct. This makes the majority baseline 86.8% (GSM8K) and 86.4% (MATH), but that baseline achieves 0% PB-F1 on incorrect solutions because it never predicts an error.

The encoding pipeline is identical to Math-Shepherd: `[context tokens] + [<sep>] + [step tokens] + [<eos>]`, same checkpoint (`gsm8k-385k_Qwen2.5-0.5b_spar-10.pt`), batch size 32, max sequence length 2048.

### 9.3 Results: ProcessBench GSM8K

Per-step binary classification (macro F1 at best threshold = 0.3 for all methods):

| Method | Best threshold | Step Accuracy | Step Macro F1 | PB-F1 |
|---|---|---|---|---|
| MLP probe seed 42 | 0.3 | 74.4% | 0.507 | 16.7% |
| MLP probe seed 43 | 0.3–0.5 | 70–76% | 0.524 | 20.1% |
| MLP probe seed 44 | 0.3 | 78.3% | 0.536 | 25.2% |
| MLP probe seed 45 | 0.4 | 77.3% | 0.545 | 27.5% |
| **MLP aggregate (4 seeds)** | — | — | **0.528 ± 0.016** | **22.4% ± 4.9%** |
| Linear probe seed 42 | 0.3 | 81.5% | 0.579 | 30.4% |
| Linear probe seed 43 | 0.3 | 81.8% | 0.582 | 30.3% |
| Linear probe seed 44 | 0.3 | 81.0% | 0.578 | 30.1% |
| Linear probe seed 45 | 0.3 | 80.4% | 0.579 | 29.9% |
| **Linear aggregate (4 seeds)** | — | — | **0.580 ± 0.002** | **30.2% ± 0.2%** |

Reference methods from the ProcessBench paper:

| Method | Model size | PB-F1 GSM8K |
|---|---|---|
| Math-Shepherd-PRM-7B | 7B | 47.9% |
| Qwen2.5-Math-7B-PRM800K | 7B | 68.2% |
| ActPRM (SOTA, Apr 2025) | 7B | ~75.0% |
| **Linear probe (ours)** | **0.5B backbone** | **30.2% ± 0.2%** |
| **MLP probe (ours)** | **0.5B backbone** | **22.4% ± 4.9%** |

### 9.4 Results: ProcessBench MATH

| Method | Step Macro F1 | PB-F1 |
|---|---|---|
| **MLP aggregate (4 seeds)** | **0.512 ± 0.016** | **13.5% ± 7.2%** |
| **Linear aggregate (4 seeds)** | **0.568 ± 0.008** | **25.7% ± 1.0%** |

MATH is harder than GSM8K: PB-F1 drops ~4–5 pp for the linear probe and ~9 pp for the MLP. The MATH split contains harder multi-step algebra and competition problems (from NuminaMath) where errors occur at more varied positions and the SSAE, trained on GSM8K-Aug, is less likely to have seen similar step patterns.

### 9.5 Interpretation

**Three findings from the ProcessBench evaluation:**

1. **Linear probe outperforms MLP on ProcessBench, reversing the Math-Shepherd pattern.** On the in-distribution Math-Shepherd eval (§7.5), the MLP and linear probe are nearly tied (74.62% vs 74.31% accuracy, a 0.31 pp gap). On ProcessBench, the linear probe leads by 7.8 pp in PB-F1 on GSM8K (30.2% vs 22.4%) and 12.2 pp on MATH (25.7% vs 13.5%). The MLP probe also shows much higher variance across seeds (4.9% std on GSM8K vs 0.2% for linear). This suggests the MLP has learned features specific to the Math-Shepherd training distribution that do not transfer cleanly to ProcessBench's Qwen2-7B-Instruct solutions. The linear probe, constrained to a single hyperplane, generalizes better.

2. **The probe is substantially below PRM baselines on PB-F1.** The best linear probe (30.2%) is 17.7 pp behind Math-Shepherd-PRM-7B (47.9%), a 7B model explicitly trained for step-level process reward modeling. The gap reflects both model scale (our backbone is 0.5B vs 7B) and task misalignment: the probe was trained on Math-Shepherd MC rollout labels and asked to perform exact first-error localization, a task that requires knowing not just that a step is wrong but that all prior steps were correct.

3. **The localization constraint is the bottleneck, not error detection.** At threshold 0.3 on GSM8K, the linear probe reaches Acc(correct) ~ 46% and Acc(error) ~ 22%, giving PB-F1 ~ 30%. As the threshold rises, Acc(correct) increases (fewer false alarms) but Acc(error) barely moves: the probe is flagging incorrect steps, just not at the right positions for exact first-error localization. The harmonic mean in PB-F1 penalizes any imbalance between the two accuracies harshly.

**On contamination:** Math-Shepherd draws from the same GSM8K problem distribution as ProcessBench's GSM8K split. Some Math-Shepherd solutions may overlap with ProcessBench training data. We cannot rule out partial contamination, but the low PB-F1 scores make contamination-driven inflation unlikely: a contaminated probe would be expected to perform well, not at 30%.

### 9.6 Hypothesis: What the Reconstruction Objective Cannot Capture

The poor ProcessBench results point toward a structural limitation of our current approach. The SSAE latent `h_c` is trained to encode the content of a step well enough to reconstruct it. It has no training signal that relates one step to the next. A step can be fluent, arithmetically correct in isolation, and faithfully reconstructed by the SSAE, yet still be wrong in the context of the full reasoning chain: it may pursue the right sub-goal by the wrong path, or arrive at a locally correct intermediate value that leads the solution away from the final answer.

This category of error is exactly what ProcessBench's first-error localization task is designed to expose. Our probe, trained on per-step binary labels, has no way to detect it: the latent `h_c` does not represent the relationship between step k and step k+1.

Our hypothesis is that adding a next-step prediction auxiliary loss to the SSAE training objective would force `h_c` to encode information about the information flow between steps, not just the content of the current one. A sparse latent that must both reconstruct step k and predict step k+1 would be pressured to encode whether the current step is a coherent continuation of the reasoning so far, and whether it sets up a viable path to the next step. This would make the latent sensitive to the kind of contextually misleading but locally valid steps that our current probe misses entirely.

This is the motivation for the Future-SSAE experiments described in §10.

## 10. Future-SSAE Validation Run

### 10.1 Experimental Setup

We ran a first validation experiment to test whether adding a next-step prediction objective can make SSAE latents encode trajectory-level information, without using correctness labels.

Both models were trained for 2 epochs on GSM8K-Aug:

- 384,620 raw problems
- 521,120 valid step-transition pairs
- Qwen2.5-0.5B backbone
- `alpha_pred = 0.1`
- single H100
- shared sparse latent `h_c` between reconstruction and prediction branches

Two variants were compared:

| Model | Prediction context |
|---|---|
| M1 | question + current step |
| M2 | question + previous step + current step |

The original SSAE objective is preserved: reconstruct the current step from context and sparse latent. The added objective asks the same latent to help predict the next step. This tests whether `h_c` can encode not only local step information, but also information about the reasoning trajectory.

### 10.2 Training Behavior

Both models trained stably.

The reconstruction loss decreased from about `2.47` to `0.88` for both M1 and M2. This shows that the original SSAE behavior is preserved: the sparse latent still captures enough current-step information for reconstruction.

The prediction branch also learned. During training, `pred_first_token_nll` dropped from about `6.1` to `3.5`, a reduction of roughly `2.6` nats. This is important because the first token of the next step has no next-step prefix available to the decoder, so improvement here indicates that gradient is flowing through the sparse latent rather than only through teacher-forced language modeling.

However, learning plateaued quickly. Between epoch 1 and epoch 2, reconstruction improved only slightly and prediction metrics were almost flat. This suggests that the current setup is not compute-limited by two epochs. It is more likely signal-limited: with `alpha_pred = 0.1` and a transition distribution biased toward shallow first-step pairs, the latent is not pushed strongly enough toward deeper trajectory encoding.

### 10.3 Validation Diagnostics

| Metric | M1 ep1 | M1 ep2 | M2 ep1 | M2 ep2 |
|---|---:|---:|---:|---:|
| val_recon_nll | 0.807 | 0.786 | 0.787 | 0.763 |
| val_pred_nll | 0.866 | 0.867 | 0.793 | 0.799 |
| val_pred_first_token_nll_idx0 | 2.928 | 2.914 | 2.938 | 2.929 |
| val_pred_first_token_nll_idx1 | 3.583 | 3.555 | 3.400 | 3.409 |
| val_pred_first_token_nll_idx2+ | 3.827 | 3.837 | 3.777 | 3.809 |
| val_pred_nll_idx0 | 0.755 | 0.748 | 0.752 | 0.748 |
| val_pred_nll_idx1 | 0.892 | 0.885 | 0.769 | 0.766 |
| val_pred_nll_idx2+ | 1.073 | 1.076 | 0.936 | 0.942 |
| val_corr_recon_pred | 0.807 | 0.797 | 0.827 | 0.819 |
| n_idx0 / n_idx1 / n_idx2+ | 281 / 179 / 169 | 281 / 179 / 169 | 281 / 179 / 169 | 281 / 179 / 169 |

### 10.4 Interpretation

The result is not a failure, but it is not yet enough for ProcessBench.

The first clear result is that the prediction branch is real. The training-time drop in first-token NLL shows that the prediction objective is using `h_c`; it is not being ignored.

The second result is that M2 is better than M1 on the transitions that matter. At epoch 2, M2 reaches `val_pred_nll_idx2+ = 0.942`, compared with `1.076` for M1. It also improves `val_pred_first_token_nll_idx1` from `3.555` to `3.409`. Adding the previous step gives the decoder useful local trajectory context, especially beyond the first transition.

The third result is that the current setup mostly learns shallow transitions. The easiest bucket is always `idx0`, which corresponds to predicting the next step after the first reasoning step. This is expected because GSM8K-Aug is shallow: more than half of the transition pairs come from `idx0`. The deeper buckets, especially `idx2+`, remain much harder.

The fourth result is that reconstruction and prediction are not competing. `val_corr_recon_pred` stays strongly positive around `0.8`. This means examples that are hard to reconstruct are also hard to predict. The latent is not being pulled in opposite directions by the two losses. Therefore, `alpha_pred = 0.1` is safe, but likely too weak.

### 10.5 Decision

We should keep the `q_prev1_current` setting and drop `q_current` for now.

M2 is consistently better on deeper transitions, and deeper transitions are the relevant regime for first-error detection. ProcessBench requires detecting when a reasoning trajectory breaks, not just predicting the first obvious step from the question.

The next experiment should increase trajectory pressure.

Recommended next run:

| Model | Prediction context | alpha_pred | Data |
|---|---|---:|---|
| M3 | q_prev1_current | 0.3 | original GSM8K-Aug pairs |
| M4 | q_prev1_current | 0.3 | downsampled idx0 or upweighted idx1/idx2+ |

The goal is to reduce the gap between shallow and deeper transitions:

```
idx0 first-token NLL should stay good
idx1 and idx2+ first-token NLL should decrease
val_corr_recon_pred should remain > -0.5
reconstruction NLL should not degrade sharply
```

### 10.6 Current Conclusion

This validation run supports the direction but exposes the bottleneck.

Future-SSAE does learn a trajectory signal, but with `alpha_pred = 0.1` and the current GSM8K-Aug transition distribution, the signal is mostly shallow. The method is working mechanically, but the next step is to push the latent harder toward deeper reasoning dynamics through stronger prediction loss and step-index rebalancing.

In short: the prediction branch is valid, the latent learns trajectory information, and the current setting is too weak and too biased toward first-step transitions to expect strong ProcessBench performance yet.

---

## 11. Overcomplete SSAE Ablation (c=4, TopK-40, Frozen Encoder)

### 11.1 Motivation

The c=1 mechanistic analysis (§8) found that the correctness signal in the SSAE latent space is not carried by any individual feature's marginal statistics (probe-delta correlation r=0.237), but is distributed across the joint activation pattern. One hypothesis is that the bottleneck at c=1 forces polysemantic encoding: features that would ideally carry a single interpretable concept are instead mixed, and a linear probe must recover correctness from an entangled representation. Expanding the dictionary to c=4 (n_latents=3584, four times the input dimension) should allow the autoencoder to disentangle features into more monosemantic units, potentially making the correctness signal more linearly accessible.

Two other weaknesses of the c=1 experiment motivated this ablation:

- **ReLU+L1 sparsity** requires tuning the L1 penalty across training, and dynamic weight adjustment adds noise to the training signal. TopK activation (keep exactly k features per sample) enforces exact structural sparsity without any penalty schedule.
- **Unfrozen encoder** at c=1 allowed the backbone to co-adapt with the sparse projector, potentially conflating representation learning with compression artifacts. Freezing the encoder forces the sparse projector to compress a fixed representation, making the latent geometry more interpretable.

### 11.2 Experimental Setup

| Setting | Value |
|---|---|
| Architecture | SSAE, c=4, n_inputs=896, n_latents=3584 |
| Activation | TopK (k=40) — keeps 40 of 3584 features active per step |
| Encoder | Frozen (Qwen2.5-0.5B backbone, no gradient) |
| dtype | bfloat16 |
| Epochs | 8 |
| Batch size | 16 (grad_accum=8, effective batch=128) |
| Training data | gsm8k_385K_train.json (385K steps, TamIA $STORE) |
| Hardware | 4x H100 80GB (TamIA, job 264277) |
| Training time | ~10.2h (8 × ~77 min/epoch) |
| Checkpoint | ssae_c4_topk40_frozen.pt (best val loss) |

The no-grad + detach fix was required for the frozen encoder: setting `requires_grad_(False)` alone is insufficient because PyTorch still allocates activation memory during the forward pass. The encoder forward is wrapped in `torch.no_grad()` and the output `h_k` is detached before entering the sparse projector. Without this, the 80GB H100 OOMs at batch_size=64.

Probe training used the same protocol as §6: 4 seeds (42–45), 50 epochs, AdamW, 70/30 class imbalance correction, evaluated on the same 50K balanced held-out set.

### 11.3 Results

**MLP probe — threshold sweep (mean ± std, 4 seeds):**

| Threshold | Accuracy | F1 correct | F1 incorrect | Macro F1 |
|---|---|---|---|---|
| 0.3 | 69.13 ± 0.09% | 0.764 ± 0.000 | 0.555 ± 0.002 | 0.659 ± 0.001 |
| 0.4 | 69.95 ± 0.15% | 0.768 ± 0.001 | 0.575 ± 0.004 | 0.671 ± 0.002 |
| 0.5 | 71.50 ± 0.18% | 0.774 ± 0.001 | 0.615 ± 0.005 | 0.694 ± 0.003 |
| 0.6 | 73.41 ± 0.15% | 0.778 ± 0.001 | 0.669 ± 0.005 | 0.723 ± 0.002 |
| **0.7** | **74.54 ± 0.03%** | **0.766 ± 0.004** | **0.720 ± 0.004** | **0.743 ± 0.000** |
| 0.8 | 72.05 ± 0.48% | 0.695 ± 0.011 | 0.742 ± 0.001 | 0.719 ± 0.006 |

**Linear probe — threshold sweep (mean ± std, 4 seeds):**

| Threshold | Accuracy | F1 correct | F1 incorrect | Macro F1 |
|---|---|---|---|---|
| 0.3 | 60.32 ± 0.92% | 0.716 ± 0.005 | 0.343 ± 0.026 | 0.529 ± 0.015 |
| 0.4 | 66.74 ± 0.60% | 0.750 ± 0.003 | 0.504 ± 0.014 | 0.627 ± 0.009 |
| 0.5 | 69.69 ± 0.25% | 0.764 ± 0.001 | 0.578 ± 0.007 | 0.671 ± 0.004 |
| 0.6 | 71.68 ± 0.21% | 0.763 ± 0.001 | 0.649 ± 0.009 | 0.706 ± 0.004 |
| **0.7** | **71.99 ± 0.35%** | **0.725 ± 0.010** | **0.715 ± 0.003** | **0.720 ± 0.003** |
| 0.8 | 65.90 ± 1.10% | 0.571 ± 0.028 | 0.717 ± 0.003 | 0.644 ± 0.016 |

### 11.4 Comparison with c=1 and Dense Baselines

| Configuration | Probe | Accuracy (t=0.7) | Macro F1 (t=0.7) |
|---|---|---|---|
| c=1, ReLU+L1, unfrozen | MLP | 74.62 ± 0.31% | 0.746 ± 0.003 |
| c=1, ReLU+L1, unfrozen | Linear | 74.31 ± 0.03% | 0.742 ± 0.001 |
| **c=4, TopK-40, frozen** | **MLP** | **74.54 ± 0.03%** | **0.743 ± 0.000** |
| **c=4, TopK-40, frozen** | **Linear** | **71.99 ± 0.35%** | **0.720 ± 0.003** |

### 11.5 Interpretation

**The MLP probe is unchanged; the linear probe is worse.** The c=4 MLP nearly exactly matches c=1 (−0.08 pp accuracy, −0.003 macro F1 — well within noise given the seed variance). The linear probe however drops 2.32 pp in accuracy and 0.022 in macro F1.

This reverses the hypothesis. The overcomplete expansion was expected to disentangle features and make the correctness signal more linearly accessible. Instead, the linear probe degrades while the MLP holds steady. The gap between linear and MLP probes has widened: at c=1 the linear probe trails the MLP by only 0.31 pp; at c=4 the gap is 2.55 pp.

A plausible explanation is that the overcomplete dictionary distributes the correctness signal across more dimensions. In a 3584-dimensional space with only 40 active features per sample, the correctness direction may require a non-axis-aligned projection that a linear probe over sparse activations cannot learn as efficiently with the same training budget. The MLP, with its first-layer nonlinearity, can still recover the signal; the linear probe cannot find it as cleanly in the sparser, higher-dimensional space.

A second factor is the frozen encoder. The c=1 results were produced with an unfrozen backbone, which may have aligned the embedding space with the probe training signal during SSAE training. With the encoder frozen, the sparse projector must compress a fixed backbone representation, and the resulting latent geometry is determined entirely by the backbone's existing structure rather than being shaped by the reconstruction-plus-probe training interaction.

**The bottom line is that overcomplete expansion with TopK does not improve probe-based correctness detection on this task.** The MLP probe result is stable across both architectures, which is reassuring, but the hypothesis that feature disentanglement would improve linear decodability is not supported here.

---

## 12. Five-Encoding Mechanistic Comparison (c=2 ReLU+L1 and Activation SAE)

### 12.1 Motivation

Sections §7–11 established two SSAE variants in isolation: c=1 (ReLU+L1, unfrozen encoder) and c=4 (TopK-40, frozen encoder). These differ in two confounded dimensions: dictionary size (1× vs 4× input) and sparsity type (soft L1 vs hard TopK). Adding c=2 ReLU+L1 with a frozen encoder fills the gap: it isolates the effect of dictionary size from sparsity type, providing a cleaner ablation ladder. An off-the-shelf Activation SAE (standard sparse autoencoder applied directly to backbone activations, without step-level conditioning) is included as a non-SSAE reference point.

The goal of this comparison is not to identify the best-performing encoding but to characterize how each encoding distributes the correctness signal across its latent dimensions.

### 12.2 c=2 Experimental Setup

| Setting | Value |
|---|---|
| Architecture | SSAE, c=2, n_inputs=896, n_latents=1792 |
| Activation | ReLU + L1 (soft sparsity -- no fixed k) |
| Encoder | Frozen (Qwen2.5-0.5B backbone, no gradient) |
| dtype | bfloat16 |
| Epochs | 8 |
| Batch size | 16 (grad_accum=8, effective batch=128) |
| Training data | gsm8k_385K_train.json (385K steps, TamIA $STORE) |
| Hardware | 4× H100 80GB (TamIA) |

ReLU+L1 rather than TopK is the deliberate choice for c=2: the soft penalty lets each step activate as many features as it needs, which is the right inductive bias for monosemanticity. TopK enforces an identical sparsity budget on every step regardless of content. For a dictionary twice the input dimension, L1 is more appropriate because the overcomplete space should be able to represent most steps with fewer active dimensions on average -- the penalty controls sparsity pressure without hard-coding it.

### 12.3 Five-Encoding Comparison

The five encodings compared are:

| Encoding | Type | Latent dim | Sparsity | Encoder |
|---|---|---|---|---|
| Dense h_k | None | 896 | None | Unfrozen (c=1 backbone) |
| SSAE c=1 | SSAE | 896 | ReLU+L1 | Unfrozen |
| SSAE c=2 ReLU+L1 | SSAE | 1792 | ReLU+L1 | Frozen |
| SSAE c=4 TopK* | SSAE | 3584 | TopK-40 | Frozen |
| Activation SAE | SAE | 896 | (post-hoc) | -- |

The comparison uses 5,000 balanced steps per encoding from the held-out Math-Shepherd eval set. Figure 12.1 shows PCA scatter (probe direction overlaid), probe score histograms, probe-weight vs. activation-delta scatter, and active dimension distributions for all five encodings simultaneously.

![Five-encoding comparison -- Math-Shepherd eval](results/mechanistic_comparison/comparison_ms.png)
*Figure 12.1: Five-encoding mechanistic comparison on 5,000 balanced Math-Shepherd eval steps. Columns left to right: Dense h_k, SSAE c=1, SSAE c=2 ReLU+L1, SSAE c=4 TopK*, Activation SAE. Rows: PCA scatter with probe direction arrow; probe score histograms (blue=correct, red=incorrect); probe weight vs. activation delta scatter; active dimension distributions.*

### 12.4 Probe-Delta Correlation Across Encodings

The key quantity is the Pearson r between per-dimension activation deltas (mean_correct[i] − mean_incorrect[i]) and probe weights w[i]. High r means the probe is essentially a scaled mean-difference classifier; low r means the probe exploits covariance structure beyond marginal statistics. This is the same diagnostic used in §8.4 for the c=1 single-encoding analysis.

| Encoding | MS r | PB r |
|---|---|---|
| Dense h_k | 0.590 | 0.201 |
| SSAE c=1 | 0.239 | 0.170 |
| **SSAE c=2 ReLU+L1** | **0.133** | **0.123** |
| SSAE c=4 TopK* | 0.634 | 0.103 |
| Activation SAE | 0.485 | 0.157 |

Three findings:

1. **SSAE c=2 has the lowest MS r (0.133) of all five encodings.** The correctness direction in c=2 is maximally orthogonal to the marginal mean-difference direction -- more so than c=1 (r=0.239) and far more so than c=4 (r=0.634). Expanding the dictionary with soft sparsity forces the probe to rely on covariance structure even more than the narrower c=1 dictionary.

2. **SSAE c=4 TopK has the highest MS r among SSAE variants (0.634), comparable to dense representations (0.590).** Hard TopK sparsity creates systematic marginal differences: by always activating exactly k=40 features per step, class-conditional mean activations become more pronounced. The probe at c=4 essentially learns the mean-difference direction, not a covariance-based one. This is the same finding as §8.4 and §11.5 from a different angle: TopK appears to make the latent space easier to read via simple statistics, but not more linearly discriminative (the linear probe accuracy actually drops at c=4 -- §11.4).

3. **On ProcessBench, all r values converge to a low range (0.103--0.201), with no clear ordering.** The structural properties learned from Math-Shepherd (whether marginal- or covariance-based) do not transfer their character to ProcessBench. All five encodings lose most of their in-distribution directional specificity OOD.

**Interpretation.** The ReLU+L1 SSAE variants (c=1 and c=2) produce latent spaces where the correctness signal is more structurally distributed than either dense representations or TopK-sparse ones. The particularly low r for c=2 (0.133) is consistent with a larger dictionary: with 1,792 dimensions available under a soft penalty, the autoencoder can spread correctness-relevant structure across many low-delta features, making individual marginal statistics less informative and forcing the probe to use joint activation patterns. This is the desired behavior for a monosemantic dictionary -- each feature should capture a narrower concept, so no single feature's mean activation cleanly separates the classes.

The Activation SAE (r=0.485 on MS) behaves similarly to Dense h_k: the post-hoc sparse decomposition of backbone activations does not break the probe's reliance on marginal differences the way step-level SSAE training does. The step-level conditioning -- requiring the sparse latent to encode only the incremental information contributed by each step -- appears to be the structural property responsible for the low r in c=1 and c=2.

---

## 13. Model-Size Ablation: Dense Hidden-State Decodability Across Scales

### 13.1 Motivation

The experiments in §7–12 used a single backbone (Qwen2.5-0.5B) throughout. Two questions were left open: does step-level correctness become more linearly decodable as the backbone scales up, and does a probe trained on Math-Shepherd transfer to ProcessBench when the backbone has no SSAE on top? This ablation addresses both with a controlled pilot: dense hidden states only (last transformer layer, last token), no SAE, four Qwen2.5 model sizes (0.5B / 1.5B / 3B / 7B), one seed.

The two representation variants -- raw and L2-normalised -- are run in parallel, motivated by the observation in §6 that L2 normalisation projects out magnitude information. Running both allows us to test whether magnitude carries correctness signal.

### 13.2 Experimental Design

**Backbone:** Qwen2.5-0.5B, 1.5B, 3B, 7B (base, not instruct). Last transformer layer, last token. Left-padded batches so position `[:, -1]` is always the final real token.

**Representation:** raw float16 hidden states saved from the extraction job; L2 normalisation applied at probe time via `--repr {raw, l2}`. Both variants share the same extracted states.

**Probe:** sklearn `LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)`. Threshold sweep on MS eval only (maximise Macro-F1); same threshold applied to ProcessBench without re-tuning (strict OOD).

**Data:**
- Math-Shepherd: 10,000 + 10,000 steps per class (balanced) for train; 5,000 + 5,000 for eval. Split IDs materialised to `splits.json` before extraction to guarantee identical examples across all four model jobs.
- ProcessBench: full GSM8K split (1,568 steps). Used as zero-shot OOD only; no threshold re-tuning.

**Metrics:** AUROC, Macro-F1 at best-threshold, positive prediction rate (PPR).

### 13.3 Results

| Model | dim | MS AUROC | MS F1 | PB AUROC | PB F1 | PB PPR |
|-------|-----|----------|-------|----------|-------|--------|
| **Raw representation** | | | | | | |
| Qwen2.5-0.5B | 896 | 0.763 | — | 0.483 | — | — |
| Qwen2.5-1.5B | 1536 | 0.790 | — | 0.649 | — | — |
| Qwen2.5-3B | 2048 | 0.805 | — | 0.634 | — | — |
| Qwen2.5-7B | 3584 | 0.835 | — | 0.496 | — | — |
| **L2-normalised** | | | | | | |
| Qwen2.5-0.5B | 896 | 0.698 | 0.648 | 0.502 | 0.492 | 0.856 |
| Qwen2.5-1.5B | 1536 | 0.701 | 0.658 | 0.572 | 0.524 | 0.840 |
| Qwen2.5-3B | 2048 | 0.709 | 0.656 | 0.542 | 0.503 | 0.955 |
| Qwen2.5-7B | 3584 | 0.691 | 0.654 | 0.433 | 0.263 | 0.191 |

### 13.4 Figure Analysis

#### Fig 01 — Probe score histograms (L2 repr)

![Probe histograms](results/ms_ablation/01_probe_histograms.png)

The four panels reveal two distinct regimes. For 0.5B and 1.5B, the MS distributions show partial separation: the incorrect peak (red) sits left of threshold, correct (blue) extends right. The PB distributions (dashed) are shifted almost entirely toward the positive (correct) side -- PPR of 0.856 and 0.840 respectively. The probe is predicting "correct" for most ProcessBench steps regardless of true label, yielding near-random discrimination despite high recall on the positive class.

The 3B panel is an extreme version: PB dashed distributions collapse almost entirely to the right (PPR=0.955). The threshold-based F1 breaks down completely; AUROC of 0.542 is the only meaningful number here.

The 7B panel shows a dramatic reversal: PB distributions shift left (PPR=0.191). The probe now predicts mostly incorrect for ProcessBench steps. PB incorrect steps happen to be classified as negative, but PB correct steps are massively misclassified too, collapsing AUROC to 0.433 (below chance). The probe direction has inverted polarity relative to what ProcessBench requires.

**The probe scores on ProcessBench are never calibrated.** The threshold tuned on balanced MS eval maps arbitrarily onto PB's different class distribution and geometry.

#### Fig 02 — PCA scatter

![PCA scatter](results/ms_ablation/02_pca_scatter.png)

The geometry explains the OOD failure. At 0.5B, MS (circles) and PB (triangles) occupy largely the same cloud; the probe arrow is approximately aligned with the PC1 axis; PB correct/incorrect triangles show weak but consistent ordering. At 1.5B, MS splits more cleanly along PC1 and PB follows partially. By 3B, PB triangles cluster tightly in one corner of the MS cloud -- the probe direction still separates MS, but the PB cluster lands in the high-score region regardless of step correctness. At 7B, the shift is most dramatic: PB triangles occupy a region geometrically anti-aligned with the MS probe direction, causing the polarity reversal visible in the histograms.

**The domain shift is geometric, not a labeling artifact.** Larger models produce representations where Math-Shepherd and ProcessBench GSM8K steps occupy increasingly divergent regions of representation space.

#### Fig 03 — AUROC vs model size

![AUROC vs model size](results/ms_ablation/03_auroc_vs_size.png)

This is the most informative figure in the set. The two curves tell completely different stories depending on representation:

- **L2 (from the main table):** MS AUROC is flat at ~0.70 across all sizes. No scaling trend. PB AUROC peaks at 1.5B (0.572) and collapses at 7B (0.433, below chance).
- **Raw (from fig 08):** MS AUROC increases monotonically -- 0.763, 0.790, 0.805, 0.835. A genuine, clean scaling signal. L2 normalisation destroys it entirely.

The curve shown here (L2) is the one the report table reflects. See fig 08 for the separation.

#### Fig 04 — Macro-F1 vs model size

![Macro-F1 vs model size](results/ms_ablation/04_macro_f1_vs_size.png)

MS F1 is essentially constant across sizes (0.648–0.658), consistent with the flat L2 AUROC. PB F1 peaks at 1.5B (0.524) then collapses to 0.263 at 7B. The F1 curves track PPR instability rather than probe quality: as PPR shifts chaotically across sizes, the threshold-based F1 fluctuates with it.

#### Fig 05 — Positive prediction rate vs model size

![PPR vs model size](results/ms_ablation/05_ppr_vs_size.png)

The most diagnostically useful figure. MS PPR is stable near 0.5 for all sizes (the balanced eval set is 50/50, and the threshold was tuned to maximise F1 on it). PB PPR is 0.856, 0.840, 0.955, 0.191 -- chaotic, not a trend. ProcessBench is not balanced (most steps in a solution are correct; only the first-error step and those after are incorrect). The base rate mismatch alone would push PB PPR high. But the 7B reversal (PPR=0.191) shows a different effect dominates: the 7B representation maps PB into the negative side of the probe.

#### Fig 06 — Confusion matrices

![Confusion matrices](results/ms_ablation/06_confusion_matrices.png)

**MS rows (top):** All four models are qualitatively similar. TP and TN are both around 63-67%, with symmetric false positive/negative rates. The probe makes genuine weak distinctions, not degenerate predictions.

**PB rows (bottom):** 0.5B, 1.5B, 3B almost exclusively predict "correct" (Pred+ column is dark for all true classes). True incorrect steps are nearly entirely misclassified as correct. At 7B the matrix flips: most PB steps are predicted incorrect, true correct steps are massively misclassified. The confusion matrix resembles a near-identity rotated 90 degrees -- the polarity inversion made explicit.

#### Fig 07 — Activation delta vs probe weights

![Delta vs weights](results/ms_ablation/07_delta_vs_weights.png)

Pearson r values: 0.5B=0.996, 1.5B=0.854, 3B=0.981, 7B=0.998.

For 0.5B, 3B, and 7B, the near-perfect correlation (r>0.98) means the probe is doing almost nothing beyond a scaled mean-difference classifier. The logistic regression finds the mean(correct) − mean(incorrect) direction and scales it by regularised magnitude. This is functionally a centroid classifier.

**1.5B is the outlier at r=0.854.** The probe at 1.5B extracts a direction that is not purely the mean-difference direction -- it exploits the covariance structure of the latent space for something beyond the first moment. This is directly consistent with 1.5B having the best OOD transfer: its representation has richer structure that the probe can use for generalisation.

The x-axis range (mean delta magnitude) also grows with model size, largest at 7B. This means 7B encodes correctness with higher-magnitude directional differences from MS training -- but those large differences are precisely the ones that do not generalise to ProcessBench, because they reflect MS-specific features amplified at scale.

#### Fig 08 — Raw vs L2 AUROC

![Raw vs L2](results/ms_ablation/08_raw_vs_l2.png)

The central methodological finding of the ablation.

**MS AUROC:**
- Raw: 0.763 → 0.790 → 0.805 → 0.835 (monotonically increasing, ~7 points per step)
- L2: 0.698 → 0.701 → 0.709 → 0.691 (flat, no trend)

L2 normalisation costs 7–15 AUROC points on MS and eliminates the scaling signal entirely. The magnitude of hidden states carries real correctness information. Larger models do not just change the direction of the correctness vector -- they increase its magnitude relative to noise.

**PB AUROC:**
- Raw: 0.483 → 0.649 → 0.634 → 0.496 (peaks 1.5B, collapses at 7B)
- L2: 0.502 → 0.572 → 0.542 → 0.433 (peaks 1.5B, collapses at 7B)

Both representations show the same OOD story. Raw is better at the sizes that matter. If only L2 had been run, the MS scaling signal would have been missed entirely.

### 13.5 Answers to the Research Questions

**Q1 — Does in-domain decodability improve with model size?**
Yes, but only with raw representations. Raw MS AUROC improves monotonically from 0.763 to 0.835 across 0.5B → 7B. L2 normalisation destroys this signal. If only L2 had been run, the conclusion would have been: no.

**Q2 — Does OOD transfer improve with model size?**
No. Transfer peaks at 1.5B (raw PB AUROC=0.649) and collapses at 7B (0.496 raw, 0.433 L2). Larger models encode correctness more strongly in-domain but the representation geometry increasingly diverges from ProcessBench.

**Q3 — Is ProcessBench geometrically shifted relative to Math-Shepherd?**
Yes, and increasingly so with model size. The PCA shows MS and PB clouds are nearly co-located at 0.5B and clearly diverged by 7B. This is not a scaling factor -- it is a change in distribution shape and orientation.

**Q4 — Does the learned probe direction separate PB correct/incorrect?**
Only weakly and only at 1.5B (raw AUROC 0.649). All other sizes are near chance or below. The probe captures a MS-specific correctness direction that partially but unreliably projects onto ProcessBench.

**Q5 — Is 0.5B too small, or is the bottleneck dataset transfer?**
Not capacity. 0.5B has reasonable MS AUROC (0.763 raw) and is no worse on PB than 7B (both ~0.50 raw). The bottleneck across all sizes is the domain gap between Math-Shepherd annotation style and ProcessBench solutions.

**Q6 — Which size for PTB/SSAE follow-up?**
**1.5B.** Best OOD transfer (PB AUROC 0.649 raw), probe goes beyond pure mean-difference direction (r=0.854 vs r>0.98 for others), and solid MS AUROC. Its representations sit at the sweet spot where correctness is linearly encoded but not yet specialised to MS-specific features that do not transfer.

### 13.6 Implication for the SAE Program

These results set a concrete OOD baseline: raw hidden-state linear probe on 1.5B achieves PB AUROC 0.649. Any SSAE-based approach on 1.5B must exceed this to be worth pursuing as an OOD method.

The failure mode here is clear: the linear probe latches onto a MS-specific combination of features that is large in norm at scale but not robust across datasets. SAE features, being semantically decomposed directions, may find a subspace that generalises better than the raw logistic direction -- because the SAE is pressured to extract directions corresponding to meaningful concepts rather than dataset-specific covariance patterns. That is the mechanistic hypothesis motivating the continuation of the SSAE program with a 1.5B backbone.

---

## 14. Perplexity as a Label-Free Anomaly Score: PSSAE Experiment

### 14.1 Motivation

All representations evaluated so far -- dense h_k, delta_h, PTB, SSAE latents -- require a trained probe to produce a correctness score. The probe needs labeled training data (Math-Shepherd MC rollout labels), which introduces a distribution dependency: the probe latches onto a MS-specific direction that does not cleanly transfer to ProcessBench solutions from a different model family.

This section asks a simpler question: can the backbone's own generative distribution serve as a correctness signal, with no labels at all? The hypothesis is that a model trained exclusively on correct reasoning transitions will assign higher perplexity to the next step when the current step is wrong, because the current step is OOD from the training manifold. This is analogous to density estimation: the encoder has learned what a valid transition looks like, and anomalous inputs produce reconstruction failures detectable as high perplexity.

### 14.2 Architecture

The trainable component is a single linear layer `enc_proj: R^896 -> R^896` (800K parameters) on top of a frozen Qwen2.5-0.5B backbone. For step k:

1. **Encode:** run the frozen backbone on `[problem + prior_steps | SEP | step_k | EOS]`, extract the last-token hidden state `h_k`, project to a sparse bottleneck: `z_k = ReLU(enc_proj(h_k))`.
2. **Decode:** inject `z_k` as a single virtual token into the decoder -- `[embed(problem) | z_k | embed(step_{k+1})]` -- and compute the teacher-forced cross-entropy loss on step k+1 tokens.

The backbone is frozen throughout. The only thing that trains is `enc_proj`. At evaluation, the decoder cross-entropy (converted to perplexity) is the anomaly score: high perplexity means `z_k` failed to summarize the current step in a way that predicts the next.

### 14.3 Training Setup

Two variants were trained in parallel on separate H100s within a single job:

| Setting | Value |
|---|---|
| Training data | `gsm8k_385K_train.json`, first 50%, positive-only filter |
| Train transitions (capped) | 400,000 |
| Optimizer | Adam, lr=1e-3 |
| Batch size | 32 (batched backbone calls) |
| Epochs | 5 |
| Backbone precision | bfloat16 (frozen) |
| Variant A | L1 coefficient = 0.01 |
| Variant B | L1 coefficient = 0.0 (ablation) |

Positive-only filter: any solution with at least one step failing `symbolic_step_judge` is excluded. Training transitions come exclusively from arithmetically correct reasoning chains.

Training CE converged from ~1.13 (epoch 1) to ~1.09 (epoch 5), a modest but consistent decrease indicating `enc_proj` learned to compress step-level information useful for next-step prediction.

### 14.4 MS OOD Results

Evaluated on the first 50% of `gsm8k_385K_valid.json` (all solutions, not positive-only). Labels from `symbolic_step_judge`. The eval set had 413 correct steps and 23 wrong steps -- a heavily imbalanced set, so the AUROC confidence intervals are wide.

| Metric | L1 (0.01) | No-L1 |
|---|---|---|
| Correct steps mean PPL | 3.59 | 3.59 |
| Wrong steps mean PPL | 6.07 | 5.72 |
| Wrong steps std PPL | 7.94 | 7.37 |
| Ratio wrong/correct | 1.693 | 1.594 |
| AUROC | **0.619** | 0.598 |
| Verdict | supported | not supported |

Both variants produce higher perplexity on wrong steps. L1 sparsity widens the gap slightly (ratio 1.693 vs 1.594) and pushes AUROC above the 0.6 threshold. The key structural observation is that wrong step perplexity has very high variance (std > mean) -- the distribution is heavy-tailed. A minority of wrong steps produce large perplexity spikes; the majority sit only slightly above the correct-step baseline. The OOD signal is real but concentrated in the tail.

### 14.5 ProcessBench Results

Evaluated on the first 50 solutions from `processbench_gsm8k.jsonl`. All steps of each solution are encoded, including steps after the first error.

| Metric | L1 (0.01) | No-L1 |
|---|---|---|
| Correct steps mean PPL | 3.38 | 3.31 |
| Wrong steps mean PPL | 4.63 | 4.65 |
| Wrong steps std PPL | 4.12 | 4.69 |
| Ratio wrong/correct | 1.370 | 1.406 |
| Argmax hit rate | 18/50 = **0.360** | 19/50 = **0.380** |
| Random baseline | 0.193 | 0.193 |
| Best F1 | 0.219 (p95) | 0.247 (p75) |

Argmax hit rate is approximately 2x the random baseline for both variants, confirming the perplexity signal generalizes to ProcessBench. No-L1 has a slightly higher hit rate and better average-case F1, but L1 has higher precision at tight thresholds (p95: 0.500 vs 0.333; p99: 0.667 vs 0.600). L1 sparsity trades recall for precision.

The PB-F1 values (best 0.247) are computed with thresholds derived from the correct-step perplexity distribution of the eval data itself (percentiles p50--p99), so they are not tuned on held-out labels.

### 14.6 Effect of L1

L1 and no-L1 produce identical correct-step perplexity (mean 3.59 on MS, 3.31-3.38 on PB). The difference is entirely in how wrong steps are encoded. L1 compresses most `z_k` dimensions toward zero; the few that remain active carry a stronger signal when a wrong step triggers them. This creates sharper anomaly spikes for the most egregious errors (higher AUROC on MS, higher precision at tight thresholds on PB) at the cost of missing moderate wrong steps whose signal is compressed below the noise floor (lower average-case F1 and argmax hit rate on PB).

The L1 variant is preferable for high-confidence error flagging. The no-L1 variant is preferable for maximizing coverage.

### 14.7 Comparison with Probe-Based Representations

The table below situates the PSSAE in the context of all representations evaluated in this project. Probe-based results are from the PTB robust probe evaluation (§10); the SSAE linear probe PB-F1 is from §9.

**MS OOD AUROC (step-level correctness):**

| Representation | Signal | MS AUROC |
|---|---|---|
| Dense h_k | linear probe (MS labels) | 0.834 |
| PTB variants | linear probe (MS labels) | 0.828--0.835 |
| Dense delta_h | linear probe (MS labels) | 0.801 |
| **PSSAE L1** | **perplexity (no labels)** | **0.619** |
| **PSSAE no-L1** | **perplexity (no labels)** | **0.598** |

The PSSAE is ~0.21 AUROC points below the probe-based methods. That gap is the cost of being label-free: the probe uses 200K+ labeled training steps; the PSSAE uses none.

**ProcessBench first-error localization:**

| Representation | Signal | Eval scope | PB-F1 |
|---|---|---|---|
| SSAE linear probe (§9) | linear probe (MS labels) | full GSM8K split | 0.302 |
| **PSSAE no-L1** | **perplexity (no labels)** | **first 50 solutions** | **0.247** |
| **PSSAE L1** | **perplexity (no labels)** | **first 50 solutions** | **0.219** |
| Dense h_k | linear probe (MS labels) | full GSM8K split | 0.0 |
| Dense delta_h | linear probe (MS labels) | full GSM8K split | 0.0 |
| PTB variants | linear probe (MS labels) | full GSM8K split | 0.0 |

**Important caveat on the ProcessBench comparison:** the SSAE probe (0.302) was evaluated on the full ProcessBench GSM8K split (~80 solutions); the PSSAE was evaluated on the first 50. These are different samples. The 5.5 pp gap between PSSAE no-L1 and the SSAE probe cannot be interpreted as a real performance difference without re-running on the same solution set.

What the table does cleanly show is that probe-based approaches trained on Math-Shepherd labels collapse entirely on ProcessBench (PB-F1=0.0 for all PTB and dense representations), while the perplexity-based PSSAE does not. The label-free approach achieves worse in-distribution discrimination but stronger OOD robustness, because it has no dataset-specific direction to overfit.

### 14.8 Summary

The PSSAE demonstrates that a single linear layer trained on positive-only transitions can produce a label-free correctness signal via perplexity. The signal is real (2x random on ProcessBench argmax, AUROC 0.619 on MS OOD with L1), moderate in strength, and more robust OOD than probe-based alternatives. L1 sparsity helps for high-confidence detections and hurts for average-case recall. The main limitation is the small wrong-step count in the MS OOD eval set (n=23), which makes the AUROC estimates noisy. A larger or more balanced evaluation would sharpen the conclusions.

---

## 15. PRM800K Layer × Token Geometry: Where and How Correctness Is Encoded (S3 Stage 2)

S3 Stage 1 concluded that incorrectness is a diffuse property at the last-layer / last-token readout
and flagged the readout choice as the open confound: mid-layer or step-internal tokens might still
carry the structure. Stage 2 pre-paid that ablation with a single multi-layer / multi-token encode and
then dissected the resulting signal. The conclusion is sharper than Stage 1: across every layer and
both tokens the correctness signal is a genuine but very low-variance, low-effect-size **linear
direction**, linearly decodable yet invisible to any variance- or neighborhood-based embedding, and it
never organizes into clusters.

### 15.1 Setup

One whole-node 4-GPU pass encoded the natural PRM800K held-out test set: 6,000 steps, balanced 3,000
correct (rating +1) and 3,000 incorrect (rating −1), capturing the **first and last token of each
step at layers 11, 17, 20, 22, 25, 28** (depth fractions 0.39 to 1.0) of Qwen2.5-7B, hidden size
3,584. The result is a 4D tensor (6,000 × 6 layers × 2 tokens × 3,584). Decodability throughout is
5-fold balanced logistic-regression accuracy with a majority floor of 0.50. Because the set is the
rating ±1 extremes, this is an in-distribution and optimistic ceiling, not directly comparable to the
ProcessBench numbers in §9 and §13.

Scripts: `scripts/encode_prm800k_multitoken_multilayer.py` (+ `merge_prm800k_multitoken_shards.py`),
`scripts/analysis/s3_prm800k_layer_projection.py`, `s3_prm800k_probe_anatomy.py`,
`s3_prm800k_minimal_subspace.py`, `s3_prm800k_supervised_view.py`. Outputs under
`results/prm800k_layers/` (gitignored).

### 15.2 Layer × token sweep

| layer (depth) | first-token acc | last-token acc |
|---|---|---|
| L11 (0.39) | 0.681 | 0.694 |
| L17 (0.61) | 0.683 | 0.737 |
| **L20 (0.71)** | 0.697 | **0.742** |
| L22 (0.79) | 0.688 | 0.727 |
| L25 (0.89) | 0.694 | 0.731 |
| L28 (1.00) | 0.682 | 0.731 |

(ROC-AUC where measured: L20/last 0.808, L28/last 0.784, L20/first 0.729, L28/first 0.709.)

Two robust effects:

- **Last token beats first at every depth.** Last-token decodability peaks at L20 (0.742) and
  plateaus near 0.73 to the final layer; first-token is flat near 0.68 throughout. Only the last token
  has attended over the entire step, so it integrates step content; the first token is essentially the
  step's opening token. The first-token signal is also qualitatively worse: row-normalizing hurts it
  and confound-residualizing barely changes it, meaning what little it has leans on magnitude and
  confounds. Actionable: the readout should be last-token (or mean-pool), never first.
- **L20 slightly beats the final layer.** L20/last exceeds L28/last on accuracy and AUC and after
  every control below (cosine 0.766 vs 0.737, residual 0.711 vs 0.691). Mid-late layers hold the most
  abstract step-validity representation before the final layer re-specializes toward next-token
  prediction.

### 15.3 What the probe reads: confound audit (L20/last)

Incorrect steps are systematically longer (median 246 vs 193 tokens) and later (step_idx 5.2 vs 4.4),
so the obvious worry is that the probe is a length/position detector. It is not, mostly:

| readout | acc | auc |
|---|---|---|
| length + position only | 0.583 | 0.627 |
| raw last-token norm only | 0.556 | 0.575 |
| full hidden state | 0.742 | 0.808 |
| cosine (row-normalized, magnitude removed) | 0.766 | 0.839 |
| residualized (length + position regressed out of every dim) | 0.711 | 0.763 |

The lift over floor is 0.242; after linearly removing length and position it is still 0.211, so about
87% of the lift is content, not confound (probe-score correlation with length is only r=+0.21). The
mechanism is **direction, not magnitude**: removing magnitude (cosine) improves decodability and the
raw norm alone is near-useless. Length and position are real contaminants worth fixing in the data,
but they are not the mechanism.

### 15.4 Why no clusters: a 0.01%-variance linear margin

The signal lives in a distributed, low-variance subspace roughly orthogonal to the dominant variance:

- Top-1 PC decodes at 0.517 (floor); top-10 PCs only 0.635. Removing the top-1 PC still leaves 0.741;
  you must strip roughly 100 to 200 PCs before decodability collapses.
- The logistic probe direction holds **0.4 of 3,584 total variance, i.e. 0.01%**, while the top
  residual PC (the kind of direction a 2D map's axes are built from) holds about 600 times more.
- The class means along the probe axis are only **d' = 1.19 standard deviations apart** (AUC 0.808),
  i.e. heavy overlap, a shift rather than two blobs.

This resolves the apparent paradox between strong linear decodability and an unstructured map. UMAP and
PCA choose axes that maximize variance and preserve neighborhoods; those axes are topic, problem, and
length, all orthogonal to correctness. A supervised projection onto the probe axis displays the shift
plainly (`supervised_view_L20_last.png`); an unsupervised map spends both dimensions on the 99.99% of
variance that is not correctness and shows nothing.

### 15.5 Minimal separating subspace and the cluster test

An L1 sparse probe recovers the full signal from a small subset of activations:

- **235 of 3,584 dims (6.6%)** retain full accuracy (0.747); about 920 dims peak at 0.774. Selecting
  on a train half and scoring the held-out half gives 0.733, so the minimal set generalizes.
- Per-dim effect sizes confirm the elimination is meaningful: median |Cohen's d| = 0.10, only 664 dims
  with |d| > 0.2. Roughly 93% of activations are statistically common to both classes and droppable.

But isolating the discriminative subspace does **not** make it cluster. Against a shuffled-label null
selected the same way:

| subspace (235 dims) | 2D-map label separability (kNN) | HDBSCAN purity |
|---|---|---|
| minimal (discriminative) | 0.660 | 0.61 |
| common (eliminated) | 0.645 | 0.56 |
| shuffled-label control | 0.654 | 0.56 |

The minimal subspace separates labels in 2D at 0.660, barely above the shuffled null of 0.654, and far
below its own linear decodability (0.747). Correctness is a distributed linear direction, not a
localizable cluster of "error neurons."

### 15.6 Takeaways

- Stage 1's "incorrectness is diffuse" now holds across all six layers and both tokens, with a
  mechanism: a low-variance (0.01%), low-effect-size (d' ≈ 1.2) linear direction. No cluster structure
  exists for any (layer, token).
- Two concrete actions: switch the readout to last-token or mean-pool (never first); use L20 rather
  than the final layer as the probe site, for a small but consistent gain.
- The natural balanced ceiling is about 0.74, optimistic because it is the rating ±1 extremes.
- This is the strongest case yet for the deferred SAE stage. Only a learned sparse basis can allocate a
  dedicated coordinate to a 0.01%-variance direction and turn this linear margin into discrete,
  interpretable, clusterable features. Caveats: residualization removes only linear confound (matched
  to the linear probe), and the held-out set is the artificial balanced extremes.

### 15.7 What the direction encodes: ruling out every confound (S3 Stage 4)

§15.1–15.6 established *where* and *how* the correctness margin sits (a 0.01%-variance linear
direction, last-token, L20≈L28). This stage asks *what it is*: genuine correctness, or a confound the
linear probe happens to track. We eliminated candidates in order, each with its own A/B control. The
readout under test throughout is the deployed L28/last DenseLinear probe; the per-step pooled score AUC
is 0.665→0.736 across Qwen2.5 1.5B→32B.

1. **Matched-fork audit** (`analyze_fork_representation_audit.py`, 1000 PRM800K forks/size,
   anchor/positive/negative sharing a prefix). The supervised margin `w·(h_neg−h_pos)` is a real,
   non-surface, **scaling** correctness signal: P(neg>pos) 0.686→0.782 across sizes; it survives
   residualizing out length+lexical (7B 0.749→0.776) and holds on the surface-matched minimal-edit
   subset (0.723). Surface explains only ~13–20% of margin variance (ridge CV R², peak 7B 0.205),
   almost all of it `length_diff`. *Falsified* the audit's own stronger hypothesis that matched
   differencing **unsupervisedly** isolates the direction (cos(μ_Δ, w)≈0.07 at every size) — same
   low-variance story as §15.4.

2. **Per-step driver analysis** (`inspect_margin_drivers.py`). The pooled score is **not** length
   (removing it nudges AUC *up*), and removing length+step_idx+numeric+has_answer together drops 7B AUC
   only 0.699→0.694. The strongest cheap correlate is `numeric` (partial corr 0.31) but it is
   label-neutral (corr with label 0.10) — a computational-step modulation, not the correctness signal.

3. **Confidence/perplexity battery — GATE A** (`encode_fork_confidence.py` +
   `analyze_confidence_battery.py`; the user's prime suspect). Surprise is ruled out harder than length.
   (i) *Sufficiency:* each of {nll_mean, nll_max, entropy_mean, logit_gap_mean} alone predicts the label
   at AUC ~0.47–0.56 vs probe 0.699. (ii) *Subsumption:* removing all four moves probe AUC 0.699→0.697
   (fraction of lift removed 0.01–0.02 at every size). (iii) *No confound:* each feature's corr with the
   label is tiny (−0.01..+0.09). (iv) *Reverse:* each feature collapses to AUC ~0.45–0.55 once the score
   is removed. Cleanest tell: with scale, raw surprise **drops** (nll_pos 0.860→0.672) while probe AUC
   **rises** — opposite trends, so they cannot be the same signal.

4. **On-policy control — GATE B** (`generate_onpolicy_steps.py` → encode → `analyze_onpolicy_probe.py`).
   The forks still use *human-written* negatives, so the probe could read "off-distribution human text"
   rather than wrong reasoning. Decisive control: test on the model's **own** generations, which are
   uniformly low-perplexity by construction. 7B, 1200 generated trajectories (53.2% incorrect by
   final-answer match), 11679 steps. The manipulation worked — on-policy step NLL 0.617 vs fork-negative
   NLL 0.804 (drop +0.187 nats) — yet the probe still separates: **trajectory AUC 0.720 (95% CI
   0.691–0.748)**, step AUC 0.615. Surprise/OOD-text is dead as an explanation.

**The honest F1 caveat (the headline metric).** At a *threshold*, the same on-policy result is
unflattering, and deliberately so: this set is 53% incorrect, so the trivial "predict-all-incorrect"
F1 = 2p/(1+p) is already 0.69 (traj) / 0.75 (step). Trajectory **oracle** F1 0.728 clears trivial 0.694,
but the deployable **val-selected** F1 0.686 dips just *under* it, and step F1 = trivial *exactly*
(0.747). A 0.72-AUC signal on a near-balanced positive-heavy set cannot push max-F1 far past the
all-positive corner. F1's prevalence-dependence works against the probe here. On the *natural* PRM800K
test (§ Stage 3, 25% incorrect) the same probe scores F1 0.58 vs a 0.40 trivial bar — a clear win. So
the probe is a useful thresholded detector; the on-policy set is the right experiment for the *causal
dissociation* but the wrong base rate for the *deployable F1 headline*, which belongs on the natural
distribution.

**Takeaway.** After ruling out length, numeric density, step position, answer-presence, model
surprise/perplexity (GATE A) **and** off-distribution-ness (GATE B), the residual is correctness. It is
genuine, distributed, low-variance, and scales with backbone size — the same content §15.4 located
geometrically. Next: name the direction `w` (logit-lens through `W_U`, per-token DLA, optional steering)
then the SAE stage; report the deployable F1/reliability statement on the natural held-out test.
Artifacts under `runs/fork_rep_audit/<tag>/` (gitignored); stage plan
`~/.claude/plans/piped-wishing-platypus.md`.

### 15.8 Is `w` causal? Activation steering during generation (S3 Stage 5)

§15.7 established that the residual is correctness, distributed and scaling. That is still
correlational: `w` reads the model's state. Stage 5 asks whether `w` is a causal lever or only a
diagnostic readout, by the test from the research plan: add `+/- alpha * s_layer * w_hat` to the
residual stream of Qwen2.5-7B during the model's own generation and measure whether final-answer
correctness moves with a dose-response, against matched-norm control directions. We inject at the
decoder block whose output is `hidden_states[L]` for L in {20, 28}: L20 is upstream so layers
21 to 28 can react (the strong test), L28 is the deployed probe's own space (a shallow readout
test). `w_hat` is the unit probe direction in each layer's space (L20 trained on the layer cache,
L28 the deployed probe), oriented toward correct so `alpha > 0` should repair and `alpha < 0`
should corrupt. `s_layer` is the median residual norm, so `alpha` is a fraction of typical
residual magnitude. Final answers are graded by `src/eval/math_grade.grade`. Code:
`scripts/build_steering_directions.py`, `scripts/s1ms_steer_generate.py`,
`scripts/analyze_steer_causality.py`, and `scripts/s1ms_steer_forks.py --directions_npz` for the
teacher-forced Tier-0 pre-check; launcher `slurm/s1_model_size/run_steer_causality_7b.sh`.

**Calibration.** `s_layer` at 7B/L20 is about 100. At `alpha = 2` the injected vector exceeds the
residual norm and destroys generation regardless of direction: P(correct) collapses to ~0, the
gradeable rate craters, the probe logit is driven to about +/-835 (baseline -1.3), and meandiff at
+2 emits length-3 outputs. The usable regime is small fractions. Steering is applied only at decode
steps (`--steer_scope generated`), leaving the problem prompt untouched, which removes a
prompt-corruption confound.

**Tier-0 readout moves.** The teacher-forced fork battery confirms the sign and that the readout is
movable: steering toward correct raises the fork margin `logp(correct) - logp(incorrect)` (probe
0.074 at alpha=0 to 0.441 at alpha=+2) and drives the probe logit down. So the direction does shift
the model's relative preference between pre-written correct and incorrect sibling continuations.

**Tier-1 behavior does not move.** On 100 PRM800K problems x 4 samples (n=400 per cell, paired to
the alpha=0 baseline by problem and sample index), generated-scope, gradeable=1.000 throughout
(P(correct) baseline 0.375, dP standard error about 0.03):

| layer | dir | alpha | dP | repair | corrupt |
|---|---|---|---|---|---|
| L20 | probe | -0.2 | +0.010 | 0.144 | 0.213 |
| L20 | probe | -0.1 | +0.010 | 0.104 | 0.147 |
| L20 | probe | +0.1 | -0.003 | 0.112 | 0.193 |
| L20 | probe | +0.2 | -0.047 | 0.084 | 0.267 |
| L20 | random | +0.2 | -0.008 | 0.116 | 0.213 |
| L28 | probe | -0.1 | +0.023 | 0.132 | 0.160 |
| L28 | probe | +0.1 | -0.010 | 0.096 | 0.187 |
| L28 | random | +0.2 | -0.013 | 0.120 | 0.233 |

Three readings, all pointing the same way:

1. **No antisymmetric repair/corrupt pattern.** A causal `w` needs `alpha > 0` to repair and
   `alpha < 0` to corrupt. The faint trend is the opposite (toward incorrect mildly helps, toward
   correct mildly hurts: L20 probe +0.2 gives dP -0.047 and the highest corruption 0.267) and every
   effect is under about 1.5 standard errors.
2. **probe is indistinguishable from random.** At each matched alpha the probe and a random
   direction of equal norm differ by at most 0.04 in dP. At +0.2 the probe corrupts slightly more
   (0.267 vs 0.213) and repairs slightly less (0.084 vs 0.116) than random.
3. **The only real effect is direction-agnostic degradation.** Larger `|alpha|` raises corruption
   for both probe and random; that is perturbation, not correctness control. The intermediate
   `alpha = 0.4` run already showed symmetric degradation (corrupt 0.529 at L20 probe +0.4), so no
   repair window hides between the clean regime (`|alpha| <= 0.2`) and destruction (`|alpha| >= 1`).

**Verdict.** For additive residual steering at L20 and L28 on 7B, `w` is a **diagnostic readout,
not a causal variable**: the readout moves (Tier-0 margin and probe logit shift with alpha) while
on-policy correctness does not, and the probe direction is no more effective than a random push.
This is exactly the disambiguation the plan set up. Caveat: this falsifies additive steering only.
It does not test activation patching (swapping the `w`-component between matched correct and
incorrect runs), which is the stronger causal probe and the one intervention that could still flip
the verdict. Otherwise the program moves to naming `w` (logit-lens through `W_U`, per-token DLA) and
the SAE stage. Artifacts under `runs/fork_rep_audit/qwen2_5_7b/steer_causality/` (gitignored).

---

## 16. Combining All Layers for the Dense Probe

§15 established that the correctness signal is a distributed linear direction present at every
layer, not a property of the last layer alone. That raises an engineering question for the
deployable pipeline of §13: if every layer carries a slice of the signal, does concatenating all
of them into one probe input beat the single last-layer probe on ProcessBench PB-F1?

An offline check on the balanced held-out set (§15 data, 6,000 steps, 5-fold out-of-fold) said
maybe. Stacking the six stored layers lifted last-token ROC-AUC from about 0.80 (best single layer,
L17, in this 5-fold setup) to 0.847 leakage-free (concatenate raw features plus strong L2) and 0.854
via a per-layer score meta-probe. Two guardrails emerged and both matter below: the gain needed
strong L2 (LogisticRegression C=0.05 clearly beat C=1.0), and it was purely linear (QDA and kNN
density underperformed the linear probe, and PCA compression to 256 dims dropped AUC to 0.83). Like
all §15 numbers this is an optimistic in-distribution ceiling on the rating ±1 extremes, not a
ProcessBench figure.

### 16.1 Setup

We repeated the small-scale §13 DenseLinear experiment (Qwen2.5-7B, `probe_train_40k` / `val_1k`,
ProcessBench PB-F1) changing only the representation. Every one of the 28 transformer layers is
captured in a single forward pass, concatenated column-wise into a 100,352-dim vector (28 × 3,584),
and z-scored per dimension with statistics fit once on the training split and reused for val and every
ProcessBench subset. Standardization is required, not cosmetic: the DenseLinear probe trains on raw
hidden states with AdamW, and layers whose norms differ by ~100x (deep layers, massive-activation
dimensions) otherwise swamp the gradient. Everything downstream is byte-identical to §13: the same
probe trainer, the same PRM800K-val balanced-accuracy threshold selection, the same first-error-scan
PB-F1 evaluator, the same aggregator.

Scripts: `scripts/encode_prm800k_multilayer.py`, `scripts/encode_processbench_multilayer.py`,
`scripts/assemble_multilayer_concat.py` (the one new piece: concat plus z-score into the existing
`{stem}_h.npy` contract), driven by `slurm/s1_model_size_multilayer/run_multilayer_dense_7b.sh`.
Offline analysis in `scripts/analysis/s3_repr_separation.py` and `s3_repr_combine.py`.

### 16.2 Result: a small oracle gain, a val-selected regression

| representation | macro F1 oracle | macro F1 val-selected |
|---|---|---|
| last layer only (§13 baseline) | 0.4132 | 0.2369 |
| all 28 layers, concatenated (wd=0) | **0.4333** | 0.1781 |
| delta | **+0.020** | −0.059 |

Per-subset oracle PB-F1: gsm8k 0.454 → 0.479, math 0.447 → 0.440, olympiadbench 0.377 → 0.419,
omnimath 0.375 → 0.396. The oracle gain is broad (three of four subsets up) and driven by
olympiadbench (+0.041) and gsm8k (+0.026).

The val-selected score dropped 0.059, concentrated in gsm8k (0.394 → 0.205). The cause is the probe,
not the representation: trained with `weight_decay=0.0`, the 100,352-dim probe fit PRM800K-val
slightly worse than the 3,584-dim single-layer probe (balanced accuracy 0.750 vs 0.766), and its
selected threshold moved from 0.5 to 0.3, which over-flags errors on ProcessBench and collapses
Acc(correct).

### 16.3 Weight-decay sweep

We swept AdamW L2 over {0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0}, reusing the cached features (seconds per
point, no re-encode; `slurm/s1_model_size_multilayer/sweep_weight_decay_7b.sh`).

| weight_decay | macro F1 oracle | macro F1 val-selected |
|---|---|---|
| 0 / 1e-4 / 1e-3 | 0.4333 | 0.1781 |
| 1e-2 | 0.4332 | 0.1774 |
| **1e-1** | 0.4319 | **0.2190** |
| 1.0 | 0.4191 | 0.1886 |

Decay below ~1e-1 is inert: AdamW's decoupled decay at lr=1e-3 is negligible, so 0 through 1e-2 give
identical numbers. Only 1e-1 bites, and it is the best trade-off: it recovers about 70% of the
val-selected drop (0.178 → 0.219, still 0.018 short of the 0.237 baseline) while holding oracle at
0.432 (+0.019). At 1.0 the probe over-regularizes and both metrics fall. L2 recovers most of the val
regression but does not push oracle further (best oracle is at wd=0). Plot:
`results/repr_combine/wd_sweep.png`.

### 16.4 Verdict

Combining all 28 layers is a small, real win, not a breakthrough. At the best configuration (wd=1e-1)
it is strictly better on oracle PB-F1 (0.432, +0.019, about 5% relative) and essentially tied on the
deployable val-selected metric (0.219, −0.018). Three observations converge on why the ceiling is
low. The offline balanced-heldout AUC gain (0.80 → 0.85) does not convert into PB-F1, so detection is
not the bottleneck; §9 and §13 already located it at first-error localization. The signal is linear
and distributed, so nothing beyond a regularized linear combination helps (offline: QDA, kNN, PCA all
lose). And static last-token geometry plateaus around 0.85 AUC regardless of how the layers are
combined. The remaining lever is a different signal class, behavioral uncertainty such as semantic
entropy of resampled continuations, not more of the same static representation.

Artifacts: `results/repr_combine/wd_sweep.png` and `results/repr_combine/combine_ranking.png`; run
outputs under `runs/s1_model_size_dense/qwen2_5_7b_multilayer/` (gitignored).

---

## 17. Side Study: Parametric Retrieval Access (WikiProfile, Qwen2.5-7B-Instruct)
*Updated: 2026-07-11*

### Context

This side study asks whether, when Qwen2.5-7B-Instruct fails a direct factual question, the correct answer is already represented in its pre-generation hidden state, and whether the difference between successful and failed retrieval is causal and fact-independent. It replaces the earlier parametric_retrieval_geometry_v0 framing (four behavioral retrieval classes, a retrieved-vs-non-retrieved probe at AUROC 0.79, SAE steering that tracked random controls): that probe predicted a behavioral outcome, not answer identity, and its CoT-state transplant used a donor that had already retrieved the answer.

### What Was Done

The experimental unit is the same-fact mixed-outcome paraphrase pair. Stage 0 (`scripts/parametric_retrieval/prga_build_prompts.py`) exploded all 2,150 WikiProfile facts into 12 direct paraphrases per fact and direction (2 seed questions x 6 instruction wrappers, all ending "Question: ...\nAnswer:") plus one canonical CoT prompt, 55,510 instances total, 390 answer-leaking prompts dropped, fact-disjoint 60/20/20 splits frozen before any GPU job. Stage 1 ran greedy-only generation (TamIA job 366552, 5m32s). Stage 2 (`prga_pairs.py`) graded outputs and selected groups with at least 2 successful and 2 failed paraphrases. Stage 3 (`prga_extract.py`, job 366553) stored the residual stream at all 29 hidden_states indices and 7 token positions for 20,170 instances (136,177 rows, 53 GB on $SCRATCH). Stage 4 scored every row with a first-token logit lens against 32-candidate sets (gold + WikiProfile MC distractors + type/category/popularity-matched answers from other facts). Experiments A-D (job 366559, 13m20s) then ran a candidate-ranking decoder, a success-vs-fail accessibility probe, same-fact activation patching with six control conditions, and access-subspace steering. Layer and alpha selection used the validation split only; test facts were evaluated once.

### Results

The dataset yielded 1,163 mixed-outcome groups (27.2% of 4,272 fact x direction groups) and 9,304 matched pairs. Direct greedy accuracy is 0.247; CoT accuracy 0.308.

**Answer identity is present before generation, including in failures.** On test mixed groups at hs_idx 27, final prompt token, the gold answer ranks first among 32 matched candidates in 37.5% of failed paraphrases (chance 3.1%) and in the top 5 in 79%. Successful paraphrases reach 62.7% hits@1. Reasoning-unlocked groups (0 of 12 direct paraphrases correct, CoT correct; 146 groups) show hits@1 0.469 pre-generation, so Experiment E (dynamic emergence during CoT) was not triggered: the spec conditioned it on the prompt-state signal being weak or absent.

**Which paraphrase succeeds is not linearly readable.** The best success-vs-fail probe over 48 layer x position cells (fact-grouped 5-fold CV) reaches AUC 0.584 and does not beat a probe on surface confounds alone (template, seed question, direction, popularity, category, prompt length: AUC 0.589). Residualizing the states drops the probe to 0.504.

**Same-fact patching is causal and fact-specific.** Transplanting the donor paraphrase's stored residual state at hs_idx 26, final prompt token (h <- h + alpha(h_donor - h), alpha=1, selected on validation) into the failed paraphrase raises the gold-minus-distractor margin by +1.34 (fact-bootstrap 95% CI [0.48, 2.21]) on test and flips 44.9% of failures to exact match, with donor-answer copying at 0.0. Successful donor states from other facts destroy the margin (-3.89 same answer type and category, -4.05 popularity-matched), norm-matched random noise gives -0.59, the self-patch control gives +0.02, and patching failed states into successful prompts breaks 48.5% of successes.

**No fact-independent access direction exists.** Directions estimated on train facts from confound-residualized paired differences (mean difference, SVD subspaces, regularized LDA, relation-conditioned means) were applied to failed test instances at the best validated configuration (LDA, hs_idx 28, alpha 4 x mean edit norm). The learned direction raises logP(gold) by +5.42 versus +5.27 for a norm-matched random direction, exact-match rates are 8.7% versus 9.0%, and the gold-minus-distractor margin is negative for every direction. The candidate-ranking decoder with frozen mean input-embedding answer targets underperforms the first-token lens (hits@1 0.05 versus 0.34-0.63), so the answer-identity claim currently rests on the lens readout.

Figure: `results/parametric_retrieval_access_v1/prga_v1_results.png`. Tables: `runs/parametric_retrieval_access_v1/exp{A,B,C,D}/`.

### Interpretation

The results match outcome C2 of the experiment specification. The model often holds the correct answer in its decision-point state when it answers wrongly, and that state causally rescues the failure, but only when it carries the same fact: the transferable object is retrieved content, not a generic access mechanism. Steering reproduces the v0 gauge result on a fair paired design: any large edit inflates gold probability without preferentially favoring the gold over distractors. "Knowing that it knows" is not encoded as one controllable direction in this model. This supports the S4 knowledge-boundary reading of the CoT-verification signal: step incorrectness detection keys on whether fact-specific content was retrieved, and no shortcut direction can substitute for the content itself.

### Next Step

Close the decoder gap: replace the frozen mean input-embedding answer targets with unembedding-based or learned answer representations so the pre-generation answer-identity claim no longer depends on the first-token lens; then test component-level patching (attention versus MLP output) at hs_idx 24-26 to localize what carries the fact-specific content.

---

## 18. Latent Transition Operator: Predicting a Step's Downstream Effect (transition_operator_v0)
*Updated: 2026-07-16*

### Context

This study asks whether a compact latent trained to predict a reasoning step's downstream computational effect represents the transition more semantically, by operation type reusable across problems, and more robustly, by correctness, than raw boundary activations, deltas, or pooling. It extends the S4 contribution-geometry thread from describing what a step's representation looks like to predicting what the step did.

### What Was Done

The experimental unit is the PRM800K matched fork: same question and same golden prefix, one +1 and one -1 continuation, so problem difficulty and prefix content are held constant across the two siblings. `scripts/transition_operator/to_build_forks.py` sampled 5,000 forks (10,000 transitions) from the raw PRM800K scan, carrying each fork's ground-truth answer and phase-2 pre-generated answer for distractors, and froze problem-disjoint train/val/test splits (3,952 / 522 / 526 forks) before extraction.

An encoder E, a two-layer transformer over `[S_{t-1}; x_1..x_{m-1}]` (the pre-step boundary state plus the step tokens with the last step token excluded to block leakage of S_t), produces a 64-dimensional latent z_t. The encoder never sees S_t, the last step token, the elicitation suffix, or any label. Two effect targets, each a post-minus-pre delta measured with an identical trailing separator token so the readout position is fixed: Target A is the boundary next-token distribution shift, Target B is the shift in an 8-candidate answer belief read through a fixed elicitation suffix ("\nSo the final answer is", selected in Stage 0). Both are computed on Qwen2.5-7B base at layer 20.

The design replaces a trained forward model with the frozen model itself as decoder (`src/analysis/transition_operator_train.py`). A trained linear map D turns z into a residual-stream edit `ĥ = S_{t-1} + D(z)` at the pre-step boundary, which is decoded through Qwen2.5-7B's own upper blocks 21-28 plus the LM head over the pre-context KV cache. The boundary token's upper-block K/V is recomputed from the patched state so later suffix tokens attend to the edit; equivalence of this fast cache-based decode with a hook-based patched forward is unit-tested (identity, propagation, and last-layer-dead-end invariants). The Stage 2 ablation trained three arms, A (Target-A KL through the frozen decoder), B (Target-B via a trained head h_B(z)), and AB, over seeds {0,1,2}, with an InfoNCE term over fork siblings (near-duplicate-effect negatives masked) on in every arm. TamIA jobs 368860 (Stage 0), 368879 (Stage 1), 369679 (Stage 2).

### Results

**Stage 0 fixed the target design.** A directional gate confirmed the answer-belief effect tracks reasoning progress: on 500 forks the gold-answer margin moves higher for correct siblings than wrong siblings, Wilcoxon p = 5.8e-17. A boundary-sufficiency oracle then patched the true post-step boundary state into the pre context and measured how much of each effect returns. Target A recovers a median 0.90 / 0.99 / 1.0 at layers 20 / 24 / 26: a single boundary edit almost fully controls the immediate next-token distribution. Target B recovers about 0 at every layer. The answer-belief shift is not carried by one late boundary position. Target A therefore stayed a frozen-decoder KL loss, and Target B became a trained predictive head h_B(z); Stage 4 causal scope is restricted to Target A.

**Stage 1 baselines cleared the operation bar that z later missed.** Raw max-pooling of the step-token states gives operation decodability macro-F1 0.45 and cross-problem operation retrieval 0.42 against 0.30 chance. High-precision symbolic operation labels (ADD/SUB/MUL/DIV/POW from a parser) cover only 21% of steps, since PRM800K MATH steps are mostly prose and casework, so both symbolic-operation and broad keyword-tag labels were carried forward.

**Stage 2 operation organization is null and below pooling.** All arms learn their targets (Target-A boundary-KL validation loss near 0.8, Target-B belief MSE near 0.16). On the identical test split (`compare_baselines_vs_z.json`), z sits at chance for cross-problem operation retrieval (0.27-0.31 versus 0.28) and below trivial pooling on operation decodability (z 0.17-0.25 versus mean/max-pool 0.35-0.375). The three arms are statistically indistinguishable, so Target B adds nothing over Target A.

**Stage 2 correctness is real but not amplified.** Correct versus wrong is balanced by fork construction, so the correctness probe is fully powered (`correctness_probes.json`). Within-fork pair accuracy, where siblings share the exact prefix and chance is 0.50, is 0.658 (A), 0.670 (B), 0.655 (AB), all 95% bootstrap CIs excluding 0.50, and survives residualizing surface features (step length, equation presence, numeric-token count, final-character type). The signal only ties the raw baselines: S_t 0.671, mean-pool 0.660, and the measured Target-A boundary logit delta 0.673. z's edge on global test AUC (0.65 versus 0.61-0.64) reflects the raw-activation probes overfitting (train AUC 0.96, test 0.64) while z's 64-dimensional probe generalizes (train 0.66, test 0.65). Target B confers no correctness advantage.

### Interpretation

The v0 hypothesis is falsified on both counts. The effect-prediction operator discards the operation structure that raw pooled activations already contain, and it preserves without amplifying the moderate correctness signal already present in the boundary state. z is not more semantic (operation organization at chance, below pooling) and it ties rather than beats on robustness (correctness). The one genuine positive is that z is a compact 64-dimensional correctness encoding that generalizes where high-dimensional raw probes overfit. The cleanest single correctness readout throughout is the measured boundary logit delta itself, echoing the Stage 0 directional gate: what a step does to the next-token distribution carries its correctness, and compressing that through an effect-prediction bottleneck neither creates nor removes the signal.

Caveats: symbolic operation labels cover only 21%, but the raw baselines clear chance on the same N, so z's operation null is a real absence of structure rather than underpowering. The A and AB arms dropped about 14% of training batches to a bf16 scaled-dot-product-attention backward overflow on long-context forks, mitigated with a gradient-safe -1e4 attention-mask fill plus a skip-and-log net; the clean, zero-skip B arm reaches the same conclusions. Results are for one model, one layer (L20), and one patch position for the oracle.

### Next Step

Two directions follow. Either accept the negative and stop v0, since the evidence points to effect-prediction being the wrong inductive bias for operation semantics rather than a tuning problem, or run the deferred Stage 3 characterization to identify what z does organize by, given it retrieves broad step-type above chance (about +0.09) while missing operation, and localize the Target-B belief effect with multi-position patching that the single-boundary oracle could not reach.

---

## 19. Limitations

- Only one SSAE checkpoint is evaluated (`gsm8k-385k_Qwen2.5-0.5b_spar-10.pt`). Different SSAE training runs may produce different latent geometries and different probe results.
- The training pool's final shard (offset 360K–450K) has a slightly elevated correct rate (53.1%), near the dataset tail. The 70/30 subsampling absorbs this, but it is not as clean as earlier offsets.
- Seed 44 underperforms the other three seeds by ~2 pp at threshold=0.5 (macro F1=0.647 vs 0.667–0.684). The cause is unknown; it narrows to 0.31 pp std at threshold=0.7.
- The probe was trained on a 70/30 subset (no duplication) of the 450K pool. The minority class (correct, 149,769 steps) caps the training size; a larger correct pool would allow more training data without oversampling.

---

## 20. Literature Survey: Mechanistic Signals for Step-Level CoT Validity (May 2026)

### 17.1 Overview

The field has converged around a central question: do intermediate CoT steps causally determine the final answer, and can mechanistic signals detect when they do not? As of May 2026, five broad approaches have emerged.

---

### 17.2 Full Paper Inventory

#### Tier 1: Most Directly Relevant (Mechanistic Signal + Step-Level CoT Validity)

**[P1] Step-Level Sparse Autoencoder for Reasoning Process Interpretation**
- Authors: Xuan Yang, Jiayu Liu, Yuhang Lai, Hao Xu, Zhenya Huang, Ning Miao
- arXiv: 2603.03031 | March 2026 | GitHub: Miaow-Lab/SSAE
- Signal: Step-level SAE features + linear probing
- Granularity: Step-level (explicit step boundary)
- Dataset: AIME 2024, math reasoning benchmarks | Model: DeepSeek-R1-Distill-Qwen-32B
- **Contribution:** Proposes step-level SAEs (SSAE) operating at reasoning-step granularity via an information bottleneck, disentangling "incremental information" from "background information" into sparse dimensions. Linear probing on SSAE features predicts step correctness and logicality. Improves AIME 2024 from 86.67% to 90.00%.

**[P2] Mechanistic Interpretability of Code Correctness in LLMs via Sparse Autoencoders**
- Authors: Kriz Tahimic, Charibeth Cheng
- arXiv: 2510.02917 | Oct 2025 | ICLR 2026
- Signal: SAE directions (error alarms F1=0.821), attention analysis, weight orthogonalization
- Granularity: Trace-level, with step-level attention analysis
- Dataset: MBPP | Model: Gemma-2
- **Contribution:** Identifies "detection directions" for incorrect code (F1: 0.821) and "steering directions." Reveals asymmetry: model reliably signals incorrect code but not correct code. Mechanisms persist through instruction tuning.

**[P3] Verifying Chain-of-Thought Reasoning via Its Computational Graph (CRV)**
- Authors: Zheng Zhao, Yeskendir Koishekenov, Xianjun Yang, Naila Murray, Nicola Cancedda
- arXiv: 2510.09312 | Oct 2025 | ICLR 2026 (Oral)
- Signal: Attribution graphs, transcoder circuits (white-box mechanistic)
- Granularity: Step-level
- **Contribution:** Builds attribution graphs using transcoders, identifies structural fingerprints of incorrect steps, trains a graph classifier. Error signatures are causal -- intervening on transcoder features corrects faulty reasoning. Error patterns are domain-specific.

**[P4] How does Chain of Thought Think? Mechanistic Interpretability via Sparse Autoencoding**
- Authors: Xi Chen, Aske Plaat, Niki van Stein
- arXiv: 2507.22928 | July 2025
- Signal: SAE features + activation patching (causal)
- Granularity: Trace-level CoT vs. non-CoT comparison; feature-level
- Dataset: GSM8K | Models: Pythia-70M, Pythia-2.8B
- **Contribution:** First feature-level causal study of CoT faithfulness. Swapping CoT-reasoning features into non-CoT runs raises answer log-probabilities significantly in the 2.8B model (negligible in 70M), establishing a capacity threshold for CoT's mechanistic effect.

**[P5] Reasoning Models Know When They're Right: Probing Hidden States for Self-Verification**
- Authors: Anqi Zhang, Yulin Chen, Jane Pan, Chen Zhao, Aurojit Panda, Jinyang Li, He He
- arXiv: 2504.05419 | April 2025 | GitHub: AngelaZZZ-611/reasoning_models_probing
- Signal: Linear probe on hidden states (step-level)
- Granularity: Step-level (per reasoning chunk)
- **Contribution:** Binary linear probe on hidden states at intermediate reasoning steps predicts final-answer correctness. Used as inference-time verifier for early-exit, reducing token count by 24% with no performance loss.

**[P6] Deep Hidden Cognition Facilitates Reliable Chain-of-Thought Reasoning**
- Authors: Zijun Chen, Wenbo Hu, Richang Hong
- arXiv: 2507.10007 | July 2025 | AAAI-26
- Signal: Attention head activations (step-level truthfulness probe)
- Granularity: Step-level
- Dataset: Math, symbolic, commonsense + multimodal
- **Contribution:** Attention head activations reflect step truthfulness up to 85% accuracy, concentrated in middle layers. Confidence predictor on these activations guides beam search, outperforming Self-Consistency and Self-Evaluation Guided Beam Search.

**[P7] LLM Reasoning as Trajectories: Step-Specific Representation Geometry and Correctness Signals**
- Authors: Lihao Sun et al. (Microsoft Research)
- arXiv: 2604.05655 | April 2026 | ACL 2026 (Main)
- Signal: Representation geometry / trajectory analysis (step-specific subspace)
- Granularity: Step-level
- **Contribution:** Models CoT as a trajectory through representation space. Step-specific subspaces become increasingly separable with layer depth. Correct/incorrect solutions diverge at late stages; mid-reasoning correctness prediction achieves ROC-AUC up to 0.87.

**[P8] Hidden States as Early Signals: Step-level Trace Evaluation and Pruning for Efficient Test-Time Scaling (STEP)**
- Authors: Zhixiang Liang, Beichen Huang, Zheng Wang, Minjia Zhang
- arXiv: 2601.09093 | Jan 2026
- Signal: Hidden state probe (step scorer)
- Granularity: Step-level
- **Contribution:** Lightweight step scorer probing hidden states estimates trace quality. GPU memory-aware pruning triggered at KV cache saturation reduces latency 45-70% while improving accuracy.

**[P9] Mechanistic Evidence for Faithfulness Decay in Chain-of-Thought Reasoning**
- Authors: Donald Ye, Max Loffgren, Om Kotadia, Linus Wong
- arXiv: 2602.11201 | Feb 2026
- Signal: Step corruption + logit drop (behavioral/mechanistic hybrid)
- Granularity: Step-level (per-step corruption)
- **Contribution:** Introduces Normalized Logit Difference Decay (NLDD): corrupts individual reasoning steps and measures confidence drop. Discovers "Reasoning Horizon" (k*) at 70-85% of chain length beyond which tokens have minimal/negative effect. Models can encode correct internal representations while failing the task.

**[P10] Knowing Before Saying: LLM Representations Encode Information About Chain-of-Thought Success Before Completion**
- Authors: Anum Afzal, Florian Matthes, Gal Chechik, Yftah Ziser
- arXiv: 2505.24362 | May 2025 | ACL 2025 Findings
- Signal: Linear probing on LLM representations (pre-generation)
- Granularity: Trace-level and per-step prefix
- **Contribution:** Probing classifier on internal representations predicts CoT success before a single token is generated (60-76.4% accuracy). Early representations can be as informative as later ones.

**[P11] Reasoning with Confidence: Efficient Verification via Uncertainty Heads (ReProbe)**
- Authors: Jingwei Ni, Ekaterina Fadeeva, Tianyi Wu, et al.
- arXiv: 2511.06209 | Nov 2025 | ACL 2026 (Main)
- Signal: Uncertainty heads (<10M params) on frozen LLM internal states
- Granularity: Step-level
- Dataset: Math, planning, QA
- **Contribution:** UHeads match or surpass PRMs up to 810x larger. Training labels from DeepSeek-R1 or self-supervised. Works across math, planning, and QA.

**[P12] Interpreting Reasoning Features via Sparse Autoencoders**
- Authors: Andrey Galichin et al. (AIRI-Institute)
- arXiv: 2503.18878 | March 2025 | GitHub: AIRI-Institute/SAE-Reasoning
- Signal: SAE features + ReasonScore + steering experiments
- Granularity: Token/step-level (reasoning moment detection)
- Model: DeepSeek-R1-Llama-8B
- **Contribution:** Uses SAEs to decompose DeepSeek-R1 activations. Introduces ReasonScore to identify features active during reasoning moments. Finds features encoding uncertainty, exploratory thinking, self-reflection. Steering: +2.2% accuracy, +20.5% reasoning trace length.

**[P13] Thought Anchors: Which LLM Reasoning Steps Matter?**
- Authors: Paul C. Bogdan, Uzay Macar, Neel Nanda, Arthur Conmy
- arXiv: 2506.19143 | June 2025
- Signal: Attention patterns + causal attribution (sentence-level)
- Granularity: Step/sentence-level
- Models: DeepSeek R1-Distill Qwen-14B, Llama-8B
- **Contribution:** Identifies "thought anchors" with outsized causal impact on reasoning trajectory via three attribution methods: counterfactual sampling, attention pattern aggregation (broadcasting heads), and causal sentence-to-sentence attribution.

**[P14] Base Models Know How to Reason, Thinking Models Learn When**
- Authors: Constantin Venhoff, Iván Arcuschin, Philip Torr, Arthur Conmy, Neel Nanda
- arXiv: 2510.07364 | Oct 2025 | NeurIPS 2025 Mechanistic Interpretability Workshop
- Signal: Top-K SAEs on sentence-level activations + steering vectors
- Granularity: Sentence/step-level
- Dataset: MMLU-Pro (430k sentences), GSM8K, MATH500
- **Contribution:** Derives unsupervised taxonomy of reasoning behaviors via SAE clustering. Reasoning mechanisms exist in base models; thinking models learn when to deploy them. Steering 12% of tokens recovers 91% of performance gap to thinking models.

**[P15] Mining Intrinsic Rewards from LLM Hidden States for Efficient Best-of-N Sampling (SWIFT)**
- Authors: Jizhou Guo, Zhaomin Wu, Hanchen Yang, Philip S. Yu
- arXiv: 2505.12225 | May 2025 | KDD 2026
- Signal: Linear layer on concatenated LLM hidden states (token-level reward)
- Granularity: Token-level reward (aggregated to trace-level)
- Dataset: MATH
- **Contribution:** ~80% per-layer accuracy in predicting reasoning correctness from hidden states. Outperforms EurusRM-7B by 12.7% on MATH with <0.005% of parameters.

**[P16] Sparse Reward Subsystem in Large Language Models**
- Authors: Guowei Xu, Mert Yuksekgonul, James Zou
- arXiv: 2602.00986 | Feb 2026
- Signal: Sparse neuron subsystem (mechanistic, intervention-based)
- Granularity: Token/state-level
- **Contribution:** Identifies "value neurons" tracking expected state value and "dopamine neurons" encoding reward prediction errors, analogous to biological reward systems. Robust across datasets, scales, architectures, and fine-tuned variants.

**[P17] Harnessing Reasoning Trajectories for Hallucination Detection (ARS)**
- Authors: Jianxiong Zhang et al.
- arXiv: 2601.17467 | Jan 2026 | ICML 2026
- Signal: Latent embedding perturbation + answer-agreement representation
- Granularity: Trace-level
- **Contribution:** Perturbs trace-boundary embeddings to generate counterfactual answers, trains detectors on answer-stability. No human annotations required.

**[P18] Resa: Transparent Reasoning Models via SAEs**
- Authors: Shangshang Wang, Julian Asilis et al.
- arXiv: 2506.09967 | June 2025
- Signal: SAE features encoding reasoning abilities
- Granularity: Reasoning-ability-level
- Dataset: AIME24, AMC23 | Models: Qwen, R1-Distill variants (1.5B)
- **Contribution:** SAE-Tuning: trains SAE to extract reasoning abilities from source model, uses it to guide SFT on target model. Retains >97% of RL-trained performance at 2000x cost reduction.

**[P19] Truth as a Trajectory: What Internal Representations Reveal About LLM Reasoning**
- Authors: Hamed Damirchi et al.
- arXiv: 2603.01326 | March 2026
- Signal: Layer-wise geometric displacement (trajectory analysis)
- Granularity: Layer-level trajectory (within a forward pass)
- **Contribution:** TaT models inference as unfolded trajectory of layer-wise geometric displacements. Uncovers geometric invariants distinguishing valid from spurious reasoning; outperforms conventional probing by mitigating lexical confounds.

**[P20] Probing the Trajectories of Reasoning Traces in Large Language Models**
- Authors: Marthe Ballon, Brecht Verbeken, Vincent Ginis, Andres Algaba
- arXiv: 2601.23163 | Jan 2026
- Signal: Behavioral probing (trace injection + next-token probabilities)
- Granularity: Token-percentile level
- Dataset: GPQA Diamond, MMLU-Pro | Models: Qwen3 (4B-14B), gpt-oss (20B, 120B)
- **Contribution:** Truncates reasoning traces at fixed token-percentiles and reinjects them. Accuracy increases monotonically with reasoning tokens (from relevant content, not length). Stronger models backtrack; weaker ones anchor to initial wrong answers.

**[P21] SAE-Guided CoT Generation: Interpretable and Inference-Optimal**
- arXiv: 2510.01528 | Oct 2025
- Signal: SAE token representations + transition graph reward
- Granularity: Token-level transition

**[P22] Feature Extraction and Steering for Enhanced CoT Reasoning**
- arXiv: 2505.15634 | May 2025
- Signal: SAE features + residual activation directions (SAE-free steering)

**[P23] Controllable LLM Reasoning via Sparse Autoencoder-Based Steering**
- arXiv: 2601.03595 | Jan 2026
- Signal: SAE strategy-specific features
- **Contribution:** Decomposes hidden states into disentangled feature space, identifies strategy-specific features. 7% absolute accuracy improvement by redirecting from erroneous reasoning paths.

**[P24] Mapping Faithful Reasoning in Language Models (Concept Walk)**
- arXiv: 2510.22362 | Oct 2025 | NeurIPS 2025 Mechanistic Interpretability Workshop
- Signal: Concept direction projection in activation space
- Granularity: Reasoning step-level
- Model: Qwen 3-4B
- **Contribution:** Traces how model's internal stance evolves per reasoning step by projecting onto concept directions. Distinguishes decorative from load-bearing reasoning steps.

**[P25] Measuring CoT Faithfulness by Unlearning Reasoning Steps (FUR)**
- Authors: Martin Tutek, Fateme Hashemi Chaleshtori, Ana Marasovic, Yonatan Belinkov
- arXiv: 2502.14829 | Feb 2025 | EMNLP 2025
- Signal: Machine unlearning + parameter intervention (step-level)
- Granularity: Step-level
- Dataset: Multi-hop MCQA (5 datasets)
- **Contribution:** Erases individual CoT step content from model parameters and measures the effect on prediction. Can precisely change model predictions by unlearning key steps.

**[P26] When Chains of Thought Don't Matter: Causal Bypass in LLMs**
- arXiv: 2602.03994 | Feb 2026
- Signal: Hidden-state patching (CoT-Mediated Influence) + behavioral scoring
- Granularity: Trace-level
- **Contribution:** Many QA items show near-total bypass (CMI ~0); logic problems show stronger mediation (CMI up to 0.56).

**[P27] PRISM: Dual View of LLM Reasoning through Semantic Flow and Latent Computation**
- arXiv: 2603.22754 | March 2026
- Signal: Token-level semantic flow + Gaussian mixture model of hidden states
- Granularity: Step-level
- **Contribution:** Identifies failed trajectories as trapped in "unproductive verification loops" or diverging into overthinking/premature commitment.

**[P28] No Answer Needed: Predicting LLM Accuracy from Question-Only Linear Probes**
- arXiv: 2509.10625 | Sep 2025
- Signal: Linear probe on pre-generation activations
- **Contribution:** Works on factual/knowledge tasks but generalizes poorly to mathematical reasoning. Early-to-mid layers suffice.

**[P29] Dissecting Logical Reasoning in LLMs: FineLogic**
- arXiv: 2506.04810 | June 2025 | EMNLP 2025 Findings
- Signal: Representation-level probing + stepwise soundness evaluation
- **Contribution:** Reveals trade-off: natural language supervision generalizes but produces non-atomic steps; symbolic supervision produces structurally sound atomic steps.

**[P30] Interpretable Reward Model via Sparse Autoencoder (SARM)**
- arXiv: 2508.08746 | Aug 2025 | AAAI 2026 (Oral)
- Signal: SAE features for reward attribution

**[P31] SAFER: Probing Safety in Reward Models with Sparse Autoencoder**
- arXiv: 2507.00665 | July 2025
- Signal: SAE features in reward model activations

**[P32] Circuit Stability Characterizes Language Model Generalization**
- Authors: Alan Sun (CMU)
- arXiv: 2505.24731 | May 2025 | ACL 2025
- Signal: Circuit stability (mechanistic circuit analysis)
- **Contribution:** CoT induces circuit stability; circuit stability predicts length, structural, and compositional generalization.

#### Tier 2: Foundational / Background Papers

| arXiv | Title | Year | Role |
|---|---|---|---|
| 2305.04388 | Language Models Don't Always Say What They Think (Turpin et al.) | 2023 | Canonical baseline: CoT faithfulness is unreliable |
| 2307.13702 | Measuring Faithfulness in Chain-of-Thought Reasoning (Lanham et al., Anthropic) | 2023 | Systematic intervention study; larger models less faithful |
| 2502.12289 | Evaluating Step-by-step Reasoning Traces: A Survey (Lee & Hockenmaier) | 2025 | Taxonomy: factuality, validity, coherence, utility |
| 2507.11473 | Chain of Thought Monitorability (Korbak et al.) | 2025 | CoT monitoring as an AI safety property |
| 2510.23966 | Measuring Chain-of-Thought Monitorability | 2025 | Pragmatic metrics for CoT monitor evaluation |
| 2603.26410 | Why Models Know But Don't Say: Faithfulness Divergence in Thinking Tokens | 2026 | 55.4% of misleading-hint cases: thinking acknowledges hints, answers do not |
| 2603.22582 | Lie to Me: Faithfulness in Open-Weight Reasoning Models | 2026 | Systematic faithfulness evaluation of Q4 2025-Q1 2026 models |

---

### 17.3 Directly Relevant Papers (Both Criteria: Mechanistic Signal + Step-Level CoT Validity)

| # | Paper | Signal | Step-Level? | Year |
|---|---|---|---|---|
| P1 | SSAE (Miaow-Lab) | SAE + linear probe | Yes (explicit) | 2026 |
| P3 | CRV (ICLR Oral) | Attribution graphs / transcoders | Yes | 2025 |
| P5 | Reasoning Models Know When They're Right | Linear probe on hidden states | Yes | 2025 |
| P6 | Deep Hidden Cognition | Attention head activations | Yes | 2025 |
| P7 | LLM Reasoning as Trajectories (Microsoft) | Representation geometry | Yes | 2026 |
| P8 | STEP | Hidden state scorer | Yes | 2026 |
| P9 | NLDD Faithfulness Decay | Step corruption + logit drop | Yes | 2026 |
| P11 | ReProbe / Uncertainty Heads | Frozen LLM internal states | Yes | 2025 |
| P13 | Thought Anchors (Nanda et al.) | Attention patterns + causal attribution | Yes | 2025 |
| P24 | Concept Walk | Concept direction activation projection | Yes | 2025 |
| P25 | FUR / Machine Unlearning | Parameter intervention | Yes | 2025 |

---

### 17.4 State-of-the-Art Summary

#### Approach 1: Sparse Autoencoders (SAEs) on Activations

The most directly relevant and rapidly growing approach. SAEs decompose LLM activations into interpretable sparse features that can be probed for step-level properties, causally intervened on (steering), or used to guide generation.

**P1 (SSAE, 2026)** is the most direct instantiation of this project's goal: explicit step-level SAEs with information bottleneck, linear probes predict step correctness and logicality. **P2** (code correctness, ICLR 2026) shows SAEs identify asymmetric error-detection directions (F1=0.821 for incorrect code, weak for correct code). **P4** (Pythia + SAE) establishes that CoT induces distinct, more interpretable feature activations only in capable (2.8B+) models -- a capacity threshold. **P12** (DeepSeek-R1 SAE analysis) identifies specific SAE features for uncertainty, exploration, self-reflection. **P14** (NeurIPS 2025 workshop) uses sentence-level SAE clustering to build an unsupervised taxonomy of reasoning behaviors.

#### Approach 2: Linear Probes / Uncertainty Heads on Hidden States

The second largest cluster. Lightweight classifiers on raw hidden states predict step-level correctness without SAE decomposition.

**P5** (hidden state probes): binary probes on hidden states at intermediate steps predict final-answer correctness, enabling 24% token savings. **P6** (attention heads): probes on attention heads achieve up to 85% accuracy for step truthfulness, concentrated in middle layers. **P7** (representation geometry, ACL 2026): correct/incorrect traces diverge at late steps, ROC-AUC 0.87 for mid-reasoning prediction. **P11** (ReProbe): <10M-parameter uncertainty heads beat PRMs up to 810x larger. **P16** (sparse reward subsystem): "value neurons" and "dopamine neurons" provide mechanistic grounding for why probes work.

Key quantitative results: ROC-AUC 0.87 (P7), 85% attention-head accuracy (P6), 24% token reduction (P5), 810x size advantage (P11).

#### Approach 3: Attribution Graphs / Circuit-Level Analysis

**P3** (CRV, ICLR 2026 Oral) is the most rigorous mechanistic approach: builds attribution graphs using transcoders, finds distinct structural fingerprints for correct vs. incorrect CoT steps, shows causal interventions can repair faulty steps, and establishes that error signatures are domain-specific. **P13** (Thought Anchors, Nanda/Conmy) uses broadcasting attention heads and causal attribution to identify "anchor" sentences with outsized impact. **P26** (Causal Bypass) measures CoT-Mediated Influence via hidden-state patching, finding many QA items have CMI ~0.

#### Approach 4: Trajectory / Geometric Analysis

**P7** (Microsoft ACL 2026): step-specific representational subspaces, ROC-AUC 0.87. **P19** (Truth as Trajectory): layer-wise geometric displacement invariants distinguishing valid from spurious reasoning. **P27** (PRISM): joint semantic flow and hidden-state Gaussian mixture modeling, identifies failed trajectory types.

#### Approach 5: Behavioral / Causal Intervention on CoT Text

**P9** (NLDD/Faithfulness Decay): step corruption + logit drop, discovers Reasoning Horizon at 70-85% of chain length. **P25** (FUR): machine-unlearning of step content from parameters, step-level parametric faithfulness. **P24** (Concept Walk): concept direction projection per step, distinguishes decorative from faithful reasoning.

---

### 17.5 Signal Performance Comparison

| Signal Type | Best Result | Scope | Training Required |
|---|---|---|---|
| SAE features + linear probe (step-level, P1) | 90.0% AIME (via steering) | Step-level | Yes (lightweight) |
| Representation geometry / trajectory (P7) | ROC-AUC 0.87 | Step-level | Minimal |
| Attention head probe (P6) | Up to 85% step truthfulness | Step-level | Yes |
| Uncertainty heads (P11) | Matches PRMs 810x larger | Step-level | Yes (self-supervised) |
| Hidden state linear probe (P5) | High accuracy, 24% token savings | Step-level | Yes |
| SAE directions for code correctness (P2) | F1 0.821 (error detection) | Token/prompt-level | Yes |
| Sparse reward neurons (P16) | ~80% per-layer (P15 SWIFT) | Token-level | Yes |
| Attribution graph classifier (P3) | Highly predictive (causal) | Step-level | Yes |
| Step corruption / NLDD (P9) | Identifies Reasoning Horizon | Step-level | No (post-hoc) |

---

### 17.6 Open Questions Most Relevant to This Project

1. **SAE probe vs. circuit analysis is unexplored.** P3 (CRV) does circuit-level analysis with transcoders, not SAEs. Combining SAE features with attribution graphs has not been attempted.

2. **Correctness direction asymmetry (P2)** holds for code: SAEs reliably detect incorrect steps but are weaker at confirming correct ones. Whether this asymmetry holds for mathematical reasoning is unknown.

3. **Capacity threshold (P4):** SAE-based CoT signals appear only in models above ~2.8B parameters (Pythia scale). The SSAE backbone (Qwen2.5-0.5B) is below this threshold -- though the SSAE's step-level training may compensate.

4. **The Reasoning Horizon (P9):** mechanistic signals may degrade after 70-85% of chain length, constraining where step-level probes are useful.

5. **Probe generalization across domains (P28)** is weak: probes trained on factual tasks fail on math. Step-level probes need to be trained on the same task distribution -- our Math-Shepherd-trained probe vs. ProcessBench is exactly this generalization gap.

6. **Attention heads vs. residual stream:** P6 finds middle-layer attention heads carry step truthfulness signals; P5 uses residual stream hidden states. Which substrate is more informative for step-level correctness is an open empirical question.

7. **Next-step prediction as auxiliary signal:** our Future-SSAE experiments (§10) address a gap not covered by any existing paper -- no work has used a next-step prediction objective to make SAE latents sensitive to trajectory-level information flow between steps.

---

## 21. Next Steps

**Step 1 follow-up (strengthen the current result):**
- Investigate why seed 44 consistently underperforms at threshold=0.5; check whether it is a training instability or a real distributional effect
- Run additional seeds (46–49) to tighten the variance estimate

**Step 2 (feature analysis and mechanistic interpretation):**
- Feature importance: identify which of the 896 dimensions carry the correctness signal most strongly (e.g. permutation importance, SHAP values)
- Geometric analysis: visualise correct vs incorrect h_c vectors (PCA, cosine similarity distributions) to test whether they form distinct directional clusters on the unit sphere
- Compare against the dense baseline `h_k` (pre-SAE projection) to isolate what sparsification contributes
- Linear probe trained on individual feature subsets to test whether correctness is locally decodable

**Longer-term dataset and model direction:**
- Find or construct a dataset where labels directly reflect mathematical step correctness (not MC rollout path viability), better aligned with what the SSAE was trained to encode
- Train an SSAE from scratch on data annotated with MC rollout labels — this would remove the mismatch between the encoder's training objective and the probe's label source

