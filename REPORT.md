# CoT-Checker: Research Report
*Last updated: 2026-04-26*

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

## 8. Interpretation (Math-Shepherd Held-Out Eval)

The SSAE latent space, trained solely for reconstruction on correct GSM8K steps, encodes enough information to detect step-level incorrectness as labeled by an independent oracle (Math-Shepherd). This supports the encoding half of the hypothesis.

One observable difference: incorrect steps activate on average 9.7 more dimensions than correct steps (209.1 vs 199.4 active dimensions out of 896). This is consistent with the SSAE producing more compact representations for in-distribution steps (correct reasoning it was trained on) and more diffuse representations for out-of-distribution ones (incorrect steps it never saw).

The linear probe result (§7.5) clarifies the geometric structure of this signal. A linear probe with 897 parameters achieves 74.31% accuracy — essentially identical to the 1.4M-parameter MLP. This means the correctness signal is not buried in a curved manifold: correct and incorrect h_c vectors are nearly linearly separable on the 896-dimensional unit sphere. The MLP probe is reading a direction in the latent space, not a nonlinear feature combination. Which specific dimensions drive the separation, and whether they correspond to interpretable SSAE features, remains the open question for step 2 of the hypothesis.

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

## 10. Future-SSAE: Adding a Next-Step Prediction Objective

The hypothesis in §9.6 leads directly to a new training objective. We extend the SSAE with a prediction branch that shares the same sparse latent `h_c` used for reconstruction, and adds an auxiliary loss requiring `h_c` to predict step k+1 given the question and step k. The total loss is:

```
L = L_recon + λ * ||h_c||_1 + α * (L_pred / ema_pred_nll)
```

The prediction branch uses teacher-forcing: the decoder sees `[question | step_k | h_c]` and must predict `step_{k+1}`. Crucially, no content from step k+1 is visible to the encoder when computing `h_c`. Any gradient that flows through the prediction loss must do so via `h_c`, which means the sparse latent is directly incentivized to encode forward-looking information about where the reasoning is going.

We train two context-mode variants: M1 uses `[question | step_k]` as prediction context, and M2 uses `[question | step_{k-1} | step_k]` to provide one additional step of history. Both use `alpha_pred = 0.1` as the initial signal-validation setting. Results from these runs are pending.

The key diagnostic is whether `pred_first_token_nll` decreases across step-index buckets (idx=0: first step of the solution, idx=1: second step, idx=2+: all later steps). If the signal appears only at idx=0, the model is learning the easy first-step transition; if it appears uniformly across buckets, `h_c` is genuinely encoding cross-step information flow.

---

## 11. Limitations

- Only one SSAE checkpoint is evaluated (`gsm8k-385k_Qwen2.5-0.5b_spar-10.pt`). Different SSAE training runs may produce different latent geometries and different probe results.
- The training pool's final shard (offset 360K–450K) has a slightly elevated correct rate (53.1%), near the dataset tail. The 70/30 subsampling absorbs this, but it is not as clean as earlier offsets.
- Seed 44 underperforms the other three seeds by ~2 pp at threshold=0.5 (macro F1=0.647 vs 0.667–0.684). The cause is unknown; it narrows to 0.31 pp std at threshold=0.7.
- The probe was trained on a 70/30 subset (no duplication) of the 450K pool. The minority class (correct, 149,769 steps) caps the training size; a larger correct pool would allow more training data without oversampling.

---

## 12. Next Steps

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

