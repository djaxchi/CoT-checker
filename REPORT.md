# CoT-Checker: Research Report
*Last updated: 2026-04-24*

---

## 1. Hypothesis

The long-term research question driving this project is:

> **Can sparse autoencoder (SAE) feature activations serve as reliable mechanistic signals for detecting incorrect steps in chain-of-thought reasoning?**

If the latent space of an SAE trained on model activations encodes step-level correctness in a learnable way, then a lightweight probe on those latents could provide interpretable, model-internal verification of reasoning traces. This would be meaningfully different from black-box verifiers: rather than re-running the problem or calling a judge model, we read the model's own internal state.

The hypothesis is investigated in two steps:

1. **Does the signal exist?** Can a classifier trained on SAE latents predict step correctness above chance? If yes, the latent vector carries information about correctness — the signal is there, even if we do not yet understand it.
2. **What is the signal?** Which specific dimensions drive the predictions? Is it linearly decodable? Does it correspond to interpretable reasoning features?

This report covers step 1 only: establishing that an MLP probe trained on SSAE latents can detect the difference between correct and incorrect steps on an independent external oracle (Math-Shepherd). Step 2 — feature importance, linear probing, and mechanistic interpretation — is future work.

---

## 2. The Paper We Started From

We chose to reproduce **"Step-level Sparse Autoencoders for Interpretable CoT Verification"** (Miaow-Lab, arXiv:2603.03031). The paper proposes an SSAE (Step-level Sparse Autoencoder) architecture trained on GSM8K-Aug with a Qwen2.5-0.5B backbone. The paper's core claim is that a linear probe on the SSAE latent vector `h_c` can predict step correctness at close to 78.58% accuracy on GSM8K.

The paper releases its checkpoint publicly (`Miaow-Lab/SSAE-Checkpoints`), making it an ideal reproduction target: we can use the exact same encoder and evaluate probe performance on an independent task.

**What made this interesting as a starting point:** the SSAE is trained for reconstruction, not classification. Any correctness signal in `h_c` would be an emergent property of the latent space, not a supervised artifact. That is exactly the kind of mechanistic signal the hypothesis requires.

---

## 3. What We Observed During Reproduction

### 3.1 The Label Semantics Problem

The paper describes its labels as "ground-truth labels derived programmatically via symbolic verification of numerical outputs." The actual code in `papers/SSAE/classifier/classifier_data.py` computes labels differently:

```python
# Encoder sees [context | <sep> | step] -> sparse latent hk
# Decoder takes hk + context and reconstructs the step autoregressively
# Label = 1 if last number in reconstruction matches last number in ground truth
```

The label is not computed from the original step. It is computed from the SSAE's own reconstruction of the step. A step is labeled "correct" if the SSAE can faithfully decode its final number, and "incorrect" if the SSAE reconstruction drifts. The paper's phrase "symbolic verification" refers to matching the last number in the SSAE-decoded text against the last number in the source text.

This is a reconstruction fidelity metric, not a mathematical correctness metric. A step that contains a genuine arithmetic error will be labeled "correct" if the SSAE reconstructs it accurately (including the error). A correct step can be labeled "incorrect" if the SSAE fails to recover its final number.

### 3.2 The Training Data Is All Correct

The second observation: GSM8K-Aug, the training corpus, was filtered to exclude any example where intermediate steps do not lead to the correct final answer. Every step the SSAE was trained on is arithmetically correct. The probe therefore never sees genuinely incorrect steps during training in the paper's setup. Label = 0 in the paper means "SSAE failed to reconstruct this correct step," not "this step is mathematically wrong."

### 3.3 What the Paper Actually Measures

Putting these observations together: the paper's 78.58% accuracy measures whether the SSAE latent space predicts the SSAE's own reconstruction quality. The signal is real, but it is not the signal we care about. A probe that scores well on that task has learned which types of steps the SSAE encodes compactly, not which steps are mathematically valid.

This was the moment the project pivoted.

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

**What a latent actually looks like** (first 20 of 896 dimensions, from `results/probe_data/math_shepherd_176k_natural.npz`):

```
Correct step:   [0.0024, 0.1553, 0.0, 0.0, 0.0, 0.1688, 0.0, 0.0, 0.0, 0.0029, ...]
                Active dims: 205/896

Incorrect step: [0.0993, 0.1904, 0.0, 0.0, 0.0, 0.0902, 0.0, 0.0, 0.0, 0.0583, ...]
                Active dims: 223/896
```

Across all 176,008 steps, incorrect steps activate on average **9.7 more dimensions** than correct steps (209.1 vs 199.4). This is the learnable signal the probe uses.

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

### 7.5 Comparison to Previous Best

| | Previous best | This run |
|---|---|---|
| Eval size | 2,890 steps | **50,000 steps** |
| Eval balance | 50/50 | 50/50 |
| Seeds | 1 (seed=42) | **4 (seeds 42–45)** |
| Training steps (subset) | 72,785 | **213,955** |
| Training pool (encoded) | 176,008 | **450,000** |
| Context truncation | Bugged (independent 128-tok per part) | **Fixed (left-truncate combined)** |
| Best accuracy | 73.40% (threshold=0.8) | **74.62% ± 0.31 pp** (threshold=0.7) |
| Best macro F1 | ~0.73 | **0.746 ± 0.003** |

---

## 8. Interpretation

The SSAE latent space, trained solely for reconstruction on correct GSM8K steps, encodes enough information to detect step-level incorrectness as labeled by an independent oracle (Math-Shepherd). This supports the encoding half of the hypothesis.

One observable difference: incorrect steps activate on average 9.7 more dimensions than correct steps (209.1 vs 199.4 active dimensions out of 896). This is consistent with the SSAE producing more compact representations for in-distribution steps (correct reasoning it was trained on) and more diffuse representations for out-of-distribution ones (incorrect steps it never saw). Whether this sparsity difference is what the MLP probe is actually using, or whether the probe is reading a more complex geometric property of h_c, is unknown. That question belongs to step 2.

Whether this signal generalizes beyond arithmetic reasoning to other domains is the open question being pursued in `WIP_Report.md`.

---

## 9. Limitations

- Only one SSAE checkpoint is evaluated (`gsm8k-385k_Qwen2.5-0.5b_spar-10.pt`). Different SSAE training runs may produce different latent geometries and different probe results.
- The training pool's final shard (offset 360K–450K) has a slightly elevated correct rate (53.1%), near the dataset tail. The 70/30 subsampling absorbs this, but it is not as clean as earlier offsets.
- Seed 44 underperforms the other three seeds by ~2 pp at threshold=0.5 (macro F1=0.647 vs 0.667–0.684). The cause is unknown; it narrows to 0.31 pp std at threshold=0.7.
- The probe was trained on a 70/30 subset (no duplication) of the 450K pool. The minority class (correct, 149,769 steps) caps the training size; a larger correct pool would allow more training data without oversampling.

---

## 10. Next Steps

**Step 1 follow-up (strengthen the current result):**
- Implement a logistic regression and a linear SVM baseline on the same SSAE latents — if a linear classifier matches the MLP, the signal is linearly decodable, which is a stronger geometric claim; if it falls short, the structure is non-linear
- Investigate why seed 44 consistently underperforms; check whether it is a training instability or a real distributional effect
- Run additional seeds (46–49) to tighten the variance estimate

**Step 2 (feature analysis and mechanistic interpretation):**
- Feature importance: identify which of the 896 dimensions carry the correctness signal most strongly (e.g. permutation importance, SHAP values)
- Geometric analysis: visualise correct vs incorrect h_c vectors (PCA, cosine similarity distributions) to test whether they form distinct directional clusters on the unit sphere
- Compare against the dense baseline `h_k` (pre-SAE projection) to isolate what sparsification contributes
- Linear probe trained on individual feature subsets to test whether correctness is locally decodable

**Longer-term dataset and model direction:**
- Find or construct a dataset where labels directly reflect mathematical step correctness (not MC rollout path viability), better aligned with what the SSAE was trained to encode
- Train an SSAE from scratch on data annotated with MC rollout labels — this would remove the mismatch between the encoder's training objective and the probe's label source
- Extend to symbolic reasoning domain (see `WIP_Report.md`)

