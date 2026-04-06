# CoT-Checker: Research Report
*Last updated: 2026-04-06*

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

Each record is encoded by the SSAE as a single sequence:

```
[context tokens (≤128)] + [<sep>] + [step tokens (≤128)] + [<eos>]
```

Both context and step are truncated independently to 128 tokens before concatenation. The SSAE encoder (Qwen2.5-0.5B backbone + sparse projector) runs a forward pass and returns the hidden state at the **last token position**, projected through the sparse bottleneck. This yields a 896-dimensional latent vector `h_c`.

The encoder applies L2 normalization to `h_c` so all vectors lie on the unit sphere. Every latent in our dataset has L2 norm exactly 1.0 (verified across all 176,008 encoded steps).

**What a latent actually looks like** (first 20 of 896 dimensions, from `results/probe_data/math_shepherd_176k_natural.npz`):

```
Correct step:   [0.0024, 0.1553, 0.0, 0.0, 0.0, 0.1688, 0.0, 0.0, 0.0, 0.0029, ...]
                Active dims: 205/896

Incorrect step: [0.0993, 0.1904, 0.0, 0.0, 0.0, 0.0902, 0.0, 0.0, 0.0, 0.0583, ...]
                Active dims: 223/896
```

Across all 176,008 steps, incorrect steps activate on average **9.7 more dimensions** than correct steps (209.1 vs 199.4). This is the learnable signal the probe uses.

### 5.4 Data Provenance and Integrity

We encode Math-Shepherd in two separate windows to guarantee clean train/eval separation:

| File | Steps | Math-Shepherd offset | max_len | Distribution |
|---|---|---|---|---|
| `math_shepherd_40k_natural.npz` | 40,126 | 0 | 256 | 28.0% correct |
| `math_shepherd_57k_balanced.npz` | 57,782 | 0 | 256 | 50.0% correct |
| `math_shepherd_176k_natural.npz` | 176,008 | 110,000 | 128 | 28.9% correct |
| `math_shepherd_eval_5k_clean.npz` | 5,000 | 100,000 | 128 | 28.9% correct (raw) |
| `math_shepherd_eval_balanced.npz` | 2,890 | — (subsampled from above) | 128 | **50.0% correct** |

**A contamination issue we found and fixed:** the original eval file (`math_shepherd_eval_5k_contaminated.npz`, now retired) was generated from the same offset-0 window as the training data. An exact row-hash check revealed 48.9% of its 5,000 rows were present verbatim in the balanced training set, and 33.4% in the natural training set. All experiments using that file produced inflated metrics and are not reported here.

The clean eval (`math_shepherd_eval_5k_clean.npz`) is from offset 100,000 — a window between the two training encodings (0–100K and 110K–286K). A row-hash check against all training files confirms 0% exact overlap.

Additionally, the 40K/57K files were encoded at max_len=256 while the 176K file uses max_len=128 (reduced to fit within MPS memory). Because of this encoding inconsistency, we do not mix the older files with the 176K data in the results below.

---

## 6. Probe Architecture and Training

**Architecture** (`src/probes/classifier.py`):

```
Linear(896 → 1024) → LayerNorm(1024) → ReLU → Dropout(0.1)
Linear(1024 → 512) → LayerNorm(512)  → ReLU → Dropout(0.1)
Linear(512 → 1)    [raw logit; sigmoid gives P(correct)]
```

**Training** (`scripts/experiment_176k_clean.py`):
- Loss: binary cross-entropy on the single logit output
- Optimizer: AdamW, lr=1e-4, weight_decay=0.01
- Scheduler: cosine annealing over 30 epochs
- Batch size: 128
- 80/20 train/val split; best val-accuracy checkpoint saved

**Decision threshold:** 0.8 at evaluation (not 0.5). A threshold of 0.8 means the probe only predicts "correct" when P(correct) ≥ 0.8, and defaults to "incorrect" otherwise. This is the conservative direction: at threshold=0.5 the probe calls 3,192 of 5,000 eval steps correct; at threshold=0.8 it calls only 1,597.

This is well-motivated by the SSAE's training regime. The SSAE was trained exclusively on correct steps. Its encoder learned to compress correct reasoning into sparse latent vectors; it never saw incorrect steps during training. A step that is genuinely correct should produce an h_c that behaves like the ones the encoder was trained on — compact, familiar. An incorrect step is out-of-distribution for the encoder, likely producing a different activation pattern. Raising the threshold means: only commit to "correct" when the probe is highly confident the latent looks like what a correct step should look like; treat everything else as incorrect by default.

Note: L2 normalization means every h_c vector has Euclidean length exactly 1.0 — all latents lie on the surface of a 896-dimensional unit sphere. This means the magnitude of activations carries no information; only the *direction* of h_c matters. Whether correct and incorrect steps are geometrically separated on this sphere (i.e., whether they form distinct directional clusters) has not been tested. That is part of the step-2 feature analysis.

---

## 7. Results

### 7.1 Dataset for This Experiment

Source: `math_shepherd_176k_natural.npz` (176,008 steps, offset 110K, max_len=128)

To train without any duplication, we take a 70/30 subset by keeping all 50,950 unique correct steps and subsampling 21,835 incorrect steps from the 125,058 available:

| Split | Steps | Correct | Incorrect |
|---|---|---|---|
| Training subset | 72,785 | 50,950 (70.0%) | 21,835 (30.0%) |
| Train (80%) | 58,228 | — | — |
| Val (20%) | 14,557 | — | — |
| **Eval (balanced, held-out)** | **2,890** | **1,445 (50.0%)** | **1,445 (50.0%)** |

The eval set is balanced 50/50: the raw clean file (`math_shepherd_eval_5k_clean.npz`) has 1,445 correct and 3,555 incorrect steps; we subsample incorrect to 1,445 to match. A balanced eval makes accuracy directly interpretable as discrimination ability — a majority classifier scores 50%, not 71%. No row in the eval set appears in any training file (verified by row-hash comparison).

### 7.2 Training Curve

Val accuracy across 30 epochs:

| Epoch | Train Loss | Val Acc |
|---|---|---|
| 1 | 0.4219 | 81.16% |
| 6 | 0.3975 | 81.34% |
| 9 | 0.3889 | **81.39%** ← best |
| 20 | 0.3289 | 80.79% |
| 30 | 0.2905 | 80.40% |

The model peaks at epoch 9 and then slightly overfits. Loss continues to fall while val accuracy plateaus and dips, a sign that 30 epochs is more than sufficient for this dataset size.

### 7.3 Held-Out Evaluation

Evaluated on `math_shepherd_eval_balanced.npz` (2,890 steps, 50/50 correct/incorrect, subsampled from offset-100K clean data, never seen during training), threshold=0.8:

```
Majority baseline : 50.00%
Probe accuracy    : 73.40%  (+23.40 pp)
```

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Correct steps | 0.779 | 0.655 | **0.711** |
| Incorrect steps | 0.702 | 0.814 | **0.754** |

**Reading these numbers:**

On a balanced eval, accuracy directly measures discrimination ability — a probe that always predicts the majority class scores 50%. The +23.4 pp gain reflects genuine signal, not baseline inflation.

The threshold of 0.8 makes the probe conservative: it only predicts "correct" when P(correct) ≥ 0.8, and defaults to "incorrect" otherwise. This produces a consistent pattern across all four metrics:

- **Incorrect recall (0.814) > correct recall (0.655).** Most incorrect steps score below the 0.8 threshold and are correctly labelled incorrect. Many correct steps also score below 0.8 — the probe is strict — and get mislabelled as incorrect.
- **Correct precision (0.779) > incorrect precision (0.702).** Because the probe is selective about predicting "correct", nearly every step it does call correct genuinely is. Incorrect precision is lower because the probe sweeps in borderline correct steps (those with P between 0.5 and 0.8) as false positives for the incorrect class.

In short: high threshold → few "correct" predictions but reliable ones (high correct precision); many "incorrect" predictions that include both true incorrect steps and borderline correct ones (lower incorrect precision, high incorrect recall).

### 7.4 Effect of Training Distribution (Directional Finding)

Before the clean eval was established, we ran distribution experiments using the contaminated eval. The exact numbers are not reliable, but the directional finding held consistently across all runs and is explained by a simple mechanism:

- **50/50 rebalancing hurts incorrect-class recall.** Training at 50/50 shifts the probe's implicit prior toward predicting "correct" more often than warranted, raising false negatives on the class that matters most for a verification task.
- **Flipping to 72/28 (favoring correct) hurts incorrect-class precision.** The probe overcounts correct steps and accumulates more false positives on the incorrect class.
- **Natural distribution (28/72) or a mild 70/30 shift** best preserves the probe's calibration to the actual deployment prior.

---

## 8. Interpretation

The SSAE latent space, trained solely for reconstruction on correct GSM8K steps, encodes enough information to detect step-level incorrectness as labeled by an independent oracle (Math-Shepherd). This supports the encoding half of the hypothesis.

One observable difference: incorrect steps activate on average 9.7 more dimensions than correct steps (209.1 vs 199.4 active dimensions out of 896). This is consistent with the SSAE producing more compact representations for in-distribution steps (correct reasoning it was trained on) and more diffuse representations for out-of-distribution ones (incorrect steps it never saw). Whether this sparsity difference is what the MLP probe is actually using, or whether the probe is reading a more complex geometric property of h_c, is unknown. That question belongs to step 2.

Whether this signal generalizes beyond arithmetic reasoning to other domains is the open question being pursued in `WIP_Report.md`.

---

## 9. Limitations

- All results are single-seed (seed=42). No variance estimates have been computed.
- Only one checkpoint is evaluated (`gsm8k-385k_Qwen2.5-0.5b_spar-10.pt`). Different SSAE training runs may produce different latent geometries.
- The 176K training steps and 5K eval steps were encoded at max_len=128 per context and per step. Steps with very long accumulated context are truncated, which may bias against later steps in multi-step problems.
- The 40K and 57K datasets (encoded at max_len=256) are not directly comparable to the 176K dataset and are excluded from results for that reason.
- The probe was trained on a 70/30 subset (no duplication) of the available 176K steps. The minority class (correct, 50,950 steps) drives the dataset size; a larger correct pool would allow more training data without oversampling.

---

## 10. Next Steps

**Step 1 follow-up (strengthen the current result):**
- Multi-seed runs to quantify variance on all reported numbers
- Re-encode the 40K/57K window at max_len=128 to enable a fair comparison across training set sizes
- Implement a logistic regression and a linear SVM baseline on the same SSAE latents — if a linear classifier matches the MLP, the signal is linearly decodable, which is a stronger geometric claim; if it falls short, the structure is non-linear
- Fix context truncation direction: truncate from the left (keep most recent prior steps, drop distant question prefix) or use a sliding window of the last N steps

**Step 2 (feature analysis and mechanistic interpretation):**
- Feature importance: identify which of the 896 dimensions carry the correctness signal most strongly (e.g. permutation importance, SHAP values)
- Geometric analysis: visualise correct vs incorrect h_c vectors (PCA, cosine similarity distributions) to test whether they form distinct directional clusters on the unit sphere
- Compare against the dense baseline `h_k` (pre-SAE projection) to isolate what sparsification contributes
- Linear probe trained on individual feature subsets to test whether correctness is locally decodable

**Longer-term dataset and model direction:**
- Find or construct a dataset where labels directly reflect mathematical step correctness (not MC rollout path viability), better aligned with what the SSAE was trained to encode
- Train an SSAE from scratch on data annotated with MC rollout labels — this would remove the mismatch between the encoder's training objective and the probe's label source
- Extend to symbolic reasoning domain (see `WIP_Report.md`)

**Caveat: context truncation truncates the wrong end.**

The current encoding truncates context from the right (the HuggingFace default), which keeps the first 128 tokens and silently drops the tail. For a 6-step problem where context = question + prior steps, this means later steps lose their most recent predecessors. Concretely, when encoding Step 6 of a rope problem (186 context tokens), the model never sees Steps 3, 4, or 5 — only the original question plus Steps 1–2:

```
Full context (186 tokens):
  "Tony wants to build the longest rope... [question]
   Step 1: Tony has 8 + 20 + 6 = 34 feet of rope.
   Step 2: He has 34 + 7 = 41 feet of rope.
   Step 3: He has 41 - 2 = 39 feet of rope.      ← dropped
   Step 4: He has 39 - 3 = 36 feet of rope.       ← dropped
   Step 5: He has 36 - 1.2 = 34.8 feet of rope."  ← dropped

What the model sees at max_len=128 (128 tokens):
  "Tony wants to build the longest rope... [question]
   Step 1: Tony has 8 + 20 + 6 = 34 feet of rope.
   Step 2: He has 34 + 7 = 41 feet..."             ← cut here
```

The correct approach is to truncate from the left — keep the most recent prior steps (directly relevant to the current step) and drop the distant question prefix if something must go. This is a known flaw in the 176K encoding. The next encoding run should reverse the truncation direction or use a sliding-window context (e.g. keep only the last 3 prior steps regardless of question length).
