# MPB Pre-Study: Experimental Setup Report

**Author:** Djalil Chikhi  
**Date:** 2026-05-21  
**Run name:** `prestudy_v1_qwen2_5_1_5b_prm800k_40k_seed42`

---

## Research Question

Can mechanistic signals extracted from a language model's internal representations serve as reliable indicators of incorrect reasoning steps in Chain-of-Thought traces, at a dramatically lower cost than training a dedicated Process Reward Model (PRM)?

The core hypothesis is that hidden states at the boundary of a reasoning step carry enough structural information about step quality that a lightweight probe trained on those representations can match or approximate the performance of a full PRM, without requiring any reward-model fine-tuning.

This pre-study is designed to answer a prior question: is the signal strong enough to bother? Before committing to large-scale mechanistic analysis, we need to verify that raw hidden states extracted from a small base model contain step-level quality information that is linearly separable.

---

## Dataset: PRM800K

**Source:** OpenAI PRM800K, a human-annotated process reward model training dataset. Each record contains a math problem, a multi-step solution trace, and per-step human ratings of each candidate completion.

**Label convention:**

| Human rating | Interpretation | Label assigned |
|:---:|:---|:---:|
| +1 | Correct step | 0 |
| -1 | Incorrect step | 1 |
| 0 | Ambiguous | Discarded |

Flagged completions are also discarded. This gives a clean binary signal: label 1 means the step is bad.

**Raw schema (OpenAI format):**  
The dataset is nested: `sample["question"]["problem"]` and `sample["label"]["steps"]`. The builder handles both the canonical nested format and flat mirrors transparently.

---

## Dataset Construction (`scripts/build_prm800k_prestudy.py`)

### Candidate extraction

For each problem, the builder walks the step sequence in order. At each step, all annotated candidate completions (excluding rating=0 and flagged) become examples. The prefix fed to the model is the concatenation of all steps selected by the human annotator up to that point, joined by double newlines.

The input format for each example:

```
Problem:
{problem text}

Previous reasoning:
{prefix of selected steps, or empty if step_idx=0}

Current step:
{candidate_step}
```

When the prefix is empty (first step of a solution), the format collapses to `Previous reasoning:\n\nCurrent step:\n...` with no trailing newline after "reasoning:". This is intentional and consistent between the builder and encoder.

Each example has a stable UID:
```
prm800k::{problem_id}::{solution_id}::{step_idx}::{completion_idx}
```

UIDs are derived from the raw dataset identifiers and are stable across reruns.

### Length filtering

After extraction, examples whose tokenized length exceeds `max_seq_len=2048` are discarded. The filter uses split tokenization: the prompt prefix and candidate step are tokenized independently and their lengths summed. This exactly mirrors what the encoder does, guaranteeing no example can pass the filter but fail at encoding time.

The BOS token is included in the prefix tokenization and no special tokens are added to the candidate, matching the encoder's behavior.

### Train/val splits

| Split | Positive (label=0) | Negative (label=1) | Total |
|:---|:---:|:---:|:---:|
| Train | 20,000 | 20,000 | 40,000 |
| Val | 500 | 500 | 1,000 |

Splits are **problem-level disjoint**: no problem_id appears in both train and val. Problems are allocated to val first, then the remainder goes to train. This prevents the probe from memorizing problem structure rather than step quality.

Seed: 42. All shuffles use a single `random.Random(42)` instance initialized at the start of the build script.

### Derived files

| File | Contents | Rows |
|:---|:---|:---:|
| `prm800k_pos_base_20k.jsonl` | Train positive examples (label=0) | 20,000 |
| `prm800k_neg_base_20k.jsonl` | Train negative examples (label=1) | 20,000 |
| `prm800k_probe_train_40k.jsonl` | Balanced train set (pos+neg interleaved) | 40,000 |
| `prm800k_mixed_train_40k.jsonl` | Same 40k examples as `probe_train_40k` (reserved for future SAE training) | 40,000 |
| `prm800k_val_1k.jsonl` | Balanced validation set | 1,000 |
| `prm800k_contrastive_forks_20.jsonl` | Nested fork records | 20 |
| `prm800k_contrastive_forks_20_flat.jsonl` | Flat fork records (one row per completion) | 40 |

### Contrastive forks

Contrastive forks are pairs of completions at the same step position within the same solution trace: one rated +1, one rated -1. They share an identical prefix. These 20 pairs form a controlled analysis set where the only variation is step quality, enabling attribution analysis without confounds from problem content or reasoning depth.

Forks are preferentially drawn from examples that do not overlap with the train or val sets.

---

## Hidden State Encoding (`scripts/encode_prm800k_hidden_states.py`)

### Model

**Qwen2.5-1.5B** (base, not instruction-tuned). Float16 inference. No fine-tuning of any kind.

Rationale: the pre-study uses the smallest capable open model to establish a lower bound. If a probe on a 1.5B base model's representations achieves non-trivial accuracy, that motivates scaling to larger or more interpretable models (e.g., Gemma-2 9B with published SAE dictionaries).

### Extraction protocol

- **Layer:** last hidden layer (`outputs.hidden_states[-1]`)
- **Token position:** last token of `candidate_step` in the full sequence
- **No truncation:** the script calls `sys.exit()` on any sequence exceeding `max_seq_len`. The builder filters these out upstream; reaching the encoder with an overlength example is treated as a bug.

The candidate boundary is computed via split tokenization before any padding is applied, so the extracted position is never affected by batch padding.

### Batching

Examples are right-padded to the longest sequence in the batch with an attention mask. The pad token ID defaults to the tokenizer's EOS token (Qwen's convention). Padding positions do not influence the hidden state at `candidate_last_token_idx` because the attention mask excludes them.

### Output format per split

| File | Shape | Dtype | Contents |
|:---|:---|:---:|:---|
| `{stem}_h.npy` | (N, 1536) | float16 | Hidden state vectors |
| `{stem}_y.npy` | (N,) | int32 | Labels (0=correct, 1=incorrect) |
| `{stem}_meta.jsonl` | N rows | JSON | UID, token counts, position index |

The hidden dimension for Qwen2.5-1.5B is 1536.

---

## Artifact Validation (`scripts/check_prestudy_artifacts.py`)

The checker runs as the final step of the SLURM job and returns a non-zero exit code on any failure. Checks include:

- Exact row counts for all 7 JSONL files
- UID uniqueness within each file
- Exact label distributions (including zero-count classes to catch label flips)
- No `rating=0` or empty `candidate_step` in any output file
- Train/val UID disjointness and problem-level disjointness (from manifest or recomputed)
- Contrastive fork structure: each fork has one pos (rating=+1, label=0) and one neg (rating=-1, label=1) with identical prefixes
- Cache file existence, shape, and absence of NaN/Inf values
- No truncated examples in encoding metadata
- Manifest completeness (required keys for both dataset and encoding manifests)

---

## Compute Infrastructure

**Cluster:** TamIA (Mila/CC SLURM)  
**Job:** `slurm/build_and_encode_prm800k_tamia.sh`  
**Resources:** 4x NVIDIA H100 80GB, 48 CPU cores, full node memory (`--mem=0`), 6-hour wall time

The job runs fully offline (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`). The Qwen2.5-1.5B model weights must be pre-cached to `$SCRATCH/hf_cache` before job submission.

All outputs are written to `$SCRATCH/cot_mech/prestudy_v1/` to avoid filling `$HOME`.

---

## What This Pre-Study Will Tell Us

Once the pipeline runs, the primary experiment is to train a linear probe on `probe_train_40k` hidden states and evaluate on `val_1k`. The key questions:

1. **Is there a linear signal?** If a logistic regression on raw last-token hidden states achieves substantially better than chance (50%) on the balanced val set, the representations contain step-quality information that is worth studying further.

2. **Does the signal generalize?** The problem-level disjoint split means val problems are unseen at train time. Val accuracy significantly above chance implies the signal is about step quality, not problem-specific patterns.

3. **How sensitive is the signal to step position?** The metadata records `step_idx` for every example. Stratifying val accuracy by step depth will show whether the signal degrades as reasoning chains grow longer.

4. **Does contrastive geometry hold?** For each of the 20 fork pairs, the positive and negative completion share the same prefix. If their hidden states are separable by the trained probe, and the probe's decision boundary aligns with the linear direction identified by the fork pairs, the signal has a coherent geometric structure worth pursuing mechanistically.

If a simple linear probe on a 1.5B base model's untuned representations achieves, say, 65%+ balanced accuracy on a problem-disjoint test set, the full mechanistic investigation (SAE features, circuit-level attribution, process reward comparison) is justified. If the signal is near chance, the approach needs a different backbone, a different extraction site, or a fundamentally different mechanistic hypothesis before scaling up.

---

## Files Produced

```
$SCRATCH/cot_mech/prestudy_v1/
├── data/
│   ├── prm800k_pos_base_20k.jsonl
│   ├── prm800k_neg_base_20k.jsonl
│   ├── prm800k_probe_train_40k.jsonl
│   ├── prm800k_mixed_train_40k.jsonl
│   ├── prm800k_val_1k.jsonl
│   ├── prm800k_contrastive_forks_20.jsonl
│   ├── prm800k_contrastive_forks_20_flat.jsonl
│   └── manifest.json
└── cache/
    └── qwen2_5_1_5b/
        ├── pos_base_20k_h.npy
        ├── pos_base_20k_y.npy
        ├── pos_base_20k_meta.jsonl
        ├── neg_base_20k_{h,y,meta}.*
        ├── probe_train_40k_{h,y,meta}.*
        ├── mixed_train_40k_{h,y,meta}.*
        ├── val_1k_{h,y,meta}.*
        ├── contrastive_forks_20_flat_{h,y,meta}.*
        └── encoding_manifest.json
```
