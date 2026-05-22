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

---

## Benchmark Protocol (v1)

This section describes the **invariant** part of the benchmark: what every candidate, present or future, must respect to produce comparable numbers. It is intentionally separated from the list of candidates so that adding a new candidate does not require editing it. If any rule below changes, bump the version (`v1` → `v2`) and freeze the previous version's results under their original version tag — numbers across versions are not directly comparable.

### Scope

- **Backbone:** Qwen2.5-1.5B base, frozen, no fine-tuning.
- **Supervision source for training/validation:** PRM800K cached hidden states (`$SCRATCH/cot_mech/prestudy_v1/cache/qwen2_5_1_5b/`).
- **Held-out evaluation source:** ProcessBench-GSM8K cached hidden states (`$SCRATCH/cot_mech/prestudy_v1/cache/qwen2_5_1_5b_processbench/`, files `pb_gsm8k_step_h.npy`, `pb_gsm8k_step_meta.jsonl`). ProcessBench is **never** used for training or threshold selection.
- **What candidates may differ in:** how the cached hidden state `h ∈ R^d` is transformed into a feature `z`, and how the final per-step score `p(non_viable) ∈ [0, 1]` is produced from `z`.
- **What candidates may not differ in:** the threshold protocol, the ProcessBench decoding rule, the metrics, the output schema, the seed.

### Invariants

1. **Label convention.** `label = 0` = viable / correct step. `label = 1` = non-viable / erroneous step. Every candidate's reported score must be `p(non_viable)`.

2. **Threshold protocol.** Sweep exactly the 10 thresholds `0.1, 0.2, ..., 1.0` on PRM800K val_1k. Score `p(non_viable)`, compute `bal_acc = 0.5 * (TPR_nonviable + TNR_viable)`, select the threshold maximizing `bal_acc`, break ties toward the **smallest** threshold.

3. **ProcessBench prediction rule.** Group rows by trace `id`, sort by `step_idx`, predict the first step where `score > threshold` (strict `>`), else `-1`.

4. **Official metrics.**
   - `Acc_error` = exact-match accuracy over traces with `label != -1`.
   - `Acc_correct` = exact-match accuracy over traces with `label == -1`.
   - `F1_PB = 2 * Acc_error * Acc_correct / (Acc_error + Acc_correct)` (0 if denominator is 0).
   - `Exact_match_all` = exact-match accuracy over all traces.
   Per-step binary F1 is **not** the headline metric.

5. **Reproducibility.** `seed = 42` for python / numpy / torch / cuda; cudnn deterministic; all shuffling uses scoped `np.random.default_rng(seed)` instances.

6. **Hard guards.** Every runner must assert:
   - `probe_train_40k` has 40000 rows, `val_1k` has 1000 rows,
   - ProcessBench `hidden_dim` matches PRM800K `hidden_dim`,
   - all PRM800K `y` arrays contain only `{0, 1}`,
   - ProcessBench meta row count equals `pb_h` row count,
   - each ProcessBench trace has one consistent label across its rows.

### Required output schema (per candidate)

Every candidate writes its outputs to `$SCRATCH/cot_mech/prestudy_v1/runs/<candidate_id>/`:

```
config.yaml             # exact CLI + device/gpu_name
representation.pt       # representation state dict (omit if candidate has none)
linear_probe.pt         # final classifier state dict (omit if candidate has none)
threshold.json          # selected_threshold, best_val_balanced_accuracy, val_f1_binary, threshold_grid
train_metrics.json      # times, latencies, hidden/latent dims, representation-specific losses
val_scores.npy          # p(non_viable) on PRM800K val_1k
pb_step_scores.jsonl    # one row per ProcessBench trace: id, label, n_steps, scores[], threshold, prediction
pb_predictions.jsonl    # compact: id, label, prediction, threshold
eval_metrics.json       # threshold, Acc_error, Acc_correct, F1_PB, Exact_match_all,
                        # eval_time_sec, mean_step_latency_ms, mean_trace_latency_ms, latency_scope
```

`eval_time_sec` and the two latency fields cover **representation encoding of ProcessBench hidden states + per-step scoring + per-trace aggregation**, with `cuda.synchronize` boundaries. They exclude disk loading and any training. The same scope is recorded inline as `latency_scope` so downstream readers cannot misinterpret it.

The leaderboard merger reads `eval_metrics.json + train_metrics.json + threshold.json` from every subdirectory of `runs/` and emits `leaderboard.csv` + `leaderboard.md`. **Any new candidate that writes the schema above is picked up automatically.**

### Scripts (shared by all candidates)

| Script | Role |
|---|---|
| `scripts/train_easy_probe_method.py` | End-to-end runner. Dispatches on `--method`. Trains representation (if any), trains final classifier (if any), selects threshold on PRM800K val_1k, scores ProcessBench, writes the full output schema. Adding a new candidate normally means adding a branch here. |
| `scripts/evaluate_processbench_from_scores.py` | Recomputes ProcessBench metrics from a saved `pb_step_scores.jsonl`. Supports threshold override. No retraining. Candidate-agnostic. |
| `scripts/merge_easy_probe_leaderboard.py` | Walks `runs/`, reads each candidate's outputs, emits CSV + Markdown leaderboard with aligned columns. Candidate-agnostic — uses no allowlist. |
| `slurm/train_easy_probes_array_tamia.sh` | TamIA SLURM launcher. Full H100 node (`--gpus=h100:4`), per-PID error handling, leaderboard merge at the end. New waves of candidates are added by editing the wave block. |

### TamIA execution (shared)

- **Node:** 1 H100 node, 4 GPUs, 16 CPU cores, 128 GB RAM.
- **Offline:** `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, `TOKENIZERS_PARALLELISM=false`. No downloads inside the job.
- **Environment:** ephemeral virtualenv in `$SLURM_TMPDIR`, `pip install --no-index` from the Compute Canada wheelhouse.
- **Parallelism:** candidates are scheduled in waves of up to 4 across `CUDA_VISIBLE_DEVICES={0,1,2,3}`. Each wave is gated: a failed candidate in wave N aborts the job before wave N+1 and before the leaderboard merge.
- **Storage:** all heavy outputs go to `$SCRATCH/cot_mech/prestudy_v1/runs/`; nothing large written to `$HOME`.

### How to add a new candidate

1. Pick a `candidate_id` (snake_case, unique).
2. Add a branch to `scripts/train_easy_probe_method.py` that produces, for the same inputs, the seven required output files using the invariant protocol above. Reuse `train_linear_probe`, `select_threshold`, `evaluate_processbench` rather than re-implementing them.
3. Append a new entry to the **Candidates** section below using the template (it is short by design).
4. Add the candidate to the appropriate wave in `slurm/train_easy_probes_array_tamia.sh`.
5. Re-run. The merge script picks it up automatically.

No change to this **Benchmark Protocol** section should be required.

---

## Candidates

Every candidate is described with the same fixed template:

- **id** — exact string passed to `--method`.
- **Purpose** — what scientific question this candidate isolates.
- **Representation** — how `h ∈ R^d` becomes `z`.
- **Representation training data** — which cached arrays, with or without labels.
- **Representation objective** — loss and hyperparameters; `none` if no representation is trained.
- **Final classifier** — what produces `p(non_viable)` from `z`.
- **Classifier training data** — defaults to PRM800K `probe_train_40k` with `val_1k` for early stopping; deviations must be stated.
- **Inference-time pipeline** — exact sequence applied to a ProcessBench step.
- **What it tests** — the comparison this candidate enables, ideally naming the candidate it is contrasted against.

Hyperparameters that are constant across all current candidates (overridable via CLI): `seed=42`, `batch_size=512`, `latent_dim = hidden_dim = 1536` (no overcomplete expansion in this stage), final-probe optimizer = AdamW with `lr=1e-3`, `weight_decay=0`, up to 50 epochs, early stopping patience 5 on val BCE. SAE training uses `epochs_sae=20`, `lr_sae=1e-3`, `l1_weight=1e-4`. AdamW everywhere.

### Wave 1 (v1)

#### `random` — RandomProbe

- **Purpose.** Sanity lower bound. Anything that does not beat this is noise.
- **Representation.** None.
- **Representation training data.** —
- **Representation objective.** None.
- **Final classifier.** None — scores drawn directly from `Uniform[0, 1)`.
- **Classifier training data.** —
- **Inference-time pipeline.** PRM800K val scores from `np.random.default_rng(42)`, ProcessBench step scores from `np.random.default_rng(43)` (offset keeps streams deterministic but independent so PB inference is timeable); standard threshold selection; standard first-error decoding.
- **What it tests.** Whether the threshold + decoding machinery alone, on noise, produces any apparent ProcessBench signal.

#### `dense_linear` — DenseLinear

- **Purpose.** The strongest interpretable baseline that is not mechanistic.
- **Representation.** Identity: `z = h`.
- **Representation training data.** —
- **Representation objective.** None.
- **Final classifier.** `nn.Linear(1536, 1)`; score = `sigmoid(logit)`.
- **Classifier training data.** PRM800K `probe_train_40k_{h,y}`, val on `val_1k_{h,y}`.
- **Inference-time pipeline.** `score = sigmoid(W @ pb_h + b)`; standard decoding.
- **What it tests.** Does the raw last-token hidden state of Qwen2.5-1.5B already carry a linearly readable non-viability signal? Reference baseline for every SAE candidate.

#### `sae_positive` — SAE-positive

- **Purpose.** Does a sparse code learned **only on viable steps** still expose a direction that a linear probe can use to detect non-viability?
- **Representation.** Single-layer SAE. `z = ReLU(W_enc h + b_enc)`, `h_hat = W_dec z + b_dec`, `W_enc ∈ R^{1536×1536}`.
- **Representation training data.** `pos_base_20k_h.npy` (20k viable). Labels not used.
- **Representation objective.** `MSE(h_hat, h) + 1e-4 * mean(|z|)`. 20 epochs.
- **Final classifier.** Fresh `nn.Linear(1536, 1)` on encoder outputs. SAE encoder frozen (`requires_grad=False`, `.eval()`).
- **Classifier training data.** `encoder(probe_train_40k_h)` with `y = probe_train_40k_y`; val on `encoder(val_1k_h)`.
- **Inference-time pipeline.** `z_pb = encoder(pb_h)`; `score = sigmoid(probe(z_pb))`; standard decoding.
- **What it tests.** Whether the "viable manifold" is informative for its complement.

#### `sae_mixed` — SAE-mixed

- **Purpose.** Effect of the SAE's training distribution while holding the architecture and objective fixed.
- **Representation.** Same SAE as `sae_positive`.
- **Representation training data.** `mixed_train_40k_h.npy`. Labels explicitly not passed in.
- **Representation objective.** `MSE(h_hat, h) + 1e-4 * mean(|z|)`. 20 epochs.
- **Final classifier.** Fresh `nn.Linear(1536, 1)`. Encoder frozen.
- **Classifier training data.** Same as `sae_positive`, but the encoder is now the mixed-trained one.
- **Inference-time pipeline.** Same as `sae_positive`, with the mixed-trained encoder.
- **What it tests.** Compared to `sae_positive`: does broadening the SAE's training distribution from viable-only to mixed help, hurt, or leave the downstream signal unchanged?

#### `sae_contrastive` — SAE-contrastive

- **Purpose.** Whether a small supervised pressure during SAE training produces a more linearly separable code, while staying scored fairly by a fresh post-hoc probe.
- **Representation.** Same SAE as `sae_mixed`, **plus** an auxiliary head `aux_logit = w_aux^T z + b_aux` used **only during representation training**. The aux head is local to the training function — never saved, loaded, or used at inference, by construction.
- **Representation training data.** `mixed_train_40k_h.npy` + `mixed_train_40k_y.npy`.
- **Representation objective.** `MSE(h_hat, h) + 1e-4 * mean(|z|) + 0.1 * BCEWithLogits(aux_logit, y)`. BCE target uses `y = 1` for non-viable. 20 epochs.
- **Final classifier.** Fresh `nn.Linear(1536, 1)`. Encoder frozen. **Aux head discarded.**
- **Classifier training data.** Same as `sae_mixed`, with the contrastive-trained encoder.
- **Inference-time pipeline.** Same as the other SAE candidates.
- **What it tests.** Compared to `sae_mixed`: does label-aware shaping of the sparse code (with the aux head dropped at inference) yield a more readable signal than purely reconstructive training?

### Candidate comparison matrix

This table is the quick-reference complement to the per-candidate cards; it is regenerated whenever a candidate is added.

| Candidate | Wave | Representation | Rep. training data | Rep. uses labels? | Final classifier |
|---|---|---|---|---|---|
| `random` | 1 (v1) | none (uniform noise) | – | – | – |
| `dense_linear` | 1 (v1) | identity (`z = h`) | – | – | fresh linear probe on `probe_train_40k` |
| `sae_positive` | 1 (v1) | SAE, ReLU bottleneck | `pos_base_20k_h` (viable only) | no | fresh linear probe on `probe_train_40k` |
| `sae_mixed` | 1 (v1) | SAE, ReLU bottleneck | `mixed_train_40k_h` | no | fresh linear probe on `probe_train_40k` |
| `sae_contrastive` | 1 (v1) | SAE + aux BCE head (discarded post-training) | `mixed_train_40k_h` + `_y` | yes (aux only) | fresh linear probe on `probe_train_40k` |

### Planned candidates (not yet implemented)

Reserved IDs and the contrast they will enable. Implementing one means adding a candidate card above and a branch in the runner — no protocol change.

| Candidate (reserved id) | Contrast against | Expected question answered |
|---|---|---|
| `sae_topk` | `sae_mixed` | Hard-sparse (top-K) vs L1-sparse code at equal latent_dim. |
| `sae_overcomplete` | `sae_mixed` | Does latent_dim > hidden_dim help? |
| `ssae` | `sae_contrastive` | Step-level SAE with a structured supervised objective. |
| `mlp_probe` | `dense_linear` | Is the signal non-linear in `h`? |
| `prm_baseline` | all of the above | External PRM reference point on the same trace decoding rule. |

