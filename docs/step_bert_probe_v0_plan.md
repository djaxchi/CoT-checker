# Step-BERT probe v0: a bidirectional step-level encoder with masked-step pretraining

Date: 2026-10-14. Experiment ID: `step_bert_probe_v0`. Predecessors: §19.1 (lookahead ceiling), §21.16 (`bidirectional_token_probe_v1`), §21.17 (`bidirectional_token_probe_v2`). Compute: **local only** (Apple M4, 16 GB, MPS). TamIA is used only as a file source.

## 1. Why this exists

Three earlier designs gave a probe access to later steps, and none of them gained from it:

| design | readout | supervision | future effect |
|---|---|---|---|
| §19.1 lookahead ceiling | linear on pooled future states / deltas | PB first-error, within-PB CV | AUROC negative in 8/8 cells; F1_PB +0.007 once the past anchor is controlled |
| §21.16 v1 | 2-layer transformer over tokens | PRM800K human first-error | full - causal: -0.026 PRM F1, -0.022 F1_PB |
| §21.17 v2 | same | ReProbe all-step DeepSeek labels | full - causal: -0.019 test F1, -0.138 F1_PB |

All three learned from labels alone. The v2 diagnosis is that context probes lean on "an error already happened nearby" (P(incorrect | earlier labeled error) = 0.438 vs 0.037 without one) instead of judging the step itself.

This design changes two things:

1. **Unit.** The probe reads one token per step, not per LLM token. A trace becomes a sequence of about 5 to 40 step tokens. The model is BERT-sized at most (a few M parameters) and trains in minutes on a laptop.
2. **Objective.** The probe is first pretrained without labels: mask a step and reconstruct it from its neighbours (BERT's masked-LM, applied to reasoning steps). Only then is it fine-tuned on labels. The label shortcut cannot shape the pretrained context model.

**Primary question.** Does a step's predictability from its context (before, after, both) carry correctness signal? And with the pretrained encoder fine-tuned, does bidirectional context beat a matched causal model on held-out labeled steps and on ProcessBench?

## 2. Data inventory (verified 2026-10-14 on TamIA)

All stores are Qwen3-8B-Base, layer 35, `readout = step_span_with_boundary`, fp16 `token_seq` (one row per step: pre-step boundary token followed by the step's span tokens; `meta.pre_step_boundary_idx` and `meta.step_start_idx` index into the row).

| role | path on TamIA (`$SCRATCH/cot_mech/...`) | size | content |
|---|---|---|---|
| train / val / test pool | `reprobe_v1/repstore/onpolicy_train_spans/{train,val}` | 20 GB | 5,483 Qwen3-8B on-policy PRM800K trajectories, 51,711 steps, **every step encoded** (coverage 1.0) and **every step labeled** (GPT-OSS-120B). `prompt_style = verifier` |
| problem text | `reprobe_v1/reprobe_train.shard0{0..3}_items.jsonl` | 66 MB | `fork_id`, `problem`, for the problem-text split |
| transfer eval | `qwen3_8b_v1/repstore/pb_step_spans/{gsm8k,math,olympiadbench,omnimath}` | 17 GB | ProcessBench, 3,400 traces, 25,676 steps, every step encoded; `label` = first-error index (-1 if none) |

Label convention in the on-policy store (checked on the meta): `step_label = 1` is **correct**, `0` is **incorrect**; trace field `label` is the first-error index. Counts: 18,564 steps in all-correct traces; 11,126 correct steps before a first error; 3,323 first-error steps; after the first error, 11,829 incorrect and 6,869 correct. So 37% of post-error steps are labeled correct, which is what lets the label decouple from "an earlier error exists".

What is **not** available: the v1/v2 token stores (294 GB) were node-local and are gone; the §19.1 `pb_full_solution` store is gone; `probe_train_full` is a balanced fork set, not trajectories. No new encoding is planned.

## 3. Stages

### S0. Fetch and pool (local, no GPU)

1. `rsync` the two span stores and the items files to `data/tamia/step_bert_probe_v0/raw/` (about 37 GB; 114 GB free locally).
2. `scripts/step_bert/pool_step_tokens.py` memmaps each shard and writes, per step, two 4096-d fp16 vectors:
   * `b_i`: the pre-step boundary state,
   * `m_i`: mean over the step span.

   Output: `data/tamia/step_bert_probe_v0/pooled/{onpolicy,pb_<subset>}.npz` with `b`, `m`, `trace_id`, `step_idx`, `y` (1 = incorrect, after flipping the on-policy convention), `first_error_idx`, `n_tokens`. About 0.85 GB for the pool and 0.4 GB for PB.
3. Verify pooled checksums and row counts against the meta (51,711 and 25,676). Then delete the raw spans.
4. **Preflight check:** the PB store predates the `prompt_style` meta field. Confirm that it used the `verifier` template, as the on-policy store did (compare `orig_step_start_idx` for one PB trace against a fresh tokenization of the verifier prefix). If the templates differ, PB is a cross-format transfer and is reported as such.

### S1. Split

`scripts/step_bert/make_split.py` builds a split by canonical problem text (NFKC, lowercase, whitespace-collapsed), 70 / 15 / 15 train / val / test, seed 1729, stratified by "trace has an error". It drops on-policy problems whose text matches a ProcessBench problem, so PB stays a clean transfer set, and writes `split.json` once. Every later stage reads it.

### S2. Masked-step pretraining (no labels)

**Model.** `src/probes/step_bert.py`:

* **Input token per step:** `LN(b_i) ++ LN(m_i)` -> Linear(8192 -> d).
* **Encoder:** pre-norm transformer, d = 256, 2 layers, 4 heads, FF 1024, dropout 0.1.
* **Positions:** relative (ALiBi bias) only, so absolute step index is not a free cue.
* **Mask:** a learned `[MASK]` embedding.

**Token definition is an ablation, not a fixed choice** (added 2026-10-14 after the subset probe check, `results/step_bert_probe_v0/token_probe_sub1000.txt`). On 1,000 subset traces, a single-step linear probe (PCA 256, logistic, problem-held-out, 1,696 test steps, prevalence 0.287, always-positive F1 0.445) gives AUROC: `m_i` 0.812, `[b_i, m_i]` 0.799, `b_{i+1}` 0.794, `c_i` 0.784, `b_i` 0.736, `b_{i+1} - b_i` 0.657. That matches the 7B leaderboard order (span pooling > last token > delta). But §19.1 showed that for *context* (neighbour) tokens, absolute states carry a persistent "error region" level that destroyed localization while deltas did not. So two token sets run through S2 and S3:

* `state`: `[b_i, m_i]` (current default).
* `contribution`: `c_i = m_i - b_i` only, so neighbours carry what each step changed and no absolute level.

A third, `state_self + contribution_context` (the target step reads `m_i`, neighbours read `c_j`), runs only if the two disagree. The S0 pooling also stores the last span token `l_i` for a `last`-token check.

**Target space.** This is fixed, never learned, to avoid collapse. Fit PCA on train-split `m_i` (or on the delta, see G0), keep 256 components, NOT whitened (variance-weighted, changed 2026-10-14 so low-variance noisy directions do not dominate), divided by one shared scale sqrt(mean component variance) so a loss of 1.0 means "no better than the mean". Loss: MSE in that space at the masked positions.

**Visibility modes.** Each training sequence draws one target step i and one mode:

* `both`: all other steps visible,
* `left`: only steps < i,
* `right`: only steps > i.

The draw is uniform, plus 15% random extra masking of other steps for robustness. One model therefore yields three scores per step at inference:

* `e_left(i)`, `e_right(i)`, `e_both(i)` = reconstruction error of step i, with i masked.

A separate **causal-only** pretrained model (mode `left` only) is the matched control for S3.

**Gate G0 (leakage, run first).** Causal LLM states downstream of step i already encode step i. The risk is that `right` reconstruction is trivial (copying step i out of `b_{i+1}`) and carries no signal. Measure ridge R² for predicting target(i) from `b_{i+1}` alone and from `m_{i+1}` alone, on train.

* If R² > 0.9, the raw tokens leak. Switch inputs and targets to contribution tokens: `c_i = m_i - b_i` (the step's within-step change; the S4 contribution idea and the §19.1 delta that avoided the pooled-state collapse). Keep `b_i` out of the masked position's neighbours.
* Record the R² either way.

**Gate G1 (zero-label signal).** On the held-out test split, compute AUROC of `e_both`, `e_left` and `e_right` for incorrect vs correct steps, with 95% problem-clustered bootstrap intervals.

* Also report each AUROC after residualizing on the structural features from v1: step index, step length, prefix length, number of later steps, later-token count.
* Also report each AUROC sliced into: steps up to and including the first error; and steps after it.
* **Pass condition:** `e_both` beats chance with its CI above 0.5 after residualization, and `e_both - e_left` > 0 with CI excluding 0. This is the clean version of the user's question: does seeing what comes after make a wrong step look more wrong?

### S3. Fine-tune with a sliding window

**Head.** An MLP over `[e_left, e_right, e_both, z_i, x_i]`:

* `z_i` is the encoder output at step i, with i unmasked.
* `x_i` is the projected input token.

**Window.** Step i attends to steps i-W .. i+W, with W in {0, 1, 2, 4, all}.

**Conditions** (identical parameter count; only the attention mask differs):

| condition | visible steps | uses scores |
|---|---|---|
| `local` | i only (W=0) | none |
| `causal` | i-W .. i (from the causal-only pretrained model) | `e_left` |
| `bidir_W` | i-W .. i+W | `e_left, e_right, e_both` |
| `bidir_W_noPT` | same, randomly initialized encoder | same |

`bidir_W_noPT` isolates what pretraining adds; it is the closest analogue of v1/v2 `full`.

**Training.**

* Loss: BCE averaged over a trace's labeled steps, then over 32 traces.
* AdamW, lr in {3e-4, 1e-4}, wd in {0.01, 0.1}, selected on val step F1 (seed 0).
* Then seeds 42, 43, 44 for the finals.
* Threshold frozen on val, applied unchanged to test and PB.

### S4. Evaluation

**Headline.** Step F1 at the val-selected threshold, plus the oracle threshold as a ceiling, against always-positive F1 on the same set. AUROC is secondary.

**Sets.**

* On-policy test (all-step labels).
* ProcessBench mean F1_PB and per-subset F1_PB. A trace's prediction is the first step over threshold.
* Known-label step AUROC on PB.

**Primary contrasts.** Paired by seed, with problem-clustered bootstrap CIs (10,000 resamples):

* `bidir_best - causal`: the decision.
* `bidir_best - bidir_noPT`: the value of pretraining.
* `causal - local`.

**Shortcut diagnostics.** Same as v2, so the results are comparable:

* Report F1/AUROC on steps after the first error separately.
* On 256 stratified test targets, compare scores with the future shuffled, the future dropped, and the final answer step removed.

**Reference rows.** The existing on-policy cells (`reprobe_v1/cells/*`) are listed for context, not as matched comparisons. They used a different split and a token-level readout.

## 4. Decision rules

* **G0 fails** (leak) and contribution tokens still leak (R² > 0.9): stop. Step-level reconstruction cannot read the future in these states. That is reportable as a negative.
* **G1 fails** (`e_both` at chance after residualization, or no gain over `e_left`): report it. Still run S3 with `bidir_noPT` vs `causal` only if cheap (it is), but the pretraining story is dead.
* **S3 GO:** `bidir_best - causal` > 0 with CI excluding 0 on on-policy test step F1, **and** not negative on mean F1_PB. Then the next step is a matched TamIA run on the v2 ReProbe set (32k traces, DeepSeek labels) once compute frees up.
* **S3 NO-GO:** report it alongside §21.16 and §21.17 as a fourth form of future access that does not help, with the pretraining contrast as the new information.

## 5. Files to write (tests first)

| file | purpose |
|---|---|
| `scripts/step_bert/pool_step_tokens.py` | S0 pooling from `ShardedRepSplit` rows |
| `scripts/step_bert/make_split.py` | S1 problem-text split, PB overlap removal |
| `src/probes/step_bert.py` | encoder, visibility masks, `[MASK]`, ALiBi, heads |
| `scripts/step_bert/pretrain.py` | S2 + G0 + G1 |
| `scripts/step_bert/finetune.py` | S3 grid and finals |
| `scripts/step_bert/evaluate.py` | S4 metrics, bootstrap, diagnostics |
| `experiments/step_bert_probe_v0/config.yaml` | every hyperparameter above |
| `tests/probes/test_step_bert.py` | masks: `left` never sees j > i, `right` never sees j < i, window W never sees \|j - i\| > W (perturb out-of-view steps, outputs must not move); `[MASK]` hides the target; PCA target is frozen |
| `tests/analysis/test_step_bert_data.py` | pooling indices against a synthetic `token_seq` shard, label flip, split disjointness, PB overlap exclusion |

## 6. Budget

* **S0:** download time, plus about 10 minutes of memmap pooling.
* **Training data:** about 3,800 train traces, 36k steps.
* **S2:** one pretraining run per mode set, about 50 epochs at seconds per epoch on MPS.
* **S3:** 4 HP fits x 2 conditions for the search, then 3 seeds x (1 + 2 + 4 windows) finals. Each fit is a few minutes.
* **Total:** well under a day of laptop time.
* **Memory:** pooled fp16 arrays fit in RAM (about 1.3 GB).

## 7. Known risks

* **Small pool.** 5.5k traces is small for pretraining. If G1 is borderline, an unsupervised-only extension is legitimate: the 22 GB `onpolicy_v1/repstore/onpolicy_step_spans` pool, unlabeled use only. But that pool is the frozen-verifier evaluation pool for the TTS work, so using it would need a note in REPORT.
* **Labels are GPT-OSS-120B judgements, not human.** PB (human first-error labels) is the only human check.
* **Leakage through causal states (G0)** is the most likely failure. It is tested before anything else.
* **Template mismatch between stores** is handled by the S0 preflight.
