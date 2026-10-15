# step_bert_probe_v1: full token states and full text instead of pooled step tokens

Date: 2026-10-15. Experiment ID: `step_bert_probe_v1`. Predecessor: `step_bert_probe_v0` (REPORT §21.22, NO-GO). Related: §21.16 (`bidirectional_token_probe_v1`, token attention on Instruct states). Compute: **TamIA, one H100 node** (changed 2026-10-15 at the user's request; no smoke job). Implementation notes that supersede §3 to §5 are in §9.

## 1. Question

v0 condensed each step into one vector (`[b_i, m_i]`, then PCA 512) before the encoder saw it. Two things could have hidden a future-context signal:

1. **Pooling.** A wrong step may show up in a few tokens (a wrong number, a sign), and later steps may refer back to exactly those tokens. Mean pooling over 33 tokens (on-policy median) blurs both.
2. **The frozen causal backbone.** Qwen3-8B states at token i never saw anything after i. A text encoder trained end to end reads the raw tokens bidirectionally from layer 0.

v1 removes each in turn, on the **same split, labels and metrics as v0**, so every row is directly comparable to the v0 table:

* **Arm A (`tok`): full token encodings.** The step encoder runs over every CoT token's frozen L35 state, not one vector per step.
* **Arm B (`txt`): full text.** A pretrained bidirectional text encoder (ModernBERT-base) is fine-tuned end to end on the raw problem and steps.

In each arm, only the attention mask changes between conditions: `local`, `causal`, `bidir`.

## 2. Data

| what | source | size | status |
|---|---|---|---|
| on-policy token states | TamIA `reprobe_v1/repstore/onpolicy_train_spans/{train,val}` | 20 GB raw, 2,542,475 tokens over 51,711 steps | deleted locally after v0, re-fetch |
| PB token states | TamIA `qwen3_8b_v1/repstore/pb_step_spans/*` | 17 GB raw | re-fetch |
| on-policy step text | TamIA reprobe_v1 trajectory files (exact file to be located in S0) | small | fetch |
| PB step text | `Qwen/ProcessBench` (local HF cache) | | present |
| split, labels | `data/tamia/step_bert_probe_v0/split.json`, pooled `y` | | reuse unchanged |

Trace lengths (on-policy): mean 463 tokens, p95 780, max 815. PB: OmniMath mean 780, max 1,953. Every trace fits in one window for both arms.

**S0 preflight checks:**

1. Concatenating the stored step spans in `step_idx` order reproduces the trace without gaps or overlaps (compare against `orig_step_start_idx`). The pre-step boundary token is kept as its own token with a type flag.
2. Step texts align one to one with stored steps (count per trace equals `n_steps`, and Qwen-tokenized length equals `n_tokens` for at least 99% of steps). Traces that fail are dropped from **both** arms and from the re-scored v0 reference rows, and the count is reported.

## 3. Arm A: token-level encoder over frozen states

**Inputs.**

* Each token's 4096-d state is projected to 256 PCA components, fit on 200k train-split tokens (not whitened), with an added token-type embedding (boundary or span).
* Projected stores: about 1.3 GB on-policy and 1.0 GB PB, fp16. The 37 GB raw spans are deleted after projection.

**Model.** The v0 encoder, run over tokens instead of steps: 2 layers, d = 256, 4 heads, ALiBi, dropout 0.1.

**Step readout.**

* Attention pooling over the step's span outputs with a learned query, concatenated with the step's boundary output.
* An MLP head on top, BCE weighted 1/n_steps per trace, as in v0.

**Masks.** Token q may attend to token k only if:

| condition | rule |
|---|---|
| `local` | k is in q's own step (span plus its boundary) |
| `causal` | k <= q |
| `bidir_W` | step(k) is in step(q) - W .. step(q) + W, W in {1, 2, all} |

**No pretraining.** v0 measured +0.000 F1 from masked-step pretraining (bidir W=2 vs noPT). Masked-token pretraining is listed as an extension only if a bidir condition wins.

**Cost.** About 1.7M train tokens per epoch, sequences of 815 tokens or fewer. Estimated a few minutes per epoch on CPU; S2 measures it.

## 4. Arm B: fine-tuned text encoder

**Model.** `answerdotai/ModernBERT-base` (149M parameters, 8,192-token context, bidirectional, trained on code and math-heavy web text).

* `bert-base-uncased` is already cached but rejected: its 512-token limit cuts PB traces, and lowercasing plus WordPiece mangle LaTeX.

**Input.** `problem [SEP] step_0 [STEP] step_1 [STEP] ...`, where `[STEP]` is a new special token closing each step.

**Readout.** The `[STEP]` token output, concatenated with the mean over the step's tokens, feeds an MLP head with the same weighted BCE.

**Masks** (4D additive attention mask, applied at every layer):

| condition | token in step t sees |
|---|---|
| `local` | problem + step t |
| `causal` | problem + steps 0..t (block-causal: full attention within the visible block) |
| `bidir` | everything |

* ModernBERT was pretrained bidirectionally. `local` and `causal` are therefore off-distribution at the start of fine-tuning, but every condition gets the same fine-tuning.
* If the 4D mask cannot be passed through cleanly, `causal` falls back to truncation (one forward pass per step, input cut after step t). That costs about 5x more.

**Training.**

* AdamW, lr in {2e-5, 5e-5}, wd 0.01, linear warmup 10%, up to 6 epochs, early stopping on val step F1.
* bf16 is off (MPS); fp32.
* MPS first. If losses go non-finite (the v0 MPS failure), fall back to CPU.

**Cost.** About 600 tokens per trace, 3.6k train traces. Estimate 10 to 20 min per epoch on MPS; S2 measures it. If one final fit exceeds 90 minutes, the arm drops to 2 seeds and the cut is reported.

## 5. Stages

1. **S0 fetch.** rsync the spans and step text over the user's open ssh session. Run the preflight checks. Project tokens to PCA 256, verify row counts, and delete the raw spans.
2. **S1 tests first.**
   * Mask tests for both arms: perturbing tokens outside a condition's view must leave that step's logit unchanged, and perturbing tokens inside it must change it.
   * Readout indices against a synthetic trace.
   * Text-to-step alignment.
3. **S2 smoke.** 200 traces, 1 epoch per condition per arm, timing recorded.
4. **S3 HP search.** Seed 0, on `causal` and `bidir_all` (Arm A) or `bidir` (Arm B). The shared HP is picked by mean val F1.
5. **S4 finals.** Seeds 42, 43, 44.
   * Arm A: `local`, `causal`, `bidir_W1`, `bidir_W2`, `bidir_Wall` (15 fits).
   * Arm B: `local`, `causal`, `bidir` (9 fits).
6. **S5 aggregate.** Same metrics and bootstrap code as v0 (`finetune.py` metrics, `overnight.py` contrasts). v0 rows are copied in as same-split references.

`scripts/step_bert/overnight_v1.py` runs S3 to S5 unattended and skips finished fits, as v0's driver did.

## 6. Metrics and decisions

**Headline.** On-policy test step F1 at the val threshold, plus oracle, against always-positive F1 (0.473). Also ProcessBench mean F1_PB at calib-20 and known-step AUROC. Same slices as v0: up to the first error, and after it.

**Primary contrasts** (paired, with a problem-clustered bootstrap):

| contrast | question |
|---|---|
| A: `bidir_best - causal` | does future context help once tokens are not pooled? |
| B: `bidir - causal` | does it help when the encoder reads text bidirectionally from scratch? |
| A `local` - v0 `local` | does pooling a step to one vector lose signal? |
| B `causal` - A `causal` | a fine-tuned text encoder against frozen Qwen3-8B states |

**GO rule (per arm, as in v0).** The `bidir_best - causal` test F1 CI must lie above 0, and the PB calib-20 difference must be >= 0.

Both arms NO-GO closes future context for this population across four designs. The A `local` - v0 `local` row still answers the pooling question either way.

## 7. Files

| file | purpose |
|---|---|
| `scripts/step_bert/project_tokens.py` | S0: spans to per-token PCA-256 store + preflight |
| `scripts/step_bert/step_text.py` | S0: step text per trace, alignment check |
| `src/probes/token_bert.py` | Arm A encoder, token masks, step readout |
| `src/probes/text_bert.py` | Arm B ModernBERT wrapper, block masks, `[STEP]` readout |
| `scripts/step_bert/finetune_tok.py`, `finetune_txt.py` | training, reusing v0 metric functions |
| `scripts/step_bert/overnight_v1.py` | S3 to S5 driver |
| `tests/probes/test_token_bert.py`, `test_text_bert.py` | mask and readout tests |

## 8. Risks

* **Memory.** ModernBERT-base fine-tuning at 2k-token PB sequences on 16 GB. Use batch-by-tokens (8k tokens per batch) and gradient accumulation.
* **Text arm is slow.** It has the most uncertain cost. Arm A runs first, so one arm finishes even if Arm B overruns.
* **Step text alignment.** If on-policy step text cannot be recovered exactly, Arm B runs on PB-style re-segmentation, and the comparison is reported as unmatched.
* **Overlap with §21.16.** Arm A is close to §21.16's design, which found full token attention hurt by 0.026 F1. Arm A differs in backbone (Base vs Instruct), labels (all-step GPT-OSS vs PRM800K human) and split (matched to v0). A negative Arm A is expected and mainly confirms §21.16 on this population; Arm B is the new test.

## 9. As implemented (2026-10-15, TamIA)

* **One job** (`slurm/step_bert_probe_v1_tamia.sh`, 4 x H100). It concatenates the span stores into node-local token arrays (`scripts/step_bert/build_token_store.py`), then `scripts/step_bert/driver_v1.py` runs both arms at once, one fit per GPU per arm. Finished fits are skipped on resubmission.
* **Arm A inputs are the full 4096-d states, not PCA 256.** On an H100 the on-policy tokens (20.8 GB fp16) sit on the GPU, so no compression is needed. The correctness signal is known to live in low-variance directions, which PCA would cut.
* **Arm A windows.** `local` and `causal` are exact masks over whole traces. `bidir_W` with finite W crops the input to steps t-W..t+W per target step, because a 2-layer mask would reach 2W. `bidir_Wall` is the full mask. Readout ids are per (crop, step), so overlapping crops in one batch never share a readout.
* **Arm B visibility is set by the input text, not a 4D mask:** local = problem + step t, causal = problem + steps 0..t (one example per target step), bidir = whole trace (one example per trace, every step read). This is exact at every layer and needs no change to ModernBERT. Up to 4 epochs, patience 2, encoder lr {2e-5, 5e-5}, head lr 1e-3, wd 0.01, bf16 autocast.
* **Step text.** On-policy text is the items files' `candidate_step` (the exact text each span was encoded from), and PB text is `Qwen/ProcessBench`. Both are built on the login node by `scripts/step_bert/build_text_table.py`, with an alignment report.
* **Search.** Arm A: causal_Wall and bidir_Wall x lr {3e-4, 1e-4} x wd {0.01, 0.1}. Arm B: causal and bidir x lr {2e-5, 5e-5}. Both use seed 0, with the shared setting picked by mean val F1. W=4 is dropped from Arm A finals.
