# judge_prompt_v1: does telling Qwen3-8B to verify the step make its states a better verifier input?

Date: 2026-10-08. Experiment ID: `judge_prompt_v1`. Predecessors: §21.12 (Instruct leaderboard), §21.15 (prm_backbone_v1), §21.18 (context_ablation_v3).

## 1. Why

§21.15 found that the same probes read from Qwen2.5-Math-PRM-7B beat Qwen3-8B on 21 of 22 cells (calib-20 F1_PB +0.071 on average). The best PRM cell reaches 0.661 and the best Instruct cell 0.610.

The Instruct states were read under the verifier template:

```
Problem:\n{p}\n\nPrevious reasoning:\n{prefix}\n\nCurrent step:\n{step}
```

That is a plain layout. It has no chat template and no instruction, so the model is never told it is judging anything. The PRM was trained to judge. Part of its advantage could therefore be elicitation rather than a better representation.

This experiment asks whether a verification prompt closes that gap. The comparison is with the Instruct verifier-template cells and with the PRM cells, under the same splits, the same layer and the same protocol.

## 2. Prompt

The prompt is Qwen3's chat template in non-thinking mode (`src/onpolicy/prompts.py`, `judge_prefix` and `JUDGE_SUFFIX`):

```
<|im_start|>system
You are a careful math verifier. You are given a problem, the previous steps of a solution,
and the current step. Decide whether the current step is correct, given the problem and the
previous steps.<|im_end|>
<|im_start|>user
Problem:
{p}

Previous steps:
{prefix or "(none)"}

Current step:
{step}

Is the current step correct? Answer Yes or No.<|im_end|>
<|im_start|>assistant
<think>

</think>

```

* The text matches `apply_chat_template` on the local Qwen3-8B tokenizer exactly.
* Prefix, step and suffix are tokenized separately, as the verifier template is. The step's token ids are therefore identical under both prompts.
* The test suite pins the string.

## 3. Readouts (one forward pass per step)

| arm | store | what it reads | what it isolates |
|---|---|---|---|
| A, judge-span | `step_spans` | pre-step boundary + the step's own tokens, as in every leaderboard store | the instruction's effect on the step states. Under causal attention the step can see the instruction but not the question. |
| B, judge-verdict | `judge_token` | the boundary row and the last token of the verdict question, where the next token is the answer | the state where an LLM judge forms its verdict. This is `last_token` on that store. |
| zero-shot | meta of `judge_token` | P(No) renormalized over {Yes, No} at the verdict token | the backbone's own verdict, with no probe |

## 4. Cells and protocol

* **Backbone:** Qwen3-8B at `hidden_states[35]` (block 34 of 36), bfloat16 forward pass, float16 store.
* **Splits:** the frozen PRM800K splits `probe_train_full` (513,810 steps), `val_5k` and `test_2k`, plus all four ProcessBench subsets.
* **Overlength:** the ProcessBench overlength skip is decided on the verifier-template length, so both stores hold the same steps.
* **Arm A** (`experiments/judge_prompt_v1/span.cells`):
  * `step_tokens x transformer:d512`, the best Instruct cell;
  * `boundary_stats x mlp:h1024x2`, the best PRM cell;
  * `boundary_stats`, `last_token` and `step_mean` with a linear learner.
* **Arm B** (`verdict.cells`): `last_token x linear` and `last_token x mlp:h1024x2`.
* **Protocol:** identical to prm_backbone_v1. RESCALE=none, 30 epochs, patience 3, batch 256, and seeds 42, 43 and 44. The lr x wd search runs on seed 42 and is reused for 43 and 44.
* **Learner-free:** `prm_geometry.py` on both stores, without kNN. LDA is the comparison with §21.15's LDA rows.

## 5. Metrics and prespecified reading

* **Headline:** ProcessBench F1_PB at calib-20, the four-subset mean, seed-averaged.
* **Also reported:** oracle F1_PB; test_2k AUROC; in-domain f1_incorrect at the val threshold against the trivial 0.667.
* **Uncertainty:** a paired trace bootstrap of calib-20 differences. Traces are resampled per subset, both sides are recomputed on the same resample, and each side is its seed mean (`scripts/analysis/judge_prompt_report.py`).
* **Precision check (from §21.15):** the PRM-minus-Instruct difference on the matched `boundary_stats x mlp:h1024x2` cell is +0.095, with bootstrap SE 0.010.

Readings, fixed before the run:

1. **The judge prompt helps the Instruct representation** if the best judge-span or judge-verdict cell beats the best Instruct cell (0.610) and the bootstrap 95% CI of the difference excludes 0.
2. **On par with the PRM** if the CI of best-judge minus best-PRM (0.661) includes 0. It **beats the PRM** if the CI lies above 0.
3. **Arm A against B.** If arm A gains and arm B does not, the instruction changes how the step is encoded. If only arm B gains, the gain comes from reading the verdict position, not from better step states.
4. **Zero-shot against probes.** If zero-shot P(No) is close to the probes, the probe adds little over asking the model.
5. **Transfer failure.** A gain in-domain (AUROC or F1) with no gain on ProcessBench is a transfer failure, not support.

## 6. Execution

Both clusters are loaded, so the run is split into small jobs across TamIA and Rorqual (`experiments/judge_prompt_v1/config.yaml`).

* **Storage constraint.** The span store (~165 GB) does not fit TamIA's shared scratch (126 GB free on 2026-10-08).
* **TamIA** (`slurm/submit_judge_prompt_tamia.sh`), whole-node jobs:
  * 4 encode jobs of 4 shards each (16 shards, ~1 h). Spans go to node-local disk. Each job derives the vector reps per shard (`scripts/derive_vector_store.py`) and keeps on shared scratch only the vectors (~45 GB) and the small judge-token store.
  * Then, in parallel: geometry (one readout per GPU), span seed 42 (4 vector cells), and verdict seed 42 plus the zero-shot cell.
  * Then span seeds 43 and 44 and verdict seeds 43 and 44.
  * Vector cells read the pre-derived vectors with `--prederived`. A test pins that this gives byte-identical vectors and scores.
* **Rorqual** (`slurm/submit_judge_prompt_rorqual.sh`), single-GPU jobs:
  * 16 encode shards into shared scratch spans.
  * Then `step_tokens x d512` at seed 42, staged to node-local disk, then seeds 43 and 44.
* **Fallback** if Rorqual stalls: `slurm/judge_prompt_seq_tamia.sh`. It re-encodes on one TamIA node and trains the d512 cell. It is not submitted by default.
* **Cross-cluster caveat.** The d512 cell reads a Rorqual encode and the vector cells read a TamIA encode. Every cell records its input fingerprints; they are not expected to match across clusters.
* **Report:** `scripts/analysis/judge_prompt_report.py` writes `results/judge_prompt_v1/leaderboard.{md,json}`.

## 7. Out of scope

* The PRM under the judge prompt.
* Thinking mode.
* Prompt wording sweeps. One prompt was fixed in advance; a null result applies to this prompt only.
