# reprobe_comparison_v2: other backbones and test-time scaling

Frozen 2026-10-30, before any run. Follows `docs/reprobe_comparison_v1_plan.md`
(Qwen3-8B step detection, done: read token on self labels .652 overall PR-AUC
against ReProbe's best published .604, paired bootstrap +0.049 [+0.030, +0.067]
against the released checkpoint).

v1 answered one question on one backbone. A paper needs two more: does the
result hold on the other backbones ReProbe reports, and does better step
detection turn into better test-time scaling, which is what ReProbe is for.

## What the paper reports beyond Qwen3-8B (v5 PDF, checked)

* "LLMs for reasoning used in our experiments include three state-of-the-art
  models: Qwen3-8B ... in non-thinking CoT mode, Qwen3-1.7B and Qwen3-32B in
  native thinking mode, and Phi-4."
* Step-level PR-AUC: Phi-4 in Table 9 (Attn+Logit only, Qwen3-8B labels),
  Qwen3-1.7B in Table 4 and Qwen3-32B in Table 5 (hidden states, GPT-OSS-120B
  labels, 10.8K training traces).
* Best-of-N: Table 3 (Qwen3-8B, N = 10, N = 5 for planning, trace score =
  minimum step score), Table 6 (Qwen3-1.7B), Table 10 (Phi-4).
* Beam search: Table 2 (Qwen3-8B only), "a beam of B = 5 partial reasoning
  trajectories ... we sample N = 5 candidate continuations".

| backbone | table | ReProbe overall | best PRM overall |
|---|---|---|---|
| Phi-4 | 9 | .448 (Attn+Logit) | .474 (Qwen2.5-Math-7B-PRM800K) |
| Qwen3-32B | 5 | .585 (hidden states) | .613 (Qwen2.5-Math-PRM-7B) |
| Qwen3-1.7B | 4 | .447 (hidden states) | .427 (Qwen2.5-Math-7B-PRM800K) |

On Phi-4 and Qwen3-32B the paper's own probe loses to a 7B PRM, so these two
backbones are where a read-token win would say the most.

## What is released (HF, checked 2026-10-30)

Recomputed AP from the stored scores of each manager:

* **Phi-4.** `rediska0123/ue_manager_Phi-4_qwen3_anno_phi4_layer1_dim512_head16_e5_lr5e-4_pos3_on_{math,gsm8k,proofnet,sci_qa,st_qa}`
  give Attn+Logit .404/.340/.155/.347/.347, equal to Table 9, and hold PRM
  step scores (Qwen2.5-Math-7B-PRM800K on MATH .474, equal to Table 9). The
  Attn+Logit planning managers carry no labels. Labels for Trips, Meetings and
  Calendar are in `rediska0123/ue_manager_Phi-4_qwen3_anno_phi4_hs_5e_on_{trip,meeting,calendar}`
  (an unpublished hidden-state probe: .782/.571/.343). Training data:
  `JingweiNi/train_prm800k_finished_qwen3_annotate_phi-4_prm` (32,480 Phi-4
  replies labeled by Qwen3-8B). Structured "- Step k:" replies, Phi-4 chat
  template. No released hidden-state checkpoint.
* **Qwen3-32B.** `JingweiNi/ue_manager_Qwen3-32B_hs_4e_on_math` (.676) and
  `JingweiNi/ue_manager_Qwen3-32B_hs_5e_on_{gsm8k,proofnet,trip,meeting,calendar,stqa,sciqa}`
  (.535/.682/.752/.698/.734/.303/.302) reproduce Table 5, with Meetings and
  Calendar swapped relative to the table. PRM scores stored for all eight
  (Qwen2.5-Math-PRM-7B MATH .668, equal to Table 5). **The traces are not
  native thinking**: the prompt is the structured non-thinking one
  (`<think>\n\n</think>\nReasoning Steps:`), replies are "- Step k:" lines,
  about 6 steps per trace. Training data:
  `JingweiNi/train_prm800k_gpt-oss-120b_annotate_qwen3_32b_finished` (10,828
  replies, 49,882 steps, 7.6% incorrect, structured). Checkpoint:
  `JingweiNi/ReProbe_Qwen3-32B_GPT-OSS-120B_anno` (last layer, head_dim 512).
* **Qwen3-1.7B.** `JingweiNi/ue_manager_Qwen3-1.7B_natural_hs_10.8k_length_1024_7e_on_*_new`
  reproduce Table 4 on 7 of 8 sets (Meetings: 96 traces, .553 against .505).
  Native thinking, sentences as steps (about 43 per trace). No PRM scores.
  Training: `JingweiNi/train_prm800k_gpt-oss-120b_annotated_qwen3_1.7b_thinking_10.8k_response_length_1024`.
* **Best-of-N pools (Qwen3-8B).** `JingweiNi/test_*_Qwen3-8B_texts_uhead_prm`,
  eight sets, 32 candidates per question, stored trace scores of ReProbe
  Attn+Logit DeepSeek and seven PRMs, gold answers, no correctness field.
  GSM8K has 1,019 questions. Exact-match grading of its first 10 candidates
  on the first 200 questions gives pass@1 93.8, pass@10 97.5, majority 95.5,
  against Table 3's 95.6 / 99.2 / 97.6. The released pool does not reproduce
  Table 3 as it stands (question subset, N or grader differ).
* No beam-search outputs and no Phi-4 or Qwen3-32B best-of-N pools.
* Table 5's PRM rows do not all match the managers' stored PRM scores: GSM8K
  Qwen2.5-Math-PRM-7B is .930 in the table and .552 in the manager (Universal
  .903 against .413, PRM800K .894 against .493), with 4.1% incorrect steps in
  that set. MATH, ProofNet and Trips match (.668, .646, .696). With the stored
  GSM8K value, Qwen2.5-Math-PRM-7B's overall is .566, not .613, so on the
  released data ReProbe (.585) is ahead of it on Qwen3-32B.

## S0 outcomes (2026-10-30)

* Gate T passes for Qwen3-32B on all eight sets (stored AP .6758 .5351 .6821
  .7517 .6982 .7342 .3030 .3016) and for Phi-4 on its five published sets
  (.4044 .3400 .1551 .3473 .3466). The Phi-4 planning managers' steps are the
  stored claims (multi-line steps), not the line splitter's.
* Gate D0 fails as anticipated: our grading of the first N candidates gives,
  against Table 3, pass@1 MATH 66.7 (92.4), GSM8K 92.0 (95.6), StrategyQA
  71.8 (86.8), ScienceQA 88.2 (92.7), Calendar 26.7 (23.5). Audited failures
  are mostly wrong answers (StrategyQA: "No" against gold True); a few MATH
  misses were unicode or tuple spellings, now handled. Best-of-N is compared
  within our harness only, and Table 3's absolute numbers are not used.
* `/scratch` is over its 1 TiB quota, so Qwen3-32B weights (65 GB) cannot be
  downloaded until space is freed; Phi-4 (29 GB) is in `$SCRATCH/hf_cache`.

## Part A: step detection on other backbones

Same protocol as v1: the released test traces and labels, PR-AUC per set, ID
mean (MATH, GSM8K, ProofNet), OOD mean (five sets), overall mean, paired trace
bootstrap, plus AUROC and F1 at the validation threshold.

Arms per backbone:

| arm | what | seeds |
|---|---|---|
| RP-pub | the published probe's stored scores (gate T) | 1 |
| RP-rel | released hidden-state checkpoint through our code (Qwen3-32B only) | 1 |
| RP | our ReProbe reimplementation retrained, ReProbe recipe, same split | 3 |
| RT | read token, v1 recipe unchanged, same split | 3 |
| PRM | stored per-step PRM scores (all sets on Qwen3-32B, five on Phi-4) | 1 |

Training split: every training problem, minus the ProcessBench-overlap
problems as in v1, 5% of problems held out as validation. The read token
recipe is not retuned for the new backbones.

**Restart rule (fixed now).** A run whose
validation AP is more than 0.10 below the median of its arm's runs is reported,
then replaced by the next seed; at most two replacements per arm. RP gets the
same rule. Test sets are not looked at to apply it.

Gates: T (stored scores reproduce the table, per set within 0.0005), and for
Qwen3-32B G0 (RP-rel within 0.01 of Table 5 on 7 of 8 sets).

Headline per backbone: RT minus RP-pub and RT minus the best PRM, overall AP,
95% CI; RT minus RP as the matched control.

Order: Phi-4 first (different family, same layout as v1, 14B), then
Qwen3-32B (same layout, needs a memory smoke test: 64 GB of weights on an
80 GB H100 with gradients through every layer), then Qwen3-1.7B (needs a
sentence-step layout read off the managers' claims; optional).

## Part B: test-time scaling on Qwen3-8B

**B1, best-of-N on the released pools.** Score every candidate of the eight
pools with RT-ds and RT-self (v1 checkpoints) and with the released
hidden-state ReProbe checkpoints (DS, self). Trace score = the maximum step
P(incorrect), the paper's minimum-step aggregation; pick the lowest. N = 10
(5 for planning), the first N candidates per question, plus the curve over
N = 1..32. Also reported: pass@1, pass@N, majority vote, the stored ReProbe
Attn+Logit and PRM arms. Paired bootstrap over questions.

Grading is ours: GSM8K numeric exact match, MATH with `src/eval/math_grade.py`,
StrategyQA yes/no, ScienceQA option match, planning sets by the gold
solution's parse. ProofNet needs an LLM judge and is left out until one runs.
Gate D0: report our pass@1, majority and stored-arm accuracies next to Table
3. If they differ, the comparison is made within our harness only and the
gap is reported, never patched.

**B2, beam search.** B = 5 beams, N = 5 candidate next steps per beam,
Qwen3-8B with ReProbe's prompt and sampling, a step ends at "\n". Arms: RT,
released ReProbe hidden states (in the loop the future does not exist, so its
masking bug is moot), and random choice from the same candidates (prices
the search itself, as in `scripts/onpolicy/online_bon.py`). Sets with
programmatic grading first (MATH, GSM8K, StrategyQA, ScienceQA). Extends
`online_bon.py` from one kept path to B.

## Order

1. S0, login node: Phi-4 and Qwen3-32B test sets from the managers (gate T),
   training splits, the eight best-of-N pools and the grader (gate D0).
   Weights for Phi-4 and Qwen3-32B go to `$SCRATCH` (`/project` is 95% full).
2. B1 scoring (one node, Qwen3-8B, existing checkpoints).
3. Phi-4 RP and RT training and scoring (one node).
4. Qwen3-32B smoke, then training (one to two nodes).
5. B2 beam search.
6. Qwen3-1.7B, if time.
