# reprobe_comparison_v1: the read token against ReProbe on ReProbe's own benchmark

Frozen 2026-10-30, before any run.

## Why

ReProbe (Ni et al., arXiv 2511.06209 v5, ACL 2026) is the published state of the
art for verifying reasoning steps from a frozen policy's internal states. It
trains a ~8.1M-parameter transformer on Qwen3-8B's final-layer hidden states and
reports step-level PR-AUC on eight test sets (Table 1). We have no number in
common with it. Our read token is scored on ProcessBench (off-policy, human
first-error labels, calib-20 F1_PB); ReProbe is scored on Qwen3-8B's own traces
with DeepSeek-R1 all-step labels. The only ReProbe-like cell we have
(`step_tokens x transformer d512`, RTH +0.046 calib-20 over it) differs from
ReProbe in layer, depth, context and training data.

This plan puts both methods on the same backbone, the same training traces and
labels, the same prompt format and the same test traces, so that the readout is
the only difference.

## What ReProbe is (checked against the paper, the code and the released files)

* Backbone Qwen3-8B, non-thinking, prompt `reprobe_prompt` (one `- Step k:` per
  line, then `<Answer>:`). Sampling T 1.0, top-k 50, top-p 0.95.
* Training data: 10.8K PRM800K train problems x 3 samples (32,484 rows, 256-token
  cap), every step labeled by DeepSeek-R1 (`rediska0123/train_prm800k_Qwen3-8B_finished`)
  or by Qwen3-8B itself (`JingweiNi/train_prm800k_Qwen3-8B_finished_self_annotate`),
  1 = incorrect.
* Head (`luh/heads/uncertainty_head_claim.py`, released configs `layer_nums: [-1]`,
  head_dim 512, 1 layer, 16 heads): Linear 4096 -> 1024, LN, GELU, Dropout,
  Linear -> 512, LN, GELU; a 2-row entity embedding marks the step's tokens; one
  TransformerEncoder layer over all prompt and reply token states; mean pool over
  the step's tokens; Linear, LN, GELU, Dropout, Linear -> 1 logit. Token t's
  feature is the state at position t - 1.
* **`mask_future_tokens: true` does not mask future tokens.** The code computes
  `src_key_pd &= cumulative_mask`, where `cumulative_mask` is True up to the
  step's last token. Positions after the step stay attended. Checked on a toy
  mask: claim at positions 2-3, padding at 0 -> ignored = [1, 0, 0, 0, 0, 0, 0].
  With batch size 1 at evaluation, every step's score sees the whole trace,
  including later steps and the final answer. ReProbe as published is
  bidirectional; the read token is causal.
* Training: BCE with pos_weight 3, lr 5e-4, batch 128 traces, 5 epochs, linear
  schedule with 5% warmup, weight decay 0.1, grad clip 1.0, xavier-uniform init,
  best checkpoint on a validation split.
* Steps at test time: `StepsExtractor` (`synthetic_dataset_generation/utils/steps_extractor.py`)
  splits the reply on "\n", drops lines starting with "Reasoning Steps:",
  "SOLUTION:", "<start of response>" or "<end of response>", and aligns tokens
  with a decode loop. The `<Answer>:` line is a step.

## Test sets (recovered, verified)

LM-Polygraph managers (`ue_manager.pth`, loaded with an allowlisted unpickler)
carry the prompt, the sampled reply and its token ids, the DeepSeek-R1 step
labels and ReProbe's Attn+Logit DeepSeek scores. Recomputed AP equals Table 1 to
the third decimal on all eight:

| set | repo | traces | labeled steps | % incorrect | Attn+Logit DS AP |
|---|---|---|---|---|---|
| MATH | JingweiNi/ue_manager_math_no_prm800k_uhead_claim_Qwen3-8B_fixed_prm_layer1_dim512_head16_e5_lr5e-4_pos3 | 728 | 6,164 | 16.1 | .465 |
| GSM8K | rediska0123/ue_manager_gsm8k_uhead_claim_Qwen3-8B_fixed_prm_layer1_dim512_head16_e5_lr5e-4_pos3 | 200 | 1,106 | 8.6 | .616 |
| ProofNet | rediska0123/ue_manager_proofnet_uhead_claim_Qwen3-8B_fixed_prm_layer1_dim512_head16_e5_lr5e-4_pos3 | 186 | 1,640 | 10.3 | .243 |
| Trips | JingweiNi/ue_manager_fixed_prm_layer1_dim512_head16_e5_lr5e-4_pos3_Qwen3-8B_on_trip_plan | 320 | 5,492 | 53.1 | .740 |
| Meetings | JingweiNi/ue_manager_fixed_prm_layer1_dim512_head16_e5_lr5e-4_pos3_Qwen3-8B_on_meeting_plan | 200 | 2,569 | 59.1 | .802 |
| Calendar | JingweiNi/ue_manager_fixed_prm_layer1_dim512_head16_e5_lr5e-4_pos3_Qwen3-8B_on_calendar_plan | 200 | 1,616 | 47.5 | .786 |
| StrategyQA | JingweiNi/ue_manager_fixed_prm_layer1_dim512_head16_e5_lr5e-4_pos3_on_st_qa_cr | 500 | 2,865 | 18.9 | .395 |
| ScienceQA | JingweiNi/ue_manager_fixed_prm_layer1_dim512_head16_e5_lr5e-4_pos3_on_sci_qa_fixed | 500 | 2,621 | 11.3 | .361 |

The five OOD managers also carry per-step scores of ten PRMs (Qwen2.5-Math-PRM-7B
and -7B-PRM800K reproduce Table 1). The MATH manager also scores
Qwen2.5-Math-PRM-72B (.608, not in the paper). No manager holds the hidden-state
probes' scores; we compute those from the released checkpoints
(`JingweiNi/ReProbe_Qwen3-8B_{DeepSeek,self}_anno`). 459 of MATH-500's 500
problems are in the MATH set.

## Arms

| arm | what | training data | context | seeds |
|---|---|---|---|---|
| RP-AL | released Attn+Logit DS scores, from the managers | ReProbe DS, all 32K | full trace | 1 (released) |
| RP-rel-ds, RP-rel-self | released hidden-state checkpoints, through our reimplementation | ReProbe DS / self, all 32K | full trace | 1 (released) |
| RP-ds, RP-self | our reimplementation, ReProbe's recipe, retrained | ReProbe DS / self, our train split | full trace, as released | 3 |
| RPc-ds, RPc-self | same, future tokens masked (what `mask_future_tokens` was meant to do) | same | causal | 3 |
| RT-ds, RT-self | read token, RTH-ICS recipe, ReProbe layout | same | causal | 3 |
| RT-zs | the existing RTH-ICS tokens (rthics_seed0-2), read inline in ReProbe layout, no retraining | PRM800K human, chat layout | causal | 3 |

RT recipe (unchanged from Stage H-C): `read_token_inline.py train --init_text " correct"
--deep --head linear --chain --start_marker --start_deep --lr 3e-3 --soft_attn no_cudnn`,
one epoch, every labeled training step (not the 100k subsample), layout
`reprobe`: `reprobe_prompt(question)` then the reply's lines joined by "\n", a
read token after each line's "\n". Each method keeps its own published recipe;
neither is tuned on the test sets.

RP and RPc heads of one seed share one backbone pass per batch (the same
traces carry both label sets), four heads per GPU.

**Training split.** All ReProbe rows whose problem is in the frozen
bidirectional_token_probe_v2 manifest (so the 752 ProcessBench-overlap problems
are excluded); validation = the v2 `dev` problems; training = every other v2
problem. All rows of a problem stay on one side. RP selects its best epoch on
validation AP (ReProbe selected on a validation split too); RT takes its one
epoch as always.

## Stages

* **S0, data (CPU, login node).** `scripts/reprobe_cmp/build_testsets.py`: the
  eight managers -> per-trace JSONL (prompt, reply tokens, steps via a port of
  `StepsExtractor`, labels, stored scores) and read-token items. Gate T: the
  ported extractor gives exactly as many steps as labels in every set, and the
  stored RP-AL scores reproduce the table above. `scripts/reprobe_cmp/build_train.py`:
  DS and self parquets joined on (question, reply), v2 split, RP trace files and
  RT items for both label sets; ProcessBench rewritten in ReProbe layout
  (`- Step k: ` plus the step with newlines flattened). Overlap audit: test
  questions against training problems and against PRM800K (the RT-zs training
  source).
* **S1, references (1 node, ~1 h).** RP-rel-ds and RP-rel-self on all eight
  sets, full-trace and causal-at-inference; RT-zs on all eight sets.
* **S2, matched training (1 to 2 nodes).** RP/RPc on GPUs 0-2 (one seed each,
  four heads), RT on GPU 3 and a second node (2 label sets x 3 seeds). Every
  trained model scores the eight test sets, validation and ProcessBench in
  ReProbe layout.
* **S3, analysis (CPU).** `scripts/reprobe_cmp/analyze.py`.
* **S4, optional.** Per-step scores of Qwen2.5-Math-PRM-7B and -7B-PRM800K on the
  three math sets (the OOD managers already have them), and best-of-N on
  ReProbe's released pools.

## Metrics

* Primary, as published: PR-AUC (average precision) of "incorrect" per test set;
  ID mean (MATH, GSM8K, ProofNet), OOD mean (five sets), overall mean (eight).
  Unlabeled steps are dropped.
* Secondary: AUROC; F1 of "incorrect" at the threshold chosen on validation,
  oracle F1, and the always-positive F1 per set (never compared across sets).
* ProcessBench: calib-20 F1_PB (leaderboard code), first step above threshold.
  Released checkpoints trained on the 752 overlap problems, so only retrained
  arms enter ProcessBench comparisons.
* Uncertainty: paired bootstrap over traces within each set (2,000 resamples),
  seed-mean scores per arm; for the overall mean, every set is resampled
  independently and the mean recomputed.

## Gates and prespecified readings

* **G0, reimplementation fidelity.** RP-rel-ds and RP-rel-self through our code
  within 0.01 AP of Table 1 on at least 7 of 8 sets each, and the overall mean
  within 0.005 (Table 1: DS .534/.650/.281/.793/.817/.789/.344/.450, overall
  .582; self .558/.673/.264/.799/.819/.785/.381/.553, overall .604). Failure
  stops S2 until explained.
* **G1, retraining fidelity.** RP-ds 3-seed overall AP within 0.03 of .582. If
  lower, the retrain is weaker than the release, and every RT result is also
  read against RP-rel-ds.
* **Headline.** RT-ds minus RP-ds, overall AP, 95% CI. Above 0: the read token
  beats ReProbe with matched data even though ReProbe sees the whole trace.
  CI containing 0: parity at a fraction of the parameters (report the count).
  Below 0: ReProbe wins on its benchmark.
* **Information-matched.** RT-ds minus RPc-ds: the comparison at equal context
  (both causal). RPc-ds minus RP-ds measures what future access buys ReProbe.
* **Domain split.** The same contrasts on the ID and OOD means. The read token
  has only seen math; an OOD loss with an ID win is a domain-transfer result, not
  a method loss.
* **Self labels.** RT-self minus RP-self, read the same way.
* **Zero-shot.** RT-zs against RP-rel-ds: what our current model does on their
  benchmark with no retraining (format and label shift together).
* **ProcessBench.** RT-ds minus RPc-ds and minus RP-ds at calib-20; also RT-ds
  against the PRM800K-trained RTH-ICS (0.651), which says whether ReProbe's
  labels transfer to human first-error labels.
* **PRMs.** Point comparison against Table 1's PRM rows; on the OOD sets also a
  paired bootstrap against the managers' per-step PRM scores.

## Code

* `src/probes/reprobe_head.py`: the head (state-dict compatible with the release),
  claim masks, full-trace and causal attention, ReProbe init.
* `src/data/reprobe_steps.py`: the `StepsExtractor` port and manager parsing.
* `scripts/reprobe_cmp/{build_testsets,build_train,reprobe_probe,analyze}.py`.
* `scripts/read_token_inline.py`: a `reprobe` layout with "\n" as the step
  separator (chat layout unchanged).
* `slurm/reprobe_cmp_tamia.sh` (`STAGE=s1|s2rp|s2rt`).

## Cost

S1 about 1 node-hour. S2: RP forward passes over ~19M training tokens per epoch,
5 epochs, ~1 h per seed per GPU; RT one epoch over all labeled steps, ~1 h per
run. About 4 node-hours in total.
