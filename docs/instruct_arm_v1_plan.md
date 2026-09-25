# instruct_arm_v1: the same verifier on the Instruct backbone

*2026-09-22. Authorised this sprint. The standing constraint in
`docs/onpolicy_tiebreak_v2_plan.md` §3.1 allowed an Instruct arm only as a
matched retrain under an identical protocol; this is that retrain.*

## 1. The question

Qwen3-8B-**Base** was chosen deliberately, because instruction tuning leaves
artifacts in the activations that a correctness probe can latch onto instead of
step content. The cost of that choice is that our absolute numbers do not line up
with anyone else's: our MATH-500 pass@1 is 45.0 where the field reports 92.4 for
Qwen3-8B instruct.

> Does the verifier get better, worse, or the same when it reads an
> instruction-tuned model, and is any change explained by activation artifacts
> rather than by step content?

Only a matched retrain answers it, because probes do not transfer across
backbones.

## 2. The reference to beat

`step_tokens x transformer:d512,l2,f2048,h8` (8,665,089 parameters), the top cell
of the Qwen3-8B-Base leaderboard, seed 42:

| metric | Base |
|---|---|
| in-domain AUROC, PRM800K balanced test | **0.8978** |
| ProcessBench F1_PB, val-selected, avg 4 subsets | 0.427 |
| ProcessBench F1_PB, oracle threshold, avg 4 subsets | 0.576 |
| ProcessBench F1_PB, calib-20, avg 4 subsets | 0.566 |

Per subset, val-selected: gsm8k 0.674, math 0.469, olympiadbench 0.235,
omnimath 0.329. The spread between val-selected and oracle, 0.427 against 0.576,
is itself the §21.9 point: most of the loss is threshold placement, not ranking.
**Report AUROC first and all three thresholds, or the comparison will be read as
a ranking result when it is a calibration result.**

## 3. What changes, and only this

| | Base arm | Instruct arm |
|---|---|---|
| backbone | `Qwen/Qwen3-8B-Base` | `Qwen/Qwen3-8B`, non-thinking mode |
| read layer | block 34 of 36 (`LAYER=35` in the scripts, the hidden-states index) | identical |
| training data | PRM800K, 513,810 steps, problem-id-disjoint, balanced | identical |
| protocol | 30 epochs, patience 3, batch 256, t_max 512, dropout 0.1, rescale none | identical |
| seeds | 42, 43, 44 | identical |
| evaluation | ProcessBench, 4 subsets | identical |

Everything except the backbone is held fixed, so the backbone is the only thing
the comparison can attribute a difference to.

**Prompting must be matched, not templated.** The PRM800K step text is encoded
under the verifier template. Wrapping it in the Instruct chat template instead
would change the object being encoded and confound the comparison with a
formatting change. Encode the same text the same way; the backbone is the
variable.

## 4. The artifact audit, run on the same encode

This is the gate the standing constraint asks for, and it costs one extra pass
over the stored states rather than a job of its own. Four measurements, Base
against Instruct at the read layer:

1. **Outlier-dimension mass.** Fraction of the state's norm carried by the top-k
   dimensions. Instruction tuning is known to sharpen these, and a probe that
   rides them is reading format, not content.
2. **Attention-sink and template mass.** How much of the step's token budget sits
   on template or sink positions rather than on content tokens.
3. **Probe-weight localisation.** Where the trained probe puts its weight. If it
   concentrates on template positions on Instruct and does not on Base, the gain
   is an artifact.
4. **Length and position residualisation.** The standard control, already used in
   §19 and the matched-fork audit, rerun on the Instruct states.

An Instruct win that survives all four is a real win. One that does not is the
reason the constraint existed.

## 5. Jobs

Four, chained with `--dependency=afterok`, because the encodes are long and a
single job risks the 24h cap.

| # | job | what | estimate |
|---|---|---|---|
| 1 | `prm_spanstore_instruct` | PRM800K step spans, `--span_only` (~157 GiB, not the 1.1 TB the two-stage path would need) | 12h |
| 2 | `pb_spanstore_instruct` | ProcessBench step spans, same encoder | 4h |
| 3 | `train_instruct_L` | transformer L, 3 seeds, in-domain AUROC and ProcessBench F1 | 3h |
| 4 | `instruct_audit` | the four artifact measurements | 1h |

Jobs 1 and 2 are independent and can run concurrently on separate nodes if both
are available, which halves the wall clock.

**Download on the login node before submitting.** Compute nodes have no internet:

```
HF_HOME=/project/aip-azouaq/$USER/hf_cache hf download Qwen/Qwen3-8B
```

## 6. The part that is not a rescore

**An Instruct verifier cannot score the existing test-time-scaling pool.** The
probe reads the hidden states of the model doing the reasoning. Our 18,188 saved
traces were written by Qwen3-8B-Base and carry Base states. A probe trained on
Instruct states has no valid reading of them; the residual geometries are not
aligned, and matching dimensionality at 4,096 does not make them comparable.

So "rerun the TTS with it" means **regenerating the pool with the Instruct
policy**: 10 samples each for GSM8K (1,319) and MATH-500, then scoring, then
rebuilding the frontier. That is the original study again, roughly a 3h
generation job plus scoring, not the cheap rescore that the Base-arm rerun is.

This is worth doing and should be stated as a benefit rather than a cost:

- It puts our numbers **on the same scale as the field for the first time**.
  Qwen3-8B instruct pass@1 is 95.6 on GSM8K and 92.4 on MATH-500, against our
  Base pool's 78.8 and 45.0.
- It answers the sharpest open question about the result. Our tie-break gain at
  N=2 is large partly because a weak policy leaves the vote with nothing to
  count. **At pass@1 of 92 on MATH-500, is there anything left for a verifier to
  do at N=2?** If the effect survives, it is a property of the rule. If it
  collapses, the honest finding is that the regime matters more than we claimed,
  and that is still a finding.

Sequencing: run §5 first and read the ProcessBench number. The generation job is
only worth a node if the Instruct verifier is at least competitive with the Base
one, and the retrain tells us that in a day.

## 7. What a null looks like

If the Instruct cell lands within noise of Base on ProcessBench AUROC, the answer
is "the backbone does not matter for this probe", which retires a standing
uncertainty and justifies regenerating the pool on Instruct purely for
comparability with the field.

If Instruct wins **and** the audit is clean, Base was a conservative choice that
cost us accuracy, and the paper should say so.

If Instruct wins **and** the audit is dirty, the constraint was right, and that is
the most publishable of the three outcomes because nobody else in this literature
has measured it.
