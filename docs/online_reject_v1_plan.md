# online_reject_v1: rewrite a step only when the checker condemns it

## Why

Guided decoding (REPORT.md §20.9) is the only experiment in this project that let
the checker act during generation, and it lost: -0.091 against plain sampling at
10.7x the generation tokens. The post-mortem put the blame on the procedure
rather than the head, and named two causes.

1. **Branching needs diversity, so it runs hot.** Five candidate steps per
   position requires temperature 1.5, and that alone cost 0.304 accuracy before
   the checker saw anything (plain 0.365, random-choice-from-the-same-pool 0.061).
   The head then recovered two thirds of it, +0.213 from random to guided with
   McNemar p=9.7e-14. It spent itself repairing damage the procedure caused.
2. **"Lowest uncertainty of five" is the wrong objective for decoding.**
   Restating the problem is never wrong; committing to a number can be. Guided
   averaged 19.5 steps against plain's 9.2, hit a 28-step cap half the time, and
   often reached no boxed answer at all.

Neither cause is about whether hidden states carry step quality. They do: the
same head ranks candidate steps well, and §20.15 shows its prefix score
discriminates from the first step (AUROC 0.659 after step 1, rising to 0.765 by
step 7). So the question this run asks is whether the signal survives a procedure
that does not create either problem.

## What changes

Sample **one** step at the policy's own temperature 1.0. Score it. Resample
**that step only** if the checker condemns it, up to `--max_retries`. Accept the
first draw that passes; if the retries run out, take the least condemned seen.

- The sampler is untouched, so there is no hole to climb out of.
- The checker is only ever asked "is this condemned", never "which of five is
  safest", so an ordinary committing step is accepted on the first draw and
  nothing rewards stalling.
- A second draw is paid for only where the checker objects, so cost is
  `1 + rejection_rate x retries` rather than a flat N. The expectation is ~1.4x
  plain, against guided's 10.7x.

## Arms

| arm | what it is |
|---|---|
| `plain` | one step per position at T=1.0, the base policy |
| `reject` | the rule above, threshold at quantile q of this cell's own offline step scores |
| `reject_blind` | the identical retry loop with the accept decision made by a coin at the measured rejection rate |

**There is deliberately no `random` arm.** The guided run needed one because
branching at raised temperature is a different sampler. This is not: accepting a
uniformly chosen one of k i.i.d. draws from the policy is one draw from the
policy, so `plain` already *is* the exact checker-blind control. `reject_blind`
is not an accuracy control but a price tag: it spends what `reject` spends while
deciding nothing, which is what makes the token comparison honest.

Two thresholds, q = 0.70 and q = 0.85, because a single threshold that happened
to work would not say whether the rule works. Quantiles rather than raw numbers
because score distributions differ wildly between heads.

## Gates

1. **Scoring path.** `--verify_against` re-scores stored trajectories through the
   live path and must match the offline scores the cell already wrote: median
   within 0.002, max within 0.02. The max tolerance sits above the measured
   bf16 batch-shape floor of 0.0064, below which no implementation can go. Stop
   if it fails; generation is too expensive to spend on an unverified scorer.
2. **`reject_blind` must be null.** If the retry loop alone moves accuracy, the
   loop is not distribution-preserving and something is wrong with the sampler
   wiring, not with the checker. Any `plain -> reject_blind` delta outside noise
   invalidates the run rather than being a finding.
3. **Stalling.** Report steps per solution, step-cap hit rate and ungradeable
   rate per arm. If `reject` inflates length the way `guided` did, the objection
   in §20.9 applies to rejection too and the result is negative regardless of
   accuracy.

## Predictions, recorded before the run

- `plain -> reject_blind`: zero, within noise. This is a wiring check.
- `plain -> reject`: positive. The size is the open question. §20.15 measured the
  offline analogue, filtering trajectories on the same prefix signal, at +0.04 to
  +0.06 in accuracy, and rejection should do at least as well because it acts at
  every step rather than once.
- Cost: 1.3x to 1.6x plain, so far below the 10.7x that sank guided that a small
  accuracy gain still clears the matched-token bar.
- Length: within a step of plain. If the rule mostly rejects steps that commit to
  a wrong number, the replacement also commits, so nothing pushes length up.

The comparison that decides it is **accuracy per thousand generation tokens
against `plain`**, and separately against self-consistency at the same token
budget, which is the bar guided failed by +0.281.

## Run

```
sbatch slurm/online_reject_tamia.sh
```

Whole-node H100, four shards, 5h allocation derived from the guided job's 8h
scaled by the token ratio. Shorten it on submit once the first shard reports.
Outputs land in `$SCRATCH/cot_mech/reprobe_v1/online_reject_v1/` with one report
per quantile at `online_reject_report_q070.json` and `_q085.json`.

## Known limits

- One cell (`step_tokens x transformer, seed42`). If the rule works, the seed and
  architecture sweep comes after, not before.
- The threshold is calibrated on the *evaluation* pool's offline score
  distribution, which is a mild transduction: it uses the shape of the
  distribution, not any label. Recalibrating on the training pool is the
  conservative variant if the result is close.
- These solutions average 9.3 steps and 470 generation tokens. §20.15 argues the
  value of acting mid-generation scales with how much trace lies after the
  decision point, so a null here would not generalise to long-form reasoning.
