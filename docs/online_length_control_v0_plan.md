# online_length_control_v0: is the rejection gain a length preference?

Frozen 2026-10-19, before the run.

§21.36: on MATH-500, rejection by `boundary_stats x mlp:h1024x2` beats its blind
control by +0.028 [-0.002, +0.058]. The checker's floor output is far more
common on short drafts (66% under 40 characters, 2.4% above 400), so rejection
keeps short steps and traces grow from 10 to 17 steps. The PRM prefers the kept
redraw in 70.2% of swaps, but that is only evidence for the checker if the PRM
does not simply prefer shorter drafts.

Design: one new arm, `reject_length`, on the same 500 MATH-500 problems with the
same sampler and retry loop (`scripts/onpolicy/online_bon.py`,
`slurm/online_blind_v4_tamia.sh ARMS=reject_length`). It condemns any draft
longer than 110 characters (the 0.667 quantile of plain-arm step lengths, so
33% of first drafts, the probe arm's first-draft rate) and, when every retry is
condemned, keeps the shortest. Every draft is scored by the panel
(`--score_all_drafts`) without influencing the rollout.

Readings, fixed now:

* **PRM agreement baseline.** Share of swaps where the PRM prefers the kept
  draft over the rejected first draft. If the length arm reaches within 5 points
  of the probe arm's 70.2%, the PRM agreement in §21.36 is a length effect and
  carries no evidence for the checker. If it is at or below 55%, the probe's
  swaps carry information beyond length.
* **Accuracy.** Length minus blind against reject minus blind (+0.028), paired
  by problem. Length within 0.01 of the probe arm: length carries the gain.
  Length minus blind at or below 0: it does not. Expect this to be unresolved at
  n = 500; the PRM reading is the primary one.
* Report extra draws per step and generated tokens against the probe arm (0.606,
  1,107).
