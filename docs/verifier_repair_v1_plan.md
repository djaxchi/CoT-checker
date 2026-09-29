# Does the verifier recognize genuine repair?

Frozen before extraction on 29 September 2026. This follows the exploratory v1 audit. No diagnostic outcome was inspected when choosing these contrasts.

The dataset has 64 numeric families, 32 each in affine equations and multiplication. Each family crosses two prefix conditions, two conclusion conditions, four candidate forms, and two stages, for 2,048 rows. Eight families per domain form a dev partition; the primary test partition contains 24 per domain. Partitions hold out numbers within templates, not templates. The builder validates arithmetic witnesses; independent human review remains pending. No probe is trained or threshold fitted on these examples.

The prefix contains either a correct intermediate calculation or a plausible incorrect one. The candidate conclusion is either the true answer or the answer inherited from the incorrect prefix. Candidate forms are:

- `direct`: only state the conclusion. A correct answer after a wrong prefix is an unsupported jump.
- `recompute`: explicitly recompute from the original problem and state the conclusion.
- `check`: prepend “Let me check the calculation.” to the recomputation.
- `repair`: prepend “Let me correct the calculation.” to the identical recomputation.

Correct and incorrect recomputations share the same framing. The repair wrapper states an intention, avoiding an explicit false claim of a past mistake in the correct-prefix controls. Wrong recomputations contain a false arithmetic equality. After each candidate, a follow-up adds one to the stated value. That operation is locally consistent even when it propagates a globally wrong value. The label fields distinguish these meanings; no headline F1 merges them.

Example: original problem `Solve 4*x + 6 = 50`; wrong prefix `Subtracting 6 from both sides gives 4*x = 48.` The repair candidates are:

- Genuine: `Let me correct the calculation. 50 - 6 = 44. Dividing by 4 gives x = 11.`
- False: `Let me correct the calculation. 50 - 6 = 48. Dividing by 4 gives x = 12.`

**Primary endpoints, separately by domain.** Average each family contrast across each probe's available seeds, then bootstrap whole families with 2,000 draws and seed 731. Report per-seed results alongside the averages.

1. Genuine versus false repair under a wrong prefix: score(false repair) minus score(genuine repair). Also report the fraction of families with the correct preference and both absolute score levels.
2. Genuine repair versus the unsupported correct-answer jump: score(direct correct answer) minus score(genuine repair), under a wrong prefix. This includes length, wording, and supplied mathematical justification; it is not a pure causal estimate of justification alone.
3. Repair-wording effect: repair minus check, for identical arithmetic and prefix, reported separately for correct and incorrect arithmetic and correct and incorrect prefixes. A preference for repair language alone is insufficient to pass endpoint 1.
4. Prefix dependence: genuine repair after wrong versus correct prefix, with the candidate text fixed.
5. Follow-up: compare correct and wrong conclusion histories, and compare a genuine-repair history with an unsupported-jump history. These are locally valid subsequent operations. Low scores do not certify that the earlier error was repaired.

No endpoint selects the best-performing probe or form. Primary candidates contain several clauses scored together as one step; repair length and segmentation are limitations. Intervals describe these numeric families and do not include seed uncertainty or multiple-testing correction.

**Likelihood analysis.** During the same forward passes, record the conditional log probability and entropy of every candidate token. Logit position t-1 predicts token t. Save average and summed negative log likelihood, plus average NLL over the mathematical portion after the wrapper. For the primary genuine/false repair pairs, compare the sign of the verifier preference with the sign of the math-NLL preference. Report all four agreement categories and paired effects. Opposite preferences demonstrate a behavioral difference on those examples; agreement or correlation does not establish that the verifier is merely a likelihood detector. These comparisons cannot independently manipulate surprise and mathematical validity.

**Execution.** Use the pinned Qwen/Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`, the existing verifier prompt and separate prefix/candidate tokenization, hidden_states[35], bfloat16 forward passes, and float16 span storage with the pre-step boundary. The original encoder performs extraction; an observer records likelihoods from the same output. Reject truncation, incomplete shards, changed bundle hashes, and mismatched row identities.

TamIA performs extraction on one four-H100 node with a one-hour allocation. Python 3.12 and offline packages match the successful original extraction: torch 2.14.0, transformers 5.14.1, numpy 2.5.3, pyyaml 6.0.3. Download the completed store and run scoring on the laptop CPU. Reuse the same 16 hash-verified checkpoints: attention-query and d128 seeds 42/43, others 42/43/44. Their original source-test validation is inherited from the hash-matched completed v1 run, not recomputed from the full training store.

Commands:

```bash
.venv/bin/python scripts/analysis/verifier_repair_experiment.py validate
.venv/bin/python scripts/analysis/verifier_repair_experiment.py score \
  --extraction results/verifier_repair_v1/extraction \
  --out results/verifier_repair_v1/scores
```

The frozen bundle lives in `experiments/verifier_repair_v1`. The batch script is `slurm/verifier_repair_tamia.sh`. The user authorized execution of this test; no commit was requested.

**Submission record.** Job 494754 was submitted on 29 September 2026 without dependencies. Source snapshot: `/home/d/dchikhi/CoT-checker-snapshots/verifier-repair-20260929`. Output: `/project/aip-azouaq/dchikhi/cot_mech/verifier_repair_v1/run_01`. Log: `/project/aip-azouaq/dchikhi/cot_mech/verifier_repair_v1/logs/verifier_repair_v1-494754.out`. The submission receipt and source manifest are in the parent `verifier_repair_v1` directory. Forty related local tests passed before interpreting any result. No commits were made.

After local scoring:

```bash
.venv/bin/python scripts/analysis/verifier_repair_analysis.py \
  --extraction results/verifier_repair_v1/extraction \
  --scores results/verifier_repair_v1/scores \
  --out results/verifier_repair_v1/analysis
```

**Local continuation.** A local completion process runs `scripts/analysis/finish_verifier_repair_local.sh 494754`. It polls the job every 60 seconds for up to six hours, stops on a failed or cancelled job, then downloads the completed extraction and executes scoring and analysis without another GPU allocation. Its log is `results/verifier_repair_v1/finish.log`; successful completion writes `results/verifier_repair_v1/local_completed_at.txt`. A queued job has no interpreted results yet. The likelihoods condition on the existing verifier-format prompt, not on a natural chat-generation prompt.
