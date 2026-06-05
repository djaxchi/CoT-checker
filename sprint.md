# Sprint Log

One row per sprint: what we hypothesized, what we tested, what we got. Keep it terse. Newest on top.

| Sprint | Dates | Hypothesis | Verdict |
|---|---|---|---|
| S2 | 2026-06-05 → | Fork-based preference supervision (ranking / triplet) shapes a better latent than reconstruction-only | ⏳ running |
| S1 | 2026-05-21 → 05-26 | SAE/SSAE latents localize the first wrong CoT step better than raw dense hidden states | ✗ not supported |

---

## S2 — Fork-based representation shaping (2026-06-05 → )

**Motivation.** S1 showed the so-called "contrastive" methods are recon + L1 + an auxiliary BCE head that is *discarded* (not contrastive learning), and no learned representation beat raw dense. PRM800K forks (same prefix, multiple rated next steps) give matched positive/negative continuations. We use them to shape the *representation*, not to train a better classifier; eval stays the locked pipeline (freeze representation → fresh probe → ProcessBench).

**Hypotheses.**
- H1 fork preference supervision > reconstruction-only.
- H2 sparse reps benefit more from preference supervision than dense AE reps.
- H3 reasoning validity is local preference structure (ranking helps).
- H4 reasoning validity is local geometry around a prefix (triplet helps).

**Design.** 2×2 matrix = {AE, SAE} × {Ranking, Triplet} → ae_rank, sae_rank, ae_triplet, sae_triplet. Baselines carried from S1 + dense AE (recon-only) + SAE-mixed. Backbone Qwen2.5-1.5B, last-token hidden states. Data: full-scale PRM800K forks (~40k train / 5k val, problem-disjoint, surplus held out). Ranking = scalar head, score(pos) > score(neg); Triplet = anchor(prefix) close to pos, far from neg in latent space. Heads discarded; only `representation.pt` saved. Eval: 4 ProcessBench subsets, macro F1_PB, val-selected + oracle thresholds.

**Results.** _pending_ — see `mpb_results.md` / leaderboard once runs land.

**Verdict.** ⏳ running.

**Next.** Build + encode forks on TamIA; run the 4-cell matrix; compare to S1 dense/SAE-mixed.

See `docs/sprint2_dataset_design.md` for the dataset + objective design note and exact commands.

---

## S1 — Full ProcessBench evaluation (2026-05-21 → 05-26)

**Hypothesis.** SAE / SSAE latents carry a step-level correctness signal that beats (or matches) the raw final-layer hidden state for first-error localization on full ProcessBench.

**Tested.** Backbone Qwen2.5-1.5B, PRM800K supervision. Methods: DenseLinear, SAE-{positive,mixed,contrastive}, SSAE-{positive,mixed,contrastive}, + audit of the original-paper SSAE ckpt (Qwen2.5-0.5B). Eval: 4 ProcessBench subsets (gsm8k/math/olympiadbench/omnimath), macro F1_PB, val-selected + oracle (0.005-grid) thresholds.

**Results (oracle macro F1_PB).** DenseLinear 0.377 · SAE-mixed 0.369 · SAE-contrastive 0.335 · SAE-positive 0.315 · paper SSAE 0.5B 0.295 · in-house SSAE all ≈0.18 · random 0.167.

**Verdict.** ✗ No learned representation beats raw dense at oracle. In-house 1.5B SSAE collapses to ≈random; the paper's 0.5B SSAE ckpt beats them → suspect is SSAE *training*, not architecture. SSAE threshold transfer (val→PB) is broken (val-selected ≈0).

**Next.** Diagnose why in-house SSAE underperforms the 0.5B ckpt; consider contrastive/ranking objectives over PRM800K sibling forks.

---

<!-- TEMPLATE — copy above the previous entry

## S<N> — <name> (<start> → <end>)

**Hypothesis.** <one sentence>

**Tested.** <models / data / methods / eval metric>

**Results.** <key numbers>

**Verdict.** <✓/✗ + one-line reason>

**Next.** <one line>

Also add a row to the index table at the top.
-->
