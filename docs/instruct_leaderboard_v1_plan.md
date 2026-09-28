**Instruct leaderboard and downstream rank transfer, v1**

*2026-09-28. Requested replication of the presentation leaderboard on Qwen3-8B Instruct, followed by a test of whether error-identification performance predicts test-time selection gains. This replaces the compact-grid priority in REPORT.md §24 with the full matched roster.*

**Scientific question**

Across a fixed family of representations and learners trained on the same data, does ProcessBench first-error identification predict final-answer selection performance on the same Instruct policy? A positive correlation would support the benchmark as a selection instrument within this family. A weak correlation or systematic rank reversal would identify a limit of that instrument. Neither outcome establishes causality or generalization across all verifier families.

**Frozen training roster and protocol**

The presentation's original grid contains 19 cells: five fixed-vector representations crossed with linear, mlp:h1024 and mlp:h1024x2, plus step_tokens with attention pooling and transformers of widths 128, 256 and 512. The later lengthfree_geom addition contributes three labeled extension cells. `experiments/instruct_leaderboard_v1/all.cells` freezes all 22. Report the 19-cell core and the 22-cell extension separately for rank transfer; neither subset depends on Instruct results.

Every cell uses Qwen3-8B Instruct `hidden_states[35]`, verifier-template states, all 513,810 frozen PRM800K training steps, val_5k for model selection and test_2k for source evaluation. Preserve raw inputs (`rescale=none`) to match the Base presentation grid and the completed Instruct d512 arm. AdamW/BCE, batch 256, at most 30 epochs, patience 3, dropout 0.1, sequence cap 512, seeds 42/43/44. Search learning rates {1e-3, 3e-4, 1e-4} and weight decays {0, 0.01} on the same 100,000-row search subset at seed 42; reuse that cell's selected hyperparameters for seeds 43/44, then fit each seed on the full training split. Preserve the source trainer's selection and early-stopping behavior.

Reuse only the completed 8,665,089-parameter d512 cell with the full architecture spelling and all three seeds, after checking protocol and input fingerprints. Exclude the earlier truncated `transformer:d512` experiment. Reuse the existing Instruct representation stores; do not re-encode them. New outputs and the source snapshot live separately from the other Instruct agent's working tree and experiments.

**Leaderboard definition**

Headline: ProcessBench first-error F1_PB at calib-20, equal mean across GSM8K, MATH, OlympiadBench and OmniMath, then mean and standard deviation across three training seeds. For each subset, use the existing 20 stratified calibration splits, 20 calibration traces per split, and 99 score-quantile threshold candidates from calibration traces only. Evaluate on the remaining traces. Use `merge_rep_grid_leaderboard.py`, not a new implementation of calibration.

Report per-subset F1, source-val-selected F1, oracle F1 and AUROC separately. Calib-20 is target-adapted; it must not be labeled source-val-selected or compared with external systems as an equal adaptation-budget result. Simple always-error-free and always-first-step-error predictions have F1_PB zero because one component accuracy is zero; retain component accuracies when diagnosing thresholds. All 66 seed runs must be complete before publishing a ranked full table. A partial table must list coverage and cannot silently omit failed cells.

**Downstream comparison, frozen before the roster is measured**

Score every trained checkpoint on identical ten-candidate Instruct pools. Preserve candidates, step segmentation, scorer context, layer, score orientation, answer canonicalization and draw subsets across cells. The current T=1.0 pool is the primary pool because it already underlies the Instruct comparison. The concurrently generated T=0.7 pool is a separately labeled policy sensitivity arm. Do not choose the pool by which yields the stronger correlation.

Primary rank-transfer endpoint: Spearman correlation across the 19 seed-averaged core cells between four-subset calib-20 F1_PB and MATH-500 best-of-4 accuracy gain over majority, with maximum step suspicion (`worst`) as the fixed trajectory aggregation. Report all cell values and the scatter, including weak cells. N=4 permits plurality overrides and has more selection headroom than N=10; N=2 cannot distinguish reranking from tie-breaking. At a fixed dataset and N, majority is constant across cells, so correlation with absolute accuracy equals correlation with its lift.

Secondary analyses: the 22-cell roster, MATH-only ProcessBench F1, source-val-selected F1, ProcessBench step AUROC, GSM8K transfer, within-question outcome AUROC, N in {1,2,3,4,5,6,8,10}, and separate tie-break/rerank/weighted-vote curves. Keep every rule fixed across the roster. Last-step and mean-step trajectory scores are sensitivity analyses, not per-cell opportunities to choose a best aggregator. N=1 has no offline selection effect and therefore no defined rank correlation when all outcomes tie.

Uncertainty should resample the same benchmark questions jointly across all cells, preserving within-question candidates and repeated seeds. Propagate ProcessBench question and calibration uncertainty separately from TTS question uncertainty; three training seeds provide a limited estimate of training variability. Report average-rank Spearman and tie-aware Kendall tau-b. Bootstrap over cells only as roster-sensitivity analysis, since these architecture variants are related and are not independent draws from a population of verifiers. Report the seed-to-seed stability of both rankings without treating it as a strict mathematical ceiling.

Predefined contrasts: last_token versus step_mean with the same learner; step_mean versus step_stats with the same learner, explicitly changing dimension; fixed mean versus attention pooling; and attention pooling versus each transformer capacity. A high overall correlation can coexist with a failure of these specific contrasts. Use paired question-level differences to inspect reversals.

**Evaluation gates and implementation work before TTS claims**

The existing training stores allow the leaderboard to proceed while the grading audit continues. Before downstream inference, resolve conflicting answer groups and use gold-independent canonicalization shared by voting and grading. Keep unparseable candidates as paid failures and a fixed question denominator. Archive train/model-selection/evaluation question overlap checks. If exposure exists, report both the full benchmark and a predefined overlap-free subset.

Template-state scoring tests benchmark-to-selection transfer with fixed checkpoint inputs. Generation-state scoring is a separate context-transfer and deployment-cost test. Do not mix contexts across cells. The existing generic score_cells_on_split path needs an explicit check for lengthfree_geom: it must apply the training-fitted length transform before inference, as the trainer does. Until that path reproduces held-out scores, extension cells cannot enter a downstream ranking. Apply the same reproduction gate to all representations and seeds.

Freeze a complete paired score matrix before correlating the leaderboard with TTS. Missing-score runs must fail validation, not drop difficult candidates. Report raw generation tokens, actual verifier work and measured latency alongside accuracy. Full PRM and confidence baselines contextualize utility but do not join the homogeneous small-head correlation roster.

**Execution**

Two full H100-node jobs train disjoint vector and sequence rosters. The sequence job uses two simultaneous processes so two approximately 163 GB preloads fit within the 500 GB node. The vector job uses node-local caches to avoid consuming the limited shared scratch space. A dependent merge validates all 66 runs, protocol fields, fingerprints and checkpoint files before producing the table. The completed d512 runs enter through validated copies on persistent storage, preserving their original files.

Outputs: `/project/aip-azouaq/dchikhi/cot_mech/instruct_leaderboard_v1/`. Existing Instruct generation-state, T=0.7 generation and PRM jobs remain independent. Training submission IDs and the source revision are recorded in the run's submission manifest. A completed leaderboard is a prerequisite for the full-roster TTS scoring stage; no correlation is claimed from the single currently completed architecture.
