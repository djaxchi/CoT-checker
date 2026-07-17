# cot_causal_graph_v0: detected vs causally influential errors in CoT, with repair paths

Status: v0 frozen 2026-07-17. Successor to transition_operator_v0 (S6): stops trying to
learn a latent explanation and instead measures, per reasoning step, three separable
quantities on the actual CoT, then asks whether they coincide. The interactive graph is
the analysis interface, not the scientific claim.

Research question: are the errors the probe detects internally the same errors that
causally influence the model's final answer, and can later self-correction (repair) be
causally localized?

Three quantities per step:
  Detection  D_i  does the linear correctness probe flag step i as wrong?
  Influence  I_i  does intervening on step i change downstream reasoning / the answer?
  Repair     R_j  does a later step j causally recover from an earlier error?

Prior art this must beat (REPORT section 20): Thought Anchors [P13] (counterfactual
resampling + attention, importance only, no internal detection signal) and CRV [P3]
(attribution-graph fingerprints). Neither has ground-truth wrong steps NOR a validated
internal detection probe, so neither can build the detection x influence cross-tab.
That cross-tab (plus repair) is the contribution.

Constraints inherited from our own results (binding):
  - S6 Stage-0 oracle: a single patched boundary state recovers the next-token
    distribution (median 0.90-1.0) but ~0 of the answer-belief shift. Boundary
    patching therefore CANNOT be an answer-level intervention in this design; it is
    excluded from v0 edges entirely (available later only as a local diagnostic).
  - S3 Stage-5: additive steering of w is null; the probe direction is a readout, not
    a lever. A change in downstream probe score after an intervention is a NODE
    DIAGNOSTIC, never evidence of a causal edge. Causal claims run through behavioral
    targets only (answer margin, rollout accuracy).
  - S6 Stage-0 gate 1: elicitation suffix frozen to "\nSo the final answer is"
    (gold rank-1 rate 0.957 on 300 golden trajectories); candidate machinery and
    gold_margin reused verbatim from src/analysis/transition_operator.py.
  - Fork audit: incorrect steps are longer and later; every edge-strength comparison
    is reported raw AND against matched controls (below), and residualized on
    swapped-step length delta where applicable.

---

## The whole experiment in one schema

```
════════════════════════════════════════════════════════════════════════════════════
 ARM F  (controlled: ground-truth error position, PRM800K forks joined to their
         complete golden trajectory; same sessions as transition_operator_v0)

   question ─ s_1 ─ ... ─ s_{t-1} ─ s_t ─ s_{t+1} ─ ... ─ s_T ─ [answer margin]
                                    │
             interventions at t:    ├─ swap_wrong   fork's rating -1 sibling (ERROR)
             (downstream text       ├─ swap_pos     alternative +1 sibling, when it
              kept FIXED =          │               differs from s_t (paraphrase-level
              teacher-forced)       │               control)
                                    └─ swap_xprob   length-matched step from another
                                                    problem (off-topic control)

 ARM G  (on-policy: model's own graded trajectories, behavioral ground truth only)

   prompt ─ g_1 ─ g_2 ─ ... ─ g_T ─ boxed answer   (graded vs gt_answer)

             interventions at i:    ├─ delete       drop step i (teacher-forced only:
                                    │               FG-deleting step i = rolling out
                                    │               from prefix i-1, which the base
                                    │               curve already covers)
                                    └─ swap_xtrace  length-matched step from another
                                                    trace (control)
             + free-generation resampling from every prefix (no intervention needed)
════════════════════════════════════════════════════════════════════════════════════
 EDGE FAMILY 1: TEACHER-FORCED (precise, conditional on the fixed continuation)
   one full forward per (trace, intervention); downstream text unchanged
   e_tf(t -> j)      = Δ mean logprob of step j tokens          (all j > t at once)
   e_tf(t -> ans)    = Δ answer margin m_T at the final boundary
   margin profile    = Δ m_j at every boundary j >= t   (propagation curve)
   probe profile     = Δ probe logit at downstream step-final tokens (DIAGNOSTIC)

 EDGE FAMILY 2: FREE-GENERATION (behavioral; step -> answer only, the original
   downstream nodes cease to exist)
   s_i  = P(correct final answer | rollouts from intact prefix s_1..s_i)   K rollouts
   s'_t = P(correct | rollouts from prefix with s_t intervened)
   e_fg(t -> ans) = s'_t - s_t          influence;  s'_t itself = recovery rate
   on incorrect Arm-G traces the s_i curve alone localizes where the trajectory
   becomes doomed (drop points), with no step labels needed
════════════════════════════════════════════════════════════════════════════════════
 NODE FEATURES (every step, both arms, from ONE teacher-forced base pass)
   probe logit (L28 deployed w,b; L20 in-space secondary)   [detection]
   answer margin m_i via frozen suffix + 8 candidates        [belief curve]
   step mean NLL, next-token entropy at boundary             [cheap covariates]
   gold labels: Arm F swapped-step identity; Arm G trajectory correctness
════════════════════════════════════════════════════════════════════════════════════
```

## Frozen constants

| item | value |
|---|---|
| model | Qwen/Qwen2.5-7B (base), bf16, one node 4x H100 |
| step join (teacher-forced paths) | `sep_join_ids`, SEP id 198, identical readout token by construction |
| Arm G prompt + step split | `generate_onpolicy_steps.py` conventions ("\n\n" blank-line steps, \boxed answer) |
| elicitation suffix | `"\nSo the final answer is"` (S6 Stage-0 winner) |
| answer candidates | k=8 via `build_candidates` (gold + wrong_finals + pre_generated + corpus pool), frozen per trace at build time |
| probe | deployed L28 `linear_probe.pt` weights via `steering/directions_L28.npz` (w, b, raw space); L20 secondary |
| detection threshold | val-selected threshold of the deployed probe; sensitivity curve over thresholds reported alongside |
| rollouts | K=8 per prefix (pilot gate may raise to 16), temperature 0.7, top_p 0.95, max_new_tokens 512, graded by `src/eval/math_grade.grade` |
| Arm F target size | >= 500 joined traces, each with >= 2 downstream golden steps after the fork |
| Arm G target size | 300 problems x 4 samples (reuse existing graded trajectories if present on cluster) |
| splits | problem-disjoint, frozen to `splits.json` before any GPU job |
| seeds | build 42; rollouts seed = stable_seed(trace_id, base) |

## Taxonomy (the deliverable cross-tab)

Assigned per error site (Arm F: the swapped step; Arm G: probe-flagged and/or
FG-localized drop steps):

```
detected + influential      genuine propagated error
detected + inert            flagged, never affects the answer
undetected + influential    dangerous hidden failure
undetected + inert          benign noise (completes the 2x2)
repaired (overlay)          later step j causally recovers: margin dips then
                            recovers after j, AND deleting j re-breaks the answer
                            (teacher-forced) or drops the recovery rate (free-gen)
```

detected   := probe logit > frozen threshold at the error step.
influential:= FG: s'_t - s_t < 0 with the K-rollout binomial CI excluding 0 (per
              site), pooled bootstrap for aggregates; TF: |Δ m_T| beyond the 95th
              percentile of the matched-control null (swap_pos where available,
              else swap_xprob).
inert      := neither criterion fires.

## Gates (kill criteria, in order)

G1 margin validity (Arm F, Stage 1). Paired over traces: Δm_T(swap_wrong) must be
   more negative than Δm_T(swap_pos) on the ~500 traces carrying a distinct +1
   sibling, and than Δm_T(swap_xprob) on all traces; Wilcoxon p < 0.01. Analogue
   of the S6 directional gate. FAIL -> TF answer edges are dropped, v0 becomes
   FG-only.
G2 FG power (Stage 2 pilot). 50 traces x K=8: bootstrap SE of s_i must resolve an
   accuracy delta of 0.15. FAIL -> K=16 on a halved trace set (same GPU budget).
G3 null calibration (Stage 1 + 2). |edge| for swap_wrong vs matched controls,
   site-level AUC. AUC <= 0.55 -> the influence measure is generic perturbation
   magnitude, not error content: kill the corresponding edge family.

## Stages

Stage 0 (local, CPU): scan raw PRM800K sessions and keep those containing BOTH a
  complete golden trajectory (finish_reason "solution", gt answer) AND fork pairs
  on the golden path; session-level pairing guarantees a real teacher-forced
  continuation at every site. (A file-level join of the separately sampled S6
  forks.jsonl x golden.jsonl was tried first: 28/911 question overlap, 1 trace.)
  Output `traces_forks.jsonl` (golden steps, fork position, wrong / alt-pos /
  xprob swap texts, frozen candidates), Arm G problem list, `splits.json`.
  Script `cg_build_traces.py`. Built 2026-07-17: 800 traces (232 val), 499 with
  a distinct +1 sibling, median 18 steps with 8 downstream of the fork.

Stage 1 (GPU, ~2 h): teacher-forced passes. Per Arm-F trace: 1 base + 3 intervened
  forwards, margin profile at boundaries >= t-1, probe/NLL/entropy node features.
  Per Arm-G trace: 1 base + 2 x T intervened forwards (delete, swap_xtrace at every
  step; margins final boundary only for interventions, full profile for base).
  Script `cg_stage1_tf.py`, sharded over 4 GPUs in-node. Evaluates G1, G3(TF).

Stage 2 (GPU, ~4 h): free-generation rollouts. Arm F: base + corrupted (+ both
  controls) at the fork point. Arm G: base curve from every prefix; swap_xtrace
  at 4 sites per trace (the 2 highest-probe steps from stage 1 + 2 random;
  `--interv_policy all` available). Pilot block (50 arm-F traces) runs first and
  prints G2 before the full run. Script `cg_stage2_fg.py`.

Stage 3 (local/login, CPU): assemble `graph.json` per trace (nodes, both edge
  families, taxonomy calls) + `crosstab.json` (the 2x2 by arm/threshold, repair
  inventory) + summary stats with bootstrap CIs. Script `cg_stage3_assemble.py`.

Stage 4 (local): `cg_explorer.py` -> self-contained `explorer.html` over
  `graph.json` files. Trace picker with taxonomy filters; vertical step chain with
  readable text; node fill = probe score; taxonomy badges; belief-curve sparkline;
  TF edges as right-side arcs, FG edges left-side; width = |Δ|, color = sign;
  click -> plain-language panel ("replacing this step with the wrong sibling drops
  the model's belief in the correct answer by ..."). A "how to read this" intro box.
  No external assets; works offline; theme follows the S4/S5 explorer pattern.

## Analyses (Stage 3 outputs, ranked)

1. Detection x influence cross-tab on Arm F (ground-truth error position): the
   fraction of labeled wrong steps that are causally inert, and whether the probe
   score predicts influence (Spearman probe-logit vs e_fg / Δm_T; report per-site
   scatter). This is the headline.
2. Hidden failures: undetected + influential rate, with examples.
3. Repair: (a) Arm F traces where swap_wrong rollouts still succeed: localize the
   recovery step via the s-curve of the corrupted branch, verify by TF deletion of
   that step; (b) Arm G incorrect-then-correct margin dips. Report a verified-repair
   inventory with quotes.
4. Propagation shape: Δm_j profiles after swap_wrong (immediate collapse vs decay
   vs late collapse), clustered only descriptively.
5. Controls: G3 AUCs, length-delta residualization of edge strengths.

## Out of scope for v0

Boundary-state patching as an intervention (pre-falsified for answer-level effects,
S6 Stage 0); attention-head or SAE-feature ablation edges (no sparse localization
exists, S1-S3); trained edge predictors; cross-model transfer; step j -> j' free-
generation edges (undefined once downstream regenerates).

## Artifacts layout

```
runs/causal_graph/
  traces_forks.jsonl  traces_onpolicy.jsonl  splits.json  build_manifest.json
  stage1/  tf_edges.parquet  node_features.parquet  gates_stage1.json
  stage2/  fg_rollouts.parquet  fg_curves.parquet  gates_stage2.json
  stage3/  graphs/<trace_id>.json  crosstab.json  summary.json
  explorer.html
```

Code: `src/analysis/causal_graph.py` (core, unit-tested), `scripts/causal_graph/`
(stages), `slurm/causal_graph_stage{1,2}_tamia.sh`, `tests/analysis/test_causal_graph.py`.
