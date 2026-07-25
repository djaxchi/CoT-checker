# Unified Representation Harness v1

*Goal: put every representation on one common ground so results are comparable
and the only thing that varies between experiments is the representation.*

Decision locked with Djalil (2026-07-25): backbone **Qwen2.5-7B base**,
Math-Shepherd **dropped**, in-domain data **PRM800K**, out-of-domain test
**ProcessBench**. Train on PRM800K, evaluate in-domain on a held-out PRM800K
test split, then evaluate OOD on ProcessBench.

---

## 1. The runner already exists

`scripts/train_easy_probe_method.py` is the harness. It already does the whole
spine in one pass:

- loads PRM800K from a cache directory using the contract `{stem}_h.npy` (float32
  `(N, d)` features) + `{stem}_y.npy` (int64 labels in `{0, 1}`);
- dispatches on `--method` (the representation registry, the single variable);
- trains one `LinearProbe` on `--probe_train_stem`, selects a threshold on
  `--val_stem` by balanced accuracy;
- scores every ProcessBench target passed via `--pb_specs name:h:meta` and emits
  first-error `F1_PB` at three thresholds: fixed 0.5, val-selected (deployable),
  oracle (ceiling), plus `Acc_error`, `Acc_correct`, `Exact_match_all`.

The metric protocol (val-selected + oracle F1, plus AUROC as secondary) and the
`afterok`-chained build -> encode -> train pipeline
(`slurm/run_dense_full_pipeline_tamia.sh`) are already in place.

## 2. Frozen spine (lock once, never vary)

| Axis | Locked value |
|---|---|
| Backbone | Qwen2.5-7B base, last-token readout, L20 primary (L28 secondary) |
| In-domain data | PRM800K, problem-id-disjoint train / val / test |
| OOD test | ProcessBench (threshold never re-tuned on it) |
| Detector | `LinearProbe` primary; MLP as capacity control |
| Cache contract | `{stem}_h.npy` + `{stem}_y.npy`; PB `pb_step_h.npy` + `pb_step_meta.jsonl` |
| Metrics | in-domain step AUROC + F1 (fixed/val/oracle); OOD F1_PB (fixed/val/oracle) |
| Single variable | `--method` (the representation registry) |

## 3. The two gaps between "exists" and "unified for our goal"

**Gap A -- no in-domain PRM800K test.** `build_prm800k_full.py` reserves only
train + val (val is used for threshold selection). The runner never reports a
clean in-domain test number; it jumps from val straight to ProcessBench. To make
"train -> in-domain -> OOD" a single command we need:

1. `build_prm800k_full.py`: reserve a third problem-disjoint split
   `prm800k_test_Nk.jsonl` alongside train/val.
2. `train_easy_probe_method.py`: add `--test_stem`; when set, run the trained
   probe on that cached split and emit in-domain step-level AUROC + F1 at
   fixed/val/oracle, written next to the PB metrics.

**Gap B -- caches are 1.5B, we want 7B.** The wired pipeline uses
`qwen2_5_1_5b`. We need the dense-full cache contract emitted at 7B
(`hidden_dim = 3584`): build the JSONL splits (tokenizer-only, backbone-agnostic),
then encode PRM800K train/val/test and ProcessBench at 7B, then run the harness.
The §15/§16 7B encodes are multilayer/multitoken under different stems and do not
satisfy the `{stem}_h.npy/_y.npy` dense-full contract, so a fresh 7B dense-full
encode is required.

## 4. Open design decision (Djalil's call)

`build_prm800k_full.py` currently **drops rating-0 and balances pos/neg 50/50**.
So "all PRM800K" through this builder means the balanced +/-1 extremes, which
REPORT.md repeatedly flags as an optimistic ceiling (~0.74). Two options for the
in-domain ground:

- **Balanced +/-1 extremes** (current builder): cleanest signal, optimistic, not
  a deployment prevalence.
- **Natural prevalence** (rating-0 folded into correct, ~25% incorrect): the
  honest headline; matches §15.7's natural PRM800K test where F1 0.58 clears a
  0.40 trivial bar. Requires a builder mode that keeps rating-0 and does not
  rebalance.

Recommendation: build BOTH as separate stems (`..._balanced`, `..._natural`),
train on balanced, report the deployable F1 on natural. This costs one extra
encode and removes the ambiguity permanently.

## 5. Representation registry: migration map

Each REPORT.md representation becomes one `--method` (or one producer of the
`{stem}_h.npy` cache). The probe/eval downstream stay byte-identical.

| REPORT.md representation | Harness entry | Status at 7B |
|---|---|---|
| Dense last-token (§13) | `dense_linear` | re-encode at 7B |
| Multi-layer concat (§16) | `dense_linear` on assembled cache | reuse `assemble_multilayer_concat.py`, re-encode |
| SSAE / SAE latents (§7-12) | `sae_positive/mixed/contrastive` | in registry; re-encode base h at 7B |
| Transition h_i - h_{i-1} (§18) | new producer -> `dense_linear` | thin wrapper, new |
| Fork-shaped objectives (Sprint 2) | `*_rank`, `*_triplet`, `dense_*` | in registry |
| Trajectory / sequence encoder | not present | genuine new build |
| Mechanistic (contrib, attention-routing) | producer -> `dense_linear` | reuse producers |

## 6. Build order

1. Extend `build_prm800k_full.py` with a disjoint `--test` split (+ optional
   `--natural` mode). Tests alongside.
2. Add `--test_stem` in-domain eval to `train_easy_probe_method.py`. Tests
   alongside.
3. Materialize PRM800K train/val/test JSONL (frozen before any encode).
4. TamIA: encode PRM800K {train, val, test} + ProcessBench at 7B into the
   dense-full cache contract.
5. Run the harness with `--method dense_linear` -> first unified leaderboard row
   (in-domain PRM800K test + OOD ProcessBench, 7B).
6. Add each further representation as a new `--method` / producer; every run is
   one more comparable row.
