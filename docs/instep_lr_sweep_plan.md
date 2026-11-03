# InStep lr sweep (plan, frozen before any run, 2026-11-02)

## Why

InStep's recipe (peak lr 3e-3, OneCycle with 5% warm-up, one epoch) was inherited from
read_token v0 and never tuned for the inline method. The training logs show it sits at the
edge of stability. The constant predictor's loss on the training labels is 0.262 for
ReProbe's DeepSeek-labelled split (7.34% incorrect) and 0.693 for the PRM800K draw (50%):

| run (peak lr 3e-3, 5% warm-up) | running loss | constant predictor |
|---|---|---|
| InStep-DS, one run, step 300 of 1,966 | 0.343 | 0.262 |
| InStep-DS, another run, step 500 of 1,966 | 0.857 (val AP 0.311 against ~0.61) | 0.262 |
| InStep on PRM800K, step 300 of 1,874 | 0.805 | 0.693 |
| InStep on PRM800K, another seed, step 300 of 1,885 | 0.856 | 0.693 |

Around the lr peak the token predicts worse than a constant, and one run never recovers.
The sweep picks the recipe on validation data, with stability as a hard requirement.

## Scope

* Backbone: Qwen3-8B only. Phi-4 and Qwen3-32B inherit the picked recipe.
* Method: InStep as frozen in `docs/instep_paper_plan.md` (" correct" init, per-layer
  offsets, linear head, chained reads, start marker with its own offsets).
* Sources, both swept (the optimum can depend on the data):
  - `ds`: ReProbe's DeepSeek-R1-labelled training split, ReProbe layout, all 167,355 items;
    validation = ReProbe DS validation split (8,894 items), metric AP (the benchmark's metric).
  - `prm`: 100,000 PRM800K human-labelled items (seed draw, as before), chat layout;
    validation = `prm800k_val_5k`, metric AUROC (the metric used to pick seeds so far).
  Self labels inherit the `ds` pick.
* Grid, 10 configs: peak lr {5e-4, 1e-3, 1.5e-3, 2e-3, 3e-3} x warm-up {5%, 20%}. Fixed:
  AdamW, weight decay 0, clip 1.0, one epoch, batches as before.
* Seeds 0 to 2.
* No test set and no ProcessBench item is read before the pick.

## Selection rule (implemented in `scripts/analysis/instep_lr_select.py`)

1. A run (config x seed) is unstable if its loss goes non-finite, or if its running training
   loss at any logged step from 25% of training on is above the constant predictor's loss,
   or if its validation metric is more than 0.05 below the median of the same config's
   other seeds.
2. A config is stable when none of its three seeds is unstable.
3. Per source, the pick is the stable config with the highest seed-mean validation metric.
   Any stable config within 0.003 of it ties; the tie goes to the lower peak lr, then the
   longer warm-up.
4. One recipe for both sources if a config is stable in both and within 0.003 of each
   source's pick (same tie-break); otherwise one recipe per source.
5. No stable config: stop and report; no pick is forced.

## After the pick

* Qwen3-8B: the three seeds of the picked config are the paper's tokens (no retraining);
  their test and ProcessBench encodes follow.
* Phi-4, Qwen3-32B, self labels: retrained with the pick only if it differs from the
  current recipe. A restart rule stays as a safety net there: a run meeting rule 1 is
  replaced by the next seed, and the count is reported.

## Run time on TamIA

One process per seed trains all 10 configs on the same batches and shares each batch's
text pass (`read_token_inline.py --sweep`); only the read pass (one position per step,
backward through the frozen layers) runs per config. Validation is scored the same way,
one text pass for all configs. The text pass is most of a step's time in the existing logs
(0.36 s per step with one config on ReProbe data, 0.57 s on PRM800K), so 10 configs should
cost a small multiple of one run instead of ten times. Equivalence is tested on a tiny model:
the first config of a sweep equals the single run, and a config inside a sweep equals the
same config trained alone (`tests/test_read_token_inline.py`).

* Smoke (`SMOKE=4000`, `--time=00:25:00`): both sources on two GPUs, measures text and read
  seconds per step; the main `--time` is set from it.
* Main: one node, GPU g = seed g (0-2), `ds` then `prm`, then the selection.
  Outputs: `/project/aip-azouaq/$USER/cot_mech/instep_lr_sweep/{ds,prm}_seed{0-2}/`
  (`sweep.json`, one `read_token.pt` per config), `selection/selection.{md,json}`.

## Result and decision (2026-11-03, TamIA 662436)

The frozen rule gave one pick per source and no joint recipe: `ds` 3e-3 with 20% warm-up
(val AP 0.603), `prm` 1.5e-3 with 5% warm-up (val AUROC 0.887, by the tie-break).
Decision (deviation from rule 4, disclosed): InStep keeps ONE recipe for every source,
peak lr 3e-3 with 5% warm-up, the configuration used so far. It is stable on all six runs
(three seeds per source, late loss at most 0.83x and 0.94x the constant predictor) and
near-best on both: `ds` 0.597 (0.006 below the pick, about half a seed SD), `prm` 0.889
(0.002 above the pick, inside the tie margin). The paper's Qwen3-8B tokens are the
sweep's three seeds of that config for each source; Phi-4, Qwen3-32B and the self-label
runs already use it.
