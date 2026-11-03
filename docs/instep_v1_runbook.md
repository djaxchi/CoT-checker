# InStep V1 runbook (2026-11-03)

Everything the paper's V1 needs: what runs, what each run produces, and every table and
figure built from it. Context and decisions: `docs/instep_paper_plan.md` (V1 run list).
Rule: on ReProbe's test sets, the ReProbe paper's printed PR-AUC is cited as given (Tier B);
the Tier A comparators get our own step scores on test, validation and ProcessBench so that
AUROC, F1 at the validation threshold, oracle F1, always-positive F1 and paired bootstraps
exist.

## Backbones and data (TamIA)

| backbone | model | data root `R` | layout | val label |
|---|---|---|---|---|
| qwen3_8b | Qwen/Qwen3-8B | `/project/aip-azouaq/$USER/cot_mech/reprobe_cmp_v1` | reprobe | ds |
| phi4 | microsoft/phi-4 | `/project/aip-azouaq/$USER/cot_mech/reprobe_cmp_v2/phi4` | reprobe_phi4 | qwen3 |
| qwen3_32b | Qwen/Qwen3-32B | `/project/aip-azouaq/$USER/cot_mech/reprobe_cmp_v2/qwen3_32b` | reprobe | oss |

Inputs under `$R/data`: `test/<set>.{traces,items}.jsonl` (8 sets: math gsm8k proofnet trips
meetings calendar strategyqa scienceqa), `rp/val.traces.jsonl` + `rt/all_val.items.jsonl`
(validation), `pb/<subset>.{traces,items}.jsonl` (ProcessBench, 4 subsets), `rt/<label>_train.items.jsonl`.
Items carry `uid = <trace_id>::<k>`, `problem`, `prefix` (earlier steps), `candidate_step`, `label`.
Weights: Qwen3-8B and both PRMs in `/project/aip-azouaq/$USER/hf_cache`; Phi-4 and Qwen3-32B in
`$SCRATCH/hf_cache`.

Score format (what `scripts/reprobe_cmp/analyze.py` reads as an `rp:` arm): one file per set,
`<dir>/<set>.jsonl`, one line per step `{"uid": "<trace_id>::<k>", "p": <higher = more likely incorrect>}`,
set names `math ... scienceqa`, `val`, `pb_gsm8k pb_math pb_olympiadbench pb_omnimath`.

Outputs: `OUT=/project/aip-azouaq/$USER/cot_mech/instep_v1`; large states in
`$SCRATCH/cot_mech/instep_v1/states/<backbone>/`.

## Runs

| id | what | backbones | code | output |
|---|---|---|---|---|
| E1 | the six paper tokens (DS seeds 0-2, PRM800K seeds 0-2, lr 3e-3, 5% warm-up) on 8 sets, PB, val, best-of-N pools; DS seed 0 also keeps passive + read states every 4th block on test, val, PB, 40k training steps | qwen3_8b | `slurm/instep_v1_tamia.sh STAGE=e1` | `$OUT/rt/instep_{ds,prm}_seed{0,1,2}`, states |
| L1 | ladder, 3 seeds, DS data: `emb`, `off`, `chain`, `nooff`; `emb` on PRM800K | qwen3_8b | `STAGE=l1` | `$OUT/rt/<variant>_<src>_seed<s>` |
| P1 | strong PRM (Qwen2.5-Math-PRM-7B) and weak PRM (Qwen2.5-Math-7B-PRM800K) on test, val, PB; gate: reproduce the scores stored in ReProbe's files | all 3 (PB: qwen3_8b) | `scripts/instep_v1/score_prm.py` | `$OUT/scores/<bb>/{prm7b,prm800k}/` |
| U1 | token confidence: MaxProb, MaxEntropy, Perplexity (LM-Polygraph claim definitions); gate: stored values | all 3 (PB: qwen3_8b) | `scripts/instep_v1/score_confidence.py` | `$OUT/scores/<bb>/{maxprob,maxent,ppl}/` |
| T1 | P(True): zero-shot P(No) under a verification prompt, per step | qwen3_8b | same script, `--ptrue` | `$OUT/scores/qwen3_8b/ptrue/` |
| B1 | passive linear: logistic on the step's last text state, layer picked on val; Phi-4 and Qwen3-32B states encoded with their existing InStep tokens | all 3 | `scripts/instep_v1/passive_fit.py`, `slurm/instep_v1_states_tamia.sh` | `$OUT/scores/<bb>/passive/` |
| S1 | per-layer separability (val AP, AUROC) of passive vs read states | qwen3_8b (+ others) | `passive_fit.py --per_layer` | `$OUT/analysis/per_layer_<bb>.json` |
| C1 | every arm through `analyze.py` per backbone (+ pairs), PB threshold-free metrics, Tier B paper rows, tables and figures | all | `scripts/instep_v1/paper_v1.py` | `$OUT/final/` |

Existing scores reused (not rerun): ReProbe released and retrained (`rp`, `rpc`) on all three
backbones, InStep tokens of Phi-4 and Qwen3-32B and the self-label InStep seeds
(`reprobe_cmp_v{1,2}/.../runs`), best-of-N and beam search grading, cost.

## Tables and figures

1. **Table 1** ReProbe benchmark, Qwen3-8B: 8 sets + ID/OOD/overall PR-AUC. Tier B rows from the
   paper; Tier A rows ours (paper number shown where it exists). Columns AUROC overall, F1 at the
   val threshold vs always-positive (per set, in the appendix). Paired deltas InStep - each
   Tier A comparator with 95% trace-bootstrap CIs.
2. **Table 2** ProcessBench (Qwen3-8B backbone): per subset first-error AUROC and PR-AUC
   (positives: labelled first-error steps; negatives: earlier steps of erroneous traces and all
   steps of correct traces; later steps dropped); calib-20 in the appendix.
3. **Table 3** cost: params, ms per trajectory (existing numbers).
4. **Fig 2** Pareto: overall PR-AUC vs trainable params (log) and vs ms per trajectory.
5. **Fig 3** backbones: overall PR-AUC per backbone for passive, ReProbe, InStep, best PRM, MaxProb.
6. **Fig 4** ladder: P(True) -> passive -> emb -> +offsets -> +chain -> InStep, and InStep without
   offsets; ReProbe overall PR-AUC (DS tokens) and ProcessBench AUROC (PRM800K tokens), 3 seeds, CIs.
7. **Fig 5** per-layer val AP of passive and read states.
8. **Fig 6** use: abstention risk-coverage on the test sets; best-of-N and beam search from existing results.
