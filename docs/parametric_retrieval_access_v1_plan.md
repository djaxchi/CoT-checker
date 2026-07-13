# parametric_retrieval_access_v1: latent factual knowledge vs retrieval access

Follow-up to `parametric_retrieval_geometry_v0`. The v0 result: a pre-generation
probe predicts retrieved-vs-non-retrieved (AUROC ~0.79) but that predicts a
behavioral outcome, not answer identity; SAE feature steering was a gauge; the
CoT-state transplant donor had already retrieved the answer. v1 stops treating
the four behavioral classes as the target and asks, in order:

1. Is the correct answer identity represented pre-generation? (Experiment A)
2. Where does it emerge across layers x positions? (Experiment B)
3. What differs between same-fact successful and failed retrieval? (B)
4. Is that difference causal? (Experiment C: same-fact direct patching)
5. Does it generalize across unseen facts without transferring answer
   content? (Experiment D: access subspace)

Experiment E (dynamic emergence during CoT) runs only if the prompt-state
answer signal is weak for reasoning-unlocked facts.

## Core experimental unit

Same-fact mixed-outcome paraphrase pairs: for one fact and direction, two
semantically equivalent direct prompts where greedy decoding succeeds on one
and fails on the other. This controls answer identity, popularity, topic,
relation, and entity types.

## Design decisions and deviations from the spec

- **Paraphrases**: 2 WikiProfile seed questions per direction (template +
  natural) x 6 instruction wrappers = 12 paraphrases per fact x direction.
  Every wrapper ends with `Question: ...\nAnswer:` so the answer-prefix token
  is standardized. `paraphrase_id = <seed>::<wrapper>`; wrapper and seed are
  recorded for confound controls. No LLM-generated paraphrases in v1 (keeps
  factual equivalence exact by construction).
- **Directions**: direct (gold = object) and reverse (gold = subject) are
  separate retrieval tasks, never pooled.
- **Decoding**: greedy only (T=0) for the main experiment, per spec 7.2.
  Sampling metadata is dropped in v1 (v0 already established instability
  rates). p_D per fact x direction = fraction of the 12 paraphrases correct
  under greedy. p_C = greedy CoT correctness on one canonical prompt per
  fact x direction (cost control; CoT per paraphrase is not needed for the
  paired design).
- **No relation column in WikiProfile**: relation-phrase positions are
  replaced by queried-entity span positions (entity = fact subject for
  direct questions, fact object for reverse questions). Relation identity
  for stratification/conditioning is proxied by (category, object_type)
  for direct and (category, subject_type) for reverse.
- **Answer leakage**: instances whose normalized gold answer is contained in
  the normalized question are dropped at build time and counted in the
  manifest.
- **Candidate sets (Experiment A)**: gold + WikiProfile MC distractors (3,
  type-matched by construction) + answers of other facts with the same
  object/subject type, preferring same category then same popularity bin,
  up to K=32. Negatives deduplicated under normalize_answer.
- **Splits**: fact-disjoint 60/20/20 stratified by gbc_bin x category, both
  directions of a fact stay in the same split. Frozen in metadata.parquet
  before any GPU job (materialize-before-extraction convention).
- **Layers**: ALL hidden_states indices 0..28 are extracted (spec: do not
  restrict to 20/24 before localization).
- **Positions** (pre-generation unless noted): entity_first, entity_last,
  entity_mean, question_last, answer_prefix (last user-message token,
  the ':' of 'Answer:'), final_prompt_token (last rendered chat token),
  first_generated_token (auxiliary, post-decision).

## Pipeline (scripts/parametric_retrieval/, prefix prga_)

| stage | script | where | outputs |
|---|---|---|---|
| 0 | `prga_build_prompts.py` | local | metadata.parquet (frozen instances + splits), build_manifest.json |
| 1 | `prga_generate.py` | TamIA 01 | generations.jsonl (greedy text + ids, direct + canonical CoT) |
| 2 | `prga_pairs.py` | local | grading.jsonl, group_outcomes.parquet, pairs.parquet, candidates.json, extraction_set.json |
| 3 | `prga_extract.py` | TamIA 02 | hidden_states/layer_{00..28}.safetensors + hs_meta.parquet |
| 4 | `prga_logitlens.py` | TamIA 02 (post) | logitlens/scores.parquet + heatmap aggregates |

Experiments B (Δh geometry), C (same-fact patching), and D (access subspace)
consume stage 3 outputs and are implemented after the stage 4 heatmaps
identify candidate layers/positions on the validation split.

Scoping note on stage 4: the first-token logit lens is the FIRST localization
analysis, not proof of full answer identity. Its outputs quantify first-token
candidate collisions and locate useful layers/positions; the sequence-level
candidate decoder (Experiment A methods B/C) is designed afterwards from
those heatmaps.

Run dir: `runs/parametric_retrieval_access_v1`. All facts (2,150) enter
stage 0; mixed-outcome selection happens after grading, requiring >=2
successful and >=2 failed paraphrases per fact x direction.

## Results (2026-07-11, jobs 366552/366553/366559, all COMPLETED)

- Dataset: direct greedy 0.247, CoT 0.308; 1,163/4,272 mixed groups (27.2%),
  9,304 pairs, 136,177 extraction rows x 29 layers.
- Exp A (lens): gold identity decodable pre-generation, including failures
  (test, mixed, hs27/final_prompt_token: failed hits@1 0.375, hits@5 0.79 vs
  chance 0.031; success 0.627). Reasoning-unlocked groups: hits@1 0.469 ->
  Experiment E gated OFF. Decoder method B with frozen mean input-embedding
  targets underperforms the lens badly (hits@1 ~0.05); needs better answer
  representations before the sequence-level claim is closed.
- Exp B: NULL. Best success-vs-fail probe AUC 0.584 does not beat the
  confound-only probe (0.589); residualized AUC 0.504. Which paraphrase
  succeeds is not linearly readable from these states.
- Exp C: POSITIVE and fact-specific. Same-fact patch (hs26, a=1.0, final
  prompt token) on test: d_margin +1.34 [0.48, 2.21], rescues 44.9% of failed
  paraphrases to exact match, donor copying 0.0. Mismatched donors DESTROY
  the margin (-3.9/-4.1); norm-matched noise -0.59; noop +0.02; reverse patch
  breaks 48.5% of successes (necessity).
- Exp D: NULL vs random (outcome C3 reproduced on the fair paired design).
  Best learned direction (LDA, hs28, a=4) d_logP(gold) +5.42 vs random +5.27,
  exact 8.7% vs 9.0%, and d_margin is NEGATIVE for all directions: a
  confidence gauge, not a factual access lever.
- Verdict: spec outcome C2. The answer representation exists pre-generation
  and successful retrieval is causally transferable ONLY with same-fact
  content; no fact-independent access subspace generalizes.

## Statistical protocol

Paired, fact-level: bootstrap CIs over facts, paired permutation tests,
McNemar for binary retrieval outcomes. Layer/position selection on the
validation split only; the full heatmap is always reported. Test facts are
touched once, after freezing layers, positions, and intervention strengths.
