# parametric_retrieval_sae_decomp: offline SAE-feature decomposition of the flip

The cheap go/no-go check (user's option "b") before committing to SAE feature
steering. prgs found the raw flip-neurons are polysemantic (steering them
off-context does nothing). Before building the full SAE-steering arm, ask
offline whether the flip even concentrates on a few interpretable SAE features.

No model forward, no generation: reuse the public BatchTopK SAE
`andyrdt/saes-qwen2.5-7b-instruct` at `resid_post_layer_27` (= hidden_states
index 28, exactly where the layer-27 MLP output lands), encode the test pairs'
donor (success) and recipient (fail) residual at the final prompt token, and
decompose `df = SAE.encode(h_donor) - SAE.encode(h_recip)`.

Measures:
- **FVU gate** — does the SAE reconstruct our final-prompt-token states
  (FVU < 0.5)? A high FVU means the SAE is off-distribution here and the whole
  approach is moot.
- **Concentration** — fraction of `|df|` L1 mass in the top-k features, and how
  many features capture 90% of `|df|`, vs a random-other-donor null.
- **Recurrence** — which "donor-added" features (df > 0) recur across facts
  (shared interpretable features vs fact-specific).

Decision: diffuse / non-recurring / high-FVU -> the SAE basis will not help,
stop. Concentrated + recurring + low FVU -> justifies the full feature-steering
build (the real Golden Gate test on monosemantic features).

## Pipeline (single short TamIA job, prefix prgd_)

| stage | script | outputs |
|-------|--------|---------|
| download | `submit.sh` (login node) | L28 SAE resid_post_layer_27/trainer_1 |
| run | `prgd_sae_decomp.py` | expH/decomp.parquet, feature_recurrence.csv, summary.json |
| figure | `prgd_viz.py` (local) | results/parametric_retrieval_sae_decomp/prgd_v1_results.png |

Reuses `BatchTopKSAE` from `scripts/public_sae/encode_public_sae.py` and HSStore
(hs_idx 28). trainer_1 => k=64, dict_size 131072.

## Caveats

- `df` at hs_idx 28 mixes fact content with paraphrase surface form (same
  limitation as the raw-neuron analysis); the random-other-donor null and the
  cross-fact recurrence are the controls. A causal feature-level test (does
  injecting the top features flip the answer) still needs a model forward and
  is deferred to the steering build if this check is positive.
