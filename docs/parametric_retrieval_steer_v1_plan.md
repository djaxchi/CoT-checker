# parametric_retrieval_steer_v1: Golden-Gate-style causal test of the flip-neurons

Follow-up to `parametric_retrieval_minimal_v1` (expF), which found a sparse set
of layer-27 MLP neurons whose donor-swap flips a fact's answer, often a single
neuron per fact. This asks the concept-level question, in the style of
Anthropic's *Scaling Monosemanticity* (Golden Gate Claude): if we **clamp a
fact's neuron to a large value on every token of unrelated prompts**, does the
model start talking about that fact's object/subject?

## Central caveat (why this is a test, not a foregone conclusion)

Scaling Monosemanticity steered **SAE dictionary features**, which are
(relatively) monosemantic; its whole premise is that **raw neurons are
polysemantic** (superposition). Here we steer **raw MLP neurons**. So the honest
prior is that a single raw neuron is unlikely to be a clean concept feature. Two
informative outcomes:

- **concept-like** — steering summons the fact off-context, specifically for its
  own fact (diagonal of the specificity matrix): a raw "knowledge neuron"
  (closer to Dai et al. 2022 than to Golden Gate).
- **contextual gate** — steering does nothing off-context: the neuron is a
  "this fact is relevant here" switch, not a concept. A clean null here is the
  trigger to escalate to the SAE-feature version (the project's andyrdt Instruct
  SAE, public_sae arm) — the raw-first / SAE-if-null plan.

## Method

- **Neuron per fact**: the rank-weighted most frequent top-attribution neuron
  across that fact's pairs, from `expF/run.parquet` (no recompute).
- **Clamp**: `ClampNeuron` sets the neuron's activation (down_proj input g) to
  `alpha * max_activation` on every token, every forward pass (prefill +
  decode), for `alpha in {none, 0, 3, 6, 10}`. max_activation is per-neuron over
  the extraction set (`NeuronStore`).
- **Prompt battery**: 10 unrelated open-ended prompts + the fact's own question
  (positive control).
- **Controls**: a random neuron (specificity) and no-clamp (base rate).
- **Detector**: normalized substring match of the fact's object/subject in the
  generation.
- **Specificity matrix** (the centerpiece): steer fact i's neuron, measure
  whether object j appears in fact i's unrelated generations. Diagonal
  dominance = fact-specific knowledge neurons; flat = polysemantic / contextual.
- **Coherence**: unique-token ratio vs alpha (steering degradation).

## Pipeline (single TamIA job, prefix prgs_)

| stage | script | outputs |
|-------|--------|---------|
| run | `prgs_steer.py --phase run` | expG/steer.parquet (sharded by fact) |
| analyze | `prgs_steer.py --analyze` | expG/steer_curves.csv, specificity_matrix.npy, summary.json |
| figure | `prgs_viz.py` | results/parametric_retrieval_steer_v1/prgs_v1_results.png |

New shared code in `prga_common.py`: `ClampNeuron` (persistent per-token neuron
clamp) and `greedy_generate` (plain greedy generation under a clamp context).

## Figure

Three panels: A. object/subject mention rate on unrelated prompts vs alpha
(target neuron vs random vs own-question control); B. specificity matrix
heatmap (diag vs off-diag); C. output coherence vs alpha.

## Caveats

- Raw neurons, not SAE features (polysemanticity prior; see above).
- Clamp value uses the neuron's max activation over the final-prompt-token
  extraction set as the scale reference, positive direction (Golden Gate
  literal); a fact whose helpful direction is a decrease would be understated.
- Detector is substring match: crude, so generations are logged for eyeballing
  and the random-neuron / off-diagonal terms give the false-positive baseline.
