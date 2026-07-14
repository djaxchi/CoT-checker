# transition_operator_v0: learning a latent transition operator for reasoning steps

Status: effect targets and Stage 0 gates frozen (2026-07-13). v0.1 design revision
(same day): the trained forward model is REPLACED by the frozen model itself as
decoder; a new Stage 0 oracle gate decides whether that arm is viable. The trained
forward model survives only as the fallback arm (Appendix A).

v0.2 review fixes (2026-07-13, Stage 0 approved for implementation; later stages wait
on Stage 0 results): (1) the patched boundary token's upper-block K/V is recomputed
from the patched state, never served from the pre-context cache; (2) the patch-layer
comparison set is L20 vs L24/L26, L28 is invalid as a patch layer for Target B (no
blocks remain to propagate the edit into suffix computations); (3) D(z) gets
naturalness audits and a norm-remedy so the result cannot silently become an
adversarial control code; (4) InfoNCE negatives are masked by measured-effect
distance so near-identical effects are never forced apart.

v0.3 revision (2026-07-14, after Stage 0 ran on 7B, job 368860). Stage 0 verdicts:
gates 1/2/4 pass; gate 5 identity check exactly 0 (patch machinery proven correct);
gate 5 recovery is ASYMMETRIC. Target A (boundary next-token) median recovery
0.898/0.988/0.999 at L20/L24/L26, i.e. a single boundary edit almost fully controls
the immediate next-token distribution. Target B (answer belief through the suffix)
median recovery ~0 at every layer. The two targets are therefore split:
  - Target A: boundary-mediated, causally decodable through D(z) at L20.
  - Target B: predictable FROM the transition representation, but NOT recoverable
    through one late patched boundary position.
Interpretation discipline (correction to the first read): the null does NOT prove the
belief effect "lives in the step tokens rather than the boundary summary". It proves
only that a SINGLE patched boundary position does not carry it; the effect could be
distributed across several step-token K/V states or other positions. Locating it needs
a later multi-position patching experiment, out of scope for v0.
Design consequences (this revision): L_B stops being a frozen-decode loss and becomes
a small trained head h_B(z) -> d_belief (MSE); L_A stays the frozen-decoder KL; Stage 2
runs a three-arm ablation (A-only, B-only, A+B); Stage 4 is causal for Target A ONLY.
Order of work is unchanged from the original plan: Stage 1 kill gate is NOT skipped;
it runs before any Stage 2 training.

Research question: can a compact latent z_t, trained to predict a step's downstream
computational effect, represent reasoning transitions more semantically (operation-like,
reusable across problems) than raw boundary activations, deltas, or pooling?

Design rule for v0.1: every component that CAN be the frozen model IS the frozen model.
Trained parameters are limited to the encoder E, a linear map D, and the small InfoNCE
projections. Removed relative to v0: the forward-model MLP (width, depth, activation),
its three heads, the PCA-64 training-target space for Target A (kept only as a
measurement space), and the auxiliary S_t head with its loss weight.

---

## The whole experiment in one schema

```
════════════════════════════════════════════════════════════════════════════════════════
 (1) DATA: one PRM800K matched fork = same question + same golden prefix,
     one +1 step and one -1 step (pool: 42,797 unique pairs; v0 samples 5k-10k forks,
     problem-disjoint train/val/test frozen to JSON before extraction)

     question ──> step_1 ──> ... ──> step_{t-1} ──┬──> step_t^correct  (rating +1)
                    (golden prefix, shared)       └──> step_t^wrong    (rating -1)
════════════════════════════════════════════════════════════════════════════════════════
 (2) EXTRACTION: Qwen/Qwen2.5-7B base (28 blocks, hidden 3584, GQA 4 KV heads),
     fp16, teacher-forced, ONE pass per branch. Steps joined by "\n" (single token,
     id 198). Residual stream read at hidden_states[20] (primary; output of block 20)
     and hidden_states[28] (final stream, diagnostic readout only). Boundary states
     additionally read at hidden_states[24] and [26] (alternate patch layers for the
     Stage 0 oracle; the blocks-21-28 K/V cache covers all three patch layers).

 token stream (post context, one branch):

   [question] \n [step_1] \n ... [step_{t-1}]  \n   x_1  x_2 ... x_{m-1}  x_m   \n
                                               │    └───── step_t ──────┘ │    │
                                               │                          │    │
        S_{t-1} = h_L at this "\n" ◄───────────┘                          │    │
        (state BEFORE the step; also                                      │    │
         the PRE readout position, see (4))                               │    │
                                                                          │    │
        H_t = h_L at x_1 .. x_{m-1} ◄─────────────────────────────────────┤    │
        (step tokens, LAST TOKEN EXCLUDED                                 │    │
         from encoder input: anti-leakage)                                │    │
                                                                          │    │
        S_t = h_L at x_m  ◄───────────────────────────────────────────────┘    │
        (state AFTER the step; oracle and                                      │
         baselines only, never encoder input)                                  │
                                                                               │
        POST readout: next-token logits at this trailing "\n" ◄────────────────┘
        (same token id as the PRE readout position, so the readout
         token is identical everywhere: format control by construction)

 the PRE context is [question] \n ... [step_{t-1}] \n  with logits read at the final
 "\n"; it is SHARED by both siblings of a fork and computed once per fork. Its K/V
 for blocks 21-28 is cached at extraction (16 KB/token fp16; ~8 MB per 500-token fork).
════════════════════════════════════════════════════════════════════════════════════════
 (3) MODEL: encoder E (trained) + linear map D (trained) + THE FROZEN MODEL ITSELF
     as decoder. No trained forward model. No text decoder (reconstruction would
     reward wording; nothing in the pipeline ever reproduces step tokens).

                 S_{t-1} ∈ R^3584          H_t = [h(x_1)..h(x_{m-1})] ∈ R^{(m-1)×3584}
                     │                                   │   (padded/masked to 192 tokens;
                     │                                   │    truncation rate reported)
                     ▼                                   ▼
              ┌───────────────┐                ┌──────────────────┐
              │ Linear 3584→512│               │ Linear 3584→512  │   (separate proj,
              │ + LayerNorm    │               │ + LayerNorm      │    shared across pos.)
              └──────┬────────┘                └────────┬─────────┘
                     │                                  │
                     └───────────┬──────────────────────┘
                                 ▼
                ENCODER E (trained, ~10M params): 2-layer Transformer,
                d_model 512, 8 heads, learned positional embeddings,
                sequence = [ s-slot(S_{t-1}) ; x_1 ; ... ; x_{m-1} ]
                                 │
                     output at the s-slot (CLS-style,
                     conditioned on prefix state)
                                 ▼
                        MLP 512→512→d_z, LayerNorm
                                 ▼
                ┌──────────────────────────────────────┐
                │   z_t ∈ R^{d_z},  d_z ∈ {32,64,128}  │   THE LATENT TRANSITION OPERATOR
                │            (default 64)              │
                └──────────────────┬───────────────────┘
                                   ▼
                    D: Linear(d_z → 3584), trained (~230k params)
                                   ▼
                   ĥ = S_{t-1} + D(z_t)
                   a residual-stream EDIT at the PRE boundary "\n",
                   in the layer-20 stream (z is literally a coordinate
                   system over boundary residual edits)
                                   ▼
        ╔═══════════════════════════════════════════════╗   ┌────────────────────────┐
        ║  FROZEN Qwen2.5-7B AS DECODER (0 trained      ║   │ h_B: trained head      │
        ║  params): blocks 21..28 + RMSNorm + LM head   ║   │ MLP(d_z → 8)  ── L_B   │
        ║  at the patched boundary over the PRE K/V.    ║   │ (Target B is NOT       │
        ║  CACHE SURGERY: cache holds only positions    ║   │  boundary-causal, gate │
        ║  STRICTLY BEFORE the boundary in blocks       ║   │  5 recovery ~0; it is  │
        ║  21-28; the boundary token's K/V there is     ║   │  PREDICTED from z,     │
        ║  recomputed fresh from ĥ (blocks 1-20 keep    ║   │  which sees H_t)       │
        ║  the original boundary K/V). No cache grads.  ║   └───────────┬────────────┘
        ╚═══════════════════════╤═══════════════════════╝               │
                                ▼                                        ▼
                predicted next-token distribution              predicted answer belief
                at the boundary  (→ L_A, KL)                   d_belief ∈ R^8 (→ L_B, MSE)

     Stage 2 three-arm ablation:  A-only (w_A=1,w_B=0) | B-only (w_A=0,w_B=1) | A+B
     side branch for the contrastive term:
        z_t ──Linear(d_z→64), L2-norm──┐
                                       ├── symmetric InfoNCE, τ = 0.07 (→ L_NCE)
        measured effect [dL_64 ; d_belief_8] ──MLP g→R^64, L2-norm──┘
════════════════════════════════════════════════════════════════════════════════════════
 (4) EFFECT TARGETS (frozen): everything is a temporal delta  POST minus PRE,
     readout token identical ("\n") in both.

  TARGET A: boundary next-token effect                            [weight 1.0]
    measured: dL_t = logits_post("\n") - logits_pre("\n")  ∈ R^152064
    loss:     L_A = KL( p_post^actual  ‖  p_post^predicted ), full vocab,
              where p_post^predicted comes from decoding ĥ through the frozen
              blocks. Format handling is now largely by construction: anything
              predictable from the PRE context alone is produced by the frozen
              decoder even at D(z)=0, so z only pays for what the STEP changed.
    the PCA-64 (train-fit, whitened) + format-residualized space of dL is KEPT
    as a MEASUREMENT space: Stage-1 baselines, the format audit, and the
    InfoNCE effect embedding live there. It is no longer a training target.

  TARGET B: answer-belief effect                                  [weight 1.0]
    fixed elicitation suffix (Stage-0 selected: "\nSo the final answer is",
    gold rank-1 0.957), teacher-forced after PRE and POST contexts; candidates =
    gold + 7 type-matched distractors (sources: phase-2 pre_generated_answer >
    wrong-branch finals > corpus same-type > integer perturbations); score =
    mean log-prob per token (Qwen splits digits, first-token lens too lossy),
    KV-cache reused
    belief = softmax over the 8 candidates (canonical order: gold first,
    distractors by pre score);  d_belief = belief_post - belief_pre ∈ R^8
    headline scalar: d_margin = [logp_gold - max logp_distractor]_post - [same]_pre
    loss (v0.3, NOT a frozen-decode loss): L_B = MSE( h_B(z_t), d_belief )
          h_B: small trained head R^{d_z} -> R^8. Stage-0 gate 5 showed a single
          patched boundary position recovers ~0 of d_belief, so B is treated as
          PREDICTABLE-FROM-z, not boundary-causal. z sees the step tokens H_t, so
          it can carry the belief effect even though the boundary state cannot.

  SIBLING CONTRAST: InfoNCE                                       [weight 0.5]
    z_t (L2-normed via Linear d_z→64) vs g([dL_64 ; d_belief]) (L2-normed),
    symmetric CLIP-style, temperature 0.07; batches are built from WHOLE FORKS
    so the sibling is available as a hard negative. EFFECT-DISTANCE MASK: any
    negative (sibling included) whose measured-effect vector has cosine > 0.9
    with the anchor's is masked out; near-identical effects are never forced
    apart. Masking rate and sibling-masked fraction are reported.
    enforces "siblings with DIFFERENT consequences stay distinguishable"

  (v0's auxiliary Ŝ_t head is DROPPED: there is no trained forward model to
   host it. S_t stays extracted: it feeds the baselines and the oracle.)

  TOTAL:  L = w_A·L_A + w_B·L_B + 0.5·L_NCE
  THREE-ARM ABLATION (Stage 2): A-only (w_A=1, w_B=0), B-only (w_A=0, w_B=1),
  A+B (w_A=w_B=1). InfoNCE on in all three. Tells us whether Target B improves
  semantic organization or merely pushes z toward answer correctness.
════════════════════════════════════════════════════════════════════════════════════════
 (5) TRAINING: AdamW lr 3e-4 cosine, wd 0.01, batch 128 forks (256 transitions),
     ~30 epochs on 10k-20k transitions, early stop on val total loss, seeds {0,1,2}.
     Trained params ~11M (E + D + InfoNCE projections). The frozen 7B sits on the
     GPU in bf16 (~15 GB); per example only 1 boundary + ~10 suffix/candidate
     positions pass through blocks 21-28 with gradients, so one H100 suffices.
     PRE-context K/V (blocks 21-28) is precomputed at extraction: ~80 GB fp16 for
     10k forks on $SCRATCH (fp8 halves it); alternative is per-epoch no-grad
     recomputation, decided at Stage 0 smoke from TamIA I/O.
════════════════════════════════════════════════════════════════════════════════════════
 (6) EVALUATION (z vs baselines, ALL baselines PCA'd to d_z for fair capacity):
     baselines: S_t | S_t - S_{t-1} | [S_{t-1};S_t] | mean-pool(H_t) | max-pool(H_t)

     a. operation decodability      linear probe on symbolic-parser + tags.parquet
                                    labels, problem-disjoint split
     b. cross-problem retrieval     same-operation precision@k, query and gallery
        (THE DECISIVE TEST)         from different problems
     c. sibling effect separation   does z separate siblings in proportion to
                                    their measured effect difference
     d. stability                   cluster ARI across the 3 seeds
     e. transfer                    a-c on held-out problems
     f. surface controls            a-c repeated after residualizing surface
                                    features (fork-audit methodology)
════════════════════════════════════════════════════════════════════════════════════════
 (7) CAUSAL (Stage 4): TARGET A ONLY (v0.3). The Target-A training operation IS a
     patch: swap z between siblings (ĥ = S_{t-1} + D(z_donor)) and decode with the
     same frozen blocks; no surrogate translation step. Controls: mismatched donors,
     random z, magnitude-matched noise (prga battery). Scope stays honest: this
     shows the decoded boundary edit carries the IMMEDIATE next-token effect, not
     that the model "uses z" internally. Target B stays PREDICTIVE: its causal
     locus is unknown (a single boundary patch recovered ~0) and would need a
     later multi-position patching study, out of scope for v0.
════════════════════════════════════════════════════════════════════════════════════════
```

---

## 1. Setting

Model: Qwen/Qwen2.5-7B (base), fp16, teacher-forced. 28 blocks, hidden 3584, GQA with
4 KV heads (head dim 128), vocab 152064. Streams read at hidden_states[20] (primary,
matching `scripts/s4_contrib_extract_forks.py`) and hidden_states[28] (diagnostic
readout only, not a patch layer); boundary states also at hidden_states[24] and [26]
as alternate patch layers for the Stage 0 oracle.

Data: PRM800K matched forks (same question + same golden prefix, one +1 and one -1
continuation). The existing pool has 42,797 unique pairs (`runs/contrib_cluster/
forks_manifest.json`); v0 samples 5k to 10k forks (10k to 20k transitions) with a fixed
seed and a problem-disjoint train/val/test split materialized to JSON before extraction.

Rendering follows the existing convention exactly: steps joined by `"\n"`
(`prefix = question + "\n" + "\n".join(prefix_steps)`, branch = `prefix + "\n" + step`).
`"\n"` is a single Qwen token (id 198).

Data plumbing change: `s4_contrib_forks.py` (or a v0 fork of it) must additionally emit
`ground_truth_answer` (present in raw PRM800K under `question.ground_truth_answer`),
the phase, and `pre_generated_answer` when present (phase 2 only).

## 2. Token positions and what is extracted

For transition t with rendered post context `P + "\n" + x_1..x_m + "\n"`:

| Symbol | Token position | Layer | Role |
|---|---|---|---|
| `S_{t-1}` | the `"\n"` immediately before `x_1` | L20, L28 | encoder input; state before the step; patch site |
| `H_t` | `x_1 .. x_{m-1}` (last step token excluded) | L20, L28 | encoder input |
| `S_t` | `x_m` (last step token) | L20, L28 | oracle and baselines only, never encoder input |
| PRE readout | final `"\n"` of `P + "\n"` (same position as `S_{t-1}`) | logits | Target A/B pre |
| POST readout | trailing `"\n"` after `x_m` | logits | Target A/B post |
| PRE K/V cache | all PRE-context positions | blocks 21-28 | frozen-decoder training input |

The last step token is excluded from `H_t` because `S_t` is by definition the layer-L
activation at that token: including it would let the encoder shortcut the effect
objectives through the model's own aggregate.

Pre quantities are stored once per fork (shared prefix). Step-token sequences are
padded/masked to 192 tokens; the truncation rate is reported in the extraction manifest.
Token boundaries are located with fast-tokenizer offset mapping, as in prior extractions.

## 3. Core definition: a step's effect

The effect of step t is the temporal delta between the model's state of expectation
AFTER the step and BEFORE it, with the wording of the measurement held fixed:

```
pre  context = P + "\n"            (shared by all siblings of a fork)
post context = P + "\n" + x_t + "\n"
```

Appending the same single separator token to both contexts means the token at the
readout position is identical everywhere, which removes the largest format confound
(identity of the last observed token) by construction.

Tokenization rule (implementation detail that is part of the spec): pieces (question,
steps) are tokenized separately and joined with the explicit separator id 198; every
context ends with an appended 198. String-level tokenization would let BPE merge a
trailing newline into the previous token, breaking both the identical-readout-token
property and the strict-prefix property (pre-context ids being a prefix of branch ids,
which is what lets the pre pass and its K/V cache be shared). Gate 4 reports how often
the two tokenizations differ.

## 4. The frozen-model decoder (v0.1 core change)

The trained forward model of v0 is replaced by the model's own upper half:

```
ĥ = S_{t-1} + D(z_t)                      D linear, d_z → 3584, trained
p_post^predicted = FrozenQwen[blocks 21..28 + RMSNorm + LM head](ĥ at the PRE
                   boundary position, attending to the PRE context's cached K/V)
```

Why this is better than a trained forward model:

- Removed hyperparameters: forward-MLP width/depth/activation, three head designs,
  the choice of Target A training space, the auxiliary loss and its weight.
- z is forced into the model's own geometry: D(z) must be a residual edit the model
  itself converts into downstream behavior, so z is directly comparable to steering
  vectors and prior patching work rather than living in an arbitrary learned space.
- Format handling largely by construction: everything predictable from the PRE
  context alone is emitted by the frozen decoder even at D(z) = 0, so the objective
  only pays z for what the step changed.
- Stage 4 unification: training, evaluation, and causal patching are the same
  operation; the Stage 4 sibling-swap needs no surrogate translation.

The honest cost, and why there is an oracle gate: in the real model a step's effect on
future tokens is mediated BOTH through the boundary state AND through attention paid
directly to the step tokens, which a single-position edit cannot reproduce. The frozen
decoder therefore has a ceiling: the fraction of the true effect recoverable from the
boundary alone. That ceiling is measurable BEFORE training (Stage 0 gate 5) by patching
the true post boundary state and seeing how much of the true post distribution comes
back. If the ceiling is low, the frozen-decoder arm caps out and the trained forward
model (Appendix A) becomes the primary arm; the ceiling itself is then a result (how
boundary-mediated are reasoning steps). Position mismatch between donor and patch site
(the donor state was computed m+1 tokens later; RoPE is relative) is part of what the
oracle measures.

Cache surgery (exact semantics of "patch at layer 20"): let C be the K/V cache of the
full PRE context.

1. Blocks 1-20: C is used unchanged, INCLUDING the boundary token (what happened below
   the patch layer is not part of the intervention).
2. Blocks 21-28: the cache is truncated to positions strictly before the boundary. The
   patched state ĥ is run through blocks 21-28 at the boundary position, attending to
   the truncated cache, and its FRESH K/V is appended. Boundary logits (Target A) are
   read from this pass.
3. Suffix and candidate tokens (Target B) run through all 28 blocks: in blocks 1-20
   they attend to the original cache (boundary included), in blocks 21-28 they attend
   to the truncated cache plus the recomputed patched-boundary K/V.

Serving the original boundary K/V to suffix tokens in the upper blocks would sever the
causal path from the patch to Target B; this is why the recompute is mandatory and
why the identity test below is a required unit test: with ĥ set to the TRUE pre
boundary state, the whole procedure must reproduce the unpatched logits exactly.

The cache carries no gradient (it does not depend on the patch), so training
backpropagates only through the patched position's path across 8 blocks (including
its recomputed K/V), at 1 boundary position plus ~10 suffix and candidate positions
per transition. GQA makes the cache small: 2 x 4 heads x 128 dims x 2 bytes = 2 KB
per token per layer, 16 KB per token for the 8 layers.

Patch-layer set: L20 (primary) vs L24 and L26 (alternates). L28 is NOT a valid patch
layer for this design: no blocks remain above it, so an edit there can influence the
immediate boundary logits (through the final norm and LM head only, a logit-lens-style
readout) but can never propagate into the suffix tokens' computation, i.e. Target B
receives no signal at all. L28 appears only as a diagnostic readout, never as an arm.

### D(z) naturalness audits (mandatory before any Stage 3 claim)

Risk: D(z) could learn adversarial steering vectors that reproduce the target logits
while lying far outside the geometry of natural boundary states; the result would be a
compact control code, not a representation of natural reasoning transitions. Reference
distributions, all computable from the extraction: natural boundary states h_L at
step boundaries (train), natural step-induced edits `delta_true = h_L(post boundary) -
S_{t-1}` per transition, and sibling donor differences `h_L^correct - h_L^wrong` at
the post boundary.

- A1 patched-state RMS: RMS(ĥ) must fall within the [1st, 99th] percentile band of
  natural boundary-state RMS; report the percentile distribution.
- A2 manifold distance: Mahalanobis distance of ĥ to the natural boundary-state
  distribution (shrinkage covariance in train PCA-256 space), compared against the
  natural states' own distance distribution.
- A3 edit magnitude: the distribution of ||D(z)|| against the distributions of
  ||delta_true|| and of sibling donor differences.
- A4 norm-matched clipping stress test: at eval, rescale D(z) to the natural median
  ||delta_true|| and separately cap it at the 95th percentile; if val L_A or L_B
  degrades by more than 20% relative, the operator depends on unnatural magnitudes.

Remedy if A4 (or grossly A1/A2) fails: add a soft hinge penalty on ||D(z)|| above the
natural 95th percentile (weight 0.1) and retrain; the audit tables appear in the
report either way.

## 5. Target A: boundary next-token effect

Measured raw effect: `dL_t = logits(post) - logits(pre)`, both read at the final
separator token, full vocab (152064).

Training loss: `L_A = KL(p_post^actual || p_post^predicted)` at the boundary readout,
full vocab, temperature 1.

Measurement space (kept from v0, no longer a training target): center `dL` on train,
train-fit PCA top 64, whiten, then residualize each PC against the frozen format
features (step length in tokens, final non-separator character class, contains display
math, ends with an equation, discourse-marker opener). Per-PC format R^2 is reported
as the format audit. This 64-dim space is where Stage-1 baseline effect-prediction,
the InfoNCE effect embedding, and diagnostics live.

## 6. Target B: answer-belief effect

Elicitation suffix: one fixed string, teacher-forced identically after `post` and `pre`
contexts for every branch and every problem. The suffix is selected in Stage 0 from a
frozen candidate list of three (`"\nThe answer is"`, `"\n# Answer\n\n"`,
`"\nSo the final answer is"`) by a frozen criterion (Stage 0 gate 1), then never
changed.

Candidate answer set per problem: gold + 7 distractors, K = 8. Distractor sources in
priority order, all type-matched to the gold answer's type (integer / decimal /
fraction / latex expression / other; measured distribution: 66.6% integer, 15.1%
fraction, 11.4% latex expr on a 3,940-answer sample):

1. `pre_generated_answer` when present and != gold (phase 2);
2. final answers of wrong branches in the same session when extractable;
3. gold answers of other problems with the same answer type (corpus-sampled, fixed seed);
4. for integers only, perturbations (sign flip, +/-1, digit swap) to fill remaining slots.

Answer normalization (frozen function): strip whitespace, strip `\$` and `\text{...}`
wrappers, canonicalize `a/b` to `\frac{a}{b}` only when the gold uses `\frac`, render
every candidate as `" " + answer` (single leading space) after the suffix.

Scoring: full teacher-forced scoring of each candidate string, mean log-prob per token
(Qwen tokenizes digits individually, e.g. `" 29"` is 3 tokens, so a first-token lens is
too lossy). KV cache is reused across the 8 candidates of one context.

Derived quantities per transition:

- `belief(c) = softmax over the 8 candidates of mean-token-logprob`, for pre and post;
  candidate order canonicalized (gold first, distractors sorted by pre score).
- Measured vector effect: `d_belief = belief_post - belief_pre` (8-dim).
- Headline scalar: `d_margin = [logp_gold - max distractor logp]_post - [same]_pre`.

Training loss (v0.3): `L_B = MSE(h_B(z_t), d_belief)`, a small trained head
`h_B: R^{d_z} -> R^8` predicting the measured belief delta from z. This is a
prediction loss, NOT a frozen-decode loss: Stage 0 gate 5 showed a single patched
boundary position recovers ~0 of d_belief, so Target B is treated as predictable from
the transition representation, not as boundary-causal. d_belief remains fully measured
at extraction (pre and post beliefs), and also feeds the InfoNCE effect embedding.

Cross-problem comparability comes from the within-set softmax (scale-free) and from the
fact that sibling contrasts share pre exactly. Never compare raw logprobs across
problems.

## 7. Losses and training protocol

```
L = w_A * L_A   (KL, frozen-decoder boundary distribution vs actual post)
  + w_B * L_B   (MSE, trained head h_B(z) vs measured d_belief)
  + 0.5 * L_NCE (symmetric InfoNCE, z vs measured-effect embedding,
                 in-batch negatives incl. guaranteed sibling, tau 0.07)

Stage 2 three-arm ablation:  A-only (w_A=1, w_B=0) | B-only (w_A=0, w_B=1) |
A+B (w_A=w_B=1).  InfoNCE stays on in all three. The comparison isolates whether
Target B improves semantic organization of z or only pushes it toward answer
correctness.
```

Trained parameters: E (~10M) + D (d_z x 3584) + InfoNCE projections. ~11M total.
z_t never sees S_t, the last step token, the suffix, or any label.

Batches are assembled from whole forks (128 forks = 256 transitions per batch) so every
transition's sibling is available as a hard negative for InfoNCE. Negatives are masked
by measured-effect distance: any in-batch negative (sibling included) whose whitened
`[dL_64 ; d_belief]` vector has cosine similarity above 0.9 with the anchor's is
excluded from the denominator, so transitions with near-identical measured effects are
never forced apart, whatever their correctness labels. The overall masking rate and the
fraction of sibling pairs masked are logged per epoch. AdamW, lr 3e-4 with
cosine decay, weight decay 0.01, ~30 epochs, early stopping on val total loss, seeds
{0, 1, 2}. Loss weights get one coarse sweep on val only if the defaults visibly
underfit one target; any change is logged.

Compute: extraction (hidden states ~11 GB + logit readouts + suffix scores + PRE K/V
cache ~80 GB fp16 for 10k forks) is the TamIA job. Training holds the frozen 7B in
bf16 plus the 11M trained parameters on one H100.

## 8. Stage 0 gates (frozen before the full run)

Run on a few hundred examples locally / single TamIA smoke before the main extraction.

1. Suffix selection: on 300 complete golden trajectories, pick the suffix whose
   gold answer is rank 1 among the 8 candidates most often. Gate: best suffix >= 60%
   rank-1. If all three fail, Target B is too noisy at 7B and v0 proceeds with
   Target A plus `d_margin` only (flagged in the report).
2. Directional sanity: on ~500 forks, paired comparison of `d_margin` for correct vs
   wrong siblings. Gate: mean `d_margin(correct) > d_margin(wrong)` with p < 0.01
   (paired test). If this fails the target does not measure reasoning progress and the
   design is revisited before training anything.
3. Format audit: report the per-PC format R^2 of Target A's measurement space. No hard
   gate, but if median R^2 > 0.5 across the top 64 PCs the residualization step is
   escalated (drop PCs instead of residualizing) and the report says so.
4. Token-boundary sanity as in prior extractions (offsets via fast-tokenizer mapping).
5. Boundary-sufficiency oracle (new in v0.1, decides the decoder arm): patch the TRUE
   post boundary state `h_L(post "\n")` into the PRE context at the pre boundary and
   decode through frozen blocks 21-28. Report
   `recovery = 1 - KL(p_post^actual || p_oracle) / KL(p_post^actual || p_pre)`
   over ~500 forks, using the full cache surgery (the oracle exercises the same
   patched-decode code path as training, including the boundary K/V recompute), and
   reported separately for Target A (boundary distribution) and Target B (answer
   belief). Gate: median Target A recovery >= 0.3 at some layer in {20, 24, 26}.
   L28 is excluded (no blocks above it; it appears only as a logit-lens diagnostic).
   If L20 fails but L24 or L26 passes, the patch layer moves there and the report
   says so. If all fail, a single-position edit cannot carry the step's effect
   (attention to step tokens bypasses the boundary), the trained forward model
   (Appendix A) becomes the primary arm, and the recovery distribution is reported
   as a mediation result in its own right.

## 9. Stage skeleton and gates

- Stage 1 baselines on frozen targets and labels: `S_t`, `S_t - S_{t-1}`,
  `[S_{t-1}; S_t]`, mean/max pooling over `H_t`, all compared at matched dimensionality
  (PCA to d_z). Operation labels: symbolic parser on the parseable subset plus existing
  heuristic tags (`runs/contrib_cluster/tags.parquet`); measure label coverage and
  manual agreement on ~200 steps first. KILL GATE: if the best raw baseline reaches
  ~90% of the label-noise ceiling on BOTH operation decodability (problem-disjoint) and
  cross-problem retrieval, the learned operator needs a different justification or v0
  stops.
- Stage 2: train E, D, h_B as specified in sections 4-7, as the THREE-ARM ablation
  (A-only, B-only, A+B; InfoNCE on in all three), 3 seeds each.
- Stage 3 semantics, all vs matched-dim baselines and all repeated after surface
  residualization (fork-audit code): operation decodability, cross-problem
  same-operation retrieval precision@k (the decisive test), sibling effect separation,
  ARI across seeds, held-out-problem transfer. The three arms are compared here to
  separate "Target B improves organization" from "Target B only adds a correctness axis".
- Stage 4 causal (Target A only): native sibling-swap, `ĥ = S_{t-1} + D(z_donor)`
  decoded by the same frozen blocks, with the prga control battery (mismatched donors,
  random z, magnitude-matched noise). Framing is fixed: this shows the decoded edit
  carries the effect; it does not show the model internally uses z.

## 10. Interpretation guardrails

- The claim is relative: z organizes transitions by downstream effect better than
  matched baselines after surface controls. Never claim z contains "effect instead of
  wording"; teacher-forced determinism guarantees wording information survives any
  effect-predictive bottleneck.
- Within-problem sibling separation is necessary but insufficient; only cross-problem
  retrieval distinguishes operation encoding from content encoding.
- The frozen-decoder objective measures the boundary-mediated share of a step's effect;
  its ceiling (gate 5) is a finding, not a nuisance. Stage 0 result: Target A is
  boundary-mediated (recovery ~0.9-1.0), Target B is not (recovery ~0 through one
  patched boundary). Do NOT overstate the B null as "belief lives in the step tokens";
  it proves only that ONE late boundary position does not carry it. The mediation locus
  of Target B is open, for a later multi-position patching study.
- No Stage 3 claim without the D(z) naturalness audits (A1-A4): reproducing the targets
  through unnatural residual edits is a control code, not a representation of natural
  reasoning transitions.
- Prior expectations from this project: content-specific effects are the likely positive
  outcome (prga v1 outcome C2); a content-independent operator effect would be the
  surprise headline and needs the full control battery before being reported.

## Appendix A: fallback arm (trained forward model, the v0 design)

Used only if Stage 0 gate 5 fails at both layers. `f(S_{t-1}, z)` =
`concat(Linear(3584->512)(S_{t-1}), z)` -> MLP(1024, GELU, 1024, GELU) with three
heads: 64-dim into Target A's whitened residualized PCA space (MSE), 8-dim into the
d_belief space (MSE), 3584-dim auxiliary for whitened `S_t - S_{t-1}` (MSE, weight
0.1). Same encoder, same InfoNCE, total ~15M trained params. Its Stage 4 requires the
extra surrogate step (patching f's implied boundary state), which is exactly what the
frozen-decoder arm eliminates.
