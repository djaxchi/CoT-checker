# external_reader_v0: does reading another model's states escape the policy's blind spots?

Frozen 2026-10-19, before any result was read.

§21.34: on MATH-500 mixed pairs where the policy's token confidence prefers the
wrong answer, the probe on the policy's own states (Qwen3-8B) is below chance
(0.441 [0.386, 0.494], question-weighted) while Qwen2.5-Math-PRM-7B's head keeps
0.629. Two explanations: the PRM reads with a different model's weights, or it
was trained for process supervision. The parent (Qwen2.5-Math-7B-Instruct)
separates them: a different model, never process-supervised.

Design. Every step of the 1,580 candidates of the 158 MATH-500 questions with a
mixed pair (T=1.0 Instruct pool, §21.13), 88,370 step items under the verifier
template (`scripts/analysis/build_tts_step_items.py`, segmentation checked
against the §21.13 score files), encoded by `scripts/encode_pooled_multilayer.py`
with four backbones: Qwen3-8B (the policy, L35), Qwen3-8B-Base (L35), the parent
(L27, L28) and the PRM (L27, L28). Each is scored by closed-form LDA fit on that
backbone's own PRM800K training vectors (backbone_geometry_v0 subsample;
Qwen3-8B-Base from its grid_vectors), step mean and last token, worst step per
trace, then the mixed-pair analysis of §21.34
(`scripts/analysis/external_reader_pairs.py`).

**GO:** the parent's LDA (best of its four blocks, chosen on all mixed pairs)
scores >= 0.55 with the 95% interval above 0.5 on the pairs where confidence
prefers the wrong answer. The reading then is that an external reader, not
process supervision, breaks the coupling. If the parent stays below 0.5 while
the PRM LDA is above it, process supervision is the cause.
