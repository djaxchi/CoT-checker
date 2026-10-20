# external_reader_v1: the PRM's separator readout on the confidence-wrong pairs

Frozen 2026-10-20, after external_reader_v0 (§21.37) and before this run.

§21.37: on the 985 MATH-500 mixed pairs where token confidence prefers the wrong
answer, LDA on the PRM's states under the verifier template reaches at best
0.557 [0.499, 0.614] (L28 step mean), against the PRM head's 0.629 (§21.34);
the parent reaches 0.451. §21.31 found the head-matching linear readout at the
native `<extra_0>` separator, not at the step's own tokens. This run reads the
pairs there.

Design: one native-template pass of Qwen2.5-Math-PRM-7B at hidden_states 20 and
28 over the backbone_geometry_v0 training subsample (60,000 rows, seed 0),
val_5k and the 88,370 TTS step items of external_reader_v0
(`slurm/external_reader_sep_tamia.sh`). Scorers: the head's P(correct) from the
same pass, and LDA fit on the pass's own training vectors at L28 separator, L20
separator and L28 last token (`scripts/analysis/external_reader_sep.py`). Worst
step per trace, question-weighted mixed-pair accuracy, question bootstrap.

Gate: the head on all mixed pairs within 0.02 of §21.34 (0.708).

**GO:** L28 separator LDA >= 0.60 with the 95% interval above 0.5 on the
confidence-wrong pairs: the linear readout at the separator keeps the PRM's
independence from confidence. Below 0.55: what the head adds over its linear
direction is what escapes confidence.
