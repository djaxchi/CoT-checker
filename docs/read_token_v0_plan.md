# read_token_v0: a learned read token on frozen Qwen3-8B

Frozen 2026-10-19, before any result was read.

§21.31: at the separator Qwen2.5-Math-PRM-7B's head reads, correctness is the
first principal component and closed-form LDA matches the head (calib-20 0.704),
while frozen Qwen3-8B's best LDA readout reaches 0.568 (L35 last token) and the
best trained Qwen3-8B cell 0.610 (d512, 8.7M parameters, 513k steps). PRM
training was full fine-tuning. Can a frozen Qwen3-8B be given a read position by
training only m appended soft-token embeddings and a linear head?

Design (`scripts/read_token_probe.py`): verifier template + step, then m soft
tokens; BCE (1 = incorrect) from a linear head on the final post-norm state of
the last soft token. The prefix runs once without gradients; the backward pass
covers only the soft positions. 100,000 PRM800K training steps (seed 0), one
epoch, AdamW lr 3e-3, one-cycle. Four configs on one node: m = 1, 4, 16 with
vocabulary-row initialisation, and m = 4 initialised from "\n\nIs this step
correct?". Evaluation: the trained head and closed-form LDA (fit on a separate
30,000-row train subsample) on the read-token state, PRM800K test_2k AUROC and
ProcessBench calib-20, through `scripts/analysis/backbone_geometry.py`.

Gate: cached left-padded read states match a full forward (relative difference
< 0.05) before training.

**GO:** the best config reaches ProcessBench calib-20 >= 0.62 with its head or
LDA, above the best Qwen3-8B cell (0.610) with at most 16 x 4096 + 4097 trained
parameters.
