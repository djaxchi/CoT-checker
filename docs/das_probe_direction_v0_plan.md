# das_probe_direction_v0: patch only the correctness-readout component

Frozen 2026-10-19, before any result was read.

§15.8 found additive steering along the probe direction no better than random
and named activation patching of the `w` component as the one untested
intervention that could still flip the "readout, not lever" verdict. §18.1's
caches make it an evaluation: on 742 PRM800K forks (Qwen2.5-7B base), swap only
the component of the wrong sibling's last-8 step-token states that lies in a
fixed subspace Q for the correct sibling's, and read the teacher-forced
gold-answer margin. Same split and recovery metric as `das_fit.py`.

Q, fit on the train forks only: the mean sibling difference (k=1), shrunk LDA on
step-mean states (k=1), the top 1, 4, 16 principal directions of the per-fork
differences, random frames (k=1, 16, five seeds), and the learned DAS subspaces
of §18.1 (reproduction gate: L20 k=16 seed 0 recovered 0.244), plus the full
state.

**GO:** the mean-difference or LDA direction recovers >= 0.15 of the held-out
margin gap at layer 12 or 20 and beats random k=1 (paired Wilcoxon p < 0.05).
A NO-GO closes the patching loophole for a single readout direction.

Script `scripts/das_branch/das_probe_direction.py`, one short H100 job, layers 12
and 20 in parallel.
