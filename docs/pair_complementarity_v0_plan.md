# pair_complementarity_v0: are hidden-state verifiers and token confidence complementary on Instruct?

Frozen 2026-10-19, before any result was read.

On the Base pool (§20.16) the verifier and answer-token margin were redundant
(phi 0.715 on 57 ties). The Instruct T=1.0 pool (§21.13) carries, per trace,
worst-step scores from 19 probes and Qwen2.5-Math-PRM-7B and DeepConf-style
token confidence. Unit: all within-question pairs of the ten candidates with one
correct and one incorrect final answer (different answer keys). Score: P(the
scorer prefers the correct one), ties counted half. Scorers: PRM worst step,
`boundary_stats x mlp_h1024x2` seed 42 worst step (best probe tie-breaker in
§21.13), mean token confidence and bottom-10% group confidence (w32), and the
unweighted within-question z-score sum of the probe and the better confidence
statistic. Question-cluster bootstrap, 2,000 resamples, MATH-500 and GSM8K.

**GO:** on MATH-500, probe + confidence beats the better of its two parts by at
least 0.02 in mixed-pair accuracy with the 95% interval above 0. Also reported:
accuracy of each scorer on the pairs where confidence prefers the wrong answer.
