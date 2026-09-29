# Local verifier diagnostics: execution and first findings

29 September 2026. TamIA job 494679 completed with exit code 0 in 2 minutes 9 seconds. I downloaded its diagnostic activations, reference scores, and the 16 selected probe checkpoints, then recomputed every diagnostic score on the MacBook Air CPU. No additional cluster job was necessary.

**Local reproduction.** All 16 checkpoints scored all 912 examples. The largest absolute probability difference from the cluster was 0.0000022650. The acceptance tolerance was 0.0001, fixed before local scoring. No joint-preference or tie outcome changed in the family metrics. Checkpoint and result-file hashes, dataset hashes, extraction manifests, and representation-store fingerprints matched. The local run inherits the hash-matched cluster source-test AUROC checks; it does not download or rerun the 513,810-row training store. The local test suite passes 33 tests.

**Files on your laptop.**

- Checkpoints: `results/verifier_signal_v1/local_assets/cells/`
- Frozen activations: `results/verifier_signal_v1/cluster_reference/store/diagnostic/`
- Cluster reference: `results/verifier_signal_v1/cluster_reference/`
- Local scores, analysis, and platform comparison: `results/verifier_signal_v1/local_cpu/`

The checkpoint download was 143,720,442 bytes; the complete cluster reference download was 80,433,001 bytes. These ignored result directories contain everything needed to replay this diagnostic offline. New text examples still require backbone extraction.

To rerun locally, choose a fresh output directory:

```bash
.venv/bin/python scripts/analysis/verifier_signal_local.py \
  --device cpu --batch-size 32 \
  --out results/verifier_signal_v1/local_cpu_repeat
```

The script checks agreement with the frozen cluster scores before completing analysis. It refuses to overwrite an existing output directory. CPU is the validated execution mode; the optional MPS path has not been tested.

**First finding: local consistency dominates the inherited-error contrast in these templates.**

For the first held-out inheritance family, the problem is `Solve 4*x + 6 = 50 for real x.` The d512 probe gives these error probabilities, averaged over its three seeds:

| Intermediate equation | Candidate | Error probability |
|---|---|---:|
| `4*x = 44` | `x = 11` | 0.0174 |
| `4*x = 44` | `x = 12` | 1.0000 |
| `4*x = 48` | `x = 11` | 0.9998 |
| `4*x = 48` | `x = 12` | 0.0358 |

The probe penalizes the correct final answer when it conflicts with the immediate prefix, and assigns a low error probability to the wrong final answer when it follows that prefix. When scored as a step itself, the wrong subtraction has error probability 0.9807. A low score on the following step therefore does not certify that the earlier mistake has disappeared.

Across all 18 plain-wording held-out inheritance families, d512 has local contrast 0.8789, with family-bootstrap interval [0.8294, 0.9243]. Its inherited-prefix contrast is 0.0255 [-0.0189, 0.0724], and its final-conclusion contrast is 0.0433 [-0.0031, 0.0936]. Attention-query shows the same ordering: local contrast 0.8872 [0.8366, 0.9317], prefix contrast 0.0382 [0.0181, 0.0613], and conclusion contrast 0.0370 [0.0185, 0.0580]. These are probability contrasts within each probe, not a calibrated ranking across probes.

**Second finding: context sensitivity varies by domain.** The following table reports the fraction of held-out families where both paired preferences point in the correct direction, averaged across training seeds. Each domain contains nine independent numeric families. Seeds are repeated trained probes, not extra independent families.

| Probe | Seeds | Affine | Inequality | Multiplication | Substitution |
|---|---|---:|---:|---:|---:|
| last_token / linear | 42,43,44 | 0.852 | 1.000 | 1.000 | 0.593 |
| step_mean / linear | 42,43,44 | 1.000 | 1.000 | 1.000 | 0.259 |
| step_tokens / attn_query | 42,43 | 1.000 | 1.000 | 1.000 | 0.500 |
| step_tokens / transformer:d128,l1,f512,h4 | 42,43 | 1.000 | 1.000 | 1.000 | 0.278 |
| step_tokens / transformer:d256,l2,f1024,h4 | 42,43,44 | 1.000 | 1.000 | 1.000 | 0.333 |
| step_tokens / transformer:d512,l2,f2048,h8 | 42,43,44 | 1.000 | 1.000 | 1.000 | 0.333 |

d512 reverses both preferences in every affine, inequality, and multiplication family for every seed, but its substitution joint-preference fraction is 0.333 [0.074, 0.630]. Attention-query reaches 0.500 [0.222, 0.778] on substitution. There is no universal chance baseline for this joint metric. Both results expose a limitation of the tested function-substitution format; this pilot does not isolate whether computation, notation, or another domain difference causes it.

**Wording remains consequential.** On inheritance examples, adding `I am certain that` shifts error probabilities by a mean absolute 0.254 for d512 and 0.257 for attention-query. On affine reversal examples, adding `Therefore` shifts them by 0.207 and 0.190. These controls change token count and wording together, so they do not identify a pure confidence effect. Strong paired semantic sensitivity coexists with substantial score sensitivity to wording.

**How I would use this result.** Preserve the record of previously flagged steps when evaluating a continuation. A later low score should not erase an earlier alarm. The most direct next experiment is to resample from the first flagged transition and compare final-answer accuracy against continuing from that prefix, using the same generation budget. First check the local-versus-inherited distinction on independently annotated natural Instruct traces. The toy result supports testing this policy, not claiming that it already improves reasoning.

We can now inspect these examples, compare seeds, audit false positives, and test alternative score aggregation locally. Testing a repair policy still requires new generator continuations. Establishing ProcessBench-to-TTS transfer still requires common candidate pools and matched compute across probes.

**Limits.** The test partition holds out numeric instances within templates. It does not hold out templates or establish natural-trace generalization. The inherited-error arm covers affine equations only. Family-bootstrap intervals do not capture training-seed uncertainty or correct for multiple comparisons; attention-query and d128 have two seeds while the others have three. These behavioral contrasts do not identify a circuit or prove that the generator uses the information decoded by the probes.

Full metrics and intervals: `results/verifier_signal_v1/local_cpu/analysis.json`. Readable primary endpoints: `results/verifier_signal_v1/local_cpu/summary.md`. Numerical agreement: `results/verifier_signal_v1/local_cpu/local_validation.json`. No plots were generated.
