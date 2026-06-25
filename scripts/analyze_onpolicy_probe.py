#!/usr/bin/env python3
"""Stage 1 verdict: does the probe separate the model's OWN correct vs incorrect steps?

Takes the encoded on-policy step items (encode_prm800k_forks.py over the
generate_onpolicy_steps.py output) plus, optionally, their confidence sidecar, and the
probe, then reports whether the probe separates steps of correct vs incorrect
trajectories among on-policy generations (where perplexity is uniformly low).

Two readouts:
  - STEP-level AUC: pooled steps, label = trajectory-incorrect, with a bootstrap CI.
  - TRAJECTORY-level AUC: mean probe score per trajectory vs its correctness (less
    label noise than the per-step outcome label).

The decisive sanity it also prints: on-policy generated steps should have much LOWER
NLL than the human fork negatives -- confirming the perplexity confound really is
removed in this experiment.

Inputs:
  --h           encoded on-policy items hidden states (_h.npy)
  --meta        their _meta.jsonl (carries role/label/fork_id via item passthrough)
  --items       {stem}_items.jsonl from generate_onpolicy_steps.py (labels + traj id)
  --probe       linear_probe.pt
  [--confidence onpolicy confidence sidecar jsonl]
  [--fork_neg_confidence fork confidence jsonl, for the NLL contrast]
  --out         output dir
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Defined locally (not imported from inspect_margin_drivers) so the analysis has NO
# matplotlib dependency: stage 4 must run in any venv, plots are optional/guarded.


def read_jsonl(path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_probe(path, hidden_dim: int):
    import torch
    obj = torch.load(path, map_location="cpu")
    state = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    wkey = next(k for k in state if k.endswith("fc.weight") or k.endswith("weight"))
    bkey = next((k for k in state if k.endswith("fc.bias") or k.endswith("bias")), None)
    w = np.asarray(state[wkey], dtype=np.float64).reshape(-1)
    if w.shape[0] != hidden_dim:
        raise ValueError(f"probe dim {w.shape[0]} != hidden {hidden_dim}")
    b = float(np.asarray(state[bkey]).reshape(-1)[0]) if bkey else 0.0
    return w, b


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """AUC of score for the positive class label==1 (incorrect)."""
    order = np.argsort(score)
    ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    n1 = label.sum(); n0 = len(label) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[label == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _f1_at(score, label, thr):
    """F1/precision/recall for the rule 'predict incorrect (1) iff score >= thr'."""
    pred = (score >= thr).astype(int)
    tp = int(((pred == 1) & (label == 1)).sum())
    fp = int(((pred == 1) & (label == 0)).sum())
    fn = int(((pred == 0) & (label == 1)).sum())
    if tp == 0:
        return 0.0, 0.0, 0.0
    prec = tp / (tp + fp); rec = tp / (tp + fn)
    return 2 * prec * rec / (prec + rec), prec, rec


def oracle_f1(score, label):
    """Max-F1 threshold on this set (a ceiling; peeks at labels). Vectorized sweep.

    Returns (f1, thr, precision, recall) where the rule is score >= thr -> incorrect.
    """
    order = np.argsort(-score)
    l = label[order].astype(float)
    P = l.sum()
    if P == 0 or P == len(l):
        return float("nan"), float("nan"), float("nan"), float("nan")
    tp = np.cumsum(l)
    fp = np.cumsum(1 - l)
    prec = tp / (tp + fp)
    rec = tp / P
    f1 = np.where(tp > 0, 2 * prec * rec / (prec + rec + 1e-12), 0.0)
    k = int(np.argmax(f1))
    thr = float(score[order][k])
    return float(f1[k]), thr, float(prec[k]), float(rec[k])


def val_selected_f1(score, label, seed=0):
    """Pick the F1-max threshold on a random val half, evaluate F1 on the test half.

    Returns (f1_test, thr, precision_test, recall_test, f1_val). The deployable number.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(label))
    half = len(label) // 2
    vi, ti = perm[:half], perm[half:]
    if label[vi].sum() in (0, len(vi)) or label[ti].sum() in (0, len(ti)):
        return (float("nan"),) * 5
    f1_val, thr, _, _ = oracle_f1(score[vi], label[vi])
    f1_t, p_t, r_t = _f1_at(score[ti], label[ti], thr)
    return f1_t, thr, p_t, r_t, f1_val


def bootstrap_auc_ci(score, label, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(label)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        l = label[idx]
        if l.sum() == 0 or l.sum() == n:
            continue
        aucs.append(auc(score[idx], l))
    if not aucs:
        return float("nan"), float("nan")
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def trajectory_scores(fork_ids, score, label):
    """Mean probe score per trajectory and its (constant) correctness label."""
    by: dict[str, list] = {}
    for fid, s, l in zip(fork_ids, score, label):
        by.setdefault(fid, [[], l])[0].append(s)
    tids = list(by.keys())
    tscore = np.array([np.mean(by[t][0]) for t in tids])
    tlabel = np.array([by[t][1] for t in tids])
    return tscore, tlabel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", required=True, type=Path)
    ap.add_argument("--meta", required=True, type=Path)
    ap.add_argument("--items", required=True, type=Path)
    ap.add_argument("--probe", required=True, type=Path)
    ap.add_argument("--confidence", type=Path, default=None)
    ap.add_argument("--fork_neg_confidence", type=Path, default=None)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--no_plots", action="store_true")
    args = ap.parse_args()
    (args.out / "plots").mkdir(parents=True, exist_ok=True)

    H = np.load(args.h).astype(np.float64)
    meta = read_jsonl(args.meta)
    items = {it["item_uid"]: it for it in read_jsonl(args.items)}
    w, b = load_probe(args.probe, H.shape[1])

    rows = [m for m in meta if items.get(m["item_uid"]) is not None]
    idx = np.array([m["row"] for m in rows])
    label = np.array([int(items[m["item_uid"]]["label"]) for m in rows])
    fork_ids = [items[m["item_uid"]]["fork_id"] for m in rows]
    score = H[idx] @ w + b
    n = len(rows)
    if label.sum() == 0 or label.sum() == n:
        sys.exit("[onpolicy] need both correct and incorrect trajectories; "
                 "got all one class (check generation diversity / temperature).")

    step_auc = auc(score, label)
    lo, hi = bootstrap_auc_ci(score, label)
    tscore, tlabel = trajectory_scores(fork_ids, score, label)
    traj_auc = auc(tscore, tlabel)
    tlo, thi = bootstrap_auc_ci(tscore, tlabel)

    # Headline metric (per Djalil): F1 at a threshold. Oracle = ceiling (peeks at
    # labels); val-selected = deployable (threshold frozen on a val half). F1 is NOT
    # prevalence-invariant, so only compare it across runs at the SAME base rate.
    s_orf1, s_orthr, s_orp, s_orr = oracle_f1(score, label)
    s_vf1, s_vthr, s_vp, s_vr, _ = val_selected_f1(score, label)
    t_orf1, t_orthr, t_orp, t_orr = oracle_f1(tscore, tlabel)
    t_vf1, t_vthr, t_vp, t_vr, _ = val_selected_f1(tscore, tlabel)

    metrics = {
        "n_steps": int(n),
        "n_trajectories": int(len(tlabel)),
        "incorrect_step_rate": float(label.mean()),
        "incorrect_traj_rate": float(tlabel.mean()),
        "step_f1_oracle": round(s_orf1, 4),
        "step_f1_oracle_pr": [round(s_orp, 4), round(s_orr, 4)],
        "step_f1_valselected": round(s_vf1, 4),
        "step_f1_valselected_pr": [round(s_vp, 4), round(s_vr, 4)],
        "trajectory_f1_oracle": round(t_orf1, 4),
        "trajectory_f1_oracle_pr": [round(t_orp, 4), round(t_orr, 4)],
        "trajectory_f1_valselected": round(t_vf1, 4),
        "trajectory_f1_valselected_pr": [round(t_vp, 4), round(t_vr, 4)],
        "step_auc": round(step_auc, 4),
        "step_auc_ci95": [round(lo, 4), round(hi, 4)],
        "trajectory_auc": round(traj_auc, 4),
        "trajectory_auc_ci95": [round(tlo, 4), round(thi, 4)],
    }

    # perplexity-removed sanity: on-policy NLL vs fork negatives
    if args.confidence is not None and args.confidence.exists():
        crows = [r for r in read_jsonl(args.confidence) if r.get("nll_mean") is not None]
        on_nll = np.array([r["nll_mean"] for r in crows], dtype=float)
        metrics["onpolicy_nll_mean"] = round(float(on_nll.mean()), 4)
        if args.fork_neg_confidence is not None and args.fork_neg_confidence.exists():
            frows = [r for r in read_jsonl(args.fork_neg_confidence)
                     if r.get("role") == "negative" and r.get("nll_mean") is not None]
            neg_nll = np.array([r["nll_mean"] for r in frows], dtype=float)
            metrics["fork_negative_nll_mean"] = round(float(neg_nll.mean()), 4)
            metrics["nll_drop_onpolicy_vs_fork_neg"] = round(
                float(neg_nll.mean() - on_nll.mean()), 4)

    # trivial "predict-all-incorrect" F1 = 2*br/(1+br): the strawman F1 must beat.
    step_triv = 2 * label.mean() / (1 + label.mean())
    traj_triv = 2 * tlabel.mean() / (1 + tlabel.mean())
    metrics["step_f1_trivial_allpos"] = round(float(step_triv), 4)
    metrics["trajectory_f1_trivial_allpos"] = round(float(traj_triv), 4)

    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    verdict = ("probe SEPARATES the model's own correct/incorrect steps "
               "(correctness signal, not surprise/distribution)"
               if traj_auc > 0.6 else
               "probe does NOT separate on-policy steps -> may be a distribution detector")
    L = [f"# Stage 1: probe on the model's OWN generations\n",
         f"- on-policy steps: {n}  ({label.mean():.1%} from incorrect trajectories)",
         f"- trajectories: {len(tlabel)}  ({tlabel.mean():.1%} incorrect)\n",
         f"## Does the probe separate correct vs incorrect (low-perplexity) own steps?",
         f"Headline = F1 (pos=incorrect). Trivial 'predict-all-incorrect' baseline must be beaten.\n",
         f"| level | F1 val-selected | F1 oracle | trivial all-pos | AUC (95% CI) |",
         f"|---|---|---|---|---|",
         f"| step | **{s_vf1:.3f}** (P {s_vp:.2f}/R {s_vr:.2f}) | {s_orf1:.3f} | "
         f"{step_triv:.3f} | {step_auc:.3f} ({lo:.3f}-{hi:.3f}) |",
         f"| trajectory | **{t_vf1:.3f}** (P {t_vp:.2f}/R {t_vr:.2f}) | {t_orf1:.3f} | "
         f"{traj_triv:.3f} | {traj_auc:.3f} ({tlo:.3f}-{thi:.3f}) |",
         f"\n-> {verdict}\n"]
    if "nll_drop_onpolicy_vs_fork_neg" in metrics:
        L += ["## Perplexity-removed sanity (the manipulation worked)",
              f"- on-policy step NLL = {metrics['onpolicy_nll_mean']:.3f} nats",
              f"- fork *negative* NLL = {metrics['fork_negative_nll_mean']:.3f} nats",
              f"- drop = {metrics['nll_drop_onpolicy_vs_fork_neg']:+.3f}  "
              "(own steps are far less surprising, yet the probe still separates them)\n"]
    L += ["## Caveat",
          "Outcome labels are trajectory-level (a wrong-answer solution may contain "
          "correct early steps), so the step-level AUC is a lower bound; the "
          "trajectory-mean AUC has less label noise."]
    (args.out / "onpolicy_report.md").write_text("\n".join(L))

    if not args.no_plots:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            GREEN, RED = "#2c7fb8", "#e6550d"
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            ax.hist(tscore[tlabel == 0], bins=30, alpha=0.6, color=GREEN, label="correct traj")
            ax.hist(tscore[tlabel == 1], bins=30, alpha=0.6, color=RED, label="incorrect traj")
            ax.set_xlabel("mean probe score (higher=incorrect)")
            ax.set_title(f"on-policy trajectory scores (AUC {traj_auc:.3f})")
            ax.legend(); fig.tight_layout()
            fig.savefig(args.out / "plots" / "onpolicy_traj_scores.png", dpi=150)
            plt.close(fig)
        except Exception as e:
            print(f"[onpolicy] plot skipped: {e}")

    print(f"[onpolicy] traj F1 val={t_vf1:.3f} oracle={t_orf1:.3f} "
          f"trivial={traj_triv:.3f}  (AUC {traj_auc:.3f} {tlo:.3f}-{thi:.3f})")
    print(f"[onpolicy] step F1 val={s_vf1:.3f} oracle={s_orf1:.3f} trivial={step_triv:.3f}")
    print(f"[onpolicy] -> {verdict}")
    print(f"[onpolicy] wrote {args.out}")


if __name__ == "__main__":
    main()
