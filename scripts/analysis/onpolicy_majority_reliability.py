#!/usr/bin/env python3
"""Does a hidden-state score help predict whether the vote is already right?

Best-of-N asks the verifier to out-rank the vote, and it does not (REPORT.md
§20.2). This asks the decision the null leaves open: conditional on the votes
already cast, is the majority answer correct? That is what an abstain, escalate
or draw-more-samples policy actually needs.

The comparison is always against a vote-only control fitted the same way, because
agreement alone predicts correctness well and any model reading both would
otherwise take credit for it. Everything is cross-validated with whole question
texts held out, since the 300 evaluation ids are 284 distinct questions, and the
paired interval resamples question texts too.

CPU only, no fitting of any representation and no generation: the features are
read off the saved score files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.analysis.onpolicy_tiebreak_baselines import (  # noqa: E402
    candidates, fitted_questions, load_pool,
)
from src.analysis.onpolicy_reliability import (  # noqa: E402
    FEATURE_SETS, f1_at, grouped_folds, problem_features, trivial_f1,
)
from src.analysis.onpolicy_tiebreak import (  # noqa: E402
    answer_groups, cluster_bootstrap, top_bloc,
)


def majority_label(rows: list[dict], bloc: list[int]) -> int:
    """Is the answer the vote points at correct?

    On a tie the bloc spans several answers and there is no single majority
    answer, so the label is the accuracy of picking among them at random,
    thresholded: these problems are exactly where the vote has nothing to say and
    a reliability model should be saying so.
    """
    if not bloc:
        return 0
    return int(np.mean([bool(rows[i]["correct"]) for i in bloc]) > 0.5)


def fit_predict(X: np.ndarray, y: np.ndarray, folds: list[np.ndarray],
                seed: int) -> np.ndarray:
    """Out-of-fold probabilities, standardised inside each fold."""
    oof = np.zeros(len(y))
    for k, test in enumerate(folds):
        train = np.concatenate([f for j, f in enumerate(folds) if j != k])
        if len(set(y[train])) < 2:
            oof[test] = float(y[train].mean())
            continue
        sc = StandardScaler().fit(X[train])
        model = LogisticRegression(max_iter=2000, random_state=seed)
        model.fit(sc.transform(X[train]), y[train])
        oof[test] = model.predict_proba(sc.transform(X[test]))[:, 1]
    return oof


def evaluate(oof: np.ndarray, y: np.ndarray, folds: list[np.ndarray]) -> dict:
    """AUROC, plus F1 at a threshold chosen out of fold and at the oracle ceiling.

    The threshold is picked on the other folds' predictions, never on the fold it
    scores, so the val-selected F1 stays honest; the oracle row is the ceiling.
    """
    grid = np.linspace(0.05, 0.95, 91)
    sel = np.zeros(len(y))
    for k, test in enumerate(folds):
        train = np.concatenate([f for j, f in enumerate(folds) if j != k])
        t = max(grid, key=lambda t: f1_at(y[train], oof[train], t))
        sel[test] = oof[test] >= t
    return {
        "auroc": float(roc_auc_score(y, oof)) if len(set(y)) == 2 else None,
        "f1_val_selected": f1_at(y, sel.astype(float), 0.5),
        "f1_oracle": max(f1_at(y, oof, t) for t in grid),
        "f1_trivial": trivial_f1(y),
        "prevalence": float(y.mean()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--onpolicy_root", type=Path,
                   default=ROOT / "cot-checker-results/onpolicy_v1")
    p.add_argument("--reprobe_root", type=Path,
                   default=ROOT / "cot-checker-results/reprobe_v1")
    p.add_argument("--out_dir", type=Path,
                   default=ROOT / "results/onpolicy_reliability")
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit_cells", type=int, default=None)
    args = p.parse_args()

    pool, question = load_pool(args.onpolicy_root)
    fitted = fitted_questions(
        args.reprobe_root / "reprobe_train_judge_traces.jsonl",
        sorted(args.reprobe_root.glob("labels.shard*.jsonl")), 0, 0.15)
    clean = {pid for pid, q in question.items() if q not in fitted}

    score_files = []
    for arm, root in (("frozen", args.onpolicy_root / "grid"),
                      ("retrained", args.reprobe_root / "grid")):
        for style in ("verifier", "generation"):
            score_files += [(arm, style, f) for f in
                            sorted(root.rglob(f"*scores*{style}*.jsonl"))]
    if args.limit_cells:
        score_files = score_files[:args.limit_cells]

    cells, digests = [], {}
    for arm, style, path in score_files:
        by_problem = candidates(path, pool)["by_problem"]
        pids = sorted(by_problem)
        feats, labels = [], []
        for pid in pids:
            rows = by_problem[pid]
            bloc, _ = top_bloc(answer_groups(rows))
            feats.append(problem_features(rows, bloc))
            labels.append(majority_label(rows, bloc))
        y_all = np.array(labels)
        row = {"arm": arm, "style": style, "cell": path.parent.name,
               "path": str(path.relative_to(ROOT))}
        for subset, keep in (("all", set(pids)), ("clean", clean & set(pids))):
            idx = [i for i, pid in enumerate(pids) if pid in keep]
            y = y_all[idx]
            groups = [question[pids[i]] for i in idx]
            folds = grouped_folds(groups, args.n_folds, args.seed)
            res, oof_store = {}, {}
            for name, keys in FEATURE_SETS.items():
                X = np.array([[feats[i][k] for k in keys] for i in idx], dtype=float)
                oof = fit_predict(X, y, folds, args.seed)
                oof_store[name] = oof
                res[name] = evaluate(oof, y, folds)
            # The only comparison that answers the question: what the scores add
            # once the votes are already in the model.
            for name in ("vote+score", "vote+length+score"):
                paired = ((oof_store[name] > 0.5).astype(float) == y).astype(float) \
                    - ((oof_store["vote"] > 0.5).astype(float) == y).astype(float)
                res[name]["accuracy_vs_vote_only"] = cluster_bootstrap(paired, groups)
            row[subset] = {"n": len(idx), "n_questions": len(set(groups)),
                           "feature_sets": res}
        cells.append(row)
        digests[row["path"]] = hashlib.sha256(path.read_bytes()).hexdigest()
        a = row["all"]["feature_sets"]
        print(f"[cell] {arm}/{style}/{path.parent.name} "
              f"vote AUROC={a['vote']['auroc']:.3f} "
              f"vote+score={a['vote+score']['auroc']:.3f} "
              f"score-only={a['score']['auroc']:.3f}", flush=True)

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "conditional reliability fits on saved scores; no representation "
                 "training, no generation",
        "target": "is the majority answer correct",
        "n_cells": len(cells),
        "protocol": {"n_folds": args.n_folds, "seed": args.seed,
                     "cv_unit": "canonical question text",
                     "model": "logistic regression, standardised per fold"},
        "cells": cells,
        "source_sha256": digests,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "reliability_results.json").write_text(
        json.dumps(report, indent=2, allow_nan=False))
    (args.out_dir / "reliability_table.md").write_text(render(report))
    print(render(report))
    print(f"[wrote] {args.out_dir}/reliability_results.json")
    print(f"[wrote] {args.out_dir}/reliability_table.md")


def render(report: dict) -> str:
    lines = []
    for subset in ("all", "clean"):
        first = report["cells"][0][subset]
        lines.append(f"\n### P(majority answer correct), {subset} problems "
                     f"(n={first['n']} over {first['n_questions']} question texts, "
                     f"prevalence {first['feature_sets']['vote']['prevalence']:.3f}, "
                     f"trivial F1 {first['feature_sets']['vote']['f1_trivial']:.3f})\n")
        lines.append("| features | AUROC (mean / max over cells) | F1 val-selected "
                     "| F1 oracle | accuracy vs vote-only | 95% CI |")
        lines.append("|---|---|---|---|---|---|")
        for name in FEATURE_SETS:
            au = np.array([c[subset]["feature_sets"][name]["auroc"]
                           for c in report["cells"]], dtype=float)
            f1 = np.array([c[subset]["feature_sets"][name]["f1_val_selected"]
                           for c in report["cells"]], dtype=float)
            orc = np.array([c[subset]["feature_sets"][name]["f1_oracle"]
                            for c in report["cells"]], dtype=float)
            delta = ci = ""
            if "accuracy_vs_vote_only" in report["cells"][0][subset]["feature_sets"][name]:
                d = np.array([c[subset]["feature_sets"][name]
                              ["accuracy_vs_vote_only"]["delta"]
                              for c in report["cells"]])
                best = max(report["cells"], key=lambda c: c[subset]["feature_sets"]
                           [name]["accuracy_vs_vote_only"]["delta"])
                b = best[subset]["feature_sets"][name]["accuracy_vs_vote_only"]
                delta = f"{d.mean():+.4f} (best cell {b['delta']:+.4f})"
                ci = f"best cell [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]"
            lines.append(f"| {name} | {au.mean():.4f} / {au.max():.4f} | "
                         f"{f1.mean():.4f} | {orc.mean():.4f} | {delta} | {ci} |")
        lines.append("")
        lines.append("The vote-only and score-only rows do not vary with the cell "
                     "except through which trajectories a cell scores; the spread "
                     "across cells is the verifier's, not the vote's.")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
