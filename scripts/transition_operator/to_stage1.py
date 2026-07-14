"""transition_operator_v0 Stage 1c: baseline decodability + retrieval, kill gate.

Loads reps_layer_<L>.npz + rows.json (to_extract_baselines) and step_labels.parquet
(to_labels), builds the five raw baselines, and evaluates each at MATCHED dimension
(PCA to --d_z) so none wins by capacity:
  S_t | S_t - S_{t-1} | [S_{t-1}; S_t] | mean-pool(H_t) | max-pool(H_t)

Two label sets: op_symbolic (5-class, precision-first) and tag_top (14-class,
recall-first). Two metrics, both PROBLEM-DISJOINT (grouped by question):
  a. operation decodability     logistic-regression macro-F1, GroupKFold, vs majority
  b. cross-problem retrieval     cosine precision@k with query and gallery from
                                 different problems, vs label-prevalence chance

KILL GATE (plan section 9): if the best raw baseline already reaches ~90% of the
achievable ceiling on BOTH metrics, a learned operator needs a different justification.
The ceiling is proxied two ways and reported; the gate is advisory, printed as a flag.

  python scripts/transition_operator/to_stage1.py --run_dir runs/transition_operator --layer 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


def build_baselines(z: dict) -> dict[str, np.ndarray]:
    return {
        "S_t": z["S_last"],
        "delta": z["S_last"] - z["S_prev"],
        "concat": np.concatenate([z["S_prev"], z["S_last"]], axis=1),
        "meanpool": z["H_mean"],
        "maxpool": z["H_max"],
    }


def matched_pca(X: np.ndarray, d: int, seed: int) -> np.ndarray:
    Xs = StandardScaler().fit_transform(X)
    d = min(d, Xs.shape[1], Xs.shape[0] - 1)
    return PCA(n_components=d, random_state=seed).fit_transform(Xs)


def decodability(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                 seed: int) -> dict:
    """Macro-F1 of a grouped-CV logistic probe vs the majority-class baseline."""
    classes, counts = np.unique(y, return_counts=True)
    majority = f1_score(y, np.full_like(y, classes[counts.argmax()]),
                        average="macro", labels=classes)
    n_splits = int(min(5, np.unique(groups).size,
                       np.min(np.bincount(pd.factorize(y)[0]))))
    if n_splits < 2:
        return {"macro_f1": float("nan"), "majority_macro_f1": float(majority),
                "n": int(len(y)), "n_classes": int(len(classes))}
    preds = np.empty_like(y)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        clf = LogisticRegression(max_iter=2000, C=1.0,
                                 class_weight="balanced")
        clf.fit(X[tr], y[tr])
        preds[te] = clf.predict(X[te])
    return {"macro_f1": float(f1_score(y, preds, average="macro")),
            "majority_macro_f1": float(majority),
            "n": int(len(y)), "n_classes": int(len(classes))}


def cross_problem_retrieval(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                            k: int = 5) -> dict:
    """Cosine precision@k where neighbors must come from a DIFFERENT problem."""
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    sim = Xn @ Xn.T
    same_problem = groups[:, None] == groups[None, :]
    sim[same_problem] = -np.inf
    np.fill_diagonal(sim, -np.inf)
    order = np.argsort(-sim, axis=1)[:, :k]
    hits = (y[order] == y[:, None])
    valid = np.isfinite(np.take_along_axis(sim, order, axis=1))
    prec = (hits & valid).sum() / np.maximum(valid.sum(), 1)
    _, counts = np.unique(y, return_counts=True)
    chance = float((counts / counts.sum() ** 1) @ (counts / counts.sum()))  # sum p_c^2
    return {"precision_at_k": float(prec), "chance": chance, "k": k,
            "n": int(len(y))}


def evaluate_label(baselines: dict, labels: pd.Series, groups: np.ndarray,
                   d_z: int, seed: int) -> dict:
    mask = labels.notna().to_numpy()
    y = labels[mask].to_numpy().astype(str)
    g = groups[mask]
    out = {}
    for name, X in baselines.items():
        Xp = matched_pca(X[mask], d_z, seed)
        out[name] = {"decodability": decodability(Xp, y, g, seed),
                     "retrieval": cross_problem_retrieval(Xp, y, g)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", type=Path, default=Path("runs/transition_operator"))
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--d_z", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--kill_frac", type=float, default=0.90)
    args = ap.parse_args()

    s1 = args.run_dir / "stage1"
    z = dict(np.load(s1 / f"reps_layer_{args.layer}.npz"))
    rows = pd.DataFrame(json.loads((s1 / "rows.json").read_text()))
    labels = pd.read_parquet(s1 / "step_labels.parquet")
    # align labels to rep row order on (fork_id, branch)
    key = ["fork_id", "branch"]
    labels = rows.merge(labels, on=key + ["question", "step_index"], how="left")
    assert len(labels) == len(rows) == z["S_prev"].shape[0], "row/label misalignment"
    groups = pd.factorize(labels["question"])[0]

    baselines = build_baselines(z)
    result = {"layer": args.layer, "d_z": args.d_z,
              "n_transitions": int(len(rows))}
    for label_name in ("op_symbolic", "tag_top"):
        col = labels[label_name].where(labels[label_name] != "NONE") \
            if label_name == "tag_top" else labels[label_name]
        result[label_name] = evaluate_label(baselines, col, groups,
                                            args.d_z, args.seed)

    # kill gate: best baseline vs a proxy ceiling.
    # decodability ceiling proxy = 1.0 macro-F1; retrieval ceiling proxy = 1.0.
    # advisory: flag if best baseline already >= kill_frac of that on BOTH.
    def best(metric_path, label):
        vals = [result[label][b][metric_path[0]][metric_path[1]]
                for b in baselines]
        vals = [v for v in vals if v == v]  # drop nan
        return max(vals) if vals else float("nan")

    gate = {}
    for label in ("op_symbolic", "tag_top"):
        bd = best(("decodability", "macro_f1"), label)
        br = best(("retrieval", "precision_at_k"), label)
        gate[label] = {
            "best_decodability_macro_f1": bd,
            "best_retrieval_p_at_k": br,
            "kill_triggered": bool(bd >= args.kill_frac and br >= args.kill_frac),
        }
    result["kill_gate"] = gate

    s1.mkdir(exist_ok=True)
    (s1 / f"stage1_baselines_layer_{args.layer}.json").write_text(
        json.dumps(result, indent=2))

    print(f"\n=== Stage 1 baselines (L{args.layer}, d_z={args.d_z}) ===")
    for label in ("op_symbolic", "tag_top"):
        print(f"\n[{label}]")
        for b in baselines:
            dd = result[label][b]["decodability"]
            rr = result[label][b]["retrieval"]
            print(f"  {b:9s} decod macroF1 {dd['macro_f1']:.3f} "
                  f"(maj {dd['majority_macro_f1']:.3f}, n={dd['n']}, "
                  f"{dd['n_classes']}cls)  retr p@{rr['k']} {rr['precision_at_k']:.3f} "
                  f"(chance {rr['chance']:.3f})")
        g = gate[label]
        print(f"  KILL best decod {g['best_decodability_macro_f1']:.3f}, "
              f"best retr {g['best_retrieval_p_at_k']:.3f} -> "
              f"{'TRIGGERED' if g['kill_triggered'] else 'not triggered'}")
    print(f"\n[stage1] wrote {s1}/stage1_baselines_layer_{args.layer}.json")


if __name__ == "__main__":
    main()
