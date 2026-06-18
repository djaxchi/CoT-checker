"""Embed the 200 tagged first-error steps and colour them by failure mode.

Takes the verified label set (failure_labels_final.jsonl), pulls each step's raw
7B hidden state via the loader, embeds with UMAP(cosine), and renders an
interactive scatter coloured by failure mode (dropdown also recolours by subset /
detected). Symbol encodes probe detected(o) / missed(x); hover shows the step
text, probe score and the judge rationale.

Because raw 7B geometry is topic-dominated, we render TWO panels:
  - raw            standardized hidden states
  - decorrelated   per-subset mean removed (topic whitened out)
so we can see whether any failure-mode structure survives topic removal.

No PCA bottleneck: UMAP and the decoder both run on the FULL hidden state
(all 3,584 dims), because a distributed failure-mode code can live in low-variance
directions that a top-k PCA would discard. The linear probe also reads all dims,
so this is the faithful test. (Pass --pca N>0 to insert a PCA(N) step for
comparison.)

We also report failure-mode-classifiability: 5-fold logistic-regression accuracy
of predicting the failure mode from the FULL hidden state, vs the majority-class
floor. This is the quantitative companion to the visual: if modes are visually
mixed, accuracy sits near the floor.

Output (results/s3_first_error/):
  - failure_scatter.html           two UMAP panels, colour-by dropdown
  - failure_mode_classifiability.csv

Usage:
    python scripts/analysis/s3_failure_mode_scatter.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from umap import UMAP

from src.data.processbench_probe_data import DEFAULT_RUN_DIR, SUBSETS, load_all
from src.eval.failure_taxonomy import FAILURE_MODES

ROOT = Path("results/s3_first_error")

# distinct colour per failure mode (10 incl. other)
MODE_COLORS = {
    "arithmetic_error": "#e6194B",
    "algebraic_transformation_error": "#f58231",
    "variable_or_entity_binding_error": "#ffe119",
    "quantity_or_unit_mismatch": "#a9a9a9",
    "unsupported_premise": "#3cb44b",
    "constraint_violation": "#42d4f4",
    "logical_inference_error": "#4363d8",
    "goal_drift": "#911eb4",
    "post_hoc_reasoning": "#f032e6",
    "other": "#000000",
}


def wrap(text: str, width: int = 90, max_chars: int = 600) -> str:
    text = (text or "")[:max_chars].replace("\n", " ")
    return "<br>".join(text[i : i + width] for i in range(0, len(text), width))


def classifiability(X: np.ndarray, y: np.ndarray) -> float:
    clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
    # only keep classes with >=5 members so 5-fold CV is defined
    keep = np.isin(y, [m for m in set(y) if (y == m).sum() >= 5])
    if keep.sum() < 10 or len(set(y[keep])) < 2:
        return float("nan")
    return float(cross_val_score(clf, X[keep], y[keep], cv=5, scoring="accuracy").mean())


def embed(X: np.ndarray, pca: int, seed: int) -> np.ndarray:
    if pca and pca > 0:
        X = PCA(min(pca, X.shape[0] - 1, X.shape[1]), random_state=seed).fit_transform(X)
    return UMAP(n_components=2, random_state=seed, metric="cosine",
                n_neighbors=15, min_dist=0.1).fit_transform(X)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=Path, default=DEFAULT_RUN_DIR)
    ap.add_argument("--labels", default="failure_labels_final.jsonl")
    ap.add_argument("--out_dir", type=Path, default=ROOT)
    ap.add_argument("--pca", type=int, default=0,
                    help="0 = use full hidden state (default); N>0 inserts PCA(N)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    labels = {json.loads(l)["sample_id"]: json.loads(l)
              for l in (ROOT / args.labels).read_text().splitlines() if l}
    rationale = {sid: r.get("rationale", "") for sid, r in labels.items()}
    print(f"[load] {len(labels)} tagged steps from {args.labels}")

    print("[load] reading run hidden states ...")
    d = load_all(args.run_dir, with_text=True)
    fe = d.is_first_error & ~d.skipped
    idx_fe = np.where(fe)[0]
    sid_of = {i: f"{d.trace_id[i]}#s{int(d.step_idx[i])}" for i in idx_fe}

    keep = [i for i in idx_fe if sid_of[i] in labels]
    miss = [s for s in labels if s not in {sid_of[i] for i in keep}]
    if miss:
        print(f"[warn] {len(miss)} labelled ids not found in run: {miss[:5]}")
    keep = np.array(keep)
    print(f"[join] matched {len(keep)}/{len(labels)} tagged steps to hidden states")

    H = d.hidden[keep]
    sub = d.subset[keep]
    detected = (d.pred_first_error == d.gold_first_error)[keep]
    score = d.score[keep]
    text = [d.step_text[i] for i in keep]
    mode = np.array([labels[sid_of[i]]["failure_mode"] for i in keep])
    sids = [sid_of[i] for i in keep]

    # ---- two feature spaces: raw-standardized and subset-decorrelated -----
    Xs = StandardScaler().fit_transform(H)
    Xd = Xs.copy()
    for s in SUBSETS:
        m = sub == s
        if m.any():
            Xd[m] -= Xs[m].mean(axis=0, keepdims=True)

    space_note = f"PCA({args.pca})" if args.pca and args.pca > 0 else "full 3,584 dims"
    print(f"[embed] UMAP raw ({space_note}) ...")
    emb_raw = embed(Xs, args.pca, args.seed)
    print(f"[embed] UMAP decorrelated ({space_note}) ...")
    emb_dec = embed(Xd, args.pca, args.seed)

    # ---- can we predict failure mode from the geometry? ------------------
    # decode on the FULL hidden state (same dims the linear probe reads), so a
    # distributed code in low-variance directions is not thrown away.
    _, counts = np.unique(mode, return_counts=True)
    floor = counts.max() / len(mode)
    acc_raw = classifiability(Xs, mode)
    acc_dec = classifiability(Xd, mode)
    print(f"\n[classifiability] predict failure mode from {space_note} "
          f"(5-fold logreg, balanced):")
    print(f"  raw            = {acc_raw:.3f}")
    print(f"  decorrelated   = {acc_dec:.3f}")
    print(f"  majority floor = {floor:.3f}")
    with (args.out_dir / "failure_mode_classifiability.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["space", "accuracy", "majority_floor", "n", "n_classes"])
        w.writerow(["raw", round(acc_raw, 4), round(floor, 4), len(mode), len(set(mode))])
        w.writerow(["decorrelated", round(acc_dec, 4), round(floor, 4), len(mode), len(set(mode))])

    # ---- interactive scatter --------------------------------------------
    hover = [
        f"<b>{sids[i]}</b> ({sub[i]})"
        f"<br>mode=<b>{mode[i]}</b> | score={score[i]:.3f} "
        f"detected={'Y' if detected[i] else 'N'}"
        f"<br>rationale: {wrap(rationale.get(sids[i], ''), max_chars=240)}"
        f"<br>---<br>{wrap(text[i])}"
        for i in range(len(keep))
    ]

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("raw (topic-dominated)",
                                        "subset-decorrelated (topic removed)"))

    # each "colour-by" scheme is a block of traces (mode/subset/detection) x
    # (detected,missed) x (raw,dec panel). dropdown toggles whole blocks.
    groups: list[tuple[str, list[int]]] = []

    def add_scheme(label, cats, cat_of_point, color_of):
        idxs = []
        for col, emb in [(1, emb_raw), (2, emb_dec)]:
            for cat in cats:
                for det_flag, sym in [(True, "circle"), (False, "x")]:
                    m = (cat_of_point == cat) & (detected == det_flag)
                    if not m.any():
                        continue
                    fig.add_trace(go.Scattergl(
                        x=emb[m, 0], y=emb[m, 1], mode="markers",
                        name=f"{cat} {'det' if det_flag else 'miss'}",
                        legendgroup=str(cat),
                        showlegend=(col == 1),
                        marker=dict(size=8, symbol=sym, color=color_of(cat),
                                    line=dict(width=0.5, color="white")),
                        text=[hover[j] for j in np.where(m)[0]],
                        hoverinfo="text", visible=False,
                    ), row=1, col=col)
                    idxs.append(len(fig.data) - 1)
        groups.append((label, idxs))

    modes_present = [m for m in FAILURE_MODES if (mode == m).any()]
    add_scheme("by failure mode", modes_present, mode,
               lambda c: MODE_COLORS.get(c, "#888"))
    add_scheme("by subset", list(SUBSETS), sub,
               lambda c: {"gsm8k": "#1b9e77", "math": "#d95f02",
                          "olympiadbench": "#7570b3", "omnimath": "#e7298a"}.get(c, "#888"))
    add_scheme("by detected/missed", [True, False], detected,
               lambda c: "#1a9850" if c else "#762a83")

    for ti in groups[0][1]:
        fig.data[ti].visible = True

    buttons = []
    for label, idxs in groups:
        vis = [False] * len(fig.data)
        for ti in idxs:
            vis[ti] = True
        buttons.append(dict(label=label, method="update",
                            args=[{"visible": vis},
                                  {"title": f"Tagged first-error steps - coloured {label}"}]))

    fig.update_layout(
        updatemenus=[dict(active=0, buttons=buttons, x=0.0, xanchor="left",
                          y=1.15, yanchor="top")],
        title=f"Tagged first-error steps - coloured by failure mode "
              f"(n={len(keep)}; symbol o=detected x=missed)",
        width=1500, height=760, template="plotly_white",
        legend=dict(title="", itemsizing="constant"),
    )
    out = args.out_dir / "failure_scatter.html"
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"\n[done] wrote -> {out}")
    print("  dropdown (top-left) switches colouring; left=raw right=decorrelated")


if __name__ == "__main__":
    main()
