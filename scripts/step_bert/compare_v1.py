"""Cross-design comparison for step_bert_probe_v1 (v0 pooled steps vs Arm A tokens vs Arm B text).

All three designs score the same on-policy rows in the same (trace_id, step_idx) order,
so contrasts across designs are paired, with the same problem-clustered bootstrap
(2000 resamples for F1, 500 for AUROC) and seed averaging as driver_v1.aggregate.
Writes <out>/compare.{json,md} and <out>/plots/{conditions,contrasts}.png.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import f1_score, roc_auc_score  # noqa: E402

SEEDS = (42, 43, 44)
COLORS = {"v0 pooled steps": "#2a78d6", "A full tokens": "#eb6834", "B ModernBERT text": "#1baf7a"}


def load(fit_dir: Path, name: str):
    fs = [fit_dir / f"final_{name}_s{s}" for s in SEEDS]
    M = [json.loads((f / "metrics.json").read_text()) for f in fs]
    return M, [np.load(f / "probs_onpolicy.npy") for f in fs]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pooled", type=Path, default=Path("data/tamia/step_bert_probe_v0/pooled/pooled_onpolicy_full.npz"))
    p.add_argument("--split", type=Path, default=Path("data/tamia/step_bert_probe_v0/split.json"))
    p.add_argument("--v0", type=Path, default=Path("runs/step_bert_probe_v0/ft"))
    p.add_argument("--v1", type=Path, default=Path("runs/step_bert_probe_v1"))
    p.add_argument("--out", type=Path, default=Path("results/step_bert_probe_v1"))
    p.add_argument("--n_boot", type=int, default=2000)
    a = p.parse_args()

    z = np.load(a.pooled)
    te = np.isin(z["trace_id"], json.loads(a.split.read_text())["test"])
    y, grp = z["y"][te].astype(int), z["problem_id"][te]
    designs = {
        "v0 pooled steps": (a.v0, {"local": "local", "causal": "causal_Wall", "bidir": "bidir_Wall"}),
        "A full tokens": (a.v1 / "tok" / "fits", {"local": "local", "causal": "causal_Wall", "bidir": "bidir_Wall"}),
        "B ModernBERT text": (a.v1 / "txt" / "fits", {"local": "local", "causal": "causal", "bidir": "bidir"}),
    }
    runs = {}
    for d, (fd, conds) in designs.items():
        for c, name in conds.items():
            M, P = load(fd, name)
            runs[(d, c)] = {"th": [m["val_threshold"] for m in M], "p": [q[te] for q in P],
                            "pb": np.array([m["processbench_mean_f1_pb_calib20"] for m in M]),
                            "pb_auc": float(np.mean([m["processbench_mean_known_step_auroc"] for m in M]))}

    def stat(k, idx, kind):
        r = runs[k]
        return float(np.mean([f1_score(y[idx], q[idx] >= t) if kind == "f1" else roc_auc_score(y[idx], q[idx])
                              for q, t in zip(r["p"], r["th"])]))

    uniq, inv = np.unique(grp, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    rng = np.random.default_rng(0)
    boots = [np.concatenate([members[g] for g in rng.integers(len(uniq), size=len(uniq))]) for _ in range(a.n_boot)]
    all_idx = np.arange(len(y))

    table = {f"{d} | {c}": {"test_f1": stat((d, c), all_idx, "f1"), "test_auroc": stat((d, c), all_idx, "auroc"),
                            "pb_f1_calib20": float(runs[(d, c)]["pb"].mean()), "pb_known_step_auroc": runs[(d, c)]["pb_auc"]}
             for d, c in runs}
    contrasts = [
        (("v0 pooled steps", "bidir"), ("v0 pooled steps", "causal")),
        (("A full tokens", "bidir"), ("A full tokens", "causal")),
        (("B ModernBERT text", "bidir"), ("B ModernBERT text", "causal")),
        (("A full tokens", "local"), ("v0 pooled steps", "local")),
        (("A full tokens", "causal"), ("v0 pooled steps", "causal")),
        (("B ModernBERT text", "causal"), ("A full tokens", "causal")),
        (("B ModernBERT text", "bidir"), ("A full tokens", "bidir")),
    ]
    out = {}
    for x, b in contrasts:
        d = {}
        for kind in ("f1", "auroc"):
            bs = [stat(x, idx, kind) - stat(b, idx, kind) for idx in boots[:500 if kind == "auroc" else a.n_boot]]
            d[kind] = {"diff": stat(x, all_idx, kind) - stat(b, all_idx, kind),
                       "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5))}
        pd = runs[x]["pb"] - runs[b]["pb"]
        d["pb_f1_calib20"] = {"diff": float(pd.mean()), "per_seed": pd.tolist()}
        out[f"{x[0]} {x[1]} - {b[0]} {b[1]}"] = d

    a.out.mkdir(parents=True, exist_ok=True)
    prev = float(y.mean())
    (a.out / "compare.json").write_text(json.dumps({"test_prevalence": prev, "always_positive_f1": 2 * prev / (1 + prev),
                                                   "conditions": table, "contrasts": out}, indent=2))
    lines = ["| design | condition | test F1 | test AUROC | PB F1_PB calib-20 | PB known-step AUROC |", "|---|---|---|---|---|---|"]
    lines += [f"| {k.split(' | ')[0]} | {k.split(' | ')[1]} | {r['test_f1']:.3f} | {r['test_auroc']:.3f} | "
              f"{r['pb_f1_calib20']:.3f} | {r['pb_known_step_auroc']:.3f} |" for k, r in table.items()]
    lines += ["", "| contrast | test F1 diff [95% CI] | test AUROC diff [95% CI] | PB calib-20 diff (per seed) |", "|---|---|---|---|"]
    lines += [f"| {k} | {d['f1']['diff']:+.4f} [{d['f1']['lo']:+.4f}, {d['f1']['hi']:+.4f}] | "
              f"{d['auroc']['diff']:+.4f} [{d['auroc']['lo']:+.4f}, {d['auroc']['hi']:+.4f}] | "
              f"{d['pb_f1_calib20']['diff']:+.4f} ({', '.join(f'{v:+.3f}' for v in d['pb_f1_calib20']['per_seed'])}) |"
              for k, d in out.items()]
    (a.out / "compare.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    plots = a.out / "plots"
    plots.mkdir(exist_ok=True)
    conds = ["local", "causal", "bidir"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, key, title, base in ((axes[0], "test_f1", "On-policy test step F1 (val threshold)", 2 * prev / (1 + prev)),
                                 (axes[1], "pb_f1_calib20", "ProcessBench mean F1_PB (calib-20)", None)):
        w = 0.26
        for i, d in enumerate(designs):
            v = [table[f"{d} | {c}"][key] for c in conds]
            bars = ax.bar(np.arange(3) + (i - 1) * w, v, w * 0.92, color=COLORS[d], label=d)
            ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=1, color="#52514e")
        if base is not None:
            ax.axhline(base, color="#52514e", lw=1, ls="--")
            ax.text(2.45, base + 0.005, f"always positive {base:.3f}", ha="right", fontsize=7, color="#52514e")
        ax.set_xticks(np.arange(3), conds)
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e5e4e0", lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylim(0.4, 0.72)
    axes[1].set_ylim(0.2, 0.6)
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=8, loc="upper center", ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(plots / "conditions.png", dpi=160)

    fig, ax = plt.subplots(figsize=(8, 4))
    keys = list(out)
    for i, k in enumerate(keys):
        d = out[k]["f1"]
        ax.errorbar(d["diff"], i, xerr=[[d["diff"] - d["lo"]], [d["hi"] - d["diff"]]], fmt="o", ms=6,
                    color=COLORS[k.split(" bidir")[0].split(" local")[0].split(" causal")[0]], capsize=3, lw=2)
    ax.axvline(0, color="#52514e", lw=1)
    ax.set_yticks(range(len(keys)), keys, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("paired test step F1 difference, 95% problem-clustered bootstrap CI")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#e5e4e0", lw=0.6)
    fig.tight_layout()
    fig.savefig(plots / "contrasts.png", dpi=160)


if __name__ == "__main__":
    main()
