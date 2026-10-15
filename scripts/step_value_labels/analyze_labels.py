#!/usr/bin/env python3
"""What the step_value_labels_v1 risk labels are worth, before any probe is trained.

Reads labeler shards (full, or the light copy without rollout text) and the frozen
label set, and asks four questions, all on CPU:

1. Reliability. Split-half: risk from the even and the odd rollouts of both cuts,
   correlated over steps, stepped up to k = 16 with Spearman-Brown. Test-retest:
   cut 0 is the bare prompt, so the two traces of a problem give two independent
   err(0) estimates of the same quantity. Signal share: 1 - mean(se^2) / var(risk).
2. Null rate of `harmful`. The rollouts of a step's two cuts are exchangeable when
   the step changes nothing, so shuffling them between the cuts gives the rate at
   which the 2-SE rule fires on noise alone, per class.
3. Validity against outcomes. On finished traces, err at the last cut (before the
   answer line) against whether the released trace answered correctly; the error
   rate around the DeepSeek first error (event study); where the largest risk of a
   trace falls relative to that first error.
4. Information beyond validity. Risk AUROC for the DeepSeek and the self labels,
   risk on steps where the two annotators disagree, benign first errors (risk <= 0)
   and harmful steps labeled correct.

Writes <out_dir>/analysis.json, analysis.md, plots, and examples.md when a
tokenizer is available (--tokenizer).
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.analysis.factorized_step_labels import spearman  # noqa: E402
from src.analysis.step_value_labels import step_risks  # noqa: E402


def cut_errors(row: dict) -> dict[int, list[bool]]:
    return {c["cut"]: [not x["correct"] for x in c["rollouts"]] for c in row["cuts"]}


def split_half(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Per step, risk from the even and from the odd rollouts of both its cuts."""
    a, b, tid = [], [], []
    for r in rows:
        e = cut_errors(r)
        ra = {x["step"]: x["risk"] for x in step_risks({c: v[0::2] for c, v in e.items()}, r["n_steps"])}
        rb = {x["step"]: x["risk"] for x in step_risks({c: v[1::2] for c, v in e.items()}, r["n_steps"])}
        for s in sorted(ra):
            a.append(ra[s]); b.append(rb[s]); tid.append(r["trace_id"])
    return np.array(a), np.array(b), tid


def spearman_brown(r: float, factor: float = 2.0) -> float:
    return factor * r / (1 + (factor - 1) * r)


def err0_pairs(rows: list[dict]) -> np.ndarray:
    """(err0 of trace 1, err0 of trace 2) for problems with two labeled traces."""
    by = defaultdict(list)
    for r in rows:
        if r["cuts"][0]["cut"] == 0:
            by[r["problem_key"]].append(r["cuts"][0]["err"])
    return np.array([v[:2] for v in by.values() if len(v) >= 2], float).reshape(-1, 2)


def null_harmful(rows: list[dict], n_perm: int = 20, seed: int = 0) -> dict[tuple[str, int], float]:
    """Per (trace, step), the share of rollout shuffles between its two cuts that the 2-SE rule calls harmful."""
    rng = np.random.default_rng(seed)
    out = {}
    for r in rows:
        e = cut_errors(r)
        for s in range(r["n_steps"]):
            if s not in e or s + 1 not in e:
                continue
            pool, kb = np.array(e[s] + e[s + 1], bool), len(e[s])
            hits = 0
            for _ in range(n_perm):
                p = rng.permutation(pool)
                hits += step_risks({0: p[:kb].tolist(), 1: p[kb:].tolist()}, 1)[0]["harmful"]
            out[(r["trace_id"], s)] = hits / n_perm
    return out


def event_study(rows: list[dict], offsets: range) -> dict[int, list[float]]:
    """err at cut fe + o around the DeepSeek first error fe (cut fe is before it, fe + 1 after)."""
    out = {o: [] for o in offsets}
    for r in rows:
        fe = r.get("first_error_ds")
        if fe is None:
            continue
        e = {c["cut"]: c["err"] for c in r["cuts"]}
        for o in offsets:
            if fe + o in e:
                out[o].append(e[fe + o])
    return out


def argmax_offset(rows: list[dict]) -> list[tuple[int, int]]:
    """(step of the largest risk minus the first error, n_steps) per error trace with a unique positive max."""
    out = []
    for r in rows:
        fe = r.get("first_error_ds")
        if fe is None or not r["steps"]:
            continue
        rk = np.array([s["risk"] for s in r["steps"]])
        if rk.max() <= 0 or (rk == rk.max()).sum() > 1:
            continue
        out.append((r["steps"][int(rk.argmax())]["step"] - fe, r["n_steps"]))
    return out


def outcome_check(rows: list[dict], finished: dict[str, bool]) -> list[tuple[float, bool]]:
    """(err at the last cut, released answer correct) for finished traces."""
    out = []
    for r in rows:
        if finished.get(r["trace_id"]) and r.get("orig_correct") is not None:
            out.append((r["cuts"][-1]["err"], bool(r["orig_correct"])))
    return out


def auroc(score: np.ndarray, y: np.ndarray) -> float | None:
    pos, neg = score[y == 1], score[y == 0]
    if not len(pos) or not len(neg):
        return None
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    for v in np.unique(allv):                       # average ranks over ties
        m = allv == v; ranks[m] = ranks[m].mean()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def cluster_boot(groups: list[str], stat, n_boot: int = 1000, seed: int = 0) -> list[float]:
    """95% CI of stat(index array), resampling whole groups (traces) with replacement."""
    rng = np.random.default_rng(seed)
    idx = defaultdict(list)
    for i, g in enumerate(groups):
        idx[g].append(i)
    keys = list(idx)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        v = stat(np.concatenate([idx[keys[j]] for j in pick]))
        if v is not None:
            vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def step_frame(rows: list[dict], finished: dict[str, bool], null: dict | None) -> list[dict]:
    out = []
    for r in rows:
        fe = r.get("first_error_ds")
        for s in r["steps"]:
            i = s["step"]
            out.append({"trace_id": r["trace_id"], "step": i, "group": r["group"], "risk": s["risk"],
                        "se": s["se"], "harmful": s["harmful"], "err_before": s["err_before"],
                        "err_after": s["err_after"], "y_ds": r["y_ds"][i], "mask_ds": r["mask_ds"][i],
                        "y_self": r["y_self"][i], "mask_self": r["mask_self"][i],
                        "finished": finished.get(r["trace_id"]),
                        "phase": "pre_error" if fe is None or i < fe else ("first_error" if i == fe else "post_error"),
                        "null_harmful": None if null is None else null.get((r["trace_id"], i))})
    return out


def analyze(rows: list[dict], finished: dict[str, bool], n_perm: int = 20) -> dict:
    A: dict = {"n_traces": len(rows)}
    # 1. reliability
    ra, rb, tid = split_half(rows)
    r_half = float(np.corrcoef(ra, rb)[0, 1])
    A["split_half"] = {"n_steps": len(ra), "pearson_half": r_half, "spearman_half": spearman(ra, rb),
                       "pearson_k16_spearman_brown": spearman_brown(r_half),
                       "ci95_half": cluster_boot(tid, lambda k: float(np.corrcoef(ra[k], rb[k])[0, 1]))}
    p = err0_pairs(rows)
    if len(p) > 2:
        r0 = float(np.corrcoef(p[:, 0], p[:, 1])[0, 1])
        A["err0_test_retest"] = {"n_problems": len(p), "pearson": r0, "mean_abs_diff": float(np.abs(p[:, 0] - p[:, 1]).mean())}
    null = null_harmful(rows, n_perm) if n_perm else None
    T = step_frame(rows, finished, null)
    rk, se = np.array([t["risk"] for t in T]), np.array([t["se"] for t in T])
    A["signal_share"] = {"all": float(1 - np.mean(se ** 2) / rk.var())}

    def cls(sel):
        x = [t for t in T if sel(t)]
        if not x:
            return {"n": 0}
        r = np.array([t["risk"] for t in x]); s = np.array([t["se"] for t in x])
        h = np.array([t["harmful"] for t in x], float)
        d = {"n": len(x), "risk_mean": float(r.mean()), "share_harmful": float(h.mean()),
             "share_harmful_ci95": cluster_boot([t["trace_id"] for t in x], lambda k: float(h[k].mean())),
             "share_benign": float(np.mean(r <= 0)), "err_before_mean": float(np.mean([t["err_before"] for t in x])),
             "signal_share": float(1 - np.mean(s ** 2) / r.var()) if r.var() > 0 else None}
        if null is not None:
            d["null_share_harmful"] = float(np.mean([t["null_harmful"] for t in x]))
        return d

    ds = lambda t: t["mask_ds"]
    A["classes"] = {
        "ds_correct": cls(lambda t: ds(t) and t["y_ds"] == 0),
        "ds_incorrect": cls(lambda t: ds(t) and t["y_ds"] == 1),
        "pre_error": cls(lambda t: t["phase"] == "pre_error"),
        "first_error": cls(lambda t: t["phase"] == "first_error"),
        "first_error_finished": cls(lambda t: t["phase"] == "first_error" and t["finished"] is True),
        "first_error_capped": cls(lambda t: t["phase"] == "first_error" and t["finished"] is False),
        "post_error": cls(lambda t: t["phase"] == "post_error"),
        "ds1_self0": cls(lambda t: ds(t) and t["mask_self"] and t["y_ds"] == 1 and t["y_self"] == 0),
        "ds0_self1": cls(lambda t: ds(t) and t["mask_self"] and t["y_ds"] == 0 and t["y_self"] == 1),
        "ds1_self1": cls(lambda t: ds(t) and t["mask_self"] and t["y_ds"] == 1 and t["y_self"] == 1),
        "ds0_self0": cls(lambda t: ds(t) and t["mask_self"] and t["y_ds"] == 0 and t["y_self"] == 0),
    }
    # 3. validity against outcomes
    oc = outcome_check(rows, finished)
    if oc:
        e, c = np.array([x[0] for x in oc]), np.array([x[1] for x in oc])
        A["outcome"] = {"n": len(oc), "err_last_cut_if_correct": float(e[c].mean()) if c.any() else None,
                        "err_last_cut_if_wrong": float(e[~c].mean()) if (~c).any() else None,
                        "auroc_err_for_wrong": auroc(e, (~c).astype(int))}
    ev = event_study(rows, range(-3, 5))
    A["event_study"] = {o: {"n": len(v), "err_mean": float(np.mean(v))} for o, v in ev.items() if v}
    am = argmax_offset(rows)
    if am:
        off = np.array([o for o, _ in am]); n = np.array([m for _, m in am])
        A["argmax_risk"] = {"n_traces": len(am), "share_at_first_error": float(np.mean(off == 0)),
                            "chance_share": float(np.mean(1 / n)), "share_before": float(np.mean(off < 0)),
                            "share_after": float(np.mean(off > 0))}
    # 4. information beyond validity
    lab = [t for t in T if t["mask_ds"]]
    y, r = np.array([t["y_ds"] for t in lab]), np.array([t["risk"] for t in lab])
    A["auroc_risk_for_ds_incorrect"] = auroc(r, y)
    fs = [t for t in T if t["mask_self"]]
    A["auroc_risk_for_self_incorrect"] = auroc(np.array([t["risk"] for t in fs]), np.array([t["y_self"] for t in fs]))
    pre = [t for t in T if t["phase"] in ("pre_error", "first_error")]
    A["auroc_risk_first_error_vs_pre_error"] = auroc(np.array([t["risk"] for t in pre]),
                                                     np.array([t["phase"] == "first_error" for t in pre], int))
    A["_steps"] = T
    return A


def plots(A: dict, out: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    T, paths = A["_steps"], []
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ev = A["event_study"]
    xs = sorted(int(o) for o in ev)
    ax.plot(xs, [ev[o]["err_mean"] for o in xs], "o-")
    ax.axvspan(0, 1, color="red", alpha=0.1, label="first error (DeepSeek)")
    ax.set_xlabel("cut relative to first error (0 = just before it)"); ax.set_ylabel("mean err(cut)")
    ax.legend(); fig.tight_layout(); p = out / "event_study.png"; fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    bins = np.linspace(-1, 1, 33)
    for ph in ("pre_error", "first_error", "post_error"):
        v = [t["risk"] for t in T if t["phase"] == ph]
        ax.hist(v, bins=bins, histtype="step", density=True, label=f"{ph} (n={len(v)})")
    ax.set_yscale("log"); ax.set_xlabel("risk = err(s+1) - err(s)"); ax.legend(fontsize=8)
    fig.tight_layout(); p = out / "risk_hist_by_phase.png"; fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    eb = np.array([t["err_before"] for t in T]); rk = np.array([t["risk"] for t in T])
    ph = np.array([t["phase"] for t in T])
    bins = np.linspace(0, 1, 11)
    for name in ("pre_error", "first_error", "post_error"):
        m = ph == name
        idx = np.clip(np.digitize(eb[m], bins) - 1, 0, 9)
        ys = [rk[m][idx == b].mean() if (idx == b).sum() >= 20 else np.nan for b in range(10)]
        ax.plot((bins[:-1] + bins[1:]) / 2, ys, "o-", label=name)
    ax.plot([0, 1], [1, 0], "k:", lw=0.8, label="ceiling 1 - err(s)")
    ax.set_xlabel("err(s), error rate before the step"); ax.set_ylabel("mean risk"); ax.legend(fontsize=8)
    fig.tight_layout(); p = out / "risk_vs_headroom.png"; fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)
    return paths


def examples(rows: list[dict], traces: dict[str, dict], tok, out: Path, n: int = 4) -> Path:
    """Harmful steps DeepSeek calls correct, and benign first errors with room to fail."""
    T = [(r, s) for r in rows for s in r["steps"]]
    harmful_ok = sorted([(r, s) for r, s in T if s["harmful"] and r["mask_ds"][s["step"]] and r["y_ds"][s["step"]] == 0],
                        key=lambda x: -x[1]["risk"])[:n]
    benign = sorted([(r, s) for r, s in T if r.get("first_error_ds") == s["step"] and s["risk"] < 0],
                    key=lambda x: x[1]["risk"])[:n]
    lines = ["# step_value_labels_v1 examples", ""]
    for title, sel in (("Harmful steps labeled correct by DeepSeek", harmful_ok),
                       ("First errors (DeepSeek) after which the policy fails less", benign)):
        lines += [f"## {title}", ""]
        for r, s in sel:
            t = traces[r["trace_id"]]; i = s["step"]
            lo, hi = t["step_spans"][i]
            prev = tok.decode(t["reply_ids"][max(0, lo - 200):lo])
            preds = {c["cut"]: [x["pred"] for x in c["rollouts"]] for c in r["cuts"]}
            lines += [f"### {r['trace_id']} step {i} of {r['n_steps']} (gold {r['gold']})", "",
                      f"risk {s['risk']:+.3f} (err {s['err_before']:.3f} -> {s['err_after']:.3f}, se {s['se']:.3f}), "
                      f"DeepSeek {r['y_ds'][i]}, self {r['y_self'][i]}, first error {r.get('first_error_ds')}", "",
                      "Context (end):", "```", prev[-600:], "```", "Step:", "```", tok.decode(t["reply_ids"][lo:hi]), "```",
                      f"answers before: {preds.get(i)}", "", f"answers after: {preds.get(i + 1)}", ""]
    p = out / "examples.md"
    p.write_text("\n".join(lines) + "\n")
    return p


def fmt(A: dict) -> str:
    c = A["classes"]
    L = ["# step_value_labels_v1: label analysis", "", f"traces: {A['n_traces']}", "",
         "## Reliability", "", "```", json.dumps({k: A.get(k) for k in ("split_half", "err0_test_retest", "signal_share")}, indent=1), "```",
         "", "## Classes", "", "| class | n | risk mean | harmful | null harmful | harmful 95% CI | risk <= 0 | err before | signal share |",
         "|---|---|---|---|---|---|---|---|---|"]
    for k, v in c.items():
        if not v.get("n"):
            continue
        ss = v.get("signal_share")
        L.append(f"| {k} | {v['n']} | {v['risk_mean']:.3f} | {v['share_harmful']:.3f} | {v.get('null_share_harmful', float('nan')):.3f} | "
                 f"[{v['share_harmful_ci95'][0]:.3f}, {v['share_harmful_ci95'][1]:.3f}] | {v['share_benign']:.3f} | "
                 f"{v['err_before_mean']:.3f} | {'' if ss is None else f'{ss:.2f}'} |")
    L += ["", "## Validity", "", "```", json.dumps({k: A.get(k) for k in ("outcome", "event_study", "argmax_risk",
          "auroc_risk_for_ds_incorrect", "auroc_risk_for_self_incorrect", "auroc_risk_first_error_vs_pre_error")}, indent=1), "```"]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--values_dir", type=Path, required=True)
    ap.add_argument("--trace_set", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--n_perm", type=int, default=20)
    ap.add_argument("--tokenizer", default=None, help="e.g. Qwen/Qwen3-8B (local cache) to write examples.md")
    a = ap.parse_args()
    rows = []
    for p in sorted(glob.glob(str(a.values_dir / "values.shard*.jsonl"))):
        rows += [json.loads(l) for l in open(p) if l.strip()]
    traces = {t["trace_id"]: t for t in json.loads(a.trace_set.read_text())["traces"]}
    finished = {k: bool(t["finished"]) for k, t in traces.items()}
    a.out_dir.mkdir(parents=True, exist_ok=True)
    A = analyze(rows, finished, a.n_perm)
    paths = plots(A, a.out_dir)
    with open(a.out_dir / "steps.jsonl", "w") as f:
        for t in A["_steps"]:
            f.write(json.dumps(t) + "\n")
    del A["_steps"]
    (a.out_dir / "analysis.json").write_text(json.dumps(A, indent=2))
    (a.out_dir / "analysis.md").write_text(fmt(A))
    if a.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
        paths.append(examples(rows, traces, tok, a.out_dir))
    print(fmt(A))
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
