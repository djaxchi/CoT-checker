#!/usr/bin/env python3
"""Merge and summarize the factorized_step_labels_v1 pilot (plan section 4).

Reads the track-1 and track-2 shards, applies the no-op gate and the track-1
sanity check, and writes summary.json and summary.md to --out_dir. Every number
here is descriptive: 30 forks size the full run, they do not test a hypothesis.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.analysis.factorized_step_labels import (POSITIONS, boot_ci,  # noqa: E402
                                                 necessity_from_lp, partial_spearman,
                                                 spearman)

GATE_MEDIAN, GATE_MAX = 0.01, 0.1          # nats, |self delta| on the continuation sum


def jl_glob(d: Path, pat: str) -> list[dict]:
    rows = []
    for p in sorted(d.glob(pat)):
        rows += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return rows


def split_boundary(r: dict) -> None:
    """Rows written before necessity_from_lp: recompute necessity from the stored
    clean and mean-condition log-probs, without the boundary token, in place."""
    start = r["segments"]["cont"][0]
    seg = {k: tuple(v) for k, v in r["segments"].items()}
    for lay in r["layers"].values():
        if "nec_boundary" in lay:
            continue
        xb = necessity_from_lp(r["lp_clean"], lay["mean"]["lp"], start, seg)
        lay["nec_sum_incl_boundary"] = lay["nec_sum"]
        lay["nec_sum"], lay["nec_ans"], lay["nec_boundary"] = xb["cont"], xb["ans"], xb["boundary_drop"]
        lay["nec_next"] = xb.get("next", float("nan"))
        lay["nec_tok"] = xb["cont"] / max(xb["n_cont_xb"], 1)
        lay["rand_includes_boundary"] = True


def fork_factors(r1: dict, r2: dict, f: dict, layers) -> dict:
    """Per-fork factor values: validity, risk, and per-token necessity split into
    step i+1 and everything after it (both without the boundary token)."""
    seg, start = r2["segments"], r2["segments"]["cont"][0]
    n_cont = seg["cont"][1] - (start + 1)
    n_next = (seg["next"][1] - (start + 1)) if "next" in seg else 0
    out = {"fork_id": f["fork_id"], "incorrect": int(f["validity"] == "incorrect"),
           "position": f["position"], "relpos": f["step"] / f["n_steps"],
           "log_n_cont": float(np.log1p(n_cont)), "n_cont": n_cont,
           "err_a": r1["risk"]["err_a"], "risk_diff": r1["risk"]["risk_diff"],
           "risk_se": r1["risk"]["risk_diff_se"]}
    for L in layers:
        v = r2["layers"][str(L)]
        nxt = v["nec_next"] if n_next > 0 else float("nan")
        later = v["nec_sum"] - (nxt if n_next > 0 else 0.0)
        out[f"nec_tok_L{L}"] = v["nec_sum"] / max(n_cont, 1)
        out[f"nec_next_tok_L{L}"] = nxt / n_next if n_next > 0 else float("nan")
        out[f"nec_later_tok_L{L}"] = later / max(n_cont - n_next, 1)
        out[f"nec_ans_L{L}"] = v["nec_ans"]
    return out


def factor_section(rows: list[dict], primary) -> dict:
    """Pairwise factor relations, raw and partial on log continuation length and
    relative position, with fork-bootstrap 95% intervals."""
    col = lambda k: np.array([r[k] for r in rows], float)  # noqa: E731
    covs = [col("log_n_cont"), col("relpos")]
    pairs = [("incorrect", "risk_diff")]
    for L in primary:
        pairs += [("incorrect", f"nec_tok_L{L}"), ("risk_diff", f"nec_tok_L{L}"),
                  ("incorrect", f"nec_next_tok_L{L}"), ("risk_diff", f"nec_next_tok_L{L}"),
                  ("incorrect", f"nec_later_tok_L{L}"), ("risk_diff", f"nec_later_tok_L{L}")]
    out = {}
    for x, y in pairs:
        X, Y = col(x), col(y)
        out[f"{x}~{y}"] = {
            "spearman": spearman(X, Y),
            "spearman_ci95": boot_ci(lambda i: spearman(X[i], Y[i]), len(X), 1000, 1),
            "partial": partial_spearman(X, Y, covs),
            "partial_ci95": boot_ci(lambda i: partial_spearman(X[i], Y[i], [c[i] for c in covs]),
                                    len(X), 1000, 2)}
    inc = [r for r in rows if r["incorrect"]]
    cor = [r for r in rows if not r["incorrect"]]
    sig = lambda rs: float(np.mean([r["risk_diff"] > 2 * r["risk_se"] for r in rs if r["risk_se"] > 0]))  # noqa: E731
    out["counts"] = {
        "incorrect_bypassed_err_a_0": sum(r["err_a"] == 0 for r in inc), "n_incorrect": len(inc),
        "correct_with_any_error": sum(r["err_a"] > 0 for r in cor), "n_correct": len(cor),
        "incorrect_risk_gt_2se_share": sig(inc) if any(r["risk_se"] > 0 for r in inc) else None,
        "correct_risk_gt_2se_share": sig(cor) if any(r["risk_se"] > 0 for r in cor) else None,
        "incorrect_risk_diff_mean": float(np.mean([r["risk_diff"] for r in inc])) if inc else None,
        "incorrect_risk_diff_sd": float(np.std([r["risk_diff"] for r in inc], ddof=1)) if len(inc) > 1 else None,
        "incorrect_risk_se_mean": float(np.mean([r["risk_se"] for r in inc])) if inc else None}
    return out


def boot_diff(a: np.ndarray, b: np.ndarray, n: int = 5000, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    d = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n)]
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fork_split", type=Path, required=True)
    p.add_argument("--track1_dir", type=Path, required=True)
    p.add_argument("--track2_dir", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--primary_layers", type=int, nargs="+", default=[9, 18])
    p.add_argument("--max_fork_rows", type=int, default=60, help="per-fork table rows in summary.md")
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    forks = json.loads(a.fork_split.read_text())["forks"]
    t1 = {r["fork_id"]: r for r in jl_glob(a.track1_dir, "track1.shard*.jsonl")}
    t2 = {r["fork_id"]: r for r in jl_glob(a.track2_dir, "track2.shard*.jsonl")}
    for r in t2.values():
        split_boundary(r)
    ids = [f["fork_id"] for f in forks]
    S: dict = {"n_forks": len(forks), "n_track1": len(t1), "n_track2": len(t2),
               "missing_track1": [i for i in ids if i not in t1],
               "missing_track2": [i for i in ids if i not in t2]}
    layers = sorted({int(L) for r in t2.values() for L in r["layers"]})

    # --- no-op gate -------------------------------------------------------------
    gate = {}
    for L in layers:
        rs = [t2[i]["layers"][str(L)] for i in ids if i in t2]
        selfd = np.abs([r["self"]["delta"]["cont"] for r in rs])
        land = max(max(r[c]["landing_max_abs"] for c in ("self", "mean", "rand")) for r in rs)
        idx = max(t2[i]["index_check_max_abs"][str(L)] for i in ids if i in t2)
        mean_nec = np.abs([r["nec_sum"] for r in rs])
        gate[L] = {"self_abs_median": float(np.median(selfd)), "self_abs_max": float(selfd.max()),
                   "landing_max_abs": land, "index_check_max_abs": idx,
                   "mean_abs_nec_median": float(np.median(mean_nec)),
                   "pass": bool(np.median(selfd) <= GATE_MEDIAN and selfd.max() <= GATE_MAX
                                and land == 0.0 and idx == 0.0)}
    floor = np.abs([t2[i]["clean2_minus_clean"]["cont"] for i in ids if i in t2])
    S["gate"] = {"thresholds": {"self_abs_median": GATE_MEDIAN, "self_abs_max": GATE_MAX,
                                "landing_max_abs": 0.0, "index_check_max_abs": 0.0},
                 "clean_repeat_floor": {"median": float(np.median(floor)), "max": float(floor.max())},
                 "per_layer": {str(L): g for L, g in gate.items()},
                 "pass_all_layers": all(g["pass"] for g in gate.values())}

    # --- track 1 ----------------------------------------------------------------
    val = {i: next(f["validity"] for f in forks if f["fork_id"] == i) for i in ids}
    pos = {i: next(f["position"] for f in forks if f["fork_id"] == i) for i in ids}
    errA = {i: t1[i]["risk"]["err_a"] for i in ids if i in t1}
    ea_inc = np.array([errA[i] for i in errA if val[i] == "incorrect"])
    ea_cor = np.array([errA[i] for i in errA if val[i] == "correct"])
    alt_status, n_trunc, n_cont = Counter(), 0, 0
    for r in t1.values():
        for alt in r["alternatives"]:
            alt_status[alt["status"]] += 1
            alt_status["equals_original"] += alt["equals_original"]
            alt_status["is_answer_line"] += alt["is_answer_line"]
        for cs in r["conditions"].values():
            n_trunc += sum(c["truncated"] for c in cs); n_cont += len(cs)
            alt_status["ungradeable"] += sum(not c["gradeable"] for c in cs)
    diffs = np.array([t1[i]["risk"]["risk_diff"] for i in ids if i in t1])
    ses = np.array([t1[i]["risk"]["risk_diff_se"] for i in ids if i in t1])
    S["track1"] = {
        "sanity": {"err_a_incorrect_mean": float(ea_inc.mean()) if len(ea_inc) else None,
                   "err_a_correct_mean": float(ea_cor.mean()) if len(ea_cor) else None,
                   "diff_incorrect_minus_correct": float(ea_inc.mean() - ea_cor.mean())
                   if len(ea_inc) and len(ea_cor) else None,
                   "diff_ci95_fork_bootstrap": boot_diff(ea_inc, ea_cor) if len(ea_inc) and len(ea_cor) else None,
                   "pass": bool(len(ea_inc) and len(ea_cor) and ea_inc.mean() > ea_cor.mean())},
        "noise": {"risk_diff_sd_across_forks": float(diffs.std(ddof=1)) if len(diffs) > 1 else None,
                  "risk_diff_se_within_fork_median": float(np.median(ses)) if len(ses) else None,
                  "risk_diff_se_within_fork_mean": float(ses.mean()) if len(ses) else None},
        "alternatives": dict(alt_status), "continuations": n_cont, "truncated": n_trunc,
        "orig_correct_vs_errA": {
            "orig_correct": float(np.mean([errA[i] for i in errA
                                           if next(f["orig_correct"] for f in forks if f["fork_id"] == i)]))
            if errA else None,
            "orig_wrong": float(np.mean([errA[i] for i in errA
                                         if not next(f["orig_correct"] for f in forks if f["fork_id"] == i)]))
            if errA else None}}

    # --- necessity --------------------------------------------------------------
    nec = {}
    for L in layers:
        v = {i: t2[i]["layers"][str(L)] for i in ids if i in t2}
        ns = np.array([v[i]["nec_sum"] for i in v])
        nec[str(L)] = {
            "nec_sum_median": float(np.median(ns)), "nec_sum_iqr": [float(np.percentile(ns, 25)),
                                                                    float(np.percentile(ns, 75))],
            "nec_tok_median": float(np.median([v[i]["nec_tok"] for i in v])),
            "nec_ans_median": float(np.median([v[i]["nec_ans"] for i in v])),
            "nec_next_median": float(np.nanmedian([v[i]["nec_next"] for i in v])),
            "rand_drop_median": float(np.median([v[i]["rand"]["drop_xb"]["cont"] if "drop_xb" in v[i]["rand"]
                                                 else -v[i]["rand"]["delta"]["cont"] for i in v])),
            "rand_includes_boundary": any(v[i].get("rand_includes_boundary", False) for i in v),
            "boundary_drop_median": float(np.median([v[i]["nec_boundary"] for i in v])),
            "boundary_drop_max": float(max(v[i]["nec_boundary"] for i in v)),
            "nec_sum_incl_boundary_median": float(np.median([v[i]["nec_sum_incl_boundary"] for i in v])),
            "nec_sum_by_validity": {va: float(np.median([v[i]["nec_sum"] for i in v if val[i] == va]))
                                    for va in ("correct", "incorrect")},
            "spearman_nec_vs_risk_diff": spearman([v[i]["nec_sum"] for i in v if i in t1],
                                                  [t1[i]["risk"]["risk_diff"] for i in v if i in t1]),
            "spearman_nec_vs_incorrect": spearman([v[i]["nec_sum"] for i in v],
                                                  [val[i] == "incorrect" for i in v])}
    S["necessity"] = nec
    S["spearman_risk_diff_vs_incorrect"] = spearman([t1[i]["risk"]["risk_diff"] for i in ids if i in t1],
                                                    [val[i] == "incorrect" for i in ids if i in t1])

    # --- cells ------------------------------------------------------------------
    cells = {}
    for va in ("correct", "incorrect"):
        for po in POSITIONS:
            c = [i for i in ids if val[i] == va and pos[i] == po]
            cells[f"{va}/{po}"] = {
                "n": len(c),
                "err_a": float(np.mean([errA[i] for i in c if i in errA])) if c else None,
                "err_c_mean": float(np.mean([t1[i]["risk"]["err_c_mean"] for i in c if i in t1])) if c else None,
                "risk_diff": float(np.mean([t1[i]["risk"]["risk_diff"] for i in c if i in t1])) if c else None,
                **{f"nec_sum_L{L}": float(np.median([t2[i]["layers"][str(L)]["nec_sum"] for i in c if i in t2]))
                   for L in layers}}
    S["cells"] = cells
    frows = [fork_factors(t1[i], t2[i], f, layers) for f in forks
             for i in [f["fork_id"]] if i in t1 and i in t2]
    primary = [L for L in a.primary_layers if L in layers]
    S["factors"] = factor_section(frows, primary)
    S["factor_cells"] = {
        f"{va}/{po}": {k: float(np.nanmedian([r[k] for r in frows if r["incorrect"] == (va == "incorrect")
                                               and r["position"] == po]))
                       for k in ["err_a", "risk_diff"] + [f"nec_{s}tok_L{L}" for L in primary
                                                          for s in ("", "next_", "later_")]}
        for va in ("correct", "incorrect") for po in POSITIONS}
    with open(a.out_dir / "fork_factors.jsonl", "w") as fh:
        for r in frows:
            fh.write(json.dumps(r) + "\n")
    (a.out_dir / "summary.json").write_text(json.dumps(S, indent=2))

    # --- markdown ---------------------------------------------------------------
    L_hdr = " | ".join(f"nec L{L}" for L in layers)
    lines = ["# factorized_step_labels_v1 pilot summary", "",
             f"Forks: {len(forks)}; track 1 rows {len(t1)}, track 2 rows {len(t2)}.", "",
             "## No-op gate (self-patch, continuation sum, nats)", "",
             "| layer | median abs self | max abs self | landing max | index check | median abs nec (mean) | pass |",
             "|---|---|---|---|---|---|---|"]
    for L, g in gate.items():
        lines.append(f"| {L} | {g['self_abs_median']:.4f} | {g['self_abs_max']:.4f} | {g['landing_max_abs']:.3g} | "
                     f"{g['index_check_max_abs']:.3g} | {g['mean_abs_nec_median']:.3f} | {g['pass']} |")
    fl = S["gate"]["clean_repeat_floor"]
    lines += ["", f"Clean repeat floor: median {fl['median']:.4f}, max {fl['max']:.4f}.", "",
              "## Track 1", "", "```", json.dumps(S["track1"], indent=1), "```", "",
              "## Necessity by layer", "", "```", json.dumps(nec, indent=1), "```", "",
              "## Cells", "", f"| cell | n | err(A) | mean err(C) | risk_diff | {L_hdr} |",
              "|---|---|---|---|---|" + "---|" * len(layers)]
    fmt = lambda x: "n/a" if x is None else f"{x:.3f}"  # noqa: E731
    for k, c in cells.items():
        lines.append(f"| {k} | {c['n']} | {fmt(c['err_a'])} | {fmt(c['err_c_mean'])} | {fmt(c['risk_diff'])} | "
                     + " | ".join(fmt(c[f'nec_sum_L{L}']) for L in layers) + " |")
    fmt2 = lambda v: "n/a" if v is None else (f"[{v[0]:+.2f}, {v[1]:+.2f}]" if isinstance(v, list) else f"{v:+.3f}")  # noqa: E731
    lines += ["", f"## Factors (primary layers {primary}; partial = on log continuation length and relative position)", "",
              "| pair | spearman | 95% CI | partial | 95% CI |", "|---|---|---|---|---|"]
    for k, v in S["factors"].items():
        if k != "counts":
            lines.append(f"| {k} | {fmt2(v['spearman'])} | {fmt2(v['spearman_ci95'])} | "
                         f"{fmt2(v['partial'])} | {fmt2(v['partial_ci95'])} |")
    lines += ["", "```", json.dumps(S["factors"]["counts"], indent=1), "```", "",
              "Factor medians by cell:", "", "```", json.dumps(S["factor_cells"], indent=1), "```"]
    lines += ["", "## Forks", "", f"| fork | cell | err(A) | err(C) | risk_diff +- se | {L_hdr} |",
              "|---|---|---|---|---|" + "---|" * len(layers)]
    for i in ids[:a.max_fork_rows]:
        r1, r2 = t1.get(i), t2.get(i)
        rk = r1["risk"] if r1 else None
        errc = " ".join(f"{e:.2f}" for e in rk["err_c"]) if rk else "n/a"
        rd = f"{rk['risk_diff']:+.3f} +- {rk['risk_diff_se']:.3f}" if rk else "n/a"
        necs = " | ".join(fmt(r2["layers"][str(L)]["nec_sum"] if r2 else None) for L in layers)
        lines.append(f"| {i} | {val[i]}/{pos[i]} | {fmt(rk['err_a'] if rk else None)} | {errc} | {rd} | {necs} |")
    (a.out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
