#!/usr/bin/env python3
"""cot_causal_graph_v0 Stage 3: assemble graphs, taxonomy, cross-tab (CPU).

Consumes stage1 (node_features, tf_edges, gates) + stage2 (fg_curves, gates) and
emits (spec: docs/cot_causal_graph_v0_plan.md):

  stage3/graphs/<trace_id>.json   one causal graph per trace (explorer input)
  stage3/crosstab.json            the detection x influence 2x2 (+ repair
                                  inventory), by influence criterion
  stage3/summary.json             headline numbers + gate echo

Decision rules (frozen): detected = probe L28 logit > val-selected threshold
(gates_stage1). Influential-TF = |d answer margin| beyond the 95th percentile of
matched-control edges (test split). Influential-FG = solve-rate drop with the
conservative CI excluding zero. Site call = FG when rollouts exist, else TF.

Usage:
  python scripts/causal_graph/cg_stage3_assemble.py --run_dir runs/causal_graph
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from scripts.encode_prm800k_hidden_states import git_commit  # noqa: E402
from src.analysis.causal_graph import (  # noqa: E402
    classify_site,
    fg_influence,
    is_influential_fg,
    is_influential_tf,
    null_quantile,
)


def load_stage_outputs(run_dir: Path):
    s1, s2 = run_dir / "stage1", run_dir / "stage2"
    nodes = pd.read_parquet(s1 / "node_features.parquet")
    edges = pd.read_parquet(s1 / "tf_edges.parquet")
    gates1 = json.loads((s1 / "gates_stage1.json").read_text())
    curves = rolls = gates2 = None
    if (s2 / "fg_curves.parquet").exists():
        curves = pd.read_parquet(s2 / "fg_curves.parquet")
        rolls = pd.read_parquet(s2 / "fg_rollouts.parquet")
        gates2 = json.loads((s2 / "gates_stage2.json").read_text())
    return nodes, edges, gates1, curves, rolls, gates2


def fg_edge_for_site(rolls: pd.DataFrame, trace_id: str, base_ctx: str,
                     interv_ctx: str) -> dict | None:
    r = rolls[rolls.trace_id == trace_id]
    base = r[r.context == base_ctx].correct.tolist()
    interv = r[r.context == interv_ctx].correct.tolist()
    if not base or not interv:
        return None
    return fg_influence(base, interv)


def build_graph(tr_nodes: pd.DataFrame, tr_edges: pd.DataFrame,
                rolls: pd.DataFrame | None, det_thresh: float,
                tf_null: float) -> dict:
    tr_nodes = tr_nodes.sort_values("step_idx")
    arm = tr_nodes.arm.iloc[0]
    trace_id = tr_nodes.trace_id.iloc[0]
    nodes = []
    for r in tr_nodes.itertuples():
        nodes.append({
            "idx": int(r.step_idx), "text": r.text,
            "probe_l28": None if pd.isna(r.probe_l28) else float(r.probe_l28),
            "detected": bool(r.probe_l28 > det_thresh) if not pd.isna(r.probe_l28) else None,
            "margin": None if pd.isna(r.margin) else float(r.margin),
            "step_logp": None if pd.isna(r.step_logp) else float(r.step_logp),
            "entropy": None if pd.isna(r.boundary_entropy) else float(r.boundary_entropy),
            "is_error_site": bool(getattr(r, "is_fork_step", False)),
        })
    graph = {
        "trace_id": trace_id, "arm": arm,
        "split": tr_nodes.split.iloc[0],
        "n_steps": len(nodes), "nodes": nodes,
        "margin_curve": [n["margin"] for n in nodes],
        "edges": [], "site": None,
    }
    if arm == "onpolicy" and "traj_correct" in tr_nodes:
        graph["traj_correct"] = bool(tr_nodes.traj_correct.iloc[0])

    for e in tr_edges.itertuples():
        base_edge = {"family": "tf", "interv": e.interv, "src": int(e.t)}
        graph["edges"].append(base_edge | {
            "dst": "answer", "kind": "margin",
            "delta": float(e.d_margin_final),
            "significant": bool(is_influential_tf(e.d_margin_final, tf_null))})
        for j, v in json.loads(e.d_logp_steps).items():
            graph["edges"].append(base_edge | {
                "dst": int(j), "kind": "logp", "delta": float(v)})
        for j, v in json.loads(e.d_probe_steps).items():
            graph["edges"].append(base_edge | {
                "dst": int(j), "kind": "probe_diag", "delta": float(v)})

    if rolls is not None:
        rr = rolls[rolls.trace_id == trace_id]
        if len(rr):
            if arm == "forks":
                for interv in ("swap_wrong", "swap_xprob", "swap_pos"):
                    fe = fg_edge_for_site(rolls, trace_id, "base", interv)
                    if fe:
                        t = int(tr_edges.t.iloc[0]) if len(tr_edges) else 0
                        graph["edges"].append({
                            "family": "fg", "interv": interv, "src": t,
                            "dst": "answer", "kind": "solve_rate",
                            "delta": fe["delta"],
                            "delta_ci_lo": fe["delta_ci_lo"],
                            "delta_ci_hi": fe["delta_ci_hi"],
                            "recovery_rate": fe["recovery_rate"],
                            "significant": is_influential_fg(fe)})
            else:
                pref = (rr[rr.context.str.startswith("prefix_")]
                        .groupby("context").correct.mean())
                curve = [float(pref.get(f"prefix_{i}", np.nan))
                         for i in range(len(nodes))]
                graph["fg_curve"] = curve
                for ctx in rr.context.unique():
                    if ctx.startswith("swap_xtrace_"):
                        i = int(ctx.split("_")[-1])
                        fe = fg_edge_for_site(rolls, trace_id, f"prefix_{i}", ctx)
                        if fe:
                            graph["edges"].append({
                                "family": "fg", "interv": "swap_xtrace",
                                "src": i, "dst": "answer", "kind": "solve_rate",
                                "delta": fe["delta"],
                                "delta_ci_lo": fe["delta_ci_lo"],
                                "delta_ci_hi": fe["delta_ci_hi"],
                                "recovery_rate": fe["recovery_rate"],
                                "significant": is_influential_fg(fe)})
    return graph


def classify_fork_site(graph: dict, tr_edges: pd.DataFrame, det_thresh: float,
                       tf_null: float) -> dict | None:
    """The taxonomy call for an arm-F trace's ground-truth error site."""
    wrong = tr_edges[tr_edges.interv == "swap_wrong"]
    if not len(wrong):
        return None
    w = wrong.iloc[0]
    detected = (not pd.isna(w.probe_at_interv_l28)
                and w.probe_at_interv_l28 > det_thresh)
    tf_infl = is_influential_tf(w.d_margin_final, tf_null)
    fg = next((e for e in graph["edges"]
               if e["family"] == "fg" and e["interv"] == "swap_wrong"), None)
    fg_infl = fg["significant"] if fg else None
    influential = fg_infl if fg_infl is not None else tf_infl
    site = {
        "t": int(w.t), "detected": bool(detected),
        "probe_at_error": float(w.probe_at_interv_l28)
        if not pd.isna(w.probe_at_interv_l28) else None,
        "influential_tf": bool(tf_infl),
        "influential_fg": fg_infl,
        "influential": bool(influential),
        "d_margin_final": float(w.d_margin_final),
        "fg_delta": fg["delta"] if fg else None,
        "recovery_rate": fg["recovery_rate"] if fg else None,
        "taxonomy": classify_site(detected, bool(influential)),
        "repaired": bool(fg and fg["recovery_rate"] >= 0.5) if fg else None,
    }
    return site


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", type=Path, default=Path("runs/causal_graph"))
    ap.add_argument("--null_q", type=float, default=0.95)
    args = ap.parse_args()

    nodes, edges, gates1, curves, rolls, gates2 = load_stage_outputs(args.run_dir)
    out = args.run_dir / "stage3"
    (out / "graphs").mkdir(parents=True, exist_ok=True)

    det = gates1.get("detection_threshold_l28", {})
    det_thresh = det.get("threshold")
    if det_thresh is None:
        det_thresh = float(np.nanquantile(nodes.probe_l28, 0.9))
        det = {"threshold": det_thresh, "fallback": "p90_all_steps"}

    fk_edges = edges[edges.arm == "forks"]
    ctrl = fk_edges[(fk_edges.interv.isin(["swap_xprob", "swap_pos"]))
                    & (fk_edges.split == "test")]
    tf_null = null_quantile(ctrl.d_margin_final.tolist(), q=args.null_q)

    sites, n_graphs = [], 0
    for trace_id, tr_nodes in nodes.groupby("trace_id"):
        tr_edges = edges[edges.trace_id == trace_id]
        graph = build_graph(tr_nodes, tr_edges, rolls, det_thresh, tf_null)
        if graph["arm"] == "forks":
            site = classify_fork_site(graph, tr_edges, det_thresh, tf_null)
            graph["site"] = site
            if site is not None and graph["split"] == "test":
                sites.append({"trace_id": trace_id} | site)
        safe = trace_id.replace("/", "_").replace(":", "_")
        (out / "graphs" / f"{safe}.json").write_text(json.dumps(graph))
        n_graphs += 1

    sdf = pd.DataFrame(sites)
    crosstab: dict = {"n_sites_test": int(len(sdf)),
                      "detection_threshold": det, "tf_null_thresh": tf_null}
    if len(sdf):
        crosstab["cells"] = sdf.taxonomy.value_counts().to_dict()
        crosstab["p_detected"] = float(sdf.detected.mean())
        crosstab["p_influential"] = float(sdf.influential.mean())
        crosstab["p_inert_errors"] = float(1 - sdf.influential.mean())
        have_both = sdf.dropna(subset=["probe_at_error"])
        if len(have_both) >= 10:
            from scipy import stats
            for eff, name in (("d_margin_final", "spearman_probe_vs_tf_effect"),
                              ("fg_delta", "spearman_probe_vs_fg_effect")):
                sub = have_both.dropna(subset=[eff])
                if len(sub) >= 10:
                    rho = stats.spearmanr(sub.probe_at_error, sub[eff].abs())
                    crosstab[name] = {"rho": float(rho.statistic),
                                      "p": float(rho.pvalue), "n": int(len(sub))}
        rep = sdf[sdf.repaired == True]  # noqa: E712
        crosstab["repair_inventory"] = {
            "n_repaired_sites": int(len(rep)),
            "trace_ids": rep.trace_id.tolist()[:50]}
    (out / "crosstab.json").write_text(json.dumps(crosstab, indent=2))

    summary = {
        "created": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "n_graphs": n_graphs,
        "gates_stage1": gates1, "gates_stage2": gates2,
        "crosstab": crosstab,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in crosstab.items()
                      if k != "repair_inventory"}, indent=2))
    print(f"[stage3] wrote {n_graphs} graphs -> {out / 'graphs'}")


if __name__ == "__main__":
    main()
