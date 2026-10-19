#!/usr/bin/env python3
"""risk_probe_v2 analysis (docs/risk_probe_v2_plan.md, sections 6 to 8).

Inputs:
  --labels       build_label_jsonl.py output (targets, folds, DeepSeek validity)
  --fits         train_risk_cell.py outputs: f<k>_<target>/oof.jsonl and
                 full_<target>_s<seed>/pb_step_scores_<sub>.jsonl
  --validity     score_cells.py outputs for the leaderboard d512 seeds:
                 v<seed>__label.jsonl, v<seed>__pb_<sub>.jsonl, v<seed>__gate.json

Reports, for each target (soft, harmful):
  E2   out-of-fold AUROC for `harmful` (all, within DeepSeek-incorrect, within
       correct), problem bootstrap;
  PB   risk alone and validity alone per seed: F1_PB calib-20, oracle F1_PB,
       AUROC over known-label steps; four-subset means, then seed means;
  PB   validity + risk, seed s with seed s:
       primary   logistic regression of DeepSeek invalidity on standardized
                 [validity logit, risk logit], fit on the label set (risk = the
                 out-of-fold score), applied to ProcessBench;
       secondary calib-20 with the weight w of z_validity + w * z_risk chosen
                 with the threshold on each calibration draw;
       delta F1_PB calib-20 against validity alone, paired bootstrap over traces
       within subsets (same resample for every seed).
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from merge_rep_grid_leaderboard import calib20_subset  # noqa: E402
from src.analysis.risk_probe_v2 import calib20_weighted  # noqa: E402
from src.eval.contextual_probe_metrics import auroc, best_pb_threshold  # noqa: E402

PB = ("gsm8k", "math", "olympiadbench", "omnimath")
TARGETS = ("soft", "harmful")
SEEDS = (42, 43, 44)
WEIGHTS = (0.0, 0.125, 0.25, 0.5, 1.0, 2.0)


def jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p) if l.strip()]


def boot_auroc(y, s, groups, n=1000, seed=0):
    y, s, groups = np.asarray(y), np.asarray(s), np.asarray(groups)
    if len(y) == 0 or y.min() == y.max():
        return {"auroc": float("nan"), "ci95": [float("nan"), float("nan")]}
    base = auroc(s, y)
    keys, inv = np.unique(groups, return_inverse=True)
    members = [np.where(inv == k)[0] for k in range(len(keys))]
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n):
        idx = np.concatenate([members[k] for k in rng.integers(0, len(keys), len(keys))])
        if y[idx].min() != y[idx].max():
            bs.append(auroc(s[idx], y[idx]))
    return {"auroc": float(base), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}


def pb_traces(rows_by_sub: dict) -> dict:
    """{sub: {id: (label, [scores in step order])}} from per-step score rows."""
    out = {}
    for sub, rows in rows_by_sub.items():
        d: dict = {}
        for r in rows:
            d.setdefault(r["id"], []).append((r["step_idx"], r["score"]))
        out[sub] = {i: [v for _, v in sorted(x)] for i, x in d.items()}
    return out


def pb_metrics(traces: dict, labels: dict) -> dict:
    """traces/labels: {sub: {id: scores}}, {sub: {id: first-error label}}."""
    res = {}
    for sub in traces:
        ids = sorted(traces[sub])
        seqs = [(labels[sub][i], traces[sub][i]) for i in ids]
        s, y = [], []
        for lab, sc in seqs:
            last = len(sc) if lab < 0 else lab + 1
            s += sc[:last]; y += [0] * (last - 1 if lab >= 0 else last) + ([1] if lab >= 0 else [])
        _, oracle = best_pb_threshold([x for _, x in seqs], [lab for lab, _ in seqs])
        res[sub] = {"f1_pb_calib20": calib20_subset(seqs), "f1_pb_oracle": float(oracle),
                    "auroc_known": float(auroc(np.array(s), np.array(y)))}
    res["mean"] = {k: float(np.mean([res[sub][k] for sub in traces])) for k in res[next(iter(traces))]}
    return res


def _boot_one(args):
    """One paired resample: calib-20 per subset for every (name, seed) score set."""
    b, seqs_by_key, labels, ids_by_sub = args
    rng = np.random.default_rng(1000 + b)
    out = {}
    for sub, ids in ids_by_sub.items():
        pick = [ids[i] for i in rng.integers(0, len(ids), len(ids))]
        for key, tr in seqs_by_key.items():
            out.setdefault(key, []).append(calib20_subset([(labels[sub][i], tr[sub][i]) for i in pick]))
    return {k: float(np.mean(v)) for k, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--fits", type=Path, required=True)
    ap.add_argument("--validity", type=Path, required=True)
    ap.add_argument("--pb_subsets", nargs="+", default=list(PB))
    ap.add_argument("--targets", nargs="+", default=list(TARGETS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--n_boot", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out_dir", type=Path, required=True)
    a = ap.parse_args()
    from sklearn.linear_model import LogisticRegression
    a.out_dir.mkdir(parents=True, exist_ok=True)
    lab = {r["uid"]: r for r in jl(a.labels)}
    uids = sorted(lab)
    y_h = np.array([lab[u]["label"] for u in uids]); y_ds = np.array([lab[u]["y_ds"] for u in uids])
    pk = np.array([lab[u]["problem_id"] for u in uids])
    out = {"gates": {}, "e2": {}, "pb": {}}

    # ---- gates: validity reproduction, OOF completeness
    out["gates"]["validity_reproduction"] = {s: json.loads((a.validity / f"v{s}__gate.json").read_text()) for s in a.seeds}
    oof = {}
    for t in a.targets:
        sc = {}
        for k in range(5):
            for r in jl(a.fits / f"f{k}_{t}" / "oof.jsonl"):
                sc[r["uid"]] = r["score"]
        out["gates"][f"oof_complete_{t}"] = len(sc) == len(uids) and set(sc) == set(uids)
        oof[t] = np.array([sc.get(u, np.nan) for u in uids])

    # ---- E2 on the label set
    inc, cor = y_ds == 1, y_ds == 0
    vlab = {s: {r["uid"]: r["logit"] for r in jl(a.validity / f"v{s}__label.jsonl")} for s in a.seeds}
    for t in a.targets:                     # scored steps only (all of them unless a smoke run)
        s, ok = oof[t], ~np.isnan(oof[t])
        out["e2"][t] = {"n_scored": int(ok.sum()), "all": boot_auroc(y_h[ok], s[ok], pk[ok]),
                        "ds_incorrect": boot_auroc(y_h[inc & ok], s[inc & ok], pk[inc & ok]),
                        "ds_correct": boot_auroc(y_h[cor & ok], s[cor & ok], pk[cor & ok])}
    v0 = np.array([vlab[a.seeds[0]].get(u, np.nan) for u in uids])
    ok = ~np.isnan(v0)
    inc, cor = inc & ok, cor & ok
    lv = (y_ds >= 0) & ok
    out["e2"]["validity_d512"] = {"auroc_ds_invalid": boot_auroc(y_ds[lv], v0[lv], pk[lv]),
                                  "harmful_ds_incorrect": boot_auroc(y_h[inc], v0[inc], pk[inc])}

    # ---- ProcessBench
    subs = a.pb_subsets
    first = a.fits / f"full_{a.targets[0]}_s{a.seeds[0]}"
    pb_labels = {sub: {r["id"]: r["label"] for r in jl(first / f"pb_step_scores_{sub}.jsonl")} for sub in subs}
    val_pb = {s: pb_traces({sub: [{"id": r["id"], "step_idx": r["step_idx"], "score": r["logit"]}
                                  for r in jl(a.validity / f"v{s}__pb_{sub}.jsonl")] for sub in subs}) for s in a.seeds}
    risk_pb = {}
    for t in a.targets:
        for s in a.seeds:
            tr = {}
            for sub in subs:
                tr[sub] = {r["id"]: r["scores"] for r in jl(a.fits / f"full_{t}_s{s}" / f"pb_step_scores_{sub}.jsonl")}
            risk_pb[(t, s)] = tr
    seqs = {}                                          # (name, seed) -> {sub: {id: scores}}
    for s in a.seeds:
        seqs[("validity", s)] = val_pb[s]
        out["pb"].setdefault("validity", {})[s] = pb_metrics(val_pb[s], pb_labels)
    for t in a.targets:
        for s in a.seeds:
            seqs[(f"risk_{t}", s)] = risk_pb[(t, s)]
            out["pb"].setdefault(f"risk_{t}", {})[s] = pb_metrics(risk_pb[(t, s)], pb_labels)
            # primary combination: fit on the label set, applied to ProcessBench
            X = np.column_stack([np.array([vlab[s].get(u, np.nan) for u in uids]), oof[t]])
            m = (y_ds >= 0) & np.isfinite(X).all(1)
            X = X[m]
            mu, sd = X.mean(0), X.std(0) + 1e-12
            clf = LogisticRegression(C=1.0, max_iter=1000).fit((X - mu) / sd, y_ds[m])
            w = clf.coef_[0] / sd
            combo = {sub: {i: (w[0] * np.asarray(val_pb[s][sub][i]) + w[1] * np.asarray(risk_pb[(t, s)][sub][i])).tolist()
                           for i in val_pb[s][sub]} for sub in subs}
            seqs[(f"combo_{t}", s)] = combo
            met = pb_metrics(combo, pb_labels)
            met["coef_std"] = clf.coef_[0].tolist()
            out["pb"].setdefault(f"combo_{t}", {})[s] = met
            # secondary: weight chosen with the threshold on the calibration draws
            zs = {sub: {i: ((np.asarray(val_pb[s][sub][i]) - mu[0]) / sd[0]).tolist() for i in val_pb[s][sub]} for sub in subs}
            zr = {sub: {i: ((np.asarray(risk_pb[(t, s)][sub][i]) - mu[1]) / sd[1]).tolist() for i in val_pb[s][sub]} for sub in subs}
            per = {}
            for sub in subs:
                ids = sorted(zs[sub])
                per[sub] = calib20_weighted([(pb_labels[sub][i], zs[sub][i]) for i in ids],
                                            [(pb_labels[sub][i], zr[sub][i]) for i in ids], WEIGHTS)
            out["pb"].setdefault(f"combo_w_{t}", {})[s] = {"per_subset": per, "f1_pb_calib20": float(np.mean(list(per.values())))}

    def seed_mean(name, key="f1_pb_calib20"):
        vals = [out["pb"][name][s]["mean"][key] if "mean" in out["pb"][name][s] else out["pb"][name][s][key]
                for s in a.seeds]
        return float(np.mean(vals))

    summary = {name: {k: seed_mean(name, k) for k in ("f1_pb_calib20", "f1_pb_oracle", "auroc_known")}
               for name in out["pb"] if not name.startswith("combo_w")}
    for t in a.targets:
        summary[f"combo_w_{t}"] = {"f1_pb_calib20": seed_mean(f"combo_w_{t}")}
    out["pb_seed_mean"] = summary

    # ---- paired bootstrap of delta F1_PB calib-20 (validity + risk, primary) - validity
    ids_by_sub = {sub: sorted(pb_labels[sub]) for sub in subs}
    keys = [k for k in seqs if k[0] == "validity" or k[0].startswith("combo_")]
    sel = {f"{n}|{s}": seqs[(n, s)] for n, s in keys}
    with ProcessPoolExecutor(a.workers) as ex:
        boots = list(ex.map(_boot_one, [(b, sel, pb_labels, ids_by_sub) for b in range(a.n_boot)]))
    out["delta_f1_pb_calib20"] = {}
    for t in a.targets:
        point = summary[f"combo_{t}"]["f1_pb_calib20"] - summary["validity"]["f1_pb_calib20"]
        d = [np.mean([bb[f"combo_{t}|{s}"] - bb[f"validity|{s}"] for s in a.seeds]) for bb in boots]
        out["delta_f1_pb_calib20"][t] = {"delta": float(point), "ci95": [float(np.percentile(d, 2.5)),
                                                                          float(np.percentile(d, 97.5))],
                                         "n_boot": a.n_boot}
    prim = out["delta_f1_pb_calib20"].get("soft")
    out["reading"] = ("risk adds to validity on ProcessBench" if prim and prim["ci95"][0] > 0
                      else "no ProcessBench gain shown")
    (a.out_dir / "summary.json").write_text(json.dumps(out, indent=1, default=str))

    lines = ["# risk_probe_v2 summary", "", f"**Reading (primary, soft target):** {out['reading']}", "",
             "## Gates", "", "```", json.dumps(out["gates"], indent=1), "```", "",
             "## ProcessBench (seed means, four-subset means)", "",
             "| score | F1_PB calib-20 | F1_PB oracle | AUROC known steps |", "|---|---|---|---|"]
    for name, r in summary.items():
        lines.append(f"| {name} | {r['f1_pb_calib20']:.3f} | {r.get('f1_pb_oracle', float('nan')):.3f} | "
                     f"{r.get('auroc_known', float('nan')):.3f} |")
    lines += ["", "## Delta F1_PB calib-20, validity + risk (primary combination) minus validity", ""]
    for t, r in out["delta_f1_pb_calib20"].items():
        lines.append(f"* {t}: {r['delta']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]")
    lines += ["", "## Label set, out of fold (AUROC for harmful)", ""]
    for t in a.targets:
        e = out["e2"][t]
        lines.append(f"* {t}: all {e['all']['auroc']:.3f} {e['all']['ci95']}, DeepSeek-incorrect "
                     f"{e['ds_incorrect']['auroc']:.3f}, DeepSeek-correct {e['ds_correct']['auroc']:.3f}")
    v = out["e2"]["validity_d512"]
    lines.append(f"* validity d512 (seed {a.seeds[0]}): DeepSeek invalidity {v['auroc_ds_invalid']['auroc']:.3f}, "
                 f"harmful within DeepSeek-incorrect {v['harmful_ds_incorrect']['auroc']:.3f}")
    (a.out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
