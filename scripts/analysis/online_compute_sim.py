#!/usr/bin/env python3
"""Spending generation compute with an online step scorer, simulated on saved pools
(read_token_v1 Stage I, docs/read_token_v1_plan.md).

Each problem has 10 saved trajectories, treated as i.i.d. draws, with a P(incorrect)
per step from each scorer (read token, RTH, PRM) and per-token DeepConf confidence.
A step's score exists once its last token is written.

  A1  early pruning: draw in order; kill a trajectory at the first step whose running
      worst score exceeds tau (tokens up to that step spent, answer lost); stop after
      K completed; majority over the completed (none: wrong). DeepConf kills at the
      first token whose running group confidence crosses its threshold.
  A2  adaptive N: stop drawing once the leading answer's (weighted) share among
      completed draws reaches theta, after at least 2 completed.
  A3  abstention: risk-coverage of the majority@10 answer (vote share, DeepConf, read
      token) and of single traces.

Thresholds and operating points are chosen on the exploratory half of the problems
(fixed hash); every reported number is on the confirmatory half, averaged over random
trajectory orders, with a paired bootstrap over problems.

    python scripts/analysis/online_compute_sim.py --out results/read_token_v1/online_sim
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
POOLS = {"t10": "instruct_t10", "t07": "instruct_t07"}
SCORERS = {"rthics_seed1": "RTH-ICS", "rthic_seed0": "RTH-IC", "rth_rth_correct_seed1": "RTH",
           "prm_qwen25_math_7b": "PRM"}
PRIMARY = "rthics_seed1"
N_TRAJ = 10


# ----------------------------------------------------------------------------------
# data

def load_cell(pool_dir: Path, conf_dir: Path, ds: str) -> dict:
    """One pool x dataset as (Q, 10) arrays, problems sorted by id."""
    rows = [json.loads(l) for f in sorted(glob.glob(str(pool_dir / f"tts_{ds}.shard*_trajectories.jsonl")))
            for l in open(f)]
    spans, conf = {}, {}
    for f in sorted(glob.glob(str(conf_dir / f"tts_{ds}.shard*_conf.npz"))):
        z = np.load(f)
        for k in z.files:
            uid, kind = k.rsplit("::", 1)
            if kind == "spans":
                spans[uid] = z[k][:, 1].astype(np.int64)
            elif kind == "conf":
                conf[uid] = z[k].astype(np.float64)
    scores = {c: {} for c in SCORERS}
    for c in SCORERS:
        for f in glob.glob(str(pool_dir / f"scores/tts_{ds}__{c}.shard*.jsonl")):
            for l in open(f):
                r = json.loads(l)
                scores[c][r["traj_uid"]] = np.asarray(r["scores"], np.float64)
    by: dict = {}
    for r in rows:
        by.setdefault(r["fork_id"], []).append(r)
    probs = sorted(p for p, rs in by.items() if len(rs) == N_TRAJ)
    cell = {"problems": probs, "keys": np.zeros((len(probs), N_TRAJ), np.int64),
            "correct": np.zeros((len(probs), N_TRAJ), bool), "key_correct": np.zeros((len(probs), N_TRAJ), bool),
            "tokens": np.zeros((len(probs), N_TRAJ)), "traj": []}
    for q, p in enumerate(probs):
        rs = sorted(by[p], key=lambda r: r.get("sample_idx", 0))
        kmap: dict = {}
        tr = []
        for i, r in enumerate(rs):
            k = r.get("answer_key")
            k = ("__none__", i) if k is None else k  # no answer: a key of its own
            j = kmap.setdefault(k, len(kmap))
            cell["keys"][q, i] = j
            cell["correct"][q, i] = bool(r["correct"])
            cell["key_correct"][q, j] |= bool(r["correct"])
            u = r["traj_uid"]
            cell["tokens"][q, i] = float(len(conf[u]))
            end = np.minimum(spans[u], len(conf[u]))
            tr.append({"uid": u, "step_end": end, "conf": conf[u],
                       "scores": {c: scores[c][u] for c in SCORERS}})
            for c in SCORERS:
                if len(scores[c][u]) != len(end):
                    raise ValueError(f"{u}: {c} has {len(scores[c][u])} scores for {len(end)} steps")
        cell["traj"].append(tr)
    return cell


def exploratory_mask(problems: list[str]) -> np.ndarray:
    return np.array([int(hashlib.sha256(p.encode()).hexdigest(), 16) % 2 == 0 for p in problems])


def group_badness(conf: np.ndarray, window: int, sign: float) -> np.ndarray:
    """Running DeepConf group confidence at every token (mean of the last `window`
    token confidences, the prefix mean before that), oriented so higher = worse."""
    c = np.cumsum(np.concatenate([[0.0], conf]))
    t = np.arange(1, len(conf) + 1)
    lo = np.maximum(t - window, 0)
    g = (c[t] - c[lo]) / (t - lo)
    return -sign * g


def running_worst(cell: dict, scorer: str, conf_rule: tuple | None = None) -> list[list[tuple]]:
    """Per trajectory: (running max of badness, the token count at which each value is
    known). conf_rule = (window, sign) for DeepConf, checked at every token."""
    out = []
    for tr in cell["traj"]:
        row = []
        for t in tr:
            if conf_rule is None:
                b, at = t["scores"][scorer], t["step_end"]
            else:
                b = group_badness(t["conf"], *conf_rule)
                at = np.arange(1, len(b) + 1)
            row.append((np.maximum.accumulate(b), at.astype(np.float64)))
        out.append(row)
    return out


def prune(worst: list[list[tuple]], tokens: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray]:
    """(completed, tokens spent) per trajectory under kill threshold tau."""
    Q = len(worst)
    comp = np.ones((Q, N_TRAJ), bool)
    cost = tokens.copy()
    for q, row in enumerate(worst):
        for i, (m, at) in enumerate(row):
            k = int(np.searchsorted(m, tau, side="right"))
            if k < len(m):
                comp[q, i] = False
                cost[q, i] = at[k]
    return comp, cost


# ----------------------------------------------------------------------------------
# simulation

def orders(Q: int, n_perm: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.argsort(rng.random((Q, n_perm, N_TRAJ)), -1)


def vote(keys_p: np.ndarray, use: np.ndarray, key_correct: np.ndarray, weight: np.ndarray | None = None):
    """Majority (or weighted) vote over the `use` positions, ties to the answer drawn
    first; (correct, any used) per (Q, P)."""
    Q, P, n = keys_p.shape
    oh = (keys_p[..., None] == np.arange(N_TRAJ)) & use[..., None]  # (Q, P, n, keys)
    w = oh * (1.0 if weight is None else weight[..., None])
    counts = w.sum(2)
    first = np.where(oh, np.arange(n)[None, None, :, None], n).min(2)
    score = counts * (4 * n) - first
    score = np.where(oh.any(2), score, -np.inf)
    chosen = score.argmax(-1)
    ok = np.take_along_axis(np.broadcast_to(key_correct[:, None, :], (Q, P, N_TRAJ)), chosen[..., None], -1)[..., 0]
    anyu = use.any(-1)
    return ok & anyu, anyu


def simulate_a1(cell: dict, comp: np.ndarray, cost: np.ndarray, K: int, perm: np.ndarray):
    """Per-problem accuracy and tokens (averaged over orders) for stop-after-K-completed."""
    take = lambda a: np.take_along_axis(np.broadcast_to(a[:, None, :], perm.shape), perm, -1)
    comp_p, cost_p, keys_p = take(comp), take(cost), take(cell["keys"])
    cum = np.cumsum(comp_p, -1)
    reached = cum >= K
    stop = np.where(reached.any(-1), reached.argmax(-1), N_TRAJ - 1)
    drawn = np.arange(N_TRAJ)[None, None, :] <= stop[..., None]
    ok, _ = vote(keys_p, comp_p & drawn, cell["key_correct"])
    return ok.mean(1), (cost_p * drawn).sum(-1).mean(1)


def simulate_a2(cell: dict, comp: np.ndarray, cost: np.ndarray, theta: float, perm: np.ndarray,
                weight: np.ndarray | None = None):
    """Adaptive N: stop at the first prefix with >= 2 completed and leading share >= theta."""
    take = lambda a: np.take_along_axis(np.broadcast_to(a[:, None, :], perm.shape), perm, -1)
    comp_p, cost_p, keys_p = take(comp), take(cost), take(cell["keys"])
    w_p = np.ones_like(cost_p) if weight is None else take(weight)
    Q, P, n = perm.shape
    oh = (keys_p[..., None] == np.arange(N_TRAJ)) & comp_p[..., None]
    cw = np.cumsum(oh * w_p[..., None], 2)  # (Q, P, prefix, keys)
    tot = cw.sum(-1)
    share = np.where(tot > 0, cw.max(-1) / np.maximum(tot, 1e-12), 0)
    ncomp = np.cumsum(comp_p, -1)
    okstop = (ncomp >= 2) & (share >= theta)
    stop = np.where(okstop.any(-1), okstop.argmax(-1), n - 1)
    drawn = np.arange(n)[None, None, :] <= stop[..., None]
    ok, _ = vote(keys_p, comp_p & drawn, cell["key_correct"], None if weight is None else w_p)
    return ok.mean(1), (cost_p * drawn).sum(-1).mean(1)


def boot_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 1000, seed: int = 0, ratio_den: np.ndarray | None = None):
    """Mean of a minus mean of b over problems (paired), 95% percentile interval; with
    ratio_den, 1 - mean(a) / mean(ratio_den) instead (a token saving)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n_boot, len(a)))
    if ratio_den is None:
        d = a[idx].mean(1) - b[idx].mean(1)
        pt = a.mean() - b.mean()
    else:
        d = 1 - a[idx].mean(1) / ratio_den[idx].mean(1)
        pt = 1 - a.mean() / ratio_den.mean()
    return float(pt), [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def risk_coverage(conf: np.ndarray, ok: np.ndarray) -> dict:
    """Accuracy at 50% and 80% coverage (most confident first) and the area under the
    risk-coverage curve (lower is better)."""
    o = np.argsort(-conf, kind="stable")
    acc = np.cumsum(ok[o]) / np.arange(1, len(o) + 1)
    n = len(o)
    return {"acc_at_50": float(acc[max(0, int(0.5 * n) - 1)]), "acc_at_80": float(acc[max(0, int(0.8 * n) - 1)]),
            "aurc": float(np.mean(1 - acc)), "acc_full": float(acc[-1])}


def within_auc(cell: dict, bad: np.ndarray) -> float:
    """Problem-macro AUROC of -bad for a correct trajectory, problems with both outcomes."""
    vals = []
    for q in range(len(cell["problems"])):
        y = cell["correct"][q]
        if y.all() or not y.any():
            continue
        s = -bad[q]
        pos, neg = s[y], s[~y]
        gt = (pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()
        vals.append(gt)
    return float(np.mean(vals))


# ----------------------------------------------------------------------------------

def run_cell(cell: dict, n_perm: int, n_tau: int, n_boot: int) -> dict:
    Q = len(cell["problems"])
    ex = exploratory_mask(cell["problems"])
    cf = ~ex
    perm = orders(Q, n_perm)
    res: dict = {"problems": Q, "exploratory": int(ex.sum()), "confirmatory": int(cf.sum())}

    # DeepConf rule on the exploratory half: window and orientation by within-problem AUROC
    best = None
    for w in (32, 64):
        for sign in (1.0, -1.0):
            bad = np.array([[group_badness(t["conf"], w, sign).max() for t in tr] for tr in cell["traj"]])
            sub = {k: (v[ex] if isinstance(v, np.ndarray) else [v[i] for i in np.flatnonzero(ex)])
                   for k, v in cell.items() if k in ("correct",)}
            sub["problems"] = [cell["problems"][i] for i in np.flatnonzero(ex)]
            a = within_auc(sub, bad[ex])
            if best is None or a > best[0]:
                best = (a, w, sign)
    res["deepconf_rule"] = {"window": best[1], "sign": best[2], "exploratory_within_auc": best[0]}
    rules = {c: ("score", c, None) for c in SCORERS}
    rules["deepconf"] = ("conf", None, (best[1], best[2]))
    worst = {name: running_worst(cell, c or PRIMARY, cr) for name, (_, c, cr) in rules.items()}
    final_bad = {name: np.array([[m[-1] for m, _ in row] for row in w]) for name, w in worst.items()}
    res["within_auc_confirmatory"] = {}
    for name, fb in final_bad.items():
        sub = {"correct": cell["correct"][cf], "problems": [cell["problems"][i] for i in np.flatnonzero(cf)]}
        res["within_auc_confirmatory"][name] = within_auc(sub, fb[cf])

    # baseline majority@n
    full = np.ones((Q, N_TRAJ), bool)
    maj = {n: simulate_a1(cell, full, cell["tokens"], n, perm) for n in range(1, N_TRAJ + 1)}
    res["majority"] = {n: {"acc_expl": float(a[ex].mean()), "acc": float(a[cf].mean()),
                           "tokens_expl": float(t[ex].mean()), "tokens": float(t[cf].mean())}
                       for n, (a, t) in maj.items()}

    # oracle pruning: wrong trajectories killed after their first step
    first_end = np.array([[t["step_end"][0] for t in tr] for tr in cell["traj"]], float)
    orc_cost = np.where(cell["correct"], cell["tokens"], first_end)
    Ks = [1, 2, 3, 4, 5, 6, 8, 10]

    def a1_grid(comp_cost_fn, taus):
        grid = []
        for tau in taus:
            comp, cost = comp_cost_fn(tau)
            for K in Ks:
                a, t = simulate_a1(cell, comp, cost, K, perm)
                grid.append({"tau": float(tau), "K": K, "acc": a, "tok": t,
                             "kill_rate": float(1 - comp[cf].mean())})
        return grid

    res["a1"], res["a2"] = {}, {}
    rows_a1 = {}
    grids = {"oracle": a1_grid(lambda tau: (cell["correct"], orc_cost), [0.0])}
    for name in rules:
        fb = final_bad[name][ex]
        taus = list(np.quantile(fb, np.linspace(0.05, 0.99, n_tau))) + [np.inf]
        grids[name] = a1_grid(lambda tau, name=name: prune(worst[name], cell["tokens"], tau), taus)
    for name, grid in grids.items():
        out = {"frontier_conf": sorted({(round(float(g["tok"][cf].mean()), 1), round(float(g["acc"][cf].mean()), 4))
                                        for g in grid})}
        for n in (4, 10):
            budget = maj[n][1][ex].mean()
            ok = [g for g in grid if g["tok"][ex].mean() <= budget + 1e-9]
            if not ok:
                continue
            g = max(ok, key=lambda g: (g["acc"][ex].mean(), -g["tok"][ex].mean()))
            dacc = boot_diff(g["acc"][cf], maj[n][0][cf], n_boot)
            dtok = boot_diff(g["tok"][cf], None, n_boot, ratio_den=maj[n][1][cf])
            out[f"matched_budget_maj{n}"] = {"tau": g["tau"], "K": g["K"], "kill_rate": g["kill_rate"],
                                             "acc": float(g["acc"][cf].mean()), "tokens": float(g["tok"][cf].mean()),
                                             "acc_minus_majority": dacc, "token_saving": dtok}
            rows_a1[(name, n)] = g
        target = maj[N_TRAJ][0][ex].mean()
        reach = [g for g in grid if g["acc"][ex].mean() >= target]
        if reach:
            g = min(reach, key=lambda g: (g["tok"][ex].mean(), -g["acc"][ex].mean()))
            out["matched_accuracy_maj10"] = {
                "tau": g["tau"], "K": g["K"], "kill_rate": g["kill_rate"],
                "acc": float(g["acc"][cf].mean()), "tokens": float(g["tok"][cf].mean()),
                "acc_minus_majority": boot_diff(g["acc"][cf], maj[N_TRAJ][0][cf], n_boot),
                "token_saving": boot_diff(g["tok"][cf], None, n_boot, ratio_den=maj[N_TRAJ][1][cf])}
            rows_a1[(name, "acc")] = g
        out["frontier_expl"] = sorted({(round(float(g["tok"][ex].mean()), 1), round(float(g["acc"][ex].mean()), 4))
                                       for g in grid})
        res["a1"][name] = out
    # paired: primary minus each competitor at the same operating rule
    res["a1_paired"] = {}
    for key in [(4,), (10,), ("acc",)]:
        k = key[0]
        if (PRIMARY, k) not in rows_a1:
            continue
        g0 = rows_a1[(PRIMARY, k)]
        for name in rules:
            if name == PRIMARY or (name, k) not in rows_a1:
                continue
            g1 = rows_a1[(name, k)]
            res["a1_paired"][f"{SCORERS.get(PRIMARY)} - {SCORERS.get(name, name)} @ {k}"] = {
                "acc": boot_diff(g0["acc"][cf], g1["acc"][cf], n_boot),
                "tokens": boot_diff(g0["tok"][cf], g1["tok"][cf], n_boot)}

    # A2 adaptive N
    thetas = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, np.inf]  # inf: never stop early (majority@10)
    for name in ["consistency"] + list(rules):
        grid = []
        for theta in thetas:
            if name == "consistency":
                a, t = simulate_a2(cell, full, cell["tokens"], theta, perm)
                grid.append({"theta": theta, "tau": None, "acc": a, "tok": t})
                continue
            fb = final_bad[name]
            # vote weights: 1 - P(incorrect) for a verifier; DeepConf's confidence as a
            # percentile rank over all trajectories (its scale is in nats)
            weight = 1 - fb if name != "deepconf" else (np.argsort(np.argsort(-fb.ravel())) / (fb.size - 1)).reshape(fb.shape)
            taus = [np.inf] + list(np.quantile(fb[ex], [0.9, 0.95, 0.99]))
            for tau in taus:
                comp, cost = prune(worst[name], cell["tokens"], tau)
                a, t = simulate_a2(cell, comp, cost, theta, perm, weight)
                grid.append({"theta": theta, "tau": float(tau), "acc": a, "tok": t})
        target = maj[N_TRAJ][0][ex].mean()
        reach = [g for g in grid if g["acc"][ex].mean() >= target]
        out = {}
        if reach:
            g = min(reach, key=lambda g: (g["tok"][ex].mean(), -g["acc"][ex].mean()))
            out["matched_accuracy_maj10"] = {
                "theta": float(g["theta"]), "tau": g["tau"], "acc": float(g["acc"][cf].mean()),
                "tokens": float(g["tok"][cf].mean()),
                "acc_minus_majority": boot_diff(g["acc"][cf], maj[N_TRAJ][0][cf], n_boot),
                "token_saving": boot_diff(g["tok"][cf], None, n_boot, ratio_den=maj[N_TRAJ][1][cf])}
        out["frontier_conf"] = sorted({(round(float(g["tok"][cf].mean()), 1), round(float(g["acc"][cf].mean()), 4))
                                       for g in grid})
        res["a2"][name] = out

    # A3 abstention, majority@10 over all ten in sample order
    use = np.ones((Q, 1, N_TRAJ), bool)
    keys1 = cell["keys"][:, None, :]
    ok10, _ = vote(keys1, use, cell["key_correct"])
    ok10 = ok10[:, 0]
    oh = cell["keys"][..., None] == np.arange(N_TRAJ)
    counts = oh.sum(1)
    first = np.where(oh, np.arange(N_TRAJ)[None, :, None], N_TRAJ).min(1)
    chosen = np.where(counts > 0, counts * 40 - first, -np.inf).argmax(-1)
    support = cell["keys"] == chosen[:, None]
    confs = {"vote_share": support.mean(1)}
    for name in rules:
        good = -final_bad[name]
        confs[name] = (good * support).sum(1) / support.sum(1)
    res["a3_majority10"] = {}
    rng = np.random.default_rng(1)
    idx_cf = np.flatnonzero(cf)
    for name, c in confs.items():
        rc = risk_coverage(c[cf], ok10[cf])
        if name != "vote_share":
            diffs = []
            for _ in range(n_boot):
                s = rng.choice(idx_cf, len(idx_cf))
                diffs.append(risk_coverage(confs["vote_share"][s], ok10[s])["aurc"] - risk_coverage(c[s], ok10[s])["aurc"])
            rc["aurc_gain_over_vote_share"] = [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
        res["a3_majority10"][name] = rc
    res["a3_single"] = {name: risk_coverage(-final_bad[name][cf].ravel(), cell["correct"][cf].ravel())
                        for name in rules}
    # paired by problem: the primary scorer's single-trace AURC against each other one
    for name in rules:
        if name == PRIMARY:
            continue
        diffs = []
        for _ in range(n_boot):
            s = rng.choice(idx_cf, len(idx_cf))
            y = cell["correct"][s].ravel()
            diffs.append(risk_coverage(-final_bad[name][s].ravel(), y)["aurc"]
                         - risk_coverage(-final_bad[PRIMARY][s].ravel(), y)["aurc"])
        res["a3_single"][name]["aurc_minus_primary_ci"] = [float(np.percentile(diffs, 2.5)),
                                                           float(np.percentile(diffs, 97.5))]
    return res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool_root", type=Path, default=ROOT / "cot-checker-results/tts_regraded_v2")
    p.add_argument("--conf_root", type=Path, default=ROOT / "cot-checker-results/tts_conf_npz")
    p.add_argument("--n_perm", type=int, default=50)
    p.add_argument("--n_tau", type=int, default=20)
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--cells", nargs="+", default=["t10:math500", "t10:gsm8k", "t07:math500", "t07:gsm8k"])
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    allres = {}
    for spec in a.cells:
        tag, ds = spec.split(":")
        cell = load_cell(a.pool_root / POOLS[tag], a.conf_root / tag, ds)
        r = run_cell(cell, a.n_perm, a.n_tau, a.n_boot)
        allres[spec] = r
        (a.out / f"{tag}_{ds}.json").write_text(json.dumps(r, indent=1, default=float))
        print(f"== {spec}: {r['problems']} problems, confirmatory {r['confirmatory']}; "
              f"DeepConf rule {r['deepconf_rule']}", flush=True)
        m10 = r["majority"][N_TRAJ]
        print(f"  majority@10 acc {m10['acc']:.4f} tokens {m10['tokens']:.0f}; "
              f"majority@4 acc {r['majority'][4]['acc']:.4f} tokens {r['majority'][4]['tokens']:.0f}", flush=True)
        print(f"  within-problem AUROC (confirmatory): "
              f"{ {k: round(v, 3) for k, v in r['within_auc_confirmatory'].items()} }", flush=True)
        for name, o in r["a1"].items():
            for k in ("matched_budget_maj4", "matched_budget_maj10", "matched_accuracy_maj10"):
                if k in o:
                    v = o[k]
                    print(f"  A1 {name:22s} {k:24s} acc {v['acc']:.4f} tok {v['tokens']:7.0f} "
                          f"dacc {v['acc_minus_majority'][0]:+.4f} {np.round(v['acc_minus_majority'][1], 4)} "
                          f"save {v['token_saving'][0]:+.3f} {np.round(v['token_saving'][1], 3)} "
                          f"(tau {v['tau']:.3g}, K {v['K']}, kill {v['kill_rate']:.2f})", flush=True)
        for k, v in r["a1_paired"].items():
            print(f"  A1 paired {k:28s} acc {v['acc'][0]:+.4f} {np.round(v['acc'][1], 4)} "
                  f"tokens {v['tokens'][0]:+.0f} {np.round(v['tokens'][1], 0)}", flush=True)
        for name, o in r["a2"].items():
            if "matched_accuracy_maj10" in o:
                v = o["matched_accuracy_maj10"]
                print(f"  A2 {name:22s} acc {v['acc']:.4f} tok {v['tokens']:7.0f} dacc {v['acc_minus_majority'][0]:+.4f} "
                      f"save {v['token_saving'][0]:+.3f} {np.round(v['token_saving'][1], 3)} "
                      f"(theta {v['theta']}, tau {v['tau']})", flush=True)
        for name, v in r["a3_majority10"].items():
            print(f"  A3 maj10 {name:22s} acc@50 {v['acc_at_50']:.4f} acc@80 {v['acc_at_80']:.4f} aurc {v['aurc']:.4f} "
                  f"{'gain CI ' + str(np.round(v['aurc_gain_over_vote_share'], 4)) if 'aurc_gain_over_vote_share' in v else ''}",
                  flush=True)
        for name, v in r["a3_single"].items():
            ci = v.get("aurc_minus_primary_ci")
            print(f"  A3 single {name:21s} acc@50 {v['acc_at_50']:.4f} acc@80 {v['acc_at_80']:.4f} aurc {v['aurc']:.4f} "
                  f"(all {v['acc_full']:.4f}){'  minus RTH-ICS CI ' + str(np.round(ci, 4)) if ci else ''}", flush=True)
    (a.out / "all.json").write_text(json.dumps(allres, indent=1, default=float))


if __name__ == "__main__":
    main()
