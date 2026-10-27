#!/usr/bin/env python3
"""M4: is RTH's tie to the policy's confidence load-bearing? (read_token_v1, Stage G)

RTH prefers the policy's confident answer on MATH-500 mixed pairs (0.530 on the
985 confidence-wrong pairs, PRM 0.629). This fits a linear readout of the policy's
confidence on the read state, removes its direction from the head and measures what
is lost on ProcessBench and gained on those pairs.

  encode   per bundle item, one forward of the policy (Qwen3-8B) on the verifier
           template + step; the logits that predict the step's tokens give
           logp_mean (mean log-probability of the step tokens, the headline
           confidence), logp_min, entropy_mean and deepconf_k20 (DeepConf's token
           confidence, minus the mean top-20 log-probability, averaged over the
           step). conf_<split>.npz, in bundle (reference) order.
  analyze  ridge of logp_mean on the final read state, fit on the bundle's train
           sample (alpha on val R2); c = the unit ridge direction. Heads scored from
           the bundle's final read state with the leaderboard code
           (prm_geometry.evaluate): w (RTH as trained), w_proj = w - (w.c) c (the
           frozen M4 test), w_decor = w - k beta with k chosen so that the score is
           uncorrelated with confidence on train (a second, stronger removal).
           Writes checkpoints with the w_proj and w_decor heads for the pair test.
           The same on the PRM's final state with its LDA direction, for contrast,
           and the confidence R2 of each representation.
  pairs    the MATH-500 mixed-pair accuracies of scored cells, as in
           scripts/analysis/pair_complementarity.py (question-weighted, all mixed
           pairs and the pairs where confidence prefers the wrong answer, question
           bootstrap), and paired differences against the first cell.
           --questions_out writes the mixed-pair questions for --fork_ids.

Readings (docs/read_token_v1_plan.md, Stage G): calib-20 down >= 0.02 with w_proj
means the coupling is load-bearing; down < 0.01 with the confidence-wrong pair
accuracy rising means a decoupled RTH is within reach.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.read_token_mech_bundle import load_bundle  # noqa: E402

PB_SUBSETS = ("gsm8k", "math", "olympiadbench", "omnimath")
CONF_KEYS = ("logp_mean", "logp_min", "entropy_mean", "deepconf_k20")


# --------------------------------------------------------------------------- encode

def step_stats(logits: torch.Tensor, targets: torch.Tensor, k: int = 20) -> dict:
    """logits (T, V) predicting targets (T,) -> the step's confidence statistics."""
    lp = torch.log_softmax(logits.float(), -1)
    tok_lp = lp.gather(1, targets[:, None])[:, 0]
    ent = -(lp.exp() * lp).sum(-1)
    topk = lp.topk(k, -1).values
    return {"logp_mean": float(tok_lp.mean()), "logp_min": float(tok_lp.min()),
            "entropy_mean": float(ent.mean()), "deepconf_k20": float((-topk.mean(-1)).mean())}


def confidence(model, tok, items: list[dict], pad: int, dev, max_tokens: int,
               chunk: int = 2048) -> dict[str, np.ndarray]:
    """Per item: the statistics of the policy's distribution over its step tokens."""
    from scripts.read_token_probe import batches
    from src.onpolicy.prompts import verifier_prefix
    pre = [tok(verifier_prefix(it["problem"], it["prefix"]), add_special_tokens=True)["input_ids"]
           for it in items]
    step = [tok(it["step"], add_special_tokens=False)["input_ids"] for it in items]
    out = {k: np.full(len(items), np.nan) for k in CONF_KEYS}
    for b in batches([len(p) + len(s) for p, s in zip(pre, step)], max_tokens, None):
        seqs = [pre[i] + step[i] for i in b]
        B, L = len(seqs), max(len(t) for t in seqs)
        ids = torch.full((B, L), pad, dtype=torch.long)
        att = torch.zeros((B, L), dtype=torch.long)
        for r, t in enumerate(seqs):
            ids[r, L - len(t):] = torch.tensor(t)
            att[r, L - len(t):] = 1
        ids, att = ids.to(dev), att.to(dev)
        with torch.no_grad():
            h = model.model(input_ids=ids, attention_mask=att,
                            position_ids=(att.cumsum(1) - 1).clamp(min=0)).last_hidden_state
            for r, i in enumerate(b):
                n = len(step[i])
                if n == 0:
                    continue
                # positions L - n - 1 .. L - 2 predict the step tokens at L - n .. L - 1
                hs, tg = h[r, L - n - 1:L - 1], ids[r, L - n:]
                parts = [step_stats(model.lm_head(hs[c:c + chunk]), tg[c:c + chunk]) for c in range(0, n, chunk)]
                if len(parts) == 1:
                    st = parts[0]
                else:  # long steps: token-weighted means, minimum of minima
                    w = np.array([min(chunk, n - c) for c in range(0, n, chunk)], float) / n
                    st = {k: float(sum(wi * p[k] for wi, p in zip(w, parts))) for k in CONF_KEYS}
                    st["logp_min"] = min(p["logp_min"] for p in parts)
                for k in CONF_KEYS:
                    out[k][i] = st[k]
    return out


def run_encode(a) -> None:
    from scripts.read_token_store import load_model, match, source_index
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok = load_model(a.model, dev)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    a.out.mkdir(parents=True, exist_ok=True)
    for spec in a.splits:
        split, source = spec.split("=", 1)
        t0 = time.perf_counter()
        b = load_bundle(a.bundle / split, arrays=("y", "index"))
        y = np.asarray(b["y"], np.int64)
        kind, index = source_index(source)
        items = match(kind, index, b["meta"], y)
        if a.smoke:
            items, y = items[:a.smoke], y[:a.smoke]
        c = confidence(model, tok, items, pad, dev, a.max_tokens)
        np.savez(a.out / f"conf_{split}.npz", y=y, index=np.asarray(b["index"])[:len(y)], **c)
        print(f"[m4 encode] {split}: {len(items)} items, logp_mean {np.nanmean(c['logp_mean']):.3f} "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)


# --------------------------------------------------------------------------- analyze

def ridge_fit(X: np.ndarray, y: np.ndarray, Xva: np.ndarray, yva: np.ndarray, dev,
              rels=(1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)) -> tuple[np.ndarray, float, float]:
    """Ridge on centered X; alpha = rel * mean eigenvalue, picked on val R2.
    Returns (beta, intercept, val R2)."""
    Xt = torch.as_tensor(np.asarray(X, np.float32), device=dev, dtype=torch.float64)
    yt = torch.as_tensor(np.asarray(y, np.float64), device=dev)
    mu, ym = Xt.mean(0), yt.mean()
    Xc = Xt - mu
    e, V = torch.linalg.eigh(Xc.T @ Xc)
    proj = V.T @ (Xc.T @ (yt - ym))
    best = (-np.inf, None, None)
    for r in rels:
        beta = V @ (proj / (e + r * e.mean()))
        icpt = float(ym - mu @ beta)
        b_np = beta.cpu().numpy()
        r2 = r2_score(yva, np.asarray(Xva, np.float64) @ b_np + icpt)
        if r2 > best[0]:
            best = (r2, b_np, icpt)
    return best[1], best[2], best[0]


def r2_score(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, np.float64)
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def heads(w: np.ndarray, beta: np.ndarray, Xtr: np.ndarray, ctr: np.ndarray) -> dict[str, np.ndarray]:
    """w, w with the unit ridge direction projected out, and w minus the multiple of beta
    that leaves the train score uncorrelated with confidence."""
    c = beta / np.linalg.norm(beta)
    s, sb = Xtr @ w, Xtr @ beta
    k = np.cov(s, ctr)[0, 1] / np.cov(sb, ctr)[0, 1]
    return {"w": w, "w_proj": w - (w @ c) * c, "w_decor": w - k * beta}


def score_all(W: dict, data: dict) -> dict:
    """Scores (higher = incorrect) of every head on every split."""
    return {name: {s: np.asarray(X, np.float64) @ w for s, (X, *_rest) in data.items()} for name, w in W.items()}


def evaluate_heads(scores: dict, data: dict, conf: dict) -> dict:
    from scripts.analysis.prm_geometry import evaluate
    out = {}
    for name, s in scores.items():
        res = evaluate(name, s["val_5k"], s["test_2k"], {k: s[k] for k in PB_SUBSETS},
                       data["val_5k"][1], data["test_2k"][1], {k: data[k][2] for k in PB_SUBSETS})
        pb_s = np.concatenate([s[k] for k in PB_SUBSETS])
        pb_c = np.concatenate([conf[k] for k in PB_SUBSETS])
        out[name] = {"pb_calib20": res["pb_avg_F1_PB_calib20"], "pb_oracle": res["pb_avg_F1_PB_oracle"],
                     "test_auroc": res["auroc_test"],
                     "corr_conf_test": float(np.corrcoef(s["test_2k"], conf["test_2k"])[0, 1]),
                     "corr_conf_pb": float(np.corrcoef(pb_s, pb_c)[0, 1]),
                     "pb_calib20_by_subset": {k: v["F1_PB_calib20"] for k, v in res["pb"].items()}}
    return out


def bundle_row(split_dir: Path, rep: str, row: int) -> np.ndarray:
    """One row of a bundle array, (N, d) float32 in reference order, without loading the others."""
    parts = sorted(split_dir.glob("part_*"))
    idx = np.concatenate([np.load(p / "index.npy") for p in parts])
    X = np.concatenate([np.asarray(np.load(p / f"{rep}.npy", mmap_mode="r")[:, row], np.float32) for p in parts])
    return X[np.argsort(idx, kind="stable")]


def load_rep(bundle: Path, conf_dir: Path, rep: str, row: int, key: str) -> tuple[dict, dict]:
    data, conf = {}, {}
    for split in ("probe_train_full", "val_5k", "test_2k") + PB_SUBSETS:
        b = load_bundle(bundle / split, arrays=("y", "index", "head_logit"))
        z = np.load(conf_dir / f"conf_{split}.npz")
        if not np.array_equal(np.asarray(z["index"]), np.asarray(b["index"])):
            raise SystemExit(f"{split}: confidence and bundle orders differ")
        data[split] = (bundle_row(bundle / split, rep, row), np.asarray(b["y"], np.int64), b["meta"],
                       np.asarray(b["head_logit"], np.float64))
        conf[split] = np.asarray(z[key], np.float64)
        if not np.isfinite(conf[split]).all():
            raise SystemExit(f"{split}: {int((~np.isfinite(conf[split])).sum())} items without confidence")
    return data, conf


def lda_direction(X: np.ndarray, y: np.ndarray, Xva: np.ndarray, yva: np.ndarray, dev) -> np.ndarray:
    from scripts.analysis.prm_geometry import SHRINKS, auroc, moments, project, shrink
    m = moments(X, y, dev)
    best = (-1.0, None)
    for s in SHRINKS:
        w = torch.linalg.solve(shrink(m["sw"], s), m["mu1"] - m["mu0"])
        av = auroc(yva, project(Xva, w, dev))
        if av > best[0]:
            best = (av, w)
    return best[1].double().cpu().numpy()


def run_analyze(a) -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    w = ck["head"]["weight"].double().numpy().reshape(-1)
    b0 = float(ck["head"]["bias"].double())
    out = {"ckpt": str(a.ckpt), "confidence": a.key, "reps": {}}
    for rep, row in (("read", 36), ("last_step", 36), ("prm", 28)):
        data, conf = load_rep(a.bundle, a.conf, rep, row, a.key)
        Xtr = data["probe_train_full"][0]
        beta, icpt, r2_val = ridge_fit(Xtr, conf["probe_train_full"], data["val_5k"][0], conf["val_5k"], dev)
        r2 = {s: r2_score(conf[s], np.asarray(data[s][0], np.float64) @ beta + icpt)
              for s in ("test_2k",) + PB_SUBSETS}
        res = {"row": row, "conf_r2_val": r2_val, "conf_r2": r2}
        if rep == "read":
            check = {s: float(np.abs(np.asarray(data[s][0], np.float64) @ w + b0 - data[s][3]).max())
                     for s in ("test_2k", "math")}
            res["head_check_max_abs_logit_diff"] = check
            ww = w
        elif rep == "prm":
            ww = lda_direction(Xtr, data["probe_train_full"][1], data["val_5k"][0], data["val_5k"][1], dev)
        else:
            res["direction"] = "none (confidence R2 only)"
            out["reps"][rep] = res
            print(f"[m4] {rep}: confidence R2 val {r2_val:.3f} test {r2['test_2k']:.3f}", flush=True)
            continue
        H = heads(ww, beta, np.asarray(Xtr, np.float64), conf["probe_train_full"])
        c = beta / np.linalg.norm(beta)
        res["cos_w_c"] = float(ww @ c / np.linalg.norm(ww))
        res["heads"] = evaluate_heads(score_all(H, data), data, conf)
        base = res["heads"]["w"]["pb_calib20"]
        for name in ("w_proj", "w_decor"):
            res["heads"][name]["calib20_drop"] = base - res["heads"][name]["pb_calib20"]
        out["reps"][rep] = res
        print(f"[m4] {rep}: confidence R2 val {r2_val:.3f} test {r2['test_2k']:.3f}; cos(w, c) "
              f"{res['cos_w_c']:+.3f}; calib-20 w {base:.4f} w_proj {res['heads']['w_proj']['pb_calib20']:.4f} "
              f"w_decor {res['heads']['w_decor']['pb_calib20']:.4f}", flush=True)
        if rep == "read":
            a.ckpt_out.mkdir(parents=True, exist_ok=True)
            for name in ("w_proj", "w_decor"):
                ck2 = dict(ck)
                ck2["head"] = {"weight": torch.tensor(H[name], dtype=ck["head"]["weight"].dtype)[None],
                               "bias": ck["head"]["bias"].clone()}
                ck2["m4_head"] = name
                d = a.ckpt_out / f"{a.ckpt.parent.name}_{name}"
                d.mkdir(parents=True, exist_ok=True)
                torch.save(ck2, d / "read_token.pt")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))


# --------------------------------------------------------------------------- pairs

def load_quality(pattern: str) -> dict[str, float]:
    """traj_uid -> 1 - worst-step suspicion, as pair_complementarity.load_scores."""
    out = {}
    for f in glob.glob(pattern):
        for r in map(json.loads, open(f)):
            if r.get("scores"):
                out[r["traj_uid"]] = 1.0 - max(r["scores"])
    return out


def mixed_pairs(trajs: dict, groups: dict) -> list[tuple]:
    from src.eval.math_grade import answer_key
    pairs = []
    for q, ids in groups.items():
        for i in ids:
            for j in ids:
                if trajs[i]["correct"] and not trajs[j]["correct"] and \
                        answer_key(trajs[i]["pred"]) != answer_key(trajs[j]["pred"]):
                    pairs.append((q, i, j))
    return pairs


def pair_metrics(trajs: dict, conf: dict, cells: dict, n_boot: int, seed: int = 0) -> dict:
    """cells: name -> traj_uid -> quality. Confidence: mean_token_conf, oriented to rank
    correct traces higher, as pair_complementarity.py."""
    us_all = [u for u, r in trajs.items() if r.get("gradeable") and u in conf
              and all(u in s for s in cells.values())]
    groups = defaultdict(list)
    for u in us_all:
        groups[trajs[u]["fork_id"]].append(u)
    cf = {u: conf[u]["mean_token_conf"] for u in us_all}
    y = np.array([trajs[u]["correct"] for u in us_all], float)
    if np.corrcoef([cf[u] for u in us_all], y)[0, 1] < 0:
        cf = {u: -v for u, v in cf.items()}
    pairs = mixed_pairs(trajs, groups)
    qs = sorted({q for q, _, _ in pairs})
    qa = np.array([qs.index(q) for q, _, _ in pairs])

    def pref(sc):
        return np.array([1.0 if sc[i] > sc[j] else 0.5 if sc[i] == sc[j] else 0.0 for _, i, j in pairs])

    def per_q(v, sel):
        return np.array([v[sel & (qa == k)].mean() for k in range(len(qs)) if (sel & (qa == k)).any()])

    P = {n: pref(s) for n, s in cells.items()}
    wrong = pref(cf) == 0.0
    rng = np.random.default_rng(seed)
    out = {"n_pairs": len(pairs), "n_questions": len(qs), "n_conf_wrong": int(wrong.sum()),
           "pair_questions": qs, "cells": {}}
    first = next(iter(cells))
    for n in cells:
        res = {}
        for tag, sel in (("all", np.ones(len(pairs), bool)), ("conf_wrong", wrong)):
            arr = per_q(P[n], sel)
            bb = rng.integers(0, len(arr), (n_boot, len(arr)))
            res[tag] = [float(arr.mean()), float(np.percentile(arr[bb].mean(1), 2.5)),
                        float(np.percentile(arr[bb].mean(1), 97.5))]
            if n != first:
                dq = per_q(P[n] - P[first], sel)
                bd = rng.integers(0, len(dq), (n_boot, len(dq)))
                res[f"{tag}_minus_{first}"] = [float(dq.mean()), float(np.percentile(dq[bd].mean(1), 2.5)),
                                               float(np.percentile(dq[bd].mean(1), 97.5))]
        res["agree_with_conf"] = float(((P[n] == 1) == (pref(cf) == 1)).mean())
        out["cells"][n] = res
        print(f"[m4 pairs] {n}: all {res['all'][0]:.3f} conf-wrong {res['conf_wrong'][0]:.3f} "
              f"[{res['conf_wrong'][1]:.3f}, {res['conf_wrong'][2]:.3f}]", flush=True)
    return out


def run_pairs(a) -> None:
    trajs = {r["traj_uid"]: r for f in glob.glob(a.trajectories) for r in map(json.loads, open(f))}
    conf = {r["traj_uid"]: r["rules"] for f in glob.glob(a.conf) for r in map(json.loads, open(f))}
    cells = {}
    for spec in a.cells:
        name, pat = spec.split("=", 1)
        cells[name] = load_quality(pat)
        if not cells[name]:
            raise SystemExit(f"no scores match {pat}")
    res = pair_metrics(trajs, conf, cells, a.n_boot)
    if a.questions_out:
        a.questions_out.parent.mkdir(parents=True, exist_ok=True)
        a.questions_out.write_text(json.dumps(res["pair_questions"]))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("encode")
    e.add_argument("--model", default="Qwen/Qwen3-8B")
    e.add_argument("--bundle", type=Path, required=True)
    e.add_argument("--splits", nargs="+", required=True, help="split=source (prm800k:PATH or pb:SUBSET:PATH)")
    e.add_argument("--max_tokens", type=int, default=16000)
    e.add_argument("--smoke", type=int, default=0)
    e.add_argument("--out", type=Path, required=True)
    z = sub.add_parser("analyze")
    z.add_argument("--bundle", type=Path, required=True)
    z.add_argument("--conf", type=Path, required=True, help="directory with conf_<split>.npz")
    z.add_argument("--ckpt", type=Path, required=True, help="the bundle's token (dq_correct)")
    z.add_argument("--key", default="logp_mean", choices=CONF_KEYS)
    z.add_argument("--ckpt_out", type=Path, required=True, help="where the projected-head tokens go")
    z.add_argument("--out", type=Path, required=True)
    q = sub.add_parser("pairs")
    q.add_argument("--trajectories", required=True, help="glob of *_trajectories.jsonl")
    q.add_argument("--conf", required=True, help="glob of *_conf.jsonl")
    q.add_argument("--cells", nargs="+", required=True, help="name=glob of score files; first = reference")
    q.add_argument("--n_boot", type=int, default=2000)
    q.add_argument("--questions_out", type=Path, default=None)
    q.add_argument("--out", type=Path, default=None)
    a = p.parse_args()
    {"encode": run_encode, "analyze": run_analyze, "pairs": run_pairs}[a.cmd](a)


if __name__ == "__main__":
    main()
