"""In-node driver for step_bert_probe_v1: HP search, finals and aggregation for both arms.

Arm A (tok) and Arm B (txt) run concurrently, one worker per GPU per arm, so each
GPU holds one fit of each arm at a time. Every fit whose metrics.json exists is
skipped, so a resubmitted job resumes.

    A search  causal_Wall, bidir_Wall x lr {3e-4, 1e-4} x wd {0.01, 0.1}, seed 0
    A finals  local, causal_Wall, bidir_W1, bidir_W2, bidir_Wall x seeds 42/43/44
    B search  causal, bidir x lr {2e-5, 5e-5}, seed 0
    B finals  local, causal, bidir x seeds 42/43/44

The shared setting per arm is the one with the best mean val F1 over its two
search conditions. Aggregation mirrors v0: seed-mean test F1 at each fit's val
threshold, problem-clustered bootstrap CIs on paired differences, per-seed PB
calib-20 differences, and the v0 GO rule on bidir_best - causal.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

HERE = Path(__file__).resolve().parent
SEEDS = (42, 43, 44)
A_GRID = [(lr, wd) for lr in (3e-4, 1e-4) for wd in (0.01, 0.1)]
B_GRID = [(lr, 0.01) for lr in (2e-5, 5e-5)]
A_FINAL = {"local": ("local", -1), "causal_Wall": ("causal", -1), "bidir_W1": ("bidir", 1),
           "bidir_W2": ("bidir", 2), "bidir_Wall": ("bidir", -1)}
B_FINAL = {"local": ("local", None), "causal": ("causal", None), "bidir": ("bidir", None)}
LOCK = threading.Lock()


def log(msg: str) -> None:
    with LOCK:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Pool:
    """One worker thread per GPU; run(jobs) blocks until all are done."""

    def __init__(self, arm: str, gpus: list[int], out: Path):
        self.arm, self.gpus, self.out = arm, gpus, out

    def run(self, jobs: list[dict]) -> list[str]:
        q, failed = queue.Queue(), []
        for j in jobs:
            if not (self.out / "fits" / j["name"] / "metrics.json").exists():
                q.put(j)
            else:
                log(f"{self.arm} skip {j['name']} (done)")

        def worker(g):
            while True:
                try:
                    j = q.get_nowait()
                except queue.Empty:
                    return
                d = self.out / "fits" / j["name"]
                d.mkdir(parents=True, exist_ok=True)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
                t0 = time.time()
                log(f"{self.arm} start {j['name']} gpu{g}")
                with (d / "fit.log").open("w") as f:
                    rc = subprocess.run(j["cmd"] + ["--out_dir", str(d)], stdout=f, stderr=subprocess.STDOUT,
                                        env=env).returncode
                log(f"{self.arm} done  {j['name']} rc={rc} {time.time() - t0:.0f}s")
                if rc != 0:
                    failed.append(j["name"])
                    with LOCK:
                        print("".join((d / "fit.log").read_text().splitlines(True)[-15:]), flush=True)

        ts = [threading.Thread(target=worker, args=(g,)) for g in self.gpus]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        return failed


def best_val(out: Path, name: str) -> float:
    return json.loads((out / "fits" / name / "metrics.json").read_text())["best_val_f1"]


def run_arm(arm: str, a, out: Path, gpus: list[int]) -> None:
    pool = Pool(arm, gpus, out)
    if arm == "tok":
        base = [sys.executable, str(HERE / "finetune_tok.py"), "--data", str(a.data), "--split", str(a.split)]
        base += shlex.split(a.tok_args)
        cond = lambda c, W: ["--condition", c, "--window", str(W)]  # noqa: E731
        grid, search_conds = A_GRID, {"causal_Wall": ("causal", -1), "bidir_Wall": ("bidir", -1)}
        finals = A_FINAL
    else:
        base = [sys.executable, str(HERE / "finetune_txt.py"), "--data", str(a.data), "--split", str(a.split),
                "--texts", str(a.texts), "--model", a.model] + shlex.split(a.txt_args)
        cond = lambda c, W: ["--condition", c]  # noqa: E731
        grid, search_conds = B_GRID, {"causal": ("causal", None), "bidir": ("bidir", None)}
        finals = B_FINAL
    hp = lambda lr, wd: ["--lr", str(lr), "--wd", str(wd)]  # noqa: E731
    search = [{"name": f"search_{n}_lr{lr}_wd{wd}", "cmd": base + cond(*c) + hp(lr, wd) + ["--seed", "0", "--no_pb"]}
              for lr, wd in grid for n, c in search_conds.items()]
    failed = pool.run(search)
    if failed:
        raise RuntimeError(f"{arm} search failed: {failed}")
    score = {(lr, wd): float(np.mean([best_val(out, f"search_{n}_lr{lr}_wd{wd}") for n in search_conds]))
             for lr, wd in grid}
    lr, wd = max(score, key=score.get)
    (out / "hp_selection.json").write_text(json.dumps(
        {"lr": lr, "wd": wd, "search": {f"{k[0]}_{k[1]}": v for k, v in score.items()}}, indent=2))
    log(f"{arm} selected lr={lr} wd={wd} ({score})")
    jobs = [{"name": f"final_{n}_s{s}", "cmd": base + cond(*c) + hp(lr, wd) + ["--seed", str(s)]}
            for s in SEEDS for n, c in finals.items()]
    failed = pool.run(jobs)
    if failed:
        raise RuntimeError(f"{arm} finals failed: {failed}")
    aggregate(arm, out, list(finals), a, lr, wd)


def aggregate(arm: str, out: Path, names: list[str], a, lr: float, wd: float, n_boot: int = 2000) -> None:
    z = np.load(a.data / "onpolicy" / "steps.npz")
    split = json.loads(a.split.read_text())
    te = np.isin(z["trace_id"], split["test"])
    y, grp = z["y"][te].astype(int), z["problem_id"][te]
    post = ((z["first_error_idx"] >= 0) & (z["step_idx"] > z["first_error_idx"]))[te]
    fit = lambda n, s: out / "fits" / f"final_{n}_s{s}"  # noqa: E731
    M = {n: [json.loads((fit(n, s) / "metrics.json").read_text()) for s in SEEDS] for n in names}
    probs = {n: [np.load(fit(n, s) / "probs_onpolicy.npy")[te] for s in SEEDS] for n in names}
    th = {n: [m["val_threshold"] for m in M[n]] for n in names}

    def stat(n, idx, kind):
        v = [f1_score(y[idx], p[idx] >= t) if kind == "f1" else roc_auc_score(y[idx], p[idx])
             for p, t in zip(probs[n], th[n])]
        return float(np.mean(v))

    uniq, inv = np.unique(grp, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    rng = np.random.default_rng(0)
    boots = [np.concatenate([members[g] for g in rng.integers(len(uniq), size=len(uniq))]) for _ in range(n_boot)]
    all_idx = np.arange(len(y))
    rows = {}
    for n in names:
        r = {}
        for sl, mask in (("all", np.ones(len(y), bool)), ("upto_first_error", ~post), ("after_first_error", post)):
            r[f"f1_{sl}"] = stat(n, all_idx[mask], "f1")
            r[f"auroc_{sl}"] = stat(n, all_idx[mask], "auroc")
        for key in ("processbench_mean_f1_pb_calib20", "processbench_mean_f1_pb_val_threshold",
                    "processbench_mean_f1_pb_oracle", "processbench_mean_known_step_auroc"):
            v = [m[key] for m in M[n]]
            r[key], r[key + "_seeds"] = float(np.mean(v)), v
        r["val_f1"] = float(np.mean([m["best_val_f1"] for m in M[n]]))
        rows[n] = r

    causal = "causal_Wall" if arm == "tok" else "causal"
    bidirs = [n for n in names if n.startswith("bidir")]
    best = max(bidirs, key=lambda n: rows[n]["val_f1"])
    contrasts = [(best, causal), (causal, "local"), (best, "local")]
    if arm == "tok" and best != "bidir_Wall":
        contrasts.insert(1, ("bidir_Wall", causal))
    diffs = {}
    for x, b in contrasts:
        d = {}
        for kind in ("f1", "auroc"):
            bs = [stat(x, idx, kind) - stat(b, idx, kind) for idx in boots[:500 if kind == "auroc" else n_boot]]
            d[kind] = {"diff": stat(x, all_idx, kind) - stat(b, all_idx, kind),
                       "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5))}
        pa = np.array(rows[x]["processbench_mean_f1_pb_calib20_seeds"])
        pb = np.array(rows[b]["processbench_mean_f1_pb_calib20_seeds"])
        d["pb_f1_calib20"] = {"diff": float((pa - pb).mean()), "per_seed": (pa - pb).tolist()}
        diffs[f"{x} - {b}"] = d
    d0 = diffs[f"{best} - {causal}"]
    go = d0["f1"]["lo"] > 0 and d0["pb_f1_calib20"]["diff"] >= 0
    prev = float(y.mean())
    summary = {"arm": arm, "lr": lr, "wd": wd, "best_bidir": best, "test_prevalence": prev,
               "always_positive_f1": 2 * prev / (1 + prev), "conditions": rows, "contrasts": diffs, "GO": bool(go)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = [f"# step_bert_probe_v1 arm {arm}\n", f"lr {lr}, wd {wd}, seeds {SEEDS}, test prevalence {prev:.3f}, "
             f"always-positive F1 {2 * prev / (1 + prev):.3f}\n",
             "| condition | test F1 | test AUROC | F1 up to 1st err | F1 after 1st err | PB F1_PB calib-20 | PB oracle | PB known-step AUROC |",
             "|---|---|---|---|---|---|---|---|"]
    for n, r in rows.items():
        lines.append(f"| {n} | {r['f1_all']:.3f} | {r['auroc_all']:.3f} | {r['f1_upto_first_error']:.3f} | "
                     f"{r['f1_after_first_error']:.3f} | {r['processbench_mean_f1_pb_calib20']:.3f} | "
                     f"{r['processbench_mean_f1_pb_oracle']:.3f} | {r['processbench_mean_known_step_auroc']:.3f} |")
    lines += ["", "| contrast | test F1 diff [95% CI] | test AUROC diff [95% CI] | PB F1_PB calib-20 diff (per seed) |",
              "|---|---|---|---|"]
    for k, d in diffs.items():
        lines.append(f"| {k} | {d['f1']['diff']:+.4f} [{d['f1']['lo']:+.4f}, {d['f1']['hi']:+.4f}] | "
                     f"{d['auroc']['diff']:+.4f} [{d['auroc']['lo']:+.4f}, {d['auroc']['hi']:+.4f}] | "
                     f"{d['pb_f1_calib20']['diff']:+.4f} ({', '.join(f'{v:+.3f}' for v in d['pb_f1_calib20']['per_seed'])}) |")
    lines += ["", f"Decision rule ({best} - {causal}: test F1 CI above 0 and PB diff >= 0): **{'GO' if go else 'NO-GO'}**"]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    log(f"{arm} aggregate written\n" + "\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--texts", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--out_root", type=Path, required=True)
    p.add_argument("--arms", nargs="+", default=["tok", "txt"])
    p.add_argument("--gpus", type=int, default=4)
    p.add_argument("--tok_args", default="", help="extra finetune_tok.py args")
    p.add_argument("--txt_args", default="", help="extra finetune_txt.py args")
    a = p.parse_args()
    errs = []

    def go(arm):
        try:
            run_arm(arm, a, a.out_root / arm, list(range(a.gpus)))
        except Exception as e:  # noqa: BLE001
            errs.append(f"{arm}: {e!r}")
            log(f"{arm} FAILED {e!r}")

    ts = [threading.Thread(target=go, args=(arm,)) for arm in a.arms]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    if errs:
        raise SystemExit("; ".join(errs))
    log("ALL DONE")


if __name__ == "__main__":
    main()
