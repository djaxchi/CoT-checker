"""Unattended driver for step_bert_probe_v0, from pretraining to the decision table.

Stages (each skips work whose output already exists, so it can be re-run):
  1. pretrain the four models (pt_{state,contribution}_{bidir,causal}), 150 epochs max, patience 10
  2. G1 eval on each                      -> results/step_bert_probe_v0/g1_<tok>_<arm>.json
  3. token set = the one whose bidir model has the higher residualized e_both
     AUROC on test (ties within 0.01 -> state)
  4. HP search: causal(W=all) and bidir(W=all), lr x wd, seed 0; one shared
     setting chosen by the mean val F1 of the two, so the contrast stays matched
  5. finals, seeds 42/43/44: local, causal(W=all), bidir W in {1,2,4,all};
     then bidir_noPT at the bidir W with the best mean val F1
  6. aggregate -> results/step_bert_probe_v0/summary.{json,md}
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "data/tamia/step_bert_probe_v0"
P = D / "pooled"
RUNS = ROOT / "runs/step_bert_probe_v0"
RES = ROOT / "results/step_bert_probe_v0"
PY = str(ROOT / ".venv/bin/python")
PB = [str(P / f"pooled_pb_{s}.npz") for s in ("gsm8k", "math", "olympiadbench", "omnimath")]
SMOKE = os.environ.get("STEP_BERT_SMOKE") == "1"   # tiny end-to-end check on the 1k subset
POOLED = str(P / ("pooled_onpolicy_sub1000.npz" if SMOKE else "pooled_onpolicy_full.npz"))
SPLIT = str(D / ("split_sub1000.json" if SMOKE else "split.json"))
if SMOKE:
    RUNS, RES = RUNS / "smoke_overnight", RES / "smoke_overnight"
PT_EPOCHS, FT_EPOCHS = ("2", "1") if SMOKE else ("150", "40")
PARALLEL = 4
SEEDS = (42, 43, 44)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd: list[str], logfile: Path) -> int:
    logfile.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="2")
    with logfile.open("w") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=ROOT).returncode


def finetune(name: str, tok: str, condition: str, window: int, pretrained: str | None,
             lr: float, wd: float, seed: int, pb: bool = True) -> Path:
    out = RUNS / "ft" / name
    if (out / "metrics.json").exists():
        return out
    cmd = [PY, "scripts/step_bert/finetune.py", "--pooled", POOLED, "--split", SPLIT, "--tokens", tok,
           "--condition", condition, "--window", str(window), "--lr", str(lr), "--wd", str(wd),
           "--seed", str(seed), "--epochs", FT_EPOCHS, "--out_dir", str(out)]
    if pretrained:
        cmd += ["--pretrained_dir", pretrained]
    if pb:
        cmd += ["--pb", *PB]
    log(f"start {name}")
    rc = run(cmd, RUNS / "logs" / f"ft_{name}.log")
    log(f"done  {name} rc={rc}")
    return out


def run_many(jobs: list[dict]) -> list[Path]:
    with ThreadPoolExecutor(PARALLEL) as ex:
        return list(ex.map(lambda j: finetune(**j), jobs))


def metrics(out: Path) -> dict:
    return json.loads((out / "metrics.json").read_text())


def main() -> None:
    RES.mkdir(parents=True, exist_ok=True)
    arms = [(t, a) for t in ("state", "contribution") for a in ("bidir", "causal")]

    # 1. pretraining (launched here if not already done), then wait
    procs = []
    for t, a in arms:
        out = RUNS / f"pt_{t}_{a}"
        if not (out / "pretrain_log.json").exists():
            modes = ["both", "left", "right"] if a == "bidir" else ["left"]
            cmd = [PY, "scripts/step_bert/pretrain.py", "--pooled", POOLED, "--pb", *PB, "--split", SPLIT,
                   "--tokens", t, "--modes", *modes, "--epochs", PT_EPOCHS, "--patience", "10", "--out_dir", str(out)]
            (RUNS / "logs").mkdir(parents=True, exist_ok=True)
            f = (RUNS / "logs" / f"pt_{t}_{a}.log").open("w")
            procs.append(subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT,
                                          env=dict(os.environ, OMP_NUM_THREADS="2")))
            log(f"launched pretraining {t}_{a}")
    while not all((RUNS / f"pt_{t}_{a}" / "pretrain_log.json").exists() for t, a in arms):
        if procs and all(p.poll() is not None for p in procs):
            missing = [f"{t}_{a}" for t, a in arms if not (RUNS / f"pt_{t}_{a}" / "pretrain_log.json").exists()]
            sys.exit(f"pretraining died; missing {missing}")
        time.sleep(60)
    log("pretraining complete")

    # 2. G1
    for t, a in arms:
        out = RES / f"g1_{t}_{a}.json"
        if not out.exists():
            run([PY, "scripts/step_bert/g1_eval.py", "--pooled", POOLED, "--split", SPLIT,
                 "--scores_dir", str(RUNS / f"pt_{t}_{a}"), "--pb", *PB, "--out", str(out)],
                RUNS / "logs" / f"g1_{t}_{a}.log")
        log(f"G1 {t}_{a}: pass={json.loads(out.read_text())['G1_pass']}")

    # 3. token set
    g = {t: json.loads((RES / f"g1_{t}_bidir.json").read_text())["slices"]["all"]["auroc"]["both_resid"]["auroc"]
         for t in ("state", "contribution")}
    tok = "contribution" if g["contribution"] > g["state"] + 0.01 else "state"
    log(f"token set {tok} (both_resid AUROC {g})")
    pt = {a: str(RUNS / f"pt_{tok}_{a}") for a in ("bidir", "causal")}

    # 4. HP search
    grid = [(lr, wd) for lr in (3e-4, 1e-4) for wd in (0.01, 0.1)]
    jobs = [dict(name=f"search_{c}_lr{lr:g}_wd{wd:g}", tok=tok, condition=c, window=-1, pretrained=pt[c],
                 lr=lr, wd=wd, seed=0, pb=False) for c in ("causal", "bidir") for lr, wd in grid]
    outs = run_many(jobs)
    score = {}
    for j, o in zip(jobs, outs):
        score.setdefault((j["lr"], j["wd"]), []).append(metrics(o)["best_val_f1"])
    lr, wd = max(score, key=lambda k: np.mean(score[k]))
    log(f"HP lr={lr} wd={wd} (mean val F1 {np.mean(score[(lr, wd)]):.4f})")
    (RES / "hp_selection.json").write_text(json.dumps(
        {"token_set": tok, "lr": lr, "wd": wd, "search": {f"{k[0]}_{k[1]}": v for k, v in score.items()}}, indent=2))

    # 5. finals
    conds = [("local", "local", 0, pt["bidir"]), ("causal_Wall", "causal", -1, pt["causal"])]
    conds += [(f"bidir_W{w if w >= 0 else 'all'}", "bidir", w, pt["bidir"]) for w in (1, 2, 4, -1)]
    jobs = [dict(name=f"final_{n}_s{s}", tok=tok, condition=c, window=w, pretrained=p, lr=lr, wd=wd, seed=s)
            for n, c, w, p in conds for s in SEEDS]
    run_many(jobs)
    bid = {n: np.mean([metrics(RUNS / "ft" / f"final_{n}_s{s}")["best_val_f1"] for s in SEEDS])
           for n, c, _, _ in conds if c == "bidir"}
    best = max(bid, key=bid.get)
    w_best = next(w for n, c, w, _ in conds if n == best)
    log(f"best bidir by val F1: {best} {bid}")
    run_many([dict(name=f"final_{best}_noPT_s{s}", tok=tok, condition="bidir", window=w_best, pretrained=None,
                   lr=lr, wd=wd, seed=s) for s in SEEDS])

    # 6. aggregate
    names = [n for n, *_ in conds] + [f"{best}_noPT"]
    aggregate(names, best, tok, lr, wd)
    log("ALL DONE")


def aggregate(names: list[str], best: str, tok: str, lr: float, wd: float, n_boot: int = 2000) -> None:
    z = np.load(POOLED)
    split = json.loads(Path(SPLIT).read_text())
    te = np.isin(z["trace_id"], split["test"])
    y, grp = z["y"][te].astype(int), z["problem_id"][te]
    post = ((z["first_error_idx"] >= 0) & (z["step_idx"] > z["first_error_idx"]))[te]
    probs = {n: [np.load(RUNS / "ft" / f"final_{n}_s{s}" / "probs_onpolicy.npy")[te] for s in SEEDS] for n in names}
    th = {n: [metrics(RUNS / "ft" / f"final_{n}_s{s}")["val_threshold"] for s in SEEDS] for n in names}
    M = {n: [metrics(RUNS / "ft" / f"final_{n}_s{s}") for s in SEEDS] for n in names}

    def stat(n, idx, kind):
        vals = []
        for p, t in zip(probs[n], th[n]):
            yy, pp = y[idx], p[idx]
            vals.append(f1_score(yy, pp >= t) if kind == "f1" else
                        (roc_auc_score(yy, pp) if 0 < yy.mean() < 1 else np.nan))
        return float(np.mean(vals))

    uniq, inv = np.unique(grp, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    rng = np.random.default_rng(0)
    boots = [np.concatenate([members[g] for g in rng.integers(len(uniq), size=len(uniq))]) for _ in range(n_boot)]
    all_idx = np.arange(len(y))

    rows = {}
    for n in names:
        r = {}
        for sl, mask in (("all", np.ones(len(y), bool)), ("upto_first_error", ~post), ("after_first_error", post)):
            idx = all_idx[mask]
            r[f"f1_{sl}"] = stat(n, idx, "f1")
            r[f"auroc_{sl}"] = stat(n, idx, "auroc")
        for key in ("processbench_mean_f1_pb_calib20", "processbench_mean_f1_pb_val_threshold",
                    "processbench_mean_f1_pb_oracle", "processbench_mean_known_step_auroc"):
            v = [m[key] for m in M[n]]
            r[key] = float(np.mean(v))
            r[key + "_seeds"] = v
        r["val_f1"] = float(np.mean([m["best_val_f1"] for m in M[n]]))
        rows[n] = r

    contrasts = [(best, "causal_Wall"), ("bidir_Wall", "causal_Wall"), (best, f"{best}_noPT"),
                 ("causal_Wall", "local"), (f"{best}_noPT", "causal_Wall")]
    diffs = {}
    for a, b in contrasts:
        d = {}
        for kind in ("f1", "auroc"):
            point = stat(a, all_idx, kind) - stat(b, all_idx, kind)
            bs = [stat(a, idx, kind) - stat(b, idx, kind) for idx in boots[:500 if kind == "auroc" else n_boot]]
            d[kind] = {"diff": point, "lo": float(np.nanpercentile(bs, 2.5)), "hi": float(np.nanpercentile(bs, 97.5))}
        pa = np.array(rows[a]["processbench_mean_f1_pb_calib20_seeds"])
        pb = np.array(rows[b]["processbench_mean_f1_pb_calib20_seeds"])
        d["pb_f1_calib20"] = {"diff": float((pa - pb).mean()), "per_seed": (pa - pb).tolist()}
        diffs[f"{a} - {b}"] = d

    d0 = diffs[f"{best} - causal_Wall"]
    go = d0["f1"]["lo"] > 0 and d0["pb_f1_calib20"]["diff"] >= 0
    prev = float(y.mean())
    summary = {"token_set": tok, "lr": lr, "wd": wd, "best_bidir": best, "test_prevalence": prev,
               "always_positive_f1": 2 * prev / (1 + prev), "conditions": rows, "contrasts": diffs, "GO": bool(go)}
    (RES / "summary.json").write_text(json.dumps(summary, indent=2))

    lines = [f"# step_bert_probe_v0 summary\n", f"token set `{tok}`, lr {lr}, wd {wd}, seeds {SEEDS}, "
             f"test prevalence {prev:.3f}, always-positive F1 {2 * prev / (1 + prev):.3f}\n",
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
                     f"{d['pb_f1_calib20']['diff']:+.4f} ({', '.join(f'{x:+.3f}' for x in d['pb_f1_calib20']['per_seed'])}) |")
    lines += ["", f"Decision rule ({best} - causal_Wall: test F1 CI above 0 and PB diff >= 0): **{'GO' if go else 'NO-GO'}**"]
    (RES / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
