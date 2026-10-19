#!/usr/bin/env python3
"""One risk_probe_v2 fit: the leaderboard's sequence cell trained on risk labels
(docs/risk_probe_v2_plan.md, sections 2 to 5).

The model, optimizer, loss, batching and early stopping are the leaderboard
trainer's own (scripts/train_rep_learner_cell.py: build_learner, SpanLoader,
train_one): AdamW + BCE, length-bucketed batches, early stopping on validation
loss. Only the data and the targets change:

  * items: the label-set step-span store (verifier template, L35), joined by uid to
    build_label_jsonl.py's rows;
  * target: `soft` (P(err_after > err_before), BCE against a probability) or
    `harmful` (the hard 2-SE label);
  * split: --fold k trains on the other problem folds and scores fold k
    (oof.jsonl); --fold full trains on every fold and scores ProcessBench
    (pb_step_scores_<subset>.jsonl, the leaderboard's per-trace format). Inside
    every training set the problems with is_val(problem_key) are the validation
    set (early stopping, hyperparameter selection by AUROC against hard harmful).

Hyperparameters: the lr x weight-decay grid is searched unless --hp_from names a
results.json of the same target; the searched model with the winning setting is
kept (it is the fit at this seed and fold). Scores are raw logits.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from train_easy_probe_method import auroc_numpy, evaluate_processbench  # noqa: E402
from train_rep_learner_cell import build_handles, train_one  # noqa: E402
from src.analysis.risk_probe_v2 import is_val  # noqa: E402
from src.harness.learners import build_learner, param_count  # noqa: E402
from src.harness.spanloader import SpanLoader  # noqa: E402
from src.repstore import split_fingerprint  # noqa: E402
from src.repstore.store import ShardedRepSplit  # noqa: E402

D512 = "transformer:d512,l2,f2048,h8"


@torch.no_grad()
def score_logits(model, collate, idx: np.ndarray, batch_size: int) -> np.ndarray:
    model.eval()
    out = np.empty(len(idx), dtype=np.float32)
    for i in range(0, len(idx), batch_size):
        b = idx[i:i + batch_size]
        xb, mb, _ = collate(b)
        out[i:i + len(b)] = model(xb, mb).float().cpu().numpy()
    return out


def chunks(idx: np.ndarray, batch_size: int) -> list[np.ndarray]:
    return [idx[i:i + batch_size] for i in range(0, len(idx), batch_size)]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store", type=Path, required=True, help="rep root holding the label split")
    p.add_argument("--stem", default="label")
    p.add_argument("--labels", type=Path, required=True, help="build_label_jsonl.py output")
    p.add_argument("--target", choices=("soft", "harmful"), required=True)
    p.add_argument("--fold", required=True, help="0..4 or full")
    p.add_argument("--pb_store", type=Path, required=True)
    p.add_argument("--pb_subsets", nargs="+", default=["gsm8k", "math", "olympiadbench", "omnimath"])
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--learner", default=D512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hp_from", type=Path, default=None)
    p.add_argument("--lr_grid", type=float, nargs="+", default=[1e-3, 3e-4, 1e-4])
    p.add_argument("--wd_grid", type=float, nargs="+", default=[0.0, 0.01])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--t_max", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--limit", type=int, default=0, help="smoke only: first N label items")
    a = p.parse_args()
    full = a.fold == "full"
    fold = None if full else int(a.fold)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    inputs = {f"label/{a.stem}": split_fingerprint(a.store / a.stem)}
    handles, meta = build_handles(ShardedRepSplit(a.store / a.stem))
    labels = {r["uid"]: r for r in map(json.loads, open(a.labels))}
    missing = [m["uid"] for m in meta if m["uid"] not in labels]
    if missing:
        sys.exit(f"[risk] {len(missing)} store items without a label row, e.g. {missing[:3]}")
    rows = [labels[m["uid"]] for m in meta]
    n = len(rows) if not a.limit else min(a.limit, len(rows))
    rows, handles = rows[:n], handles[:n]
    tgt = np.array([r["soft"] if a.target == "soft" else float(r["label"]) for r in rows], np.float32)
    hard = np.array([int(r["label"]) for r in rows])
    folds = np.array([int(r["fold"]) for r in rows])
    val_mask = np.array([is_val(r["problem_id"]) for r in rows])
    in_train_folds = np.ones(n, bool) if full else folds != fold
    tr_idx = np.where(in_train_folds & ~val_mask)[0]
    va_idx = np.where(in_train_folds & val_mask)[0]
    te_idx = np.array([], dtype=np.int64) if full else np.where(folds == fold)[0]
    handles = [(rs, li, ss, nt, float(t)) for (rs, li, ss, nt, _), t in zip(handles, tgt)]
    loader = SpanLoader(handles, a.t_max, device, preload=True)
    d = loader.d
    print(f"[risk] target={a.target} fold={a.fold} seed={a.seed} train={len(tr_idx)} val={len(va_idx)} "
          f"test={len(te_idx)} harmful(train)={int(hard[tr_idx].sum())} device={device.type}", flush=True)

    plan = lambda rng: loader.batches(a.batch_size, rng, bucketed=True, subset=tr_idx)  # noqa: E731
    val_plan = chunks(va_idx, a.batch_size)

    def fit(lr, wd):
        torch.manual_seed(a.seed)
        m = build_learner(a.learner, d, t_max=a.t_max, dropout=a.dropout).to(device)
        m, vloss = train_one(m, loader.collate, plan, len(va_idx), val_plan, loader.collate, a.epochs,
                             lr, wd, a.patience, a.seed)
        return m, vloss, float(auroc_numpy(hard[va_idx], score_logits(m, loader.collate, va_idx, a.batch_size)))

    trials, model, best = [], None, None
    if a.hp_from is not None:
        prior = json.loads(a.hp_from.read_text())
        if (prior["target"], prior["learner"]) != (a.target, a.learner):
            sys.exit(f"[risk] --hp_from is {prior['target']} x {prior['learner']}, not {a.target} x {a.learner}")
        best = prior["hp"]["selected"]
        model, vloss, vauc = fit(best["lr"], best["weight_decay"])
        trials = [{"lr": best["lr"], "weight_decay": best["weight_decay"], "val_loss": vloss, "val_auroc": vauc}]
    else:
        best_auc = -1.0
        for lr in a.lr_grid:
            for wd in a.wd_grid:
                m, vloss, vauc = fit(lr, wd)
                trials.append({"lr": lr, "weight_decay": wd, "val_loss": vloss, "val_auroc": vauc})
                print(f"[hp] lr={lr} wd={wd} val_loss={vloss:.4f} val_auroc={vauc:.4f}", flush=True)
                if vauc > best_auc:
                    best_auc, best, model = vauc, {"lr": lr, "weight_decay": wd}, m
                else:
                    del m
    res = {"target": a.target, "fold": a.fold, "seed": a.seed, "learner": a.learner, "dim": int(d), "n_params": param_count(model),
           "inputs": inputs, "labels_file": str(a.labels), "n_train": int(len(tr_idx)), "n_val": int(len(va_idx)),
           "n_test": int(len(te_idx)),
           "hp": {"selected": best, "trials": trials, "reused_from": str(a.hp_from) if a.hp_from else None},
           "protocol": {"epochs": a.epochs, "patience": a.patience, "batch_size": a.batch_size, "t_max": a.t_max,
                        "dropout": a.dropout, "bucketed": True, "rescale": "none", "limit": a.limit},
           "val_auroc_harmful": max(t["val_auroc"] for t in trials) if a.hp_from is None else trials[0]["val_auroc"],
           "split_problems": {k: sorted({rows[i]["problem_id"] for i in ix})
                              for k, ix in (("train", tr_idx), ("val", va_idx), ("test", te_idx))}}

    if not full:
        s = score_logits(model, loader.collate, te_idx, a.batch_size)
        res["test_auroc_harmful"] = float(auroc_numpy(hard[te_idx], s))
        with open(a.out_dir / "oof.jsonl", "w") as f:
            for i, v in zip(te_idx, s):
                f.write(json.dumps({"uid": rows[i]["uid"], "score": float(v)}) + "\n")
        print(f"[risk] fold {fold}: test AUROC harmful {res['test_auroc_harmful']:.4f}", flush=True)
    else:
        for sub in a.pb_subsets:
            sd = a.pb_store / sub
            if not sd.exists():
                sys.exit(f"[risk] missing ProcessBench subset {sd}")
            inputs[f"pb/{sub}"] = split_fingerprint(sd)
            h, m = build_handles(ShardedRepSplit(sd))
            pl = SpanLoader(h, a.t_max, device, preload=True)
            s = score_logits(model, pl.collate, np.arange(len(h)), a.batch_size)
            pb_rows, _ = evaluate_processbench(s, m, 0.0)
            with open(a.out_dir / f"pb_step_scores_{sub}.jsonl", "w") as f:
                for r in pb_rows:
                    f.write(json.dumps(r) + "\n")
            print(f"[risk] scored ProcessBench {sub}: {len(pb_rows)} traces", flush=True)
    res["wall_seconds"] = round(time.time() - t0, 1)
    (a.out_dir / "results.json").write_text(json.dumps(res, indent=1))
    torch.save(model.state_dict(), a.out_dir / "model.pt")
    print(f"[risk] done in {res['wall_seconds']}s", flush=True)


if __name__ == "__main__":
    main()
