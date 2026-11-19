#!/usr/bin/env python3
"""STEP on InStep (PRM800K)'s training data (reviewer W2): the same MLP, loss and optimizer as
step_probe.py (STEP, Liang et al. 2026), trained for seed s on the last-layer state at the last
token of each of the 100,000 PRM800K steps InStep (PRM800K) seed s trains on, with their human
labels; early stopping on PRM800K's validation items (as InStep (PRM800K) selects on them).

States are read_token_inline.py encode outputs (`passive_final`, the post-norm last-layer text
state at the step's last token, chat layout as InStep's training): --train_states holds the union
of the three draws (one or more parts, --train_names), each seed takes its draw's uids
(prm_review_data.py draws); --val_states/--val_name the validation items (labels from the npz).
ProcessBench states (reprobe layout, as InStep reads ProcessBench) in --pb_states, pb_<subset>.
Scores: <out>/seed<s>/pb_<subset>.jsonl and val.jsonl ({"uid", "p"}), summary in <out>/summary.json.

    python scripts/instep_v1/step_prm.py --train_states $ST/prm_union --train_names union_p0,union_p1 \\
        --draws $W/data --val_states $ST/prm_union --val_name val5k --pb_states $ST/qwen3_8b \\
        --out $OUT/scores/qwen3_8b/step_prm
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
from scripts.instep_v1.passive_fit import load_set, load_x  # noqa: E402
from scripts.instep_v1.step_probe import DEFAULTS, KEY, n_params, predict, train_seed  # noqa: E402
from scripts.reprobe_cmp.analyze import auroc, fast_ap  # noqa: E402

PB = ("gsm8k", "math", "olympiadbench", "omnimath")


def load_union(states: Path, names: list[str]) -> tuple[dict, np.ndarray, np.ndarray]:
    """{uid: row}, X (n, d) float32, y (n,) over the kept rows of every part."""
    xs, ys, uids = [], [], []
    for nm in names:
        u, y, kept = load_set(states, nm)
        xs.append(load_x(states, nm, KEY)[kept])
        ys.append(y[kept])
        uids += [x for x, k in zip(u, kept) if k]
    row = {u: i for i, u in enumerate(uids)}
    if len(row) != len(uids):
        raise SystemExit("duplicate uids across the training parts")
    return row, np.concatenate(xs), np.concatenate(ys)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train_states", type=Path, required=True)
    p.add_argument("--train_names", required=True)
    p.add_argument("--draws", type=Path, required=True, help="dir with draw_seed<s>.uids.json")
    p.add_argument("--val_states", type=Path, required=True)
    p.add_argument("--val_name", required=True)
    p.add_argument("--pb_states", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seeds", default="0,1,2")
    for k, v in DEFAULTS.items():
        p.add_argument(f"--{k}", type=type(v), default=v)
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.perf_counter()
    row, X, Y = load_union(a.train_states, a.train_names.split(","))
    vu, vy, vk = load_set(a.val_states, a.val_name)
    Xva = torch.from_numpy(np.nan_to_num(load_x(a.val_states, a.val_name, KEY)[vk])).to(dev)
    yva = torch.from_numpy(vy[vk].astype(np.float32)).to(dev)
    pb = {}
    for s in PB:
        u, y, k = load_set(a.pb_states, f"pb_{s}")
        pb[s] = ([x for x, kk in zip(u, k) if kk], torch.from_numpy(np.nan_to_num(load_x(a.pb_states, f"pb_{s}", KEY)[k])))
    summary = {"key": KEY, "hparams": {k: getattr(a, k) for k in DEFAULTS}, "params": n_params(X.shape[1], a.hidden),
               "n_val": int(len(yva)), "per_seed": {}}
    for seed in [int(s) for s in a.seeds.split(",")]:
        uids = json.loads((a.draws / f"draw_seed{seed}.uids.json").read_text())
        miss = [u for u in uids if u not in row]
        if miss:
            raise SystemExit(f"seed {seed}: {len(miss)} drawn uids have no state (e.g. {miss[:3]})")
        idx = np.array([row[u] for u in uids])
        Xtr = torch.from_numpy(np.nan_to_num(X[idx])).to(dev)
        ytr = torch.from_numpy(Y[idx].astype(np.float32)).to(dev)
        rate = float(ytr.mean())
        print(f"[step_prm] seed {seed}: {len(idx)} training steps, positive rate {rate:.4f}", flush=True)
        model, info = train_seed(Xtr, ytr, Xva, yva, seed, a)
        sv = torch.sigmoid(predict(model, Xva)).cpu().numpy()
        info.update({"n_train": int(len(idx)), "train_pos_rate": rate, "val_ap": fast_ap(yva.cpu().numpy(), sv),
                     "val_auroc": auroc(yva.cpu().numpy(), sv)})
        od = a.out / f"seed{seed}"
        od.mkdir(parents=True, exist_ok=True)
        with open(od / "val.jsonl", "w") as f:
            for u, v in zip([x for x, k in zip(vu, vk) if k], sv):
                f.write(json.dumps({"uid": u, "p": float(v)}) + "\n")
        for s, (u, x) in pb.items():
            sc = torch.sigmoid(predict(model, x.to(dev))).cpu().numpy().astype(np.float64)
            with open(od / f"pb_{s}.jsonl", "w") as f:
                for uu, v in zip(u, sc):
                    f.write(json.dumps({"uid": uu, "p": float(v)}) + "\n")
        summary["per_seed"][str(seed)] = info
        print(f"[step_prm] seed {seed}: best epoch {info['best_epoch']}, val AUROC {info['val_auroc']:.4f} "
              f"AP {info['val_ap']:.4f}", flush=True)
        del Xtr, ytr
    summary["seconds"] = time.perf_counter() - t0
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"[step_prm] done in {summary['seconds']:.0f}s -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
