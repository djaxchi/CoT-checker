#!/usr/bin/env python3
"""Score store splits with existing contextual-probe checkpoints (risk_probe_v0).

Loads each `best.pt` into the §21.18 `local`-arm architecture (train_contextual_
token_probe.py defaults, 2,638,593 parameters; checked) and scores every step of
the requested splits through that script's own TrajStore and score_split, so the
path is the one that wrote the saved predictions the reproduction gate compares
against. Writes <out_dir>/<name>.jsonl: split, trace_id, step, logit.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from train_contextual_token_probe import TrajStore, score_split  # noqa: E402
from src.probes.contextual_token_probe import ContextualTokenProbe  # noqa: E402

N_PARAMS = 2_638_593


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--ckpt", nargs="+", required=True, help="name=path/to/best.pt")
    ap.add_argument("--splits", nargs="+", required=True)
    ap.add_argument("--arm", default="local")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--cost_budget", type=int, default=48_000_000)
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "4")))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    store = TrajStore(a.store, splits=set(a.splits))
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for spec in a.ckpt:
        name, path = spec.split("=", 1)
        ck = torch.load(path, map_location=device, weights_only=False)
        model = ContextualTokenProbe(d_in=store.dim, d=256, layers=2, heads=4, ff=1024, dropout=0.1).to(device)
        n = sum(p.numel() for p in model.parameters())
        if n != N_PARAMS:
            sys.exit(f"[score] {name}: architecture has {n} params, expected {N_PARAMS}")
        model.load_state_dict(ck["model"])
        rows = 0
        with open(a.out_dir / f"{name}.jsonl", "w") as f:
            for sp in a.splits:
                ids = store.ids(sp)
                sc = score_split(model, store, ids, a.arm, device, a.cost_budget)
                for tid in ids:
                    for k, v in enumerate(sc[tid]):
                        f.write(json.dumps({"split": sp, "trace_id": tid, "step": k, "logit": v}) + "\n")
                        rows += 1
        print(f"[score] {name}: {rows} rows from {path} (epoch {ck.get('epoch')})", flush=True)


if __name__ == "__main__":
    main()
