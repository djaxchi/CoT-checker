#!/usr/bin/env python3
"""Train contextual token probes (several arms, one seed, one HP setting) and
score every step of the evaluation splits with each arm's best checkpoint.

bidirectional_token_probe_v1 (plan sections 7-10). All arms in one process see
the SAME effective batches in the same order (sampler depends only on seed and
epoch) and start from the SAME initial weights (seeded before each model is
built; every arm has the same parameter count). Each arm then has its own
optimizer and early-stopping state.

Objective: unweighted BCE, mean over a trajectory's labeled steps, then mean over
the 32 trajectories of an effective batch; microbatches/cropped views carry
per-target weights so the split never changes the objective.

Selection: after every epoch, best incorrect-step F1 over all thresholds on
labeled dev steps (higher threshold on ties). Patience 5. Checkpoints: best.pt
and last.pt per arm (last.pt also every --ckpt_every effective batches, with
optimizer, RNG and sampler position). A re-run with the same out_dir resumes.

Outputs per arm under out_dir/<arm>/: best.pt, last.pt, history.json,
predictions.jsonl.gz (all steps of --score_splits), done.json.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
import sys
import threading
import queue
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.eval.contextual_probe_metrics import best_f1_threshold  # noqa: E402
from src.probes.contextual_token_probe import (  # noqa: E402
    ContextualTokenProbe, collate, gather_scores, target_weights, weighted_bce,
)


# ------------------------------------------------------------------ store

class TrajStore:
    """Read-only view over the sharded trajectory store."""

    def __init__(self, root: Path, splits: set[str] | None = None, label_override: Path | None = None):
        self.h, self.items = [], {}
        for sd in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("shard_")
                         and not p.name.endswith(".tmp")):
            if not (sd / "DONE").exists():
                raise SystemExit(f"[FATAL] incomplete shard {sd}")
            k = len(self.h)
            self.h.append(np.load(sd / "h.npy", mmap_mode="r"))
            lengths = np.load(sd / "lengths.npy")
            off = np.concatenate([[0], np.cumsum(lengths)])
            metas = [json.loads(l) for l in open(sd / "meta.jsonl")]
            labs = [json.loads(l) for l in open(sd / "step_labels.jsonl")]
            for i, (m, lb) in enumerate(zip(metas, labs)):
                assert m["trace_id"] == lb["trace_id"]
                if splits is not None and m["split"] not in splits:
                    continue
                self.items[m["trace_id"]] = {
                    "shard": k, "a": int(off[i]), "b": int(off[i + 1]), "split": m["split"],
                    "step_starts": m["step_starts"], "step_ends": m["step_ends"],
                    "y": lb["y"], "label_mask": lb["label_mask"]}
        self.dim = self.h[0].shape[1] if self.h else 0
        self.n_overridden = 0
        if label_override is not None:
            # replace step labels (e.g. self-annotation) for traces listed in the
            # override dir; everything else, incl. evaluation sets, keeps the store's
            for f in sorted(Path(label_override).glob("*.jsonl")):
                for l in open(f):
                    r = json.loads(l)
                    it = self.items.get(r["trace_id"])
                    if it is not None:
                        if len(r["y"]) != len(it["y"]):
                            raise SystemExit(f"[FATAL] override length mismatch {r['trace_id']}")
                        it["y"], it["label_mask"] = r["y"], r["label_mask"]
                        self.n_overridden += 1

    def ids(self, split: str) -> list[str]:
        return sorted(t for t, v in self.items.items() if v["split"] == split)

    def load(self, tid: str) -> dict:
        it = self.items[tid]
        H = torch.from_numpy(np.array(self.h[it["shard"]][it["a"]:it["b"]]))
        return {"H": H, "step_starts": it["step_starts"], "step_ends": it["step_ends"],
                "trace_id": tid}


def labeled(it: dict) -> dict[int, int]:
    return {k: int(y) for k, (y, m) in enumerate(zip(it["y"], it["label_mask"])) if m}


# ------------------------------------------------------------------ sampler

def epoch_batches(ids: list[str], lengths: dict[str, int], seed: int, epoch: int,
                  batch: int = 32, bucket: int = 50) -> list[list[str]]:
    """Length-bucketed, deterministic in (seed, epoch); independent of the arm."""
    rng = np.random.default_rng([seed, epoch])
    idx = list(ids)
    rng.shuffle(idx)
    out = []
    span = batch * bucket
    for i in range(0, len(idx), span):
        chunk = sorted(idx[i:i + span], key=lambda t: (lengths[t], t))
        out.extend(chunk[j:j + batch] for j in range(0, len(chunk), batch))
    order = rng.permutation(len(out))
    return [out[i] for i in order]


def view_rows(tr: dict, arm: str, target: int | None) -> int:
    """Probe rows of the view that scores `target` (None = whole trace)."""
    ss, se = tr["step_starts"], tr["step_ends"]
    c = len(ss) - 1 if (arm != "future1" or target is None) else min(target + 1, len(ss) - 1)
    return ss[0] + sum(e - s + 1 for s, e in zip(ss[:c + 1], se[:c + 1]))


def microbatches(trs: list[dict], arm: str, targets: list[list[int]], cost_budget: int):
    """Yield [(trace index, targets)] chunks whose padded attention cost
    (sequences * max_rows^2) stays within budget. future1 views of one trace may
    be spread over several chunks; a trace appears at most once per chunk."""
    units = []
    for i in sorted(range(len(trs)), key=lambda i: -trs[i]["H"].shape[0]):
        if not targets[i]:
            continue
        if arm == "future1":
            units.extend((i, [t], view_rows(trs[i], arm, t)) for t in sorted(targets[i], reverse=True))
        else:
            units.append((i, list(targets[i]), view_rows(trs[i], arm, None)))
    cur: dict[int, list[int]] = {}
    n_seq, mx = 0, 0
    for i, t, r in units:
        ns, m2 = n_seq + 1, max(mx, r)
        if cur and ns * m2 * m2 > cost_budget:
            yield [(k, sorted(v)) for k, v in cur.items()]
            cur, ns, m2 = {}, 1, r
        cur.setdefault(i, []).extend(t)
        n_seq, mx = ns, m2
    if cur:
        yield [(k, sorted(v)) for k, v in cur.items()]


class Prefetch:
    """Background thread that loads the next effective batches from the memmaps."""

    def __init__(self, store: TrajStore, batches: list[list[str]], start: int, depth: int = 3):
        self.q: queue.Queue = queue.Queue(depth)
        self.t = threading.Thread(target=self._run, args=(store, batches, start), daemon=True)
        self.t.start()

    def _run(self, store, batches, start):
        for bi in range(start, len(batches)):
            self.q.put((bi, [store.load(t) for t in batches[bi]]))
        self.q.put(None)

    def __iter__(self):
        while True:
            x = self.q.get()
            if x is None:
                return
            yield x


# ------------------------------------------------------------------ scoring

@torch.no_grad()
def score_split(model, store: TrajStore, ids: list[str], arm: str, device, cost_budget: int,
                batch: int = 32) -> dict[str, list[float]]:
    """Logits for EVERY step of every trace (labels are not used)."""
    model.eval()
    out: dict[str, list[float]] = {}
    ids = sorted(ids, key=lambda t: store.items[t]["b"] - store.items[t]["a"])
    for i in range(0, len(ids), batch):
        trs = [store.load(t) for t in ids[i:i + batch]]
        tg = [list(range(len(tr["step_starts"]))) for tr in trs]
        for mb in microbatches(trs, arm, tg, cost_budget):
            b = collate([trs[j] for j, _ in mb], arm, [t for _, t in mb]).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                lg = model(b)
            for ti, s, v in gather_scores(b, lg.float()):
                tr = trs[mb[ti][0]]
                out.setdefault(tr["trace_id"], [float("nan")] * len(tr["step_starts"]))[s] = v
    for tid, v in out.items():
        assert not any(np.isnan(v)), f"unscored step in {tid}"
    return out


def dev_score(model, store, dev_ids, arm, device, cost_budget) -> tuple[float, float]:
    sc = score_split(model, store, dev_ids, arm, device, cost_budget)
    s, y = [], []
    for tid in dev_ids:
        for k, lab in labeled(store.items[tid]).items():
            s.append(sc[tid][k]); y.append(lab)
    thr, f1 = best_f1_threshold(np.asarray(s), np.asarray(y))
    return f1, thr


# ------------------------------------------------------------------ checkpoints

def atomic_save(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def rng_state() -> dict:
    return {"py": random.getstate(), "np": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def set_rng_state(s: dict) -> None:
    random.setstate(s["py"]); np.random.set_state(s["np"]); torch.set_rng_state(s["torch"])
    if s.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def state_hash(sd: dict) -> str:
    h = hashlib.sha1()
    for k in sorted(sd):
        h.update(k.encode()); h.update(sd[k].detach().cpu().float().numpy().tobytes())
    return h.hexdigest()[:16]


# ------------------------------------------------------------------ main

def build_model(d_in: int, seed: int, a) -> ContextualTokenProbe:
    torch.manual_seed(seed)
    return ContextualTokenProbe(d_in=d_in, d=a.d_model, layers=a.layers, heads=a.heads,
                                ff=a.ff, dropout=a.dropout)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--wd", type=float, required=True)
    ap.add_argument("--max_epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--eff_batch", type=int, default=32)
    ap.add_argument("--cost_budget", type=int, default=48_000_000)
    ap.add_argument("--ckpt_every", type=int, default=400)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--ff", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--train_split", default="train")
    ap.add_argument("--dev_split", default="dev")
    ap.add_argument("--limit_train", type=int, default=0, help="bounded pilot/smoke only")
    ap.add_argument("--limit_dev", type=int, default=0, help="smoke only")
    ap.add_argument("--score_splits", nargs="*",
                    default=["dev", "calib", "test", "pb_gsm8k", "pb_math", "pb_olympiadbench", "pb_omnimath"])
    ap.add_argument("--bucket_by", choices=["tokens", "step"], default="tokens",
                    help="length used for length-bucketed batches")
    ap.add_argument("--label_override", type=Path, default=None,
                    help="dir of <split>.jsonl with trace_id, y, label_mask replacing store labels")
    ap.add_argument("--max_hours", type=float, default=0.0,
                    help="stop cleanly (resumable) after this much wall time; 0 = no limit")
    a = ap.parse_args()

    t_start = time.time()
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "4")))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    store = TrajStore(a.store, label_override=a.label_override)
    train_ids = store.ids(a.train_split)
    dev_ids = store.ids(a.dev_split)
    if a.limit_train:
        train_ids = sorted(train_ids, key=lambda t: hashlib.sha1(t.encode()).hexdigest())[:a.limit_train]
    if a.limit_dev:
        dev_ids = sorted(dev_ids, key=lambda t: hashlib.sha1(t.encode()).hexdigest())[:a.limit_dev]
    train_ids = [t for t in train_ids if labeled(store.items[t])]
    if a.bucket_by == "step":  # identical across context views (context_ablation_v3)
        lengths = {t: sum(e - st for st, e in zip(store.items[t]["step_starts"], store.items[t]["step_ends"]))
                   for t in train_ids}
    else:
        lengths = {t: store.items[t]["b"] - store.items[t]["a"] for t in train_ids}
    a.out_dir.mkdir(parents=True, exist_ok=True)
    cfg = {**{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
           "n_train_traces": len(train_ids), "n_dev_traces": len(dev_ids),
           "n_label_overridden": store.n_overridden,
           "torch": torch.__version__, "cuda": torch.version.cuda,
           "sdpa_backends": {"flash": torch.backends.cuda.flash_sdp_enabled(),
                             "mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
                             "math": torch.backends.cuda.math_sdp_enabled()},
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
           "autocast": "bfloat16" if device.type == "cuda" else "none",
           "slurm_job": os.environ.get("SLURM_JOB_ID")}
    (a.out_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    arms = {}
    for arm in a.arms:
        d = a.out_dir / arm
        d.mkdir(exist_ok=True)
        m = build_model(store.dim, a.seed, a).to(device)
        opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=a.wd)
        st = {"epoch": 0, "batch": 0, "best_f1": -1.0, "best_thr": None, "best_epoch": -1,
              "bad": 0, "stopped": False, "history": [], "init_hash": state_hash(m.state_dict())}
        if (d / "last.pt").exists():
            ck = torch.load(d / "last.pt", map_location=device, weights_only=False)
            m.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); st = ck["state"]
            set_rng_state(ck["rng"])
            print(f"[resume] {arm} epoch {st['epoch']} batch {st['batch']}", flush=True)
        arms[arm] = {"dir": d, "model": m, "opt": opt, "st": st}
    n_params = {k: sum(p.numel() for p in v["model"].parameters()) for k, v in arms.items()}
    assert len(set(n_params.values())) == 1, n_params
    assert len({v["st"]["init_hash"] for v in arms.values()}) == 1, "arms must share init"
    print(f"[init] arms={a.arms} params={next(iter(n_params.values())):,} train={len(train_ids)} "
          f"dev={len(dev_ids)} device={device}", flush=True)

    def save_last(arm):
        A = arms[arm]
        atomic_save({"model": A["model"].state_dict(), "opt": A["opt"].state_dict(),
                     "state": A["st"], "rng": rng_state()}, A["dir"] / "last.pt")

    # all arms advance in lockstep on the same batches
    epoch = min(v["st"]["epoch"] for v in arms.values())
    start_batch = min(v["st"]["batch"] for v in arms.values() if v["st"]["epoch"] == epoch)
    timed_out = False
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    while epoch < a.max_epochs and not all(v["st"]["stopped"] for v in arms.values()):
        batches = epoch_batches(train_ids, lengths, a.seed, epoch, a.eff_batch)
        t0 = time.time()
        losses = {k: [] for k in arms}
        t_wait, t_arm = 0.0, {k: 0.0 for k in arms}
        tw = time.time()
        for bi, trs in Prefetch(store, batches, start_batch):
            t_wait += time.time() - tw
            labs = [labeled(store.items[t["trace_id"]]) for t in trs]
            tg = [sorted(l) for l in labs]
            w = target_weights([len(l) for l in labs], len(trs))
            for arm, A in arms.items():
                if A["st"]["stopped"]:
                    continue
                ta = time.time()
                A["model"].train()
                A["opt"].zero_grad(set_to_none=True)
                tot = 0.0
                for mb in microbatches(trs, arm, tg, a.cost_budget):
                    b = collate([trs[j] for j, _ in mb], arm, [t for _, t in mb]).to(device)
                    y = {(k, s): labs[j][s] for k, (j, t) in enumerate(mb) for s in t}
                    ww = {(k, s): w[j] for k, (j, t) in enumerate(mb) for s in t}
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                        lg = A["model"](b)
                    loss = weighted_bce(b, lg.float(), y, ww)
                    loss.backward()
                    tot += float(loss.detach())
                gn = torch.nn.utils.clip_grad_norm_(A["model"].parameters(), 1.0)
                if not torch.isfinite(gn):
                    raise SystemExit(f"[FATAL] non-finite grad norm ({arm}, epoch {epoch}, batch {bi})")
                A["opt"].step()
                A["st"]["batch"] = bi + 1
                losses[arm].append(tot)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t_arm[arm] += time.time() - ta
            if (bi + 1) % a.ckpt_every == 0:
                for arm in arms:
                    if not arms[arm]["st"]["stopped"]:
                        save_last(arm)
                el = time.time() - t0
                print(f"[ep {epoch}] batch {bi + 1}/{len(batches)} {el:.0f}s data_wait={t_wait:.0f}s "
                      + " ".join(f"{k}={np.mean(v[-50:]):.4f}/{t_arm[k]:.0f}s" for k, v in losses.items() if v), flush=True)
            if a.max_hours and time.time() - t_start > a.max_hours * 3600:
                timed_out = True
                break
            tw = time.time()
        if timed_out:
            for arm in arms:
                if not arms[arm]["st"]["stopped"]:
                    save_last(arm)
            print("[timeout] saved resumable state", flush=True)
            sys.exit(3)
        ep_time = time.time() - t0
        for arm, A in arms.items():
            st = A["st"]
            if st["stopped"]:
                continue
            te = time.time()
            f1, thr = dev_score(A["model"], store, dev_ids, arm, device, a.cost_budget)
            st["history"].append({"epoch": epoch, "train_loss": float(np.mean(losses[arm])) if losses[arm] else None,
                                  "dev_f1": f1, "dev_thr": thr, "epoch_s": ep_time,
                                  "dev_eval_s": time.time() - te})
            if f1 > st["best_f1"]:
                st.update(best_f1=f1, best_thr=thr, best_epoch=epoch, bad=0)
                atomic_save({"model": A["model"].state_dict(), "epoch": epoch, "dev_f1": f1,
                             "dev_thr": thr}, A["dir"] / "best.pt")
            else:
                st["bad"] += 1
                if st["bad"] >= a.patience:
                    st["stopped"] = True
            st["epoch"], st["batch"] = epoch + 1, 0
            save_last(arm)
            (A["dir"] / "history.json").write_text(json.dumps(st["history"], indent=2))
            print(f"[ep {epoch}] {arm} loss={st['history'][-1]['train_loss']:.4f} dev_f1={f1:.4f} "
                  f"best={st['best_f1']:.4f}@{st['best_epoch']} bad={st['bad']}", flush=True)
        epoch += 1
        start_batch = 0

    # ---------------------------------------------------------- final scoring
    for arm, A in arms.items():
        d = A["dir"]
        if (d / "done.json").exists():
            continue
        ck = torch.load(d / "best.pt", map_location=device, weights_only=False)
        A["model"].load_state_dict(ck["model"])
        ck_hash = state_hash(ck["model"])
        n_rows = 0
        with gzip.open(d / "predictions.jsonl.gz.tmp", "wt") as f:
            for sp in a.score_splits:
                ids = store.ids(sp)
                if not ids:
                    continue
                sc = score_split(A["model"], store, ids, arm, device, a.cost_budget)
                for tid in ids:
                    it = store.items[tid]
                    T = len(it["step_starts"])
                    for k in range(T):
                        hor = {"local": k, "causal": k, "future1": min(k + 1, T - 1), "full": T - 1}[arm]
                        f.write(json.dumps({"split": sp, "trace_id": tid, "step": k, "n_steps": T,
                                            "logit": sc[tid][k], "visible_through_step": hor,
                                            "label_known": bool(it["label_mask"][k]),
                                            "y": int(it["y"][k])}) + "\n")
                        n_rows += 1
        os.replace(d / "predictions.jsonl.gz.tmp", d / "predictions.jsonl.gz")
        done = {"arm": arm, "seed": a.seed, "lr": a.lr, "wd": a.wd, "best_epoch": ck["epoch"],
                "dev_f1": ck["dev_f1"], "dev_thr": ck["dev_thr"], "checkpoint_hash": ck_hash,
                "init_hash": A["st"]["init_hash"], "epochs_run": A["st"]["epoch"],
                "stopped_early": A["st"]["stopped"], "n_prediction_rows": n_rows,
                "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
                "wall_s": time.time() - t_start}
        (d / "done.json").write_text(json.dumps(done, indent=2))
        print(f"[done] {arm} {done}", flush=True)


if __name__ == "__main__":
    main()
