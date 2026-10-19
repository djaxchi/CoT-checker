"""Arm B of step_bert_probe_v1: fine-tune a text encoder (ModernBERT-base) on raw step text.

Conditions differ only in which steps are in the input (src/probes/text_bert.py):
local = problem + step t, causal = problem + steps 0..t, bidir = whole trace.
local and causal build one example per target step; bidir builds one example per
trace and reads every step, which sees exactly the same tokens as one example per
step would.

Loss: BCE, each step weighted 1/n_steps(trace) (mean 1 over train). AdamW with
linear warmup (6%) and linear decay; encoder lr from --lr, head lr 1e-3. Early
stopping and the threshold use val step F1. Writes the same outputs as Arm A.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import v1_common as C  # noqa: E402
from src.probes.text_bert import TextStepClassifier, build_example, span_window  # noqa: E402


def tokenize_collection(tok, steps, texts: dict, max_prob: int, max_step: int):
    """Per trace (in steps.npz order): (problem ids, [step ids]).

    The text keeps only the steps present in the store. A few PB traces (9 of
    3,400) are stored without their trailing steps, so Arm A never sees them
    either; any other mismatch is an error.
    """
    runs = C.trace_runs(steps["trace_id"])
    tids = list(runs)
    for t in tids:
        s0, n = runs[t]
        if len(texts[t]["steps"]) < n or not np.array_equal(steps["step_idx"][s0:s0 + n], np.arange(n)):
            raise ValueError(f"{t}: {len(texts[t]['steps'])} texts vs stored steps {steps['step_idx'][s0:s0 + n]}")
    probs = tok([texts[t]["problem"] for t in tids], add_special_tokens=False)["input_ids"]
    flat = [s.strip() for t in tids for s in texts[t]["steps"][:runs[t][1]]]
    flat_ids = tok(flat, add_special_tokens=False)["input_ids"]
    out, k = {}, 0
    for t, p in zip(tids, probs):
        n = runs[t][1]
        out[t] = (p[:max_prob], [ids[:max_step] for ids in flat_ids[k:k + n]])
        k += n
    return runs, out


def make_units(runs: dict, rs, condition: str):
    """(trace id, start row, n, target or -1 for read-all)."""
    inv = {v: k for k, v in runs.items()}
    if condition == "bidir":
        return [(inv[r], r[0], r[1], -1) for r in rs]
    return [(inv[r], r[0], r[1], t) for r in rs for t in range(r[1])]


def materialize(units, toks, condition, cls, sep):
    ex = []
    for tid, s0, n, t in units:
        lo, hi = span_window(n, max(t, 0), condition)
        ids, reads = build_example(*toks[tid], lo, hi, cls, sep)
        if t >= 0:
            reads = [r for r in reads if r[0] == t]
        ex.append((ids, [(s0 + k, a, e, sp) for k, a, e, sp in reads]))
    return ex


def batches(ex, budget: int, rng=None):
    lens = np.array([len(e[0]) for e in ex])
    order = np.argsort(lens, kind="stable")
    out, cur, mx = [], [], 0
    for k in order:
        m = max(mx, lens[k])
        if cur and m * (len(cur) + 1) > budget:
            out.append(cur)
            cur, m = [], lens[k]
        cur.append(int(k))
        mx = m
    if cur:
        out.append(cur)
    if rng is not None:
        rng.shuffle(out)
    return out


def collate(ex, idx, pad: int, dev):
    sel = [ex[k] for k in idx]
    L = max(len(e[0]) for e in sel)
    ids = torch.full((len(sel), L), pad, dtype=torch.long)
    att = torch.zeros(len(sel), L, dtype=torch.long)
    rows, rb, ra, re, rs = [], [], [], [], []
    for i, (e, reads) in enumerate(sel):
        ids[i, :len(e)] = torch.tensor(e)
        att[i, :len(e)] = 1
        for row, a, b, sp in reads:
            rows.append(row); rb.append(i); ra.append(a); re.append(b); rs.append(sp)
    t = lambda v: torch.tensor(v, dtype=torch.long, device=dev)  # noqa: E731
    return ids.to(dev), att.to(dev), t(rb), t(ra), t(re), t(rs), t(rows)


@torch.no_grad()
def predict(model, ex, n_rows: int, pad: int, dev, budget: int, amp: bool) -> np.ndarray:
    model.eval()
    out = np.full(n_rows, np.nan, dtype=np.float32)
    for idx in batches(ex, budget):
        ids, att, rb, ra, re, rs, rows = collate(ex, idx, pad, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logit = model(ids, att, rb, ra, re, rs)
        out[rows.cpu().numpy()] = torch.sigmoid(logit.float()).cpu().numpy()
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True, help="dir with onpolicy/ and pb_<subset>/ steps.npz")
    p.add_argument("--texts", type=Path, required=True, help="dir with onpolicy.json and pb_<subset>.json")
    p.add_argument("--model", required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--condition", choices=["local", "causal", "bidir"], required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--no_pb", action="store_true")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--budget", type=int, default=32768, help="padded tokens per batch")
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--head_lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--max_prob", type=int, default=512)
    p.add_argument("--max_step", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    from transformers import AutoModel, AutoTokenizer
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    rng = np.random.default_rng(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = dev.type == "cuda"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    cls, sep, pad = tok.cls_token_id, tok.sep_token_id, tok.pad_token_id
    steps = dict(np.load(args.data / "onpolicy" / "steps.npz"))
    runs, toks = tokenize_collection(tok, steps, json.loads((args.texts / "onpolicy.json").read_text()),
                                     args.max_prob, args.max_step)
    _, parts = C.split_runs(steps, args.split)
    y = torch.from_numpy(steps["y"].astype(np.float32)).to(dev)
    w_np = C.step_weights(parts["train"])
    w = torch.from_numpy(np.pad(w_np, (0, len(steps["y"]) - len(w_np)))).to(dev)

    enc = AutoModel.from_pretrained(args.model, attn_implementation="sdpa")
    model = TextStepClassifier(enc).to(dev)
    opt = torch.optim.AdamW([{"params": model.encoder.parameters(), "lr": args.lr},
                             {"params": model.head.parameters(), "lr": args.head_lr}], weight_decay=args.wd)
    ex_tr = materialize(make_units(runs, parts["train"], args.condition), toks, args.condition, cls, sep)
    ex_va = materialize(make_units(runs, parts["val"], args.condition), toks, args.condition, cls, sep)
    n_batches = len(batches(ex_tr, args.budget))
    total, warm = args.epochs * n_batches, max(1, int(0.06 * args.epochs * n_batches))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, max(0.0, (total - s) / max(1, total - warm))))
    print(f"train examples {len(ex_tr)} tokens {sum(len(e[0]) for e in ex_tr)} batches/epoch {n_batches}", flush=True)

    log, best, best_state, bad = [], -1.0, None, 0
    for ep in range(args.epochs):
        model.train()
        t1, tot, nb = time.time(), 0.0, 0
        for idx in batches(ex_tr, args.budget, rng):
            ids, att, rb, ra, re, rs, rows = collate(ex_tr, idx, pad, dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                logit = model(ids, att, rb, ra, re, rs)
            loss = (F.binary_cross_entropy_with_logits(logit.float(), y[rows], reduction="none") * w[rows]).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {ep}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot, nb = tot + loss.item(), nb + 1
        pv = predict(model, ex_va, len(steps["y"]), pad, dev, args.budget, amp)
        vf1, vauc = C.val_f1(steps, pv, parts)
        log.append({"epoch": ep, "train": tot / nb, "val_f1": vf1, "val_auroc": vauc, "sec": round(time.time() - t1, 1)})
        print(json.dumps(log[-1]), flush=True)
        if vf1 > best + 1e-4:
            best, bad = vf1, 0
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)

    ex_all = materialize(make_units(runs, list(runs.values()), args.condition), toks, args.condition, cls, sep)
    p_all = predict(model, ex_all, len(steps["y"]), pad, dev, args.budget, amp)
    pb = []
    if not args.no_pb:
        for s in C.PB_SUBSETS:
            zp = dict(np.load(args.data / f"pb_{s}" / "steps.npz"))
            rp, tp = tokenize_collection(tok, zp, json.loads((args.texts / f"pb_{s}.json").read_text()),
                                         args.max_prob, args.max_step)
            exp = materialize(make_units(rp, list(rp.values()), args.condition), tp, args.condition, cls, sep)
            pb.append((f"pb_{s}", zp, predict(model, exp, len(zp["y"]), pad, dev, args.budget, amp)))
    C.finalize(args.out_dir, args, steps, p_all, parts, log, best, pb)


if __name__ == "__main__":
    main()
