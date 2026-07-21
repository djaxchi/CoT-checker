#!/usr/bin/env python3
"""das_branch_subspace_v0 Phase 3: fit the shared-U DAS subspace (GPU).

Gated on the positive whole-span oracle. Three modes:

  extract : for a layer L, cache per fork the correct/wrong step-span states (last-k
            aligned), the wrong context + candidates, and the unpatched wrong/correct
            gold margins. One pass, no training.
  fit     : train one orthonormal U (d x k_sub, one seed) on the TRAIN split to raise
            the gold margin under the correct->wrong subspace interchange; evaluate
            held-out (VAL) margin recovery against an untrained same-k random subspace.
  report  : aggregate fit metrics across (L, k, seed); DAS-vs-random deltas and the
            cross-seed subspace overlap (identifiability).

Success = held-out recovery that beats the random subspace, paired CI excluding 0
across seeds, with a cross-seed-consistent U (high principal-angle overlap).

Usage:
  python scripts/das_branch/das_fit.py --mode extract --layer 12 --k_tokens 8 ...
  python scripts/das_branch/das_fit.py --mode fit --layer 12 --k_sub 16 --seed 0 ...
  python scripts/das_branch/das_fit.py --mode report --out_dir runs/das_train
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from scripts.encode_prm800k_hidden_states import read_jsonl  # noqa: E402
from src.analysis.das_span import (  # noqa: E402
    aligned_positions,
    fork_span_ids,
    gold_margin,
    span_candidate_logprobs,
    suffix_ids,
)
from src.analysis.das_train import (  # noqa: E402
    SubspaceU,
    interchange_states,
    margin_match_loss,
    span_candidate_logprobs_grad,
    subspace_overlap,
)

MIN_SPAN = 4


def cache_path(out_dir: Path, layer: int) -> Path:
    return out_dir / f"cache_L{layer}.pt"


def do_extract(args, model, tok, device) -> None:
    import torch

    traces = read_jsonl(Path(args.run_dir) / "traces_forks.jsonl")
    sfx = suffix_ids(tok)
    pad = tok.pad_token_id or tok.eos_token_id
    items = []
    t0 = time.perf_counter()
    for n, tr in enumerate(traces):
        if not tr.get("gt_answer"):
            continue
        try:
            wids, wlo, whi = fork_span_ids(tok, tr, "wrong")
            cids, clo, chi = fork_span_ids(tok, tr, "correct")
        except Exception:
            continue
        if (whi - wlo) < MIN_SPAN or (chi - clo) < MIN_SPAN:
            continue
        ilo, ihi, dlo, dhi = aligned_positions((wlo, whi), (clo, chi), "lastk",
                                               args.k_tokens)
        with torch.no_grad():
            ho = model(input_ids=torch.tensor([cids], device=device),
                       output_hidden_states=True).hidden_states[args.layer]
            donor = ho[0, dlo:dhi, :].to(torch.float16).cpu()
            hw = model(input_ids=torch.tensor([wids], device=device),
                       output_hidden_states=True).hidden_states[args.layer]
            base = hw[0, ilo:ihi, :].to(torch.float16).cpu()
        cand_ids = [tok(c, add_special_tokens=False)["input_ids"]
                    for c in tr["candidates"]]
        lp_wrong = span_candidate_logprobs(model, wids + sfx, cand_ids, pad, device)
        lp_correct = span_candidate_logprobs(model, cids + sfx, cand_ids, pad, device)
        items.append({"trace_id": tr["trace_id"], "split": tr.get("split", "test"),
                      "base": base, "donor": donor, "ctx_ids": wids + sfx,
                      "cand_ids": cand_ids, "inject": (ilo, ihi),
                      "m_wrong": gold_margin(lp_wrong),
                      "m_correct": gold_margin(lp_correct),
                      "cand_lp_correct": lp_correct})  # donor belief = training target
        if (n + 1) % 100 == 0:
            print(f"[extract L{args.layer}] {n + 1}/{len(traces)} kept {len(items)} "
                  f"({(time.perf_counter() - t0) / (n + 1):.2f}s/trace)", flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(items, cache_path(args.out_dir, args.layer))
    print(f"[extract L{args.layer}] saved {len(items)} items", flush=True)


def _val_recovery(model, items, pad, device, layer, u) -> dict:
    """Mean margin recovery on a set of items using subspace u (None -> full-state
    oracle upper bound). Returns per-fork deltas + recovery fraction."""
    import numpy as np
    import torch

    deltas, gaps = [], []
    for it in items:
        base = it["base"].to(device).float()
        donor = it["donor"].to(device).float()
        ilo, ihi = it["inject"]
        with torch.no_grad():
            states = donor if u is None else interchange_states(base, donor, u())
            m = gold_margin(span_candidate_logprobs(
                model, it["ctx_ids"], it["cand_ids"], pad, device,
                layer=layer, lo=ilo, hi=ihi, states=states))
        deltas.append(m - it["m_wrong"])
        gaps.append(it["m_correct"] - it["m_wrong"])
    denom = float(np.mean(gaps))
    return {"mean_delta": float(np.mean(deltas)),
            "recovery": float(np.mean(deltas) / denom) if denom != 0 else float("nan"),
            "deltas": deltas}


def do_fit(args, model, tok, device) -> None:
    import numpy as np
    import torch
    from scipy import stats

    items = torch.load(cache_path(args.out_dir, args.layer), weights_only=False)
    # drop pathologically long contexts: the grad graph + full-seq logits scale with
    # length, and a handful of very long forks can OOM an 80GB card.
    before = len(items)
    items = [it for it in items
             if len(it["ctx_ids"]) + max(len(c) for c in it["cand_ids"]) <= args.max_ctx]
    if before != len(items):
        print(f"[fit] kept {len(items)}/{before} within max_ctx={args.max_ctx}",
              flush=True)
    train = [it for it in items if it["split"] == "train"]
    val = [it for it in items if it["split"] != "train"]
    if not train:  # splits.json may label val/test only; fall back to a deterministic cut
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(items))
        cut = int(0.7 * len(items))
        train = [items[i] for i in idx[:cut]]
        val = [items[i] for i in idx[cut:]]
    d = items[0]["base"].shape[1]
    pad = tok.pad_token_id or tok.eos_token_id

    u = SubspaceU(d, args.k_sub, seed=args.seed).to(device)
    opt = torch.optim.Adam(u.parameters(), lr=args.lr)
    order = list(range(len(train)))
    t0 = time.perf_counter()
    for epoch in range(args.epochs):
        np.random.default_rng(args.seed + epoch).shuffle(order)
        running = 0.0
        for si, oi in enumerate(order):
            it = train[oi]
            base = it["base"].to(device).float()
            donor = it["donor"].to(device).float()
            ilo, ihi = it["inject"]
            states = interchange_states(base, donor, u())
            lp = span_candidate_logprobs_grad(model, it["ctx_ids"], it["cand_ids"],
                                              pad, device, args.layer, ilo, ihi, states)
            target = torch.tensor(it["cand_lp_correct"], device=device)
            loss = margin_match_loss(lp, target)
            if not torch.isfinite(loss):    # skip a bad step rather than poison U
                opt.zero_grad()
                continue
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(u.parameters(), 1.0)
            # a NaN from the QR backward would otherwise write NaN into U (and every
            # later forward); only step when the gradient is finite.
            if torch.isfinite(gnorm):
                opt.step()
                running += float(loss.detach())
            opt.zero_grad()
        print(f"[fit L{args.layer} k{args.k_sub} s{args.seed}] epoch {epoch} "
              f"loss {running / len(train):.4f} "
              f"({(time.perf_counter() - t0):.0f}s)", flush=True)

    das = _val_recovery(model, val, pad, device, args.layer, u)
    rand = _val_recovery(model, val, pad, device, args.layer,
                         SubspaceU(d, args.k_sub, seed=args.seed + 1000).to(device))
    oracle = _val_recovery(model, val, pad, device, args.layer, None)
    train_rec = _val_recovery(model, train[:200], pad, device, args.layer, u)
    # donor-belief loss floor (the CE minimum = donor entropy) to read the loss vs
    floor = float(np.mean([float(-(torch.softmax(torch.tensor(it["cand_lp_correct"]), 0)
                                   * torch.log_softmax(torch.tensor(it["cand_lp_correct"]), 0)).sum())
                           for it in val]))
    dd, rd = np.array(das["deltas"]), np.array(rand["deltas"])
    p = float(stats.wilcoxon(dd, rd, alternative="greater").pvalue) \
        if len(dd) >= 10 and (dd - rd).any() else float("nan")
    torch.save(u.state_dict(), args.out_dir / f"U_L{args.layer}_k{args.k_sub}_s{args.seed}.pt")
    metrics = {"layer": args.layer, "k_sub": args.k_sub, "seed": args.seed,
               "n_train": len(train), "n_val": len(val),
               "das_recovery": das["recovery"], "random_recovery": rand["recovery"],
               "oracle_recovery": oracle["recovery"], "train_recovery": train_rec["recovery"],
               "das_mean_delta": das["mean_delta"], "random_mean_delta": rand["mean_delta"],
               "donor_entropy_floor": floor, "final_train_loss": running / max(1, len(train)),
               "p_das_gt_random": p}
    (args.out_dir / f"metrics_L{args.layer}_k{args.k_sub}_s{args.seed}.json").write_text(
        json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)


def do_report(args) -> None:
    import torch

    d = args.out_dir
    metrics = [json.loads(p.read_text()) for p in sorted(d.glob("metrics_L*.json"))]
    by = {}
    for m in metrics:
        by.setdefault((m["layer"], m["k_sub"]), []).append(m)
    out = {"configs": []}
    for (L, k), ms in sorted(by.items()):
        us = []
        for m in ms:
            f = d / f"U_L{L}_k{k}_s{m['seed']}.pt"
            if f.exists():
                sd = torch.load(f, weights_only=True)
                w = sd["weight"]
                us.append(torch.linalg.qr(w)[0])
        overlaps = [subspace_overlap(us[i], us[j])
                    for i in range(len(us)) for j in range(i + 1, len(us))]
        import numpy as np
        out["configs"].append({
            "layer": L, "k_sub": k, "seeds": len(ms),
            "das_recovery_mean": float(np.mean([m["das_recovery"] for m in ms])),
            "train_recovery_mean": float(np.mean([m.get("train_recovery", float("nan"))
                                                  for m in ms])),
            "random_recovery_mean": float(np.mean([m["random_recovery"] for m in ms])),
            "oracle_recovery_mean": float(np.mean([m["oracle_recovery"] for m in ms])),
            "final_train_loss_mean": float(np.mean([m.get("final_train_loss", float("nan"))
                                                    for m in ms])),
            "donor_entropy_floor_mean": float(np.mean([m.get("donor_entropy_floor", float("nan"))
                                                       for m in ms])),
            "das_minus_random_mean": float(np.mean(
                [m["das_recovery"] - m["random_recovery"] for m in ms])),
            "p_das_gt_random_max": max((m["p_das_gt_random"] for m in ms), default=float("nan")),
            "cross_seed_overlap_mean": float(np.mean(overlaps)) if overlaps else float("nan")})
    (d / "gates_das.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["extract", "fit", "report"], required=True)
    ap.add_argument("--run_dir", type=Path, default=Path("runs/causal_graph"))
    ap.add_argument("--out_dir", type=Path, default=Path("runs/das_train"))
    ap.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-7B")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--k_tokens", type=int, default=8)
    ap.add_argument("--k_sub", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--max_ctx", type=int, default=1024)
    ap.add_argument("--local_files_only", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "report":
        do_report(args)
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                        local_files_only=args.local_files_only)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, torch_dtype=torch.bfloat16,
        local_files_only=args.local_files_only).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    if args.mode == "extract":
        do_extract(args, model, tok, device)
    else:
        do_fit(args, model, tok, device)


if __name__ == "__main__":
    main()
