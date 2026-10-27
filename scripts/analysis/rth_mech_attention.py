#!/usr/bin/env python3
"""M3: where the read token looks, per layer (read_token_v1 mechanistic study, Stage G).

For each item of a bundle split (PRM800K test_2k by default) and each read token
given (RTH, BL1, ...), the attention probabilities of the last read position are
summed over six segments of the verifier template:

  sink     position 0 (Qwen's attention sink)
  problem  "Problem:\\n" and the problem text (minus position 0)
  prior    "\\n\\nPrevious reasoning:\\n" and the previous steps
  header   "\\n\\nCurrent step:\\n"
  step     the current step's tokens
  self     the read token(s)

per layer and head. Segments are found from the tokenizer's character offsets on
the prefix string (a token belongs to the segment its first character is in); the
token ids are exactly scripts/read_token_probe.py tokenize()'s.

The prefix runs once per batch under SDPA, as in training; only the read token's
own pass runs under eager attention (which returns the probabilities), on a copy of
the cache, and the same pass under SDPA gives the gate: the eager and SDPA head
logits of each token must agree. --plain adds the untrained " correct" token (no
offsets, no head) as a reference.

Outputs: m3_attention_<name>.npz (mass (N, layers, heads, 6) float16, ntok (N, 6),
y, logit_eager, logit_sdpa) and m3_attention_summary.json (mean mass per segment and
layer over items and heads, overall and by label, and the gate).

    python scripts/analysis/rth_mech_attention.py --bundle <mech>/bundle --split test_2k \
        --source prm800k:<data>/prm800k_test_2k.jsonl \
        --tokens rth=<tokens>/dq_correct/read_token.pt bl1=<tokens>/bl1_correct_seed0/read_token.pt \
        --plain --out <mech_study>
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.read_token_mech_bundle import load_bundle  # noqa: E402
from scripts.read_token_probe import ReadToken, batches, read_hooks  # noqa: E402
from scripts.read_token_store import load_model, load_read_token, match, source_index  # noqa: E402
from src.onpolicy.prompts import verifier_prefix  # noqa: E402

SEGMENTS = ("sink", "problem", "prior", "header", "step", "self")
HEADER = "\n\nCurrent step:\n"


def segment_ids(tok, it: dict) -> tuple[list[int], list[int]]:
    """tokenize()'s ids for the item and the segment index of each token."""
    text = verifier_prefix(it["problem"], it["prefix"])
    enc = tok(text, add_special_tokens=True, return_offsets_mapping=True)
    p_end = len("Problem:\n") + len(it["problem"])
    h_start = text.rfind(HEADER)
    seg = [1 if s < p_end else (2 if s < h_start else 3) for s, _ in enc["offset_mapping"]]
    seg[0] = 0
    step = tok(it["step"], add_special_tokens=False)["input_ids"]
    return list(enc["input_ids"]) + list(step), seg + [4] * len(step)


def prefix_pass(model, ids_list: list[list[int]], pad: int, dev):
    """The left-padded prefix forward of read_all_states: (attention mask, cache, lengths)."""
    B, L = len(ids_list), max(len(t) for t in ids_list)
    ids = torch.full((B, L), pad, dtype=torch.long)
    att = torch.zeros((B, L), dtype=torch.long)
    for r, t in enumerate(ids_list):
        ids[r, L - len(t):] = torch.tensor(t)
        att[r, L - len(t):] = 1
    ids, att = ids.to(dev), att.to(dev)
    with torch.no_grad():
        out = model.model(input_ids=ids, attention_mask=att, position_ids=(att.cumsum(1) - 1).clamp(min=0),
                          use_cache=True)
    return att, out.past_key_values, att.sum(1)


def soft_pass(model, rt: ReadToken, att, cache, lens, dev, eager: bool):
    """The read token's pass on a copy of the cache: (last read state (B, d), the last read
    position's attention per layer (B, layers, heads, L + m), or None under SDPA)."""
    B, m = att.shape[0], rt.m
    soft_pos = lens[:, None] + torch.arange(m, device=dev)[None]
    att2 = torch.cat([att, torch.ones((B, m), dtype=att.dtype, device=dev)], 1)
    emb = rt.soft.to(model.dtype)[None].expand(B, -1, -1)
    impl = model.config._attn_implementation
    if eager:
        model.set_attn_implementation("eager")
    try:
        with torch.no_grad(), read_hooks(model, rt):
            out = model.model(inputs_embeds=emb, attention_mask=att2, position_ids=soft_pos,
                              past_key_values=copy.deepcopy(cache), use_cache=True, output_attentions=eager)
    finally:
        model.set_attn_implementation(impl)
    A = torch.stack([a[:, :, -1].float() for a in out.attentions], 1) if eager else None
    return out.last_hidden_state[:, -1].float(), A


def segment_mass(A: torch.Tensor, segs: list[list[int]], m: int) -> torch.Tensor:
    """A (B, layers, heads, L + m) for left-padded rows -> (B, layers, heads, 6)."""
    B, nl, nh, K = A.shape
    L = K - m
    idx = torch.full((B, K), len(SEGMENTS), dtype=torch.long)  # padding goes to a spare slot
    for r, s in enumerate(segs):
        idx[r, L - len(s):L] = torch.tensor(s)
    idx[:, L:] = SEGMENTS.index("self")
    idx = idx.to(A.device)[:, None, None].expand(B, nl, nh, K)
    out = torch.zeros((B, nl, nh, len(SEGMENTS) + 1), device=A.device)
    return out.scatter_add_(3, idx, A)[..., :len(SEGMENTS)]


def plain_token(model, tok, text: str, dev) -> ReadToken:
    """The untrained token: the embedding of the last token of `text`, no offsets, an unused head."""
    i = tok(text, add_special_tokens=False)["input_ids"][-1]
    e = model.get_input_embeddings().weight[i].detach().float().cpu()[None]
    return ReadToken(1, model.config.hidden_size, e).to(dev).eval()


def summarize(name: str, mass: np.ndarray, ntok: np.ndarray, y: np.ndarray, lg_e, lg_s) -> dict:
    mean = mass.astype(np.float32).mean(2)  # over heads -> (N, layers, 6)

    def by(sel):
        return {s: mean[sel, :, k].mean(0).round(5).tolist() for k, s in enumerate(SEGMENTS)}
    out = {"mass": by(np.ones(len(y), bool)), "mass_correct": by(y == 0), "mass_incorrect": by(y == 1),
           "mean_tokens": {s: float(ntok[:, k].mean()) for k, s in enumerate(SEGMENTS)}}
    if np.isfinite(lg_e).all():
        d = np.abs(lg_e - lg_s)
        out["gate"] = {"max_abs_logit_diff": float(d.max()), "mean_abs_logit_diff": float(d.mean()),
                       "pearson": float(np.corrcoef(lg_e, lg_s)[0, 1])}
    print(f"[m3] {name}: " + " ".join(f"{s} {np.mean(out['mass'][s]):.3f}" for s in SEGMENTS)
          + (f" gate {out['gate']}" if "gate" in out else ""), flush=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--split", default="test_2k")
    p.add_argument("--source", required=True, help="prm800k:PATH or pb:SUBSET:PATH, as for the bundle")
    p.add_argument("--tokens", nargs="+", required=True, help="name=read_token.pt")
    p.add_argument("--plain", action="store_true", help="also the untrained --plain_text token")
    p.add_argument("--plain_text", default=" correct")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max_tokens", type=int, default=16000)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    b = load_bundle(a.bundle / a.split, arrays=("y",))
    y = np.asarray(b["y"], np.int64)
    meta = b["meta"]
    if a.limit:
        y, meta = y[:a.limit], meta[:a.limit]
    kind, index = source_index(a.source)
    items = match(kind, index, meta, y)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok = load_model(a.model, dev)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    arms = {}
    for spec in a.tokens:
        name, path = spec.split("=", 1)
        if not Path(path).is_file():
            print(f"[m3] {name}: {path} missing, skipped", flush=True)
            continue
        arms[name] = load_read_token(Path(path), model.config.hidden_size, dev, model)[0]
    if a.plain:
        arms["plain"] = plain_token(model, tok, a.plain_text, dev)
    seg = [segment_ids(tok, it) for it in items]
    N, nl, nh = len(items), model.config.num_hidden_layers, model.config.num_attention_heads
    res = {n: {"mass": np.zeros((N, nl, nh, len(SEGMENTS)), np.float16),
               "lg_e": np.full(N, np.nan), "lg_s": np.full(N, np.nan)} for n in arms}
    ntok = np.array([[s.count(k) for k in range(len(SEGMENTS) - 1)] + [1] for _, s in seg])
    for bi, rows in enumerate(batches([len(ids) for ids, _ in seg], a.max_tokens, None)):
        att, cache, lens = prefix_pass(model, [seg[r][0] for r in rows], pad, dev)
        for name, rt in arms.items():
            h_e, A = soft_pass(model, rt, att, cache, lens, dev, eager=True)
            res[name]["mass"][rows] = segment_mass(A, [seg[r][1] for r in rows], rt.m).cpu().numpy()
            if name != "plain":
                h_s, _ = soft_pass(model, rt, att, cache, lens, dev, eager=False)
                with torch.no_grad():
                    res[name]["lg_e"][rows] = rt.logit(h_e).double().cpu().numpy()
                    res[name]["lg_s"][rows] = rt.logit(h_s).double().cpu().numpy()
        del cache
        if bi % 20 == 0:
            print(f"[m3] batch {bi}, {time.perf_counter() - t0:.0f}s", flush=True)
    summary = {"split": a.split, "n_items": N, "segments": list(SEGMENTS), "tokens": {}}
    for name, r in res.items():
        np.savez(a.out / f"m3_attention_{name}.npz", mass=r["mass"], ntok=ntok, y=y,
                 logit_eager=r["lg_e"], logit_sdpa=r["lg_s"])
        summary["tokens"][name] = summarize(name, r["mass"], ntok, y, r["lg_e"], r["lg_s"])
    (a.out / "m3_attention_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"[m3] done in {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
