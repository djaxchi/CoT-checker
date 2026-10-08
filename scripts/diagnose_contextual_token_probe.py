#!/usr/bin/env python3
"""Bounded future-reliance diagnostics for the final `full` probes (plan 11).

Selects up to 256 known-label evaluation targets with >= 2 later steps,
stratified by dataset/subset and label (seed 1729, independent of predictions),
and scores each target under:
  unchanged     the original trace
  shuffled      same prefix and target, later steps in a shuffled order (re-encoded)
  no_answer     same prefix and target, explicit final-answer text removed from
                later steps (re-encoded); rules: a PRM800K "# Answer" block and
                \\boxed{...} (balanced braces). Steps emptied by the rule are
                dropped. Ineligible if no later text matches a rule.
  future_hidden the unchanged states with every later step cropped from the view
Text modifications are re-encoded with the frozen backbone; target/prefix token
ids must be identical and their states equal within tolerance (both recorded).
Causal models are scored alongside as a leakage sanity check: their target
scores must not move under any variant.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from encode_trajectory_token_store import LayerCapture, forward_states, load_jsonl, tokenize_solution  # noqa: E402
from src.probes.contextual_token_probe import ContextualTokenProbe, collate  # noqa: E402

DATASETS = ("test", "pb_gsm8k", "pb_math", "pb_olympiadbench", "pb_omnimath")
ANSWER_BLOCK = re.compile(r"\s*#\s*Answer\b.*\Z", re.S)


def strip_boxed(text: str) -> str:
    out, i = [], 0
    while True:
        j = text.find("\\boxed{", i)
        if j < 0:
            out.append(text[i:]); break
        out.append(text[i:j])
        k, depth = j + len("\\boxed{"), 1
        while k < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[k], 0)
            k += 1
        i = k
    return "".join(out)


def remove_answer(steps: list[str]) -> tuple[list[str], bool]:
    new, changed = [], False
    for s in steps:
        t = strip_boxed(ANSWER_BLOCK.sub("", s))
        changed |= t != s
        if t.strip():
            new.append(t)
    return new, changed


def select_targets(manifest: Path, n_total: int, seed: int) -> list[dict]:
    cells = {}
    for ds in DATASETS:
        for l in open(manifest / "meta" / f"{ds}.jsonl"):
            m = json.loads(l)
            if not m.get("retained", True):
                continue
            T = len(m["y"])
            for k in range(T - 2):
                if m["label_mask"][k]:
                    cells.setdefault((ds, m["y"][k]), []).append((m["trace_id"], k))
    rng = random.Random(seed)
    keys = sorted(cells)
    per = n_total // len(keys)
    chosen = []
    for c in keys:
        pool = sorted(cells[c]); rng.shuffle(pool)
        chosen.extend({"dataset": c[0], "y": c[1], "trace_id": t, "step": k} for t, k in pool[:per])
    # fill the remainder deterministically from cells with spare items
    spare = [{"dataset": c[0], "y": c[1], "trace_id": t, "step": k}
             for c in keys for t, k in sorted(cells[c])[per:]]
    rng.shuffle(spare)
    chosen.extend(spare[:max(0, n_total - len(chosen))])
    return chosen, {f"{c[0]}|y={c[1]}": len(cells[c]) for c in keys}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--fits_root", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model_name_or_path", default="Qwen/Qwen3-8B")
    ap.add_argument("--layer", type=int, default=35)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1729)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model_dtype", default="bfloat16")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = torch.device(a.device)
    inputs = {}
    for ds in DATASETS:
        for r in load_jsonl(a.manifest / "inputs" / f"{ds}.jsonl"):
            inputs[r["trace_id"]] = r
    targets, cell_sizes = select_targets(a.manifest, a.n, a.seed)
    tok = AutoTokenizer.from_pretrained(a.model_name_or_path, local_files_only=True)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    lm = AutoModelForCausalLM.from_pretrained(a.model_name_or_path, local_files_only=True,
                                              torch_dtype=getattr(torch, a.model_dtype), attn_implementation="sdpa").to(device).eval()
    cap = LayerCapture(lm, a.layer)

    probes = {}
    for sd in a.seeds:
        for arm in ("full", "causal"):
            ck = torch.load(a.fits_root / f"final_s{sd}_{arm}" / arm / "best.pt", map_location=device, weights_only=False)
            cfg = json.loads((a.fits_root / f"final_s{sd}_{arm}" / "config.json").read_text())
            m = ContextualTokenProbe(d_in=lm.config.hidden_size, d=cfg["d_model"], layers=cfg["layers"], heads=cfg["heads"],
                                     ff=cfg["ff"], dropout=0.0).to(device)
            m.load_state_dict(ck["model"]); m.eval()
            probes[(arm, sd)] = m

    def encode(problem, steps):
        ids, ss, se = tokenize_solution(tok, problem, steps)
        H = forward_states(lm, cap, [ids], pad_id, device)[0].to(torch.float16).float().cpu()
        return ids, ss, se, H

    @torch.no_grad()
    def score(H, ss, se, i, crop_after=None):
        if crop_after is not None:
            ss, se = ss[:crop_after + 1], se[:crop_after + 1]
            H = H[:se[-1]]
        tr = {"H": H, "step_starts": ss, "step_ends": se}
        out = {}
        for (arm, sd), m in probes.items():
            b = collate([tr], "full" if arm == "full" else "causal", [[i]]).to(device)
            out[f"{arm}/{sd}"] = float(m(b)[b.q_valid][0])
        return out

    rows = []
    for t in targets:
        tr = inputs[t["trace_id"]]
        i = t["step"]
        steps = tr["steps"]
        ids0, ss0, se0, H0 = encode(tr["problem"], steps)
        row = {**t, "n_steps": len(steps), "variants": {}}
        row["variants"]["unchanged"] = score(H0, ss0, se0, i)
        row["variants"]["future_hidden"] = score(H0, ss0, se0, i, crop_after=i)
        rng = random.Random(int(hashlib.sha1(f"{a.seed}:{t['trace_id']}:{i}".encode()).hexdigest(), 16))
        later = steps[i + 1:]
        perm = list(range(len(later)))
        while perm == sorted(perm):
            rng.shuffle(perm)
        variants = {"shuffled": (steps[:i + 1] + [later[p] for p in perm], True)}
        na, changed = remove_answer(later)
        variants["no_answer"] = (steps[:i + 1] + na, changed and len(na) >= 1)
        for name, (st, ok) in variants.items():
            if not ok:
                row["variants"][name] = None
                continue
            ids, ss, se, H = encode(tr["problem"], st)
            cut = se0[i]
            same_ids = ids[:cut] == ids0[:cut] and ss[:i + 1] == ss0[:i + 1] and se[:i + 1] == se0[:i + 1]
            dev = float((H[:cut] - H0[:cut]).abs().max() / H0[:cut].abs().mean())
            row["variants"][name] = score(H, ss, se, i)
            row.setdefault("checks", {})[name] = {"prefix_target_ids_identical": bool(same_ids),
                                                  "prefix_target_state_max_dev_rel": dev}
        rows.append(row)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"cell_sizes": cell_sizes, "n_selected": len(rows),
                                 "rules": {"answer_block": ANSWER_BLOCK.pattern, "boxed": "\\boxed{...} balanced"},
                                 "rows": rows}, indent=1))
    print(f"[diag] {len(rows)} targets -> {a.out}")


if __name__ == "__main__":
    main()
