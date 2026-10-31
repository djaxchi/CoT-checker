#!/usr/bin/env python3
"""Step-level beam search with a step verifier in the loop (reprobe_comparison_v2, B2).

ReProbe's Table 2 setting on Qwen3-8B: "a beam of B = 5 partial reasoning trajectories
at each step. From each partial trajectory, we sample N = 5 candidate continuations ...
We assign a step-level correctness score to each candidate and retain the top-B expanded
trajectories with the highest scores." Here:

  * the policy is Qwen3-8B under ReProbe's prompt (src/onpolicy/prompts.py reprobe_prompt),
    sampling T 1.0, top-p 0.95, top-k 50 (ReProbe's test-time sampling); a step is one line,
    generation stops at "\\n"; the root draws B x N candidates;
  * a candidate's score is its new step's P(incorrect) given the trajectory so far, and
    the B lowest are kept; a kept candidate whose step starts with "<Answer>:" (or that
    ends the turn) is finished and leaves the beam; search stops when no beam is active or
    after --max_steps;
  * the answer is the finished trajectory with the lowest worst-step score (the best-of-N
    aggregation); if none finished, the active one with the lowest;
  * verifiers share the policy's own forward model: `rt` (a read token, read inline in the
    reprobe layout exactly as scripts/read_token_inline.py encodes), `rp` (a released
    ReProbe hidden-state probe; in the loop there is no future to see) and `random`
    (uniform scores: the same search with no verifier, which prices the search itself).

Arms run on the same questions with the same per-question sampling seed. Grading is
src/eval/reprobe_bon_grade.py. One JSON line per (question, arm) in --out.

    python scripts/reprobe_cmp/beam_search.py --pools $SCRATCH/cot_mech/reprobe_bon/pools \\
        --set gsm8k --limit 100 --arm rt_self=rt:$U/tokens/rt_self_seed0/read_token.pt \\
        rp_self=rp:$R/released/self random=random --out $R/beam/gsm8k.jsonl
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
from src.eval.reprobe_bon_grade import grade  # noqa: E402
from src.onpolicy.prompts import reprobe_prompt  # noqa: E402

SKIP = ("Reasoning Steps:", "SOLUTION:", "<start of response>")


class RTScorer:
    """P(incorrect) of each trajectory's last step from an inline read token."""

    def __init__(self, model, tok, ckpt: Path, dev):
        from scripts.read_token_inline import marker_from_ckpt
        from scripts.read_token_probe import read_token_from_ckpt
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.model, self.tok, self.dev = model, tok, dev
        self.rt, self.mk = read_token_from_ckpt(ck, model, dev), marker_from_ckpt(ck, dev)
        self.chain, self.soft_attn = bool(ck.get("chain")), ck.get("soft_attn")
        self.pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def __call__(self, problem: str, trajs: list[list[str]]) -> np.ndarray:
        from scripts.read_token_inline import build_sequences, inline_read, member_index, row
        items = [{"uid": str(i), "id": str(i), "problem": problem, "prefix": "\n".join(t[:-1]),
                  "step": t[-1], "label": 0} for i, t in enumerate(trajs)]
        seqs, _ = build_sequences(self.tok, items, "reprobe", chain=self.chain)
        out = np.full(len(items), np.nan)
        with torch.no_grad():
            h = inline_read(self.model, self.rt, [row(s, self.mk) for s in seqs], self.pad, self.dev,
                            grad=False, soft_attn=self.soft_attn, mk=self.mk, chain=self.chain)
            h = h[member_index(seqs, self.dev)]
            p = torch.sigmoid(self.rt.logit(h)).float().cpu().numpy()
        out[[i for s in seqs for i in s["members"]]] = p
        return out


class RPScorer:
    """P(incorrect) of each trajectory's last step from a released ReProbe probe."""

    def __init__(self, model, tok, release: Path, dev):
        from scripts.reprobe_cmp.reprobe_probe import head_from_release
        self.model, self.tok, self.dev = model, tok, dev
        self.head = head_from_release(release, dev, model.config.hidden_size)
        self.pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def __call__(self, problem: str, trajs: list[list[str]]) -> np.ndarray:
        from scripts.reprobe_cmp.reprobe_probe import score_traces
        from src.probes.reprobe_head import step_feature_index
        ptoks = self.tok(reprobe_prompt(problem), add_special_tokens=False)["input_ids"]
        traces = []
        for i, t in enumerate(trajs):
            before = "\n".join(t[:-1]) + "\n" if len(t) > 1 else ""
            enc = self.tok(before + t[-1], add_special_tokens=False, return_offsets_mapping=True)
            last = [j for j, (s, e) in enumerate(enc["offset_mapping"]) if e > len(before)]
            ids = ptoks + list(enc["input_ids"])
            pos = [q for q in step_feature_index(len(ptoks), last) if 0 <= q < len(ids) - 1]
            traces.append({"trace_id": str(i), "ids": ids, "ctx": len(ptoks), "steps": [pos], "labels": [-1]})
        sc = score_traces(self.model, {"p": (self.head, "visible")}, traces, self.dev, self.pad, 24000, 160000)["p"]
        return np.array([sc.get(f"{i}::0", 1.0) for i in range(len(trajs))])


class RandomScorer:
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def __call__(self, problem: str, trajs: list[list[str]]) -> np.ndarray:
        return self.rng.random(len(trajs))


def sample_steps(model, tok, contexts: list[str], n: int, max_new: int, dev) -> list[tuple[str, bool]]:
    """n continuations of every context, each cut at its first newline:
    (step text, the turn ended inside it), context-major."""
    tok.padding_side = "left"
    enc = tok(contexts, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
    with torch.no_grad():
        out = model.generate(**enc, do_sample=True, temperature=1.0, top_p=0.95, top_k=50,
                             num_return_sequences=n, max_new_tokens=max_new, stop_strings=["\n"],
                             tokenizer=tok, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    res = []
    for r in out[:, enc["input_ids"].shape[1]:]:
        text = tok.decode(r, skip_special_tokens=False)
        ended = "<|im_end|>" in text.split("\n")[0] or "<|endoftext|>" in text.split("\n")[0]
        step = text.split("\n")[0].replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
        res.append((step, ended))
    return res


def is_final(step: str, ended: bool) -> bool:
    return ended or step.startswith("<Answer>:") or step.startswith("<end of response>")


def search(model, tok, scorer, problem: str, B: int, N: int, max_steps: int, max_new: int, dev) -> dict:
    prompt = reprobe_prompt(problem)
    active = [{"steps": [], "scores": []}]
    finished, generated, n_scored = [], 0, 0
    for _ in range(max_steps):
        if not active:
            break
        ctx = [prompt + "".join(s + "\n" for s in b["steps"]) for b in active]
        draws = sample_steps(model, tok, ctx, N * B if len(active) == 1 and not active[0]["steps"] else N,
                             max_new, dev)
        per = len(draws) // len(ctx)
        cands = []
        for j, (step, ended) in enumerate(draws):
            b = active[j // per]
            generated += 1
            if step.startswith(SKIP):  # a format line, not a step: the beam stays as it was
                continue
            cands.append((b, step, ended))
        if not cands:
            break
        real = [c for c in cands if c[1]]
        sc = scorer(problem, [c[0]["steps"] + [c[1]] for c in real]) if real else np.array([])
        n_scored += len(real)
        it = iter(sc)
        scores = [float(next(it)) if c[1] else 1.0 for c in cands]
        keep = np.argsort(scores, kind="stable")[:B]
        active = []
        for k in keep:
            b, step, ended = cands[k]
            nb = {"steps": b["steps"] + ([step] if step else []), "scores": b["scores"] + [scores[k]]}
            (finished if is_final(step, ended) else active).append(nb)
    pool = finished or active
    best = min(pool, key=lambda b: max(b["scores"]) if b["scores"] else 1.0)
    return {"steps": best["steps"], "scores": best["scores"], "finished": bool(finished),
            "n_finished": len(finished), "step_draws": generated, "scored": n_scored}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pools", type=Path, required=True, help="the best-of-N pools (questions and gold answers)")
    p.add_argument("--set", required=True)
    p.add_argument("--arm", nargs="+", required=True, help="name=rt:CKPT | name=rp:RELEASE_DIR | name=random")
    p.add_argument("--limit", type=int, default=100, help="first N questions of the set")
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--B", type=int, default=5)
    p.add_argument("--N", type=int, default=5)
    p.add_argument("--max_steps", type=int, default=30)
    p.add_argument("--max_new_tokens", type=int, default=192)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    from scripts.reprobe_cmp.bon_d0 import load_pool
    from scripts.reprobe_cmp.reprobe_probe import load_model
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.enable_cudnn_sdp(False)
    model, tok, _ = load_model(a.model, dev)
    df = load_pool(a.pools, a.set)
    qs = df[df.k == 0].sort_values("qi").iloc[a.offset:a.offset + a.limit]
    arms = {}
    for spec in a.arm:
        name, _, rest = spec.partition("=")
        kind, _, path = rest.partition(":")
        arms[name] = (RTScorer(model, tok, Path(path), dev) if kind == "rt" else
                      RPScorer(model, tok, Path(path), dev) if kind == "rp" else RandomScorer(a.seed))
    done = set()
    if a.out.exists():
        done = {(r["qi"], r["arm"]) for r in map(json.loads, open(a.out))}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "a") as f:
        for r in qs.to_dict("records"):
            for name, scorer in arms.items():
                if (int(r["qi"]), name) in done:
                    continue
                seed = a.seed * 100003 + int(r["qi"])
                torch.manual_seed(seed)
                if isinstance(scorer, RandomScorer):
                    scorer.rng = np.random.default_rng(seed)
                t0 = time.perf_counter()
                res = search(model, tok, scorer, r["q"], a.B, a.N, a.max_steps, a.max_new_tokens, dev)
                reply = "\n".join(res["steps"])
                ok, key = grade(a.set, reply, str(r["answer"]), r["q"])
                f.write(json.dumps({"set": a.set, "qi": int(r["qi"]), "arm": name, "correct": bool(ok), "key": key,
                                    "seconds": round(time.perf_counter() - t0, 1), **res}, ensure_ascii=False) + "\n")
                f.flush()
                print(f"[beam] {a.set} q{r['qi']} {name}: correct {ok} steps {len(res['steps'])} "
                      f"draws {res['step_draws']} ({time.perf_counter() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
