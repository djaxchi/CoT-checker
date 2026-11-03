#!/usr/bin/env python3
"""InStep V1 runs U1 and T1 (docs/instep_v1_runbook.md): training-free step scores.

U1, token confidence of the backbone on its own trace (default phase `u1`). One
teacher-forced forward over prompt + reply per trace; per step, over the step's reply
tokens (`steps[k].aligned_token_ids`), LM-Polygraph's claim-level estimators as ReProbe
stored them (lm_polygraph/estimators/claim/{max_probability,token_entropy,perplexity}.py):
  maxprob  MaximumClaimProbability  -sum_t log p(y_t | y_<t, x)
  maxent   MaxTokenEntropyClaim     max_t H(p(. | y_<t, x)), full-vocabulary entropy
  ppl      PerplexityClaim          -mean_t log p(y_t | y_<t, x)
from the raw (temperature 1) logits. Higher = more uncertain = more likely incorrect, the
orientation of the stored values. Trace files are read with reprobe_probe.load_traces,
whose step positions are the logit rows that predict the step's tokens.

T1, P(True) (phase `ptrue`): for every item (uid, problem, prefix, candidate_step) a Qwen3
chat prompt in non-thinking mode asks "Is the current step correct? Answer Yes or No.";
the score is sigmoid(logit_No - logit_Yes) at the answer position (P(No) renormalized
over {Yes, No}). `--shard i/n` scores items i::n of every set into a shard directory;
with `--claim_dir` a process keeps claiming unclaimed shards of n (one model load per GPU,
early finishers take more work); phase `merge` joins the shards.

Phase `gate`: per backbone and test set, Spearman and PR-AUC of our U1 scores against the
values stored in ReProbe's managers (trace files' `scores`), plus val AUROC / AP of every
tag present; written to <out>/scores/gate_confidence.json.

Scores: <out>/scores/<backbone>/<tag>/<set>.jsonl, one line per step {"uid", "p"}.

    python scripts/instep_v1/score_confidence.py u1 --model Qwen/Qwen3-8B --backbone qwen3_8b \\
        --sets math:<R>/data/test/math.traces.jsonl val:<R>/data/rp/val.traces.jsonl --out <OUT>
    python scripts/instep_v1/score_confidence.py ptrue --shard 0/4 --sets math:<R>/data/test/math.items.jsonl --out <OUT>
    python scripts/instep_v1/score_confidence.py merge --backbone qwen3_8b --tag ptrue --out <OUT>
    python scripts/instep_v1/score_confidence.py gate --out <OUT> --data qwen3_8b:<R1>/data:ds ...
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.reprobe_cmp.reprobe_probe import load_model, load_traces  # noqa: E402

U1_TAGS = ("maxprob", "maxent", "ppl")
STORED = {"maxprob": "MaximumClaimProbability", "maxent": "MaxTokenEntropyClaim", "ppl": "PerplexityClaim"}
TEST = ("math", "gsm8k", "proofnet", "trips", "meetings", "calendar", "strategyqa", "scienceqa")

PTRUE_SYSTEM = ("You are a careful verifier of step-by-step reasoning. The tasks include math "
                "problems, proofs, planning and scheduling, and commonsense and science "
                "questions. You are given a question, the previous steps of a solution, and the "
                "current step. Decide whether the current step is correct, given the question "
                "and the previous steps.")
PTRUE_QUESTION = "Is the current step correct? Answer Yes or No."


def jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p) if l.strip()]


# ---------------------------------------------------------------- estimators
def token_stats(logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(log p(target), entropy) per row of (N, V) logits, in float32. Entropy is
    LM-Polygraph's EntropyCalculator: -sum p log p over the finite log-probabilities."""
    lp = torch.log_softmax(logits.float(), dim=-1)
    ll = lp.gather(-1, targets.view(-1, 1)).squeeze(-1)
    fin = torch.isfinite(lp)
    ent = -torch.where(fin, lp.exp() * lp, torch.zeros_like(lp)).sum(-1)
    return ll, ent


def claim_scores(ll: np.ndarray, ent: np.ndarray) -> dict[str, float]:
    """The three claim estimators over one step's token log-likelihoods and entropies.
    A step without tokens gets 0 for all three (counted by the caller)."""
    if len(ll) == 0:
        return {t: 0.0 for t in U1_TAGS}
    return {"maxprob": float(-np.sum(ll)), "maxent": float(np.max(ent)), "ppl": float(-np.mean(ll))}


def length_batches(lengths: list[int], max_tokens: int, max_rows: int = 256) -> list[list[int]]:
    """Indices sorted by length, cut into batches of at most max_tokens padded tokens."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    out, cur, T = [], [], 0
    for i in order:
        T2 = max(T, lengths[i])
        if cur and ((len(cur) + 1) * T2 > max_tokens or len(cur) >= max_rows):
            out.append(cur)
            cur, T2 = [], lengths[i]
        cur.append(i)
        T = T2
    if cur:
        out.append(cur)
    return out


def padded(seqs: list[list[int]], pad: int, mult: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-padded (ids, attention mask), length rounded up to `mult` (few kernel shapes)."""
    T = -(-max(len(s) for s in seqs) // mult) * mult
    ids = torch.full((len(seqs), T), pad, dtype=torch.long)
    att = torch.zeros((len(seqs), T), dtype=torch.long)
    for b, s in enumerate(seqs):
        ids[b, :len(s)] = torch.tensor(s, dtype=torch.long)
        att[b, :len(s)] = 1
    return ids, att


def decoder_and_head(model):
    return model.get_decoder(), model.get_output_embeddings()


@torch.no_grad()
def u1_scores(model, traces: list[dict], dev, pad: int, max_tokens: int, chunk: int = 2048) -> tuple[dict, int]:
    """{tag: {uid: score}} for every step of every trace, and the number of empty steps."""
    dec, head = decoder_and_head(model)
    out = {t: {} for t in U1_TAGS}
    empty = 0
    for mb in length_batches([len(t["ids"]) for t in traces], max_tokens):
        batch = [traces[i] for i in mb]
        ids, att = padded([t["ids"] for t in batch], pad)
        ids, att = ids.to(dev), att.to(dev)
        h = dec(input_ids=ids, attention_mask=att, use_cache=False).last_hidden_state
        rows, cols = [], []
        for b, t in enumerate(batch):
            q = sorted({p for s in t["steps"] for p in s})
            rows += [b] * len(q)
            cols += q
        if not rows:
            continue
        r = torch.tensor(rows, device=dev)
        c = torch.tensor(cols, device=dev)
        lls, ents = [], []
        for s in range(0, len(rows), chunk):
            rr, cc = r[s:s + chunk], c[s:s + chunk]
            ll, ent = token_stats(head(h[rr, cc]), ids[rr, cc + 1])
            lls.append(ll)
            ents.append(ent)
        ll = torch.cat(lls).cpu().numpy()
        ent = torch.cat(ents).cpu().numpy()
        where = {(b, q): i for i, (b, q) in enumerate(zip(rows, cols))}
        for b, t in enumerate(batch):
            for k, s in enumerate(t["steps"]):
                ix = [where[(b, q)] for q in s]
                empty += not ix
                for tag, v in claim_scores(ll[ix], ent[ix]).items():
                    out[tag][f"{t['trace_id']}::{k}"] = v
    return out, empty


# ---------------------------------------------------------------- P(True)
def ptrue_prompt(problem: str, prefix: str, step: str) -> str:
    """Qwen3 chat template (system + user, non-thinking assistant turn) as a literal; equal to
    apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)."""
    user = (f"Question:\n{problem}\n\nPrevious steps:\n{prefix if prefix else '(none)'}\n\n"
            f"Current step:\n{step}\n\n{PTRUE_QUESTION}")
    return (f"<|im_start|>system\n{PTRUE_SYSTEM}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n")


def yes_no_ids(tok) -> tuple[int, int]:
    y, n = tok.encode("Yes", add_special_tokens=False), tok.encode("No", add_special_tokens=False)
    if len(y) != 1 or len(n) != 1:
        raise SystemExit(f"Yes/No are not single tokens: {y} {n}")
    return y[0], n[0]


def ptrue_from_logits(logit_yes: torch.Tensor, logit_no: torch.Tensor) -> torch.Tensor:
    """P(No) renormalized over {Yes, No}."""
    return torch.sigmoid(logit_no.float() - logit_yes.float())


@torch.no_grad()
def ptrue_scores(model, tok, items: list[dict], dev, pad: int, max_tokens: int) -> dict:
    dec, head = decoder_and_head(model)
    yes, no = yes_no_ids(tok)
    enc = [tok(ptrue_prompt(it["problem"], it["prefix"], it["candidate_step"]),
               add_special_tokens=False)["input_ids"] for it in items]
    out = {}
    for mb in length_batches([len(e) for e in enc], max_tokens):
        ids, att = padded([enc[i] for i in mb], pad)
        h = dec(input_ids=ids.to(dev), attention_mask=att.to(dev), use_cache=False).last_hidden_state
        last = torch.tensor([len(enc[i]) - 1 for i in mb], device=dev)
        lg = head(h[torch.arange(len(mb), device=dev), last])
        p = ptrue_from_logits(lg[:, yes], lg[:, no]).cpu().numpy()
        for i, v in zip(mb, p):
            out[items[i]["uid"]] = float(v)
    return out


# ---------------------------------------------------------------- io
def write_scores(path: Path, scores: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for uid, p in scores.items():
            f.write(json.dumps({"uid": uid, "p": p}) + "\n")
    tmp.replace(path)


def parse_sets(specs: list[str]) -> list[tuple[str, Path]]:
    return [(s.split(":", 1)[0], Path(s.split(":", 1)[1])) for s in specs]


def parse_shard(s: str) -> tuple[int, int]:
    i, n = (int(x) for x in s.split("/"))
    if not 0 <= i < n:
        raise ValueError(s)
    return i, n


def merge_shards(shard_dir: Path, dst: Path) -> list[str]:
    """Join <set>.<i>of<n>.jsonl shards into <dst>/<set>.jsonl; returns the merged sets.
    Refuses a set whose shards are incomplete."""
    by = {}
    for f in shard_dir.glob("*of*.jsonl"):
        name, tail = f.name[:-len(".jsonl")].rsplit(".", 1)
        i, n = (int(x) for x in tail.split("of"))
        by.setdefault((name, n), {})[i] = f
    done = []
    for (name, n), fs in sorted(by.items()):
        if len(fs) != n:
            raise SystemExit(f"{name}: {len(fs)} of {n} shards in {shard_dir}")
        sc = {}
        for i in range(n):
            sc.update({r["uid"]: r["p"] for r in jl(fs[i])})
        write_scores(dst / f"{name}.jsonl", sc)
        done.append(name)
    return done


# ---------------------------------------------------------------- gate
def spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).correlation)


def ap(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, s)) if 0 < y.mean() < 1 else float("nan")


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if 0 < y.mean() < 1 else float("nan")


def gate_set(traces: list[dict], ours: dict[str, dict], min_rho: float = 0.95, max_dap: float = 0.01) -> dict:
    """Per tag: Spearman (all steps) and AP on labeled steps, ours vs stored."""
    res = {}
    for tag, est in STORED.items():
        if tag not in ours or not any(est in t.get("scores", {}) for t in traces):
            continue
        a, b, y = [], [], []
        miss = 0
        for t in traces:
            st = t["scores"].get(est)
            if st is None or len(st) != len(t["steps"]):
                miss += len(t["steps"])
                continue
            for k, v in enumerate(st):
                u = f"{t['trace_id']}::{k}"
                if u not in ours[tag]:
                    miss += 1
                    continue
                a.append(ours[tag][u])
                b.append(float(v))
                y.append(int(t["labels"][k]))
        a, b, y = np.array(a), np.array(b), np.array(y)
        m = y >= 0
        r = {"n": int(len(a)), "missing": miss, "spearman": spearman(a, b),
             "ap_ours": ap(y[m], a[m]), "ap_stored": ap(y[m], b[m]),
             "auroc_ours": auroc(y[m], a[m]), "auroc_stored": auroc(y[m], b[m])}
        r["pass"] = bool(r["spearman"] >= min_rho and abs(r["ap_ours"] - r["ap_stored"]) <= max_dap and miss == 0)
        res[tag] = r
    return res


def gate(a) -> None:
    out = {"criteria": "per set: Spearman >= 0.95 against stored and |AP ours - AP stored| <= 0.01",
           "backbones": {}}
    for spec in a.data:
        bb, data, vlab = spec.split(":")
        data = Path(data)
        root = a.out / "scores" / bb
        res = {"test": {}, "val": {}}
        for s in TEST:
            ours = {t: {r["uid"]: r["p"] for r in jl(root / t / f"{s}.jsonl")}
                    for t in U1_TAGS if (root / t / f"{s}.jsonl").exists()}
            if ours:
                res["test"][s] = gate_set(jl(data / "test" / f"{s}.traces.jsonl"), ours)
        vp = data / "rp" / "val.traces.jsonl"
        if vp.exists():
            y = {f"{t['trace_id']}::{k}": int(v) for t in jl(vp) for k, v in enumerate(t["labels"][vlab])}
            for tag_dir in sorted(p for p in root.glob("*") if (p / "val.jsonl").exists()):
                sc = {r["uid"]: r["p"] for r in jl(tag_dir / "val.jsonl")}
                u = [k for k, v in y.items() if v >= 0 and k in sc]
                yy, ss = np.array([y[k] for k in u]), np.array([sc[k] for k in u])
                res["val"][tag_dir.name] = {"n": len(u), "missing_labeled": sum(v >= 0 for v in y.values()) - len(u),
                                            "auroc": auroc(yy, ss), "ap": ap(yy, ss), "prevalence": float(yy.mean())}
        res["all_pass"] = all(r["pass"] for s in res["test"].values() for r in s.values())
        out["backbones"][bb] = res
    a.out.joinpath("scores").mkdir(parents=True, exist_ok=True)
    (a.out / "scores" / "gate_confidence.json").write_text(json.dumps(out, indent=1))
    for bb, res in out["backbones"].items():
        print(f"== {bb} all_pass={res['all_pass']}")
        for s, rr in res["test"].items():
            print(f"  {s:11s} " + "  ".join(
                f"{t}: rho {r['spearman']:.3f} AP {r['ap_ours']:.3f}/{r['ap_stored']:.3f}{'' if r['pass'] else ' FAIL'}"
                for t, r in rr.items()))
        for t, r in res["val"].items():
            print(f"  val {t}: AUROC {r['auroc']:.3f} AP {r['ap']:.3f} (n {r['n']}, prev {r['prevalence']:.3f})")


# ---------------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("phase", choices=["u1", "ptrue", "merge", "gate"])
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--backbone", default="qwen3_8b")
    p.add_argument("--sets", nargs="*", default=[], help="name:path (u1: trace files; ptrue: items files)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--shard", default="0/1", help="ptrue: i/n, items i::n of every set")
    p.add_argument("--claim_dir", type=Path, help="ptrue: claim shards of --shard's n from this dir")
    p.add_argument("--tag", default="ptrue", help="merge: tag whose shards to join")
    p.add_argument("--data", nargs="*", default=[], help="gate: backbone:<R>/data:<val label>")
    p.add_argument("--max_tokens", type=int, default=32768, help="padded tokens per forward")
    p.add_argument("--smoke", type=int, default=0, help="first N traces / items of every set")
    a = p.parse_args()
    if a.phase == "gate":
        return gate(a)
    if a.phase == "merge":
        root = a.out / "scores" / a.backbone / a.tag
        print(f"[merge] {merge_shards(root / '_shards', root)}", flush=True)
        return
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.perf_counter()
    model, tok, pad = load_model(a.model, dev)
    print(f"[load] {a.model} in {time.perf_counter() - t0:.0f}s", flush=True)
    root = a.out / "scores" / a.backbone
    if a.phase == "u1":
        for name, path in parse_sets(a.sets):
            t0 = time.perf_counter()
            traces = load_traces(path, tok, a.smoke)
            sc, empty = u1_scores(model, traces, dev, pad, a.max_tokens)
            for tag, s in sc.items():
                write_scores(root / tag / f"{name}.jsonl", s)
            ntok = sum(len(t["ids"]) for t in traces)
            print(f"[u1] {name}: {len(traces)} traces, {len(sc['maxprob'])} steps ({empty} empty), "
                  f"{ntok} tokens in {time.perf_counter() - t0:.0f}s", flush=True)
    else:
        for i, n in claimed_shards(a):
            t1 = time.perf_counter()
            for name, path in parse_sets(a.sets):
                t0 = time.perf_counter()
                items = jl(path)
                if a.smoke:
                    items = items[:a.smoke]
                items = items[i::n]
                sc = ptrue_scores(model, tok, items, dev, pad, a.max_tokens)
                write_scores(root / "ptrue" / "_shards" / f"{name}.{i}of{n}.jsonl", sc)
                print(f"[ptrue] {name} shard {i}/{n}: {len(items)} items in {time.perf_counter() - t0:.0f}s",
                      flush=True)
            print(f"[ptrue] shard {i}/{n} done in {time.perf_counter() - t1:.0f}s", flush=True)


def claimed_shards(a):
    """--shard i/n alone: that shard. With --claim_dir: every shard of n not yet claimed by
    another process (atomic mkdir), so GPUs that finish early take more work."""
    i, n = parse_shard(a.shard)
    if a.claim_dir is None:
        yield i, n
        return
    a.claim_dir.mkdir(parents=True, exist_ok=True)
    for j in range(n):
        try:
            (a.claim_dir / str(j)).mkdir()
        except FileExistsError:
            continue
        yield j, n


if __name__ == "__main__":
    main()
