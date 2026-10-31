#!/usr/bin/env python3
"""The read token against the PRM while the policy generates (read_token_v1 Stage J).

Qwen3-8B samples solutions to MATH-500 problems (chat layout, T = 1.0, top-p 0.95,
top-k 50, 4,096-token cap, the instruct_t10 pool's settings) with continuous batching:
B rows, a finished row refilled with the next job. Arms, all on one engine
(scripts/live_engine.py: a layer loop over per-row key/value columns, the policy's decode
calls captured as CUDA graphs):

  plain      generation only
  read       the read token (chain + start marker) rides on the decode call that feeds
             each step's separator: a marker and a read slot next to the text slot,
             P(step incorrect) out of the same call
  read_stop  the same, and a row stops (abstains) once its worst step score exceeds tau
  prm        after the decode call that feeds a separator, Qwen2.5-Math-PRM-7B extends
             its own cache of the row by the step's tokens and <extra_0> (one eager call
             for the rows that finished a step) and reads its head there
  prm_stop   the same with the stop rule

The final step is read after an appended separator (the offline scorers append one).
Every job draws its uniforms from its own generator, so the arms sample the same text
unless the logits differ.

Validation, before the timed arms: pool trajectories replayed through the live path
(teacher-forced) against the pool's saved scores (rthics_seed1, prm_qwen25_math_7b);
after them: the no-stop arms' live scores against offline scoring of the same texts.

    python scripts/analysis/live_compare.py --inline_ckpt tokens/rthics_seed1/read_token.pt \\
        --trajectories <pool>/tts_math500.shard*_trajectories.jsonl \\
        --pool_scores <pool>/scores --out <dir>/live_shard0 --shard_idx 0 --num_shards 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.analysis.rth_cost import PRM_SYSTEM  # noqa: E402
from scripts.generate_onpolicy_steps import split_into_steps  # noqa: E402
from scripts.inline_fast import MARKER, READ, TEXT  # noqa: E402
from scripts.live_engine import (  # noqa: E402
    FIELDS, Engine, GraphRunner, RowCache, StepState, bucket, sample)
from scripts.onpolicy.score_traces_with_read_token import read_jsonl  # noqa: E402
from src.onpolicy.prompts import chat_prompt  # noqa: E402

SEP = "\n\n"
PRM_SEP = "<extra_0>"
ARMS = ("plain", "read", "read_stop", "prm", "prm_stop")


def exploratory(problem_id: str) -> bool:
    """online_compute_sim's split: thresholds were picked on these problems."""
    return int(hashlib.sha256(problem_id.encode()).hexdigest(), 16) % 2 == 0


def job_seed(fork_id: str, sample_idx: int, seed: int) -> int:
    return int(hashlib.sha256(f"{seed}:{fork_id}:{sample_idx}".encode()).hexdigest(), 16) % (2 ** 63)


def load_prm(name: str, dev):
    """Qwen2.5-Math-PRM-7B: the Qwen2 backbone through AutoModelForCausalLM (as in H4) and
    its two-way head (Linear, ReLU, Linear) read from the checkpoint's score.* tensors."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from scripts.read_token_store import load_model
    model, tok = load_model(name, dev)
    d = model.config.hidden_size
    sd = {}
    for f in sorted(Path(snapshot_download(name, local_files_only=True)).glob("*.safetensors")):
        with safe_open(str(f), "pt") as s:
            for k in s.keys():
                if k.startswith("score."):
                    sd[k[len("score."):]] = s.get_tensor(k)
    head = torch.nn.Sequential(torch.nn.Linear(d, d), torch.nn.ReLU(), torch.nn.Linear(d, 2))
    head.load_state_dict(sd)  # strict: a missing tensor fails here, not as a random head
    return model, tok, head.to(dev, model.dtype).eval()


def prm_prefix(ptok, problem: str) -> list[int]:
    """The PRM template's tokens before the first step."""
    mark = "\x00STEPS\x00"
    text = ptok.apply_chat_template(
        [{"role": "system", "content": PRM_SYSTEM}, {"role": "user", "content": problem},
         {"role": "assistant", "content": mark}], tokenize=False, add_generation_prompt=False)
    return ptok(text[:text.index(mark)], add_special_tokens=False)["input_ids"]


class Row:
    def __init__(self, job: dict, P: int):
        self.job, self.t, self.gen = job, P, []
        self.next = None  # sampled, not yet fed
        self.st, self.k, self.spos, self.step0 = StepState(), 0, P, 0
        self.scores, self.worst, self.lat = [], 0.0, []
        self.rng = np.random.default_rng(job["seed"])
        self.prm_n, self.end, self.t0 = 0, None, time.perf_counter()


class Live:
    """One arm over a job list with B rows. pol: Engine of the policy (with the read token);
    graphs: {"plain": GraphRunner, "read": GraphRunner} over pol_cache, or None to run
    eagerly; prm: (Engine, RowCache, head, tokenizer) for the prm arms."""

    def __init__(self, pol, pol_cache, tok, vocab, graphs, prm, a):
        self.pol, self.cache, self.tok, self.vocab, self.graphs, self.prm, self.a = \
            pol, pol_cache, tok, vocab, graphs, prm, a
        self.sep_id = tok(SEP, add_special_tokens=False)["input_ids"]
        if len(self.sep_id) != 1:
            raise ValueError(f"the separator is {len(self.sep_id)} tokens")
        self.sep_id = self.sep_id[0]
        self.eos = {i for i in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"),
                                tok.convert_tokens_to_ids("<|endoftext|>")) if i is not None}
        self.dev = pol.dev
        if prm is not None:
            self.prm_sep = prm[3].convert_tokens_to_ids(PRM_SEP)

    # ---- model calls ----------------------------------------------------------------
    def _out_plain(self, h):
        return self.pol.model.lm_head(h[:, 0]).float(), None

    def _out_read(self, h):
        return self.pol.model.lm_head(h[:, 0]).float(), self.pol.rt.head(h.float()).squeeze(-1)

    def policy_call(self, host: torch.Tensor, ncol: int, with_read: bool):
        if self.graphs is not None:
            return self.graphs["read" if with_read else "plain"].run(host, ncol)
        st = dict(zip(FIELDS, host.to(self.dev).unbind(0)))
        h = self.pol.forward(self.cache, ncol=ncol, **st)
        return (self._out_read if with_read else self._out_plain)(h)

    def prefill(self, b: int, ids: list[int]):
        """The prompt into row b of the policy cache, eagerly: last-position logits (V,)."""
        self.cache.reset(b)
        n = len(ids)
        c = self.cache.rows(b, b + 1)
        ar = torch.arange(n, device=self.dev)[None]
        h = self.pol.forward(c, torch.tensor([ids], device=self.dev), torch.full_like(ar, TEXT), ar, ar,
                             ar + 1, torch.zeros_like(ar), bucket(n, self.a.col_step))
        self.cache.n[b] = n
        return self.pol.model.lm_head(h[:, -1]).float()[0]

    def prm_call(self, rows: list[int], chunks: list[list[int]]) -> list[float]:
        """Extend each row's PRM cache by its chunk, P(incorrect) at each chunk's last token."""
        eng, pc, head, _ = self.prm
        S = max(map(len, chunks))
        n0 = [pc.n[b] for b in rows]
        end = max(n + len(ch) for n, ch in zip(n0, chunks))
        if end > pc.cap - 1:
            raise ValueError(f"PRM cache full: a row needs {end} columns of {pc.cap - 1}")
        ncol = bucket(end, self.a.col_step)
        sub = pc.rows(rows[0], rows[0] + 1) if len(rows) == 1 else pc.gather(rows, ncol, junk=True)
        ids = torch.zeros(len(rows), S, dtype=torch.long)
        kind, pos, qlim = torch.zeros_like(ids), torch.zeros_like(ids), torch.zeros_like(ids)
        cols = torch.full((len(rows), S), sub.cap - 1, dtype=torch.long)  # pad slots: a junk column
        for i, ch in enumerate(chunks):
            m = len(ch)
            ids[i, :m] = torch.tensor(ch)
            kind[i, :m] = TEXT
            cols[i, :m] = n0[i] + torch.arange(m)
            pos[i, :m] = n0[i] + torch.arange(m)
            qlim[i, :m] = pos[i, :m] + 1
        t = [x.to(self.dev) for x in (ids, kind, pos, cols, qlim, torch.zeros_like(ids))]
        h = eng.forward(sub, *t, ncol=ncol)
        last = torch.tensor([len(ch) - 1 for ch in chunks], device=self.dev)
        p = torch.softmax(head(h[torch.arange(len(rows), device=self.dev), last]).float(), -1)[:, 0]
        if len(rows) > 1:
            pc.put_back(sub, rows, ncol)
        for b, ch in zip(rows, chunks):
            pc.n[b] += len(ch)
        return p.cpu().tolist()

    # ---- the loop -------------------------------------------------------------------
    def run(self, arm: str, jobs: list[dict], B: int, tau: float) -> tuple[list[dict], dict]:
        scorer = "read" if arm.startswith("read") else "prm" if arm.startswith("prm") else None
        stop = arm.endswith("_stop")
        full, full_prm = self.cache, self.prm
        if self.cache.B != B:  # the first B rows (graphs, if any, are for the full cache)
            if self.graphs is not None:
                raise ValueError(f"graphs were captured for {self.cache.B} rows, not {B}")
            self.cache = self.cache.rows(0, B)
            self.cache.n = [0] * B
            if self.prm is not None:
                self.prm = (self.prm[0], self.prm[1].rows(0, B)) + self.prm[2:]
                self.prm[1].n = [0] * B
        try:
            return self._run(scorer, stop, jobs, B, tau)
        finally:
            self.cache, self.prm = full, full_prm

    def _run(self, scorer, stop, jobs, B, tau):
        a = self.a
        rows: list[Row | None] = [None] * B
        queue = deque(jobs)
        out, st = [], {"iters": 0, "calls_s1": 0, "calls_s3": 0, "sec_s1": 0.0, "sec_s3": 0.0,
                       "prefill_sec": 0.0, "prm_calls": 0, "prm_rows": 0, "prm_sec": 0.0, "prm_gpu_sec": 0.0,
                       "prm_tokens": 0, "reads": 0, "gen_tokens": 0}
        host = {S: torch.zeros(len(FIELDS), B, S, dtype=torch.long).pin_memory() if self.dev.type == "cuda"
                else torch.zeros(len(FIELDS), B, S, dtype=torch.long) for S in (1, 3)}
        if self.prm is not None:
            for b in range(B):
                self.prm[1].reset(b)

        def refill(b):
            job = queue.popleft()
            r = Row(job, len(job["prompt_ids"]))
            t0 = time.perf_counter()
            logits = self.prefill(b, job["prompt_ids"])
            if scorer == "prm":
                pc = self.prm[1]
                pc.reset(b)
                self.prm_call([b], [job["prm_prefix"]])  # P at the prefix's last token is unused
            u = torch.tensor([r.rng.random()], device=self.dev)
            nxt = int(sample(logits[None], u, a.temperature, a.top_k, a.top_p)[0])
            r.next = job["forced"][0] if job.get("forced") else nxt
            st["prefill_sec"] += time.perf_counter() - t0
            rows[b] = r

        def finish(b, why):
            r = rows[b]
            self.cache.reset(b)
            r.end = why
            out.append({"jid": r.job["jid"], "fork_id": r.job["fork_id"], "sample_idx": r.job["sample_idx"],
                        "problem": r.job["problem"], "gold": r.job["gold"], "gen_ids": r.gen,
                        "solution": self.tok.decode(r.gen, skip_special_tokens=True),
                        "n_gen": len(r.gen), "scores": r.scores, "end": why,
                        "abstain_step": len(r.scores) - 1 if why == "abstain" else None,
                        "decision_lat": r.lat, "sec": time.perf_counter() - r.t0, "prompt_style": "chat"})
            rows[b] = None

        torch.cuda.synchronize() if self.dev.type == "cuda" else None
        t_start = time.perf_counter()
        for b in range(B):
            if queue:
                refill(b)
        t_prev = time.perf_counter()
        while any(r is not None for r in rows):
            plan = {}  # b -> (token to feed or None, read slots?, row ends?, step finished?)
            for b, r in enumerate(rows):
                if r is None:
                    continue
                x = r.next
                if x in self.eos or len(r.gen) >= a.max_new_tokens:
                    # the last step is read after an appended separator, as offline
                    feed = self.sep_id if scorer == "read" and r.st.content else None
                    plan[b] = (feed, feed is not None, True, r.st.content)
                else:
                    bnd = r.st.feed(self.vocab[x]) if scorer is not None else False
                    plan[b] = (x, scorer == "read" and bnd, False, bnd)
            with_read = any(p[1] for p in plan.values())
            S = 3 if with_read else 1
            feeders = [b for b, p in plan.items() if p[0] is not None]
            if feeders:
                hb = host[S]
                arr = hb.numpy()
                arr[:] = 0
                for b in range(B):
                    arr[3, b] = self.cache.n[b] + np.arange(S)
                for b in feeders:
                    r, (feed, rd, _, _) = rows[b], plan[b]
                    arr[0, b, 0], arr[1, b, 0], arr[2, b, 0], arr[4, b, 0] = feed, TEXT, r.t, r.t + 1
                    if rd:
                        arr[1, b, 1], arr[2, b, 1], arr[4, b, 1], arr[5, b, 1] = MARKER, r.spos, r.spos, r.k
                        arr[1, b, 2], arr[2, b, 2], arr[4, b, 2], arr[5, b, 2] = READ, r.t + 1, r.t + 1, r.k
                ncol = bucket(max(self.cache.n) + S, a.col_step)
                t0 = time.perf_counter()
                logits, rlog = self.policy_call(hb, ncol, scorer == "read")
                u = torch.tensor([rows[b].rng.random() if b in plan else 0.0 for b in range(B)],
                                 device=self.dev)
                toks = sample(logits, u, a.temperature, a.top_k, a.top_p).cpu().tolist()
                pr = torch.sigmoid(rlog[:, 2].double()).cpu().tolist() if with_read else None
                dt = time.perf_counter() - t0
                st["iters"] += 1
                st[f"calls_s{S}"] += 1
                st[f"sec_s{S}"] += dt
            now = time.perf_counter()
            pending = []
            for b, (feed, rd, end, flag) in plan.items():
                r = rows[b]
                if feed is not None:
                    self.cache.n[b] += 3 if rd else 1
                    r.t += 1
                    if not end:
                        r.gen.append(feed)
                        st["gen_tokens"] += 1
                if rd:
                    p = pr[b]
                    r.scores.append(p)
                    r.worst = max(r.worst, p)
                    r.k += 1
                    r.spos = r.t
                    r.lat.append(now - t_prev)
                    st["reads"] += 1
                    if stop and r.worst > tau:
                        finish(b, "abstain")
                        continue
                if scorer == "prm" and flag:
                    text = self.tok.decode(r.gen[r.step0:], skip_special_tokens=True).strip()
                    r.step0 = len(r.gen)
                    if text:
                        pending.append(b)
                        r.pending_chunk = self.prm[3](text, add_special_tokens=False)["input_ids"] + [self.prm_sep]
                if end and b not in pending:
                    finish(b, "eos" if r.next in self.eos else "cap")
                    continue
                if not end:
                    r.next = r.job["forced"][len(r.gen)] if r.job.get("forced") else toks[b]
            if pending:
                t0 = time.perf_counter()
                if self.dev.type == "cuda":
                    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    e0.record()
                ps = self.prm_call(pending, [rows[b].pending_chunk for b in pending])
                if self.dev.type == "cuda":
                    e1.record()
                    e1.synchronize()
                    st["prm_gpu_sec"] += e0.elapsed_time(e1) / 1e3
                st["prm_sec"] += time.perf_counter() - t0
                st["prm_calls"] += 1
                st["prm_rows"] += len(pending)
                st["prm_tokens"] += sum(len(rows[b].pending_chunk) for b in pending)
                done = time.perf_counter()
                for b, p in zip(pending, ps):
                    r = rows[b]
                    r.scores.append(p)
                    r.worst = max(r.worst, p)
                    r.k += 1
                    r.lat.append(done - t_prev)
                    if stop and r.worst > tau:
                        finish(b, "abstain")
                    elif plan[b][2]:
                        finish(b, "eos" if r.next in self.eos else "cap")
            for b in range(B):
                if rows[b] is None and queue:
                    refill(b)
            t_prev = time.perf_counter()
        torch.cuda.synchronize() if self.dev.type == "cuda" else None
        st["wall_sec"] = time.perf_counter() - t_start
        st["jobs"] = len(out)
        st["tokens_per_sec"] = st["gen_tokens"] / st["wall_sec"]
        return sorted(out, key=lambda r: r["jid"]), st


def make_jobs(problems: list[dict], samples: int, tok, ptok, seed: int, jid0: int = 0) -> list[dict]:
    jobs = []
    for p in problems:
        for s in range(samples):
            jobs.append({"jid": jid0 + len(jobs), "fork_id": p["fork_id"], "sample_idx": s,
                         "problem": p["problem"], "gold": p["gold"],
                         "prompt_ids": tok(chat_prompt(p["problem"]), add_special_tokens=False)["input_ids"],
                         "prm_prefix": prm_prefix(ptok, p["problem"]) if ptok is not None else None,
                         "seed": job_seed(p["fork_id"], s, seed)})
    return jobs


def forced_jobs(rows: list[dict], tok, ptok, eos: int) -> list[dict]:
    """Pool trajectories as teacher-forced jobs: the token ids the offline scorers read
    (prompt + steps joined by the separator + a final separator), then EOS."""
    from scripts.read_token_inline import encode_text
    jobs = []
    for r in rows:
        steps = split_into_steps(r["solution"])
        prompt = chat_prompt(r["problem"])
        ids, _ = encode_text(tok, prompt + SEP.join(steps) + SEP)
        P = len(tok(prompt, add_special_tokens=False)["input_ids"])
        jobs.append({"jid": len(jobs), "fork_id": r["fork_id"], "sample_idx": r.get("sample_idx", 0),
                     "problem": r["problem"], "gold": r["gold"], "prompt_ids": ids[:P],
                     "prm_prefix": prm_prefix(ptok, r["problem"]), "seed": 0,
                     "forced": ids[P:] + [eos], "traj_uid": r["traj_uid"], "n_steps": len(steps)})
    return jobs


def diff_summary(a: list[float], b: list[float]) -> dict:
    d = np.abs(np.asarray(a) - np.asarray(b))
    return {"steps": int(len(d)), "mean_abs_diff": float(d.mean()) if len(d) else None,
            "max_abs_diff": float(d.max()) if len(d) else None,
            "p99_abs_diff": float(np.quantile(d, 0.99)) if len(d) else None}


def offline_prm(prm, rows: list[dict], col_step: int) -> list[list[float]]:
    """The PRM over each whole template in one eager call (no incremental cache)."""
    eng, _, head, ptok = prm
    sep = ptok.convert_tokens_to_ids(PRM_SEP)
    out = []
    for r in rows:
        steps = split_into_steps(r["solution"])
        ids = prm_prefix(ptok, r["problem"])
        for s in steps:
            ids += ptok(s, add_special_tokens=False)["input_ids"] + [sep]
        n = len(ids)
        c = RowCache(eng.model, 1, bucket(n, col_step) + 1, eng.dev)
        ar = torch.arange(n, device=eng.dev)[None]
        x = torch.tensor([ids], device=eng.dev)
        h = eng.forward(c, x, torch.full_like(ar, TEXT), ar, ar, ar + 1, torch.zeros_like(ar), bucket(n, col_step))
        at = (x[0] == sep).nonzero().squeeze(-1)
        out.append(torch.softmax(head(h[0, at]).float(), -1)[:, 0].cpu().tolist())
        del c
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--prm_model", default="Qwen/Qwen2.5-Math-PRM-7B")
    p.add_argument("--inline_ckpt", type=Path, required=True)
    p.add_argument("--trajectories", type=Path, nargs="+", required=True, help="the pool: problems and replay rows")
    p.add_argument("--pool_scores", type=Path, nargs="+", required=True,
                   help="dirs with tts_math500__<cell>.shard*.jsonl (each cell may live in a different one)")
    p.add_argument("--read_cell", default="rthics_seed1")
    p.add_argument("--prm_cell", default="prm_qwen25_math_7b")
    p.add_argument("--tau_read", type=float, default=0.854, help="80%% answered on the exploratory half")
    p.add_argument("--tau_prm", type=float, default=0.418, help="80%% answered on the exploratory half")
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    p.add_argument("--batch", type=int, nargs="+", default=[1, 16])
    p.add_argument("--problems_b1", type=int, default=16, help="confirmatory problems for B = 1 (this shard)")
    p.add_argument("--problems_bn", type=int, default=64, help="confirmatory problems for B > 1 (this shard)")
    p.add_argument("--samples", type=int, default=2, help="samples per problem for B > 1")
    p.add_argument("--n_replay", type=int, default=20)
    p.add_argument("--max_new_tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--col_step", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--no_graphs", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    from scripts.read_token_inline import marker_from_ckpt
    from scripts.read_token_probe import soft_kernels
    from scripts.read_token_store import load_model, load_read_token
    from src.eval.math_grade import grade

    dev = torch.device("cuda")
    a.out.mkdir(parents=True, exist_ok=True)
    pool = [r for path in a.trajectories for r in read_jsonl(path)
            if r.get("gradeable") and (r.get("solution") or "").strip()]
    seen, problems = set(), []
    for r in sorted(pool, key=lambda r: r["fork_id"]):
        if r["fork_id"] not in seen and not exploratory(r["fork_id"]):
            seen.add(r["fork_id"])
            problems.append({"fork_id": r["fork_id"], "problem": r["problem"], "gold": r["gold"]})
    problems = problems[a.shard_idx::a.num_shards]

    model, tok = load_model(a.model, dev)
    rt, ck = load_read_token(a.inline_ckpt, model.config.hidden_size, dev, model)
    mk = marker_from_ckpt(ck, dev)
    if not ck.get("chain") or mk is None:
        raise SystemExit("Stage J reads the chained token with a start marker (RTH-ICS)")
    pm, ptok, phead = load_prm(a.prm_model, dev)
    vocab = [tok.decode([i]) for i in range(len(tok))]
    pol = Engine(model, rt, mk)
    peng = Engine(pm)
    Bmax = max(a.batch)
    cap = bucket(a.max_new_tokens + 1024, a.col_step) + 1  # prompt + text + 2 columns per step, + junk
    pcache = RowCache(model, Bmax, cap, dev)
    # the PRM re-tokenizes each step and adds a separator per step: more headroom than the policy
    prm_cap = bucket(a.max_new_tokens + 2048, a.col_step) + 1
    prm_cache = RowCache(pm, Bmax, prm_cap, dev)
    res = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "args": {k: str(v) for k, v in vars(a).items()},
           "problems": len(problems), "policy_param_bytes": sum(q.numel() * q.element_size() for q in model.parameters()),
           "prm_param_bytes": sum(q.numel() * q.element_size() for q in pm.parameters()) + sum(
               q.numel() * q.element_size() for q in phead.parameters()),
           "kv_bytes_per_row_col": {"policy": pcache.nbytes() / (Bmax * cap), "prm": prm_cache.nbytes() / (Bmax * prm_cap)}}
    print(f"[live] {len(problems)} confirmatory problems in shard {a.shard_idx}/{a.num_shards}", flush=True)

    with soft_kernels("no_cudnn"):
        runs = {}
        for B in a.batch:
            cache = pcache.rows(0, B) if B < Bmax else pcache
            cache.n = [0] * B
            graphs = None
            live_args = a
            prm = (peng, prm_cache.rows(0, B) if B < Bmax else prm_cache, phead, ptok)
            prm[1].n = [0] * B
            if not a.no_graphs:
                lv = Live(pol, cache, tok, vocab, None, prm, a)
                graphs = {"plain": GraphRunner(pol, cache, lv._out_plain), "read": GraphRunner(pol, cache, lv._out_read)}
                t0 = time.perf_counter()
                for ncol in range(a.col_step, cap - 1, a.col_step):
                    graphs["plain"]._graph(1, ncol)
                    graphs["read"]._graph(1, ncol)
                    graphs["read"]._graph(3, ncol)
                res[f"b{B}_capture_sec"] = time.perf_counter() - t0
                print(f"[b{B}] captured {len(graphs['plain'].graphs) + len(graphs['read'].graphs)} graphs in "
                      f"{time.perf_counter() - t0:.0f}s", flush=True)
            live = Live(pol, cache, tok, vocab, graphs, prm, live_args)

            if B == 1 and a.n_replay:
                # teacher-forced pool trajectories through the live path vs the pool's scores
                ref = {}
                for cell in (a.read_cell, a.prm_cell):
                    for d in a.pool_scores:
                        for f in sorted(d.glob(f"tts_math500__{cell}.shard*.jsonl")):
                            for r in read_jsonl(f):
                                ref.setdefault(cell, {})[r["traj_uid"]] = r["scores"]
                    if not ref.get(cell):
                        raise SystemExit(f"no pool scores for {cell} under {a.pool_scores}")
                rep_rows = [r for r in pool if not exploratory(r["fork_id"]) and r["traj_uid"] in ref.get(a.read_cell, {})
                            and r["traj_uid"] in ref.get(a.prm_cell, {})]
                rep_rows = rep_rows[a.shard_idx::a.num_shards][:a.n_replay]
                fj = forced_jobs(rep_rows, tok, ptok, tok.convert_tokens_to_ids("<|im_end|>"))
                rep = {}
                for arm, cell in (("read", a.read_cell), ("prm", a.prm_cell)):
                    o, _ = live.run(arm, fj, 1, float("inf"))
                    x, y, bad = [], [], 0
                    for j, r in zip(fj, o):
                        want = ref[cell][j["traj_uid"]]
                        if len(r["scores"]) != len(want):
                            bad += 1
                            continue
                        x += r["scores"]
                        y += want
                    rep[arm] = {**diff_summary(x, y), "trajectories": len(fj), "step_count_mismatch": bad}
                    print(f"[replay] {arm} vs pool {cell}: {rep[arm]}", flush=True)
                res["replay_vs_pool"] = rep

            nprob = a.problems_b1 if B == 1 else a.problems_bn
            jobs = make_jobs(problems[:nprob], 1 if B == 1 else a.samples, tok, ptok, a.seed)
            # warm the eager paths (prefill, PRM) on two jobs before timing
            live.run("prm", jobs[:2], B, float("inf"))
            for arm in a.arms:
                tau = a.tau_read if arm == "read_stop" else a.tau_prm if arm == "prm_stop" else float("inf")
                out, st = live.run(arm, jobs, B, tau)
                for r in out:
                    if r["end"] != "abstain":
                        g = grade(r["solution"], r["gold"])
                        r["correct"], r["gradeable"], r["pred"] = g["correct"], g["gradeable"], g["pred"]
                ans = [r for r in out if r["end"] != "abstain"]
                st["answered"] = len(ans) / len(out)
                st["acc_answered"] = float(np.mean([r["correct"] for r in ans])) if ans else None
                st["decision_lat_median"] = float(np.median([x for r in out for x in r["decision_lat"]])) \
                    if any(r["decision_lat"] for r in out) else None
                res[f"b{B}_{arm}"] = st
                runs[(B, arm)] = out
                with open(a.out / f"b{B}_{arm}.jsonl", "w") as f:
                    for r in out:
                        f.write(json.dumps(r) + "\n")
                print(f"[b{B}] {arm}: {json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in st.items()})}",
                      flush=True)

            # live vs offline on the no-stop arms' texts
            from scripts.onpolicy.score_traces_with_read_token import score_inline
            pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
            chk = {}
            if (B, "read") in runs:
                rws = [dict(r, traj_uid=str(r["jid"])) for r in runs[(B, "read")] if r["solution"].strip()]
                off, _ = score_inline(model, tok, rt, rws, pad, dev, 16384, "no_cudnn", mk, chain=True)
                x, y, bad = [], [], 0
                for r, o in zip(rws, off):
                    if len(r["scores"]) != len(o):
                        bad += 1
                        continue
                    x += r["scores"]
                    y += list(o)
                chk["read"] = {**diff_summary(x, y), "trajectories": len(rws), "step_count_mismatch": bad}
            if (B, "prm") in runs:
                rws = [r for r in runs[(B, "prm")] if r["solution"].strip()]
                off = offline_prm(prm, rws, a.col_step)
                x, y, bad = [], [], 0
                for r, o in zip(rws, off):
                    if len(r["scores"]) != len(o):
                        bad += 1
                        continue
                    x += r["scores"]
                    y += o
                chk["prm"] = {**diff_summary(x, y), "trajectories": len(rws), "step_count_mismatch": bad}
            res[f"b{B}_live_vs_offline"] = chk
            print(f"[b{B}] live vs offline: {chk}", flush=True)
            (a.out / "summary.json").write_text(json.dumps(res, indent=1))
    (a.out / "summary.json").write_text(json.dumps(res, indent=1))
    print(f"[out] {a.out}", flush=True)


if __name__ == "__main__":
    main()
