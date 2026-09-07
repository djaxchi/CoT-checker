#!/usr/bin/env python3
"""Step-level guided decoding: let the checker choose the next step as it is written.

Everything measured so far reranks *finished* solutions. This is the other mode
ReProbe defines, and the one worth wanting:

    Q_online(r_t) = 1 - U(r_t | r_<t, x)

At each position the policy proposes N candidate next steps, the head scores
each, and the best one is kept and extended. The paper uses N=5 at temperature
1.5; both are flags here and default to those values.

WHY THERE IS A RANDOM ARM. Branching five ways at every step and keeping any one
of them is already a different sampler from writing one step and moving on: it
raises the effective temperature, then filters. So "guided beats plain sampling"
does not show the head did anything, because the search alone could produce it.
The comparison that isolates the head is guided against *random choice from the
same candidate pool*, and that arm is not optional here.

Three arms, sharing problems and seeds so the comparison is paired per problem:

    plain    one step sampled per position, no branching   (the base policy)
    random   N candidates per position, uniform choice     (search, no checker)
    guided   N candidates per position, lowest U wins      (search + checker)

plain -> random measures the search. random -> guided measures the checker. Only
the second is a claim about the verifier.

CORRECTNESS. The head was trained on step spans encoded under `verifier_prefix`
at one layer, so scoring online has to reproduce that exactly or the numbers mean
nothing. `--verify_against` re-scores stored trajectories through this file's own
scoring path and compares against the offline scores that cell already wrote; it
must agree to within tolerance before any generation run is trusted.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.harness.learners import build_learner  # noqa: E402
from scripts.onpolicy.score_cells_on_split import cell_stats  # noqa: E402
from src.onpolicy.prompts import generation_prefix, verifier_prefix  # noqa: E402
from src.eval.math_grade import grade  # noqa: E402

ARMS = ("plain", "random", "guided")
STEP_SEP = "\n\n"


# ---------------------------------------------------------------------------
# checker
# ---------------------------------------------------------------------------

class Checker:
    """A trained cell, rebuilt and applied to freshly generated step text.

    Holds the backbone too, since scoring needs hidden states of the step's
    tokens under the verifier template, not the generation context.
    """

    def __init__(self, cell_dir: Path, backbone, tokenizer, layer: int,
                 stats: dict | None, device: str, t_max: int = 512):
        res = json.loads((cell_dir / "results.json").read_text())
        self.rep = res["rep"]
        if self.rep != "step_tokens":
            raise ValueError(
                f"online decoding needs a per-step sequence head; cell rep is {self.rep!r}. "
                "Pooled readouts (last_token, step_mean, ...) score a step too, but this "
                "script has only been verified for step_tokens.")
        self.learner = res["learner"]
        self.dim = int(res["dim"])
        self.t_max = int(res.get("protocol", {}).get("t_max", t_max))
        self.model = build_learner(self.learner, self.dim, t_max=self.t_max)
        self.model.load_state_dict(torch.load(cell_dir / "model.pt", map_location=device))
        self.model.to(device).eval()
        self.backbone = backbone
        self.tok = tokenizer
        self.layer = layer
        self.stats = stats
        self.device = device

    def _rescale(self, x: torch.Tensor) -> torch.Tensor:
        if not self.stats:
            return x
        mu = torch.as_tensor(self.stats["mean"], dtype=x.dtype, device=x.device)
        sd = torch.as_tensor(self.stats["std"], dtype=x.dtype, device=x.device)
        return (x - mu) / sd

    @torch.no_grad()
    def score_steps(self, problem: str, prior_steps: list[str],
                    candidates: list[str]) -> list[float]:
        """P(this step is wrong) for each candidate continuation of the same prefix.

        Tokenisation mirrors scripts/encode_processbench_token_store.py exactly:
        the prefix is tokenised WITH special tokens, the step WITHOUT, and the id
        lists are concatenated rather than the strings. Tokenising the joined
        string instead merges tokens across the prefix/step boundary and drops
        the BOS, which moved scores by ~0.011 and failed the verification gate.
        """
        ctx = verifier_prefix(problem, STEP_SEP.join(prior_steps))
        prefix_ids = self.tok(ctx, add_special_tokens=True,
                              truncation=False)["input_ids"]
        n_ctx = len(prefix_ids)

        rows = []
        for c in candidates:
            step_ids = self.tok(c, add_special_tokens=False,
                                truncation=False)["input_ids"]
            if not step_ids:                       # empty candidate: score its boundary
                step_ids = prefix_ids[-1:]
            rows.append(prefix_ids + step_ids)

        # Right padding, so a step span is always [n_ctx, n_ctx + len(step_ids)).
        t_in = max(len(r) for r in rows)
        pad = self.tok.pad_token_id or self.tok.eos_token_id
        ids = torch.full((len(rows), t_in), pad, dtype=torch.long, device=self.device)
        att = torch.zeros((len(rows), t_in), dtype=torch.long, device=self.device)
        for i, r in enumerate(rows):
            ids[i, : len(r)] = torch.tensor(r, device=self.device)
            att[i, : len(r)] = 1
        h = self.backbone(input_ids=ids, attention_mask=att,
                          output_hidden_states=True).hidden_states[self.layer]

        seqs = []
        for i, r in enumerate(rows):
            # The store holds float16 (encode_processbench_token_store.py opens
            # h.npy as np.float16 and casts with .to(torch.float16)), so the head
            # was fitted on activations rounded to that precision. Keeping full
            # precision here is more accurate but not what the head saw, and it
            # was the entire 0.011 discrepancy: the tokenisation was already
            # identical, which is why fixing that moved the number not at all.
            span = h[i, n_ctx: len(r)].to(torch.float16).float()
            seqs.append(self._rescale(span)[: self.t_max])

        t = max(x.shape[0] for x in seqs)
        batch = torch.zeros(len(seqs), t, self.dim, device=self.device)
        mask = torch.zeros(len(seqs), t, dtype=torch.bool, device=self.device)
        for i, x in enumerate(seqs):
            batch[i, : x.shape[0]] = x
            mask[i, : x.shape[0]] = True
        logits = self.model(batch, mask)
        if logits.ndim > 1:
            logits = logits.squeeze(-1)
        return torch.sigmoid(logits).float().cpu().tolist()


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------

def sample_candidates(backbone, tok, problem: str, prior_steps: list[str],
                      n: int, temperature: float, top_p: float,
                      max_new_tokens: int, device: str) -> tuple[list[str], int]:
    """n candidate next steps, each cut at the first blank line.

    The step splitter downstream segments on blank lines, so a candidate must be
    exactly one step; generating past the boundary and truncating keeps the
    sampler's own distribution rather than forcing an early stop token.
    """
    ctx = generation_prefix(problem, STEP_SEP.join(prior_steps))
    # add_special_tokens defaults to True, matching what
    # scripts/generate_onpolicy_steps.py sends to model.generate. Dropping the
    # BOS here would sample from a different distribution than the policy whose
    # behaviour every baseline in this study describes.
    enc = tok(ctx, return_tensors="pt").to(device)
    with torch.no_grad():
        out = backbone.generate(
            **enc, do_sample=True, temperature=temperature, top_p=top_p,
            num_return_sequences=n, max_new_tokens=max_new_tokens,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    n_ctx = enc["input_ids"].shape[1]
    cands, generated = [], 0
    for row in out:
        new = row[n_ctx:]
        # Every sampled token is paid for, including the candidates thrown away.
        # Counting only the kept branch would understate branching by ~Nx and
        # make guided decoding look cheap when it is the opposite.
        generated += int((new != (tok.pad_token_id or tok.eos_token_id)).sum())
        text = tok.decode(new, skip_special_tokens=True)
        cands.append(text.split(STEP_SEP)[0].strip())
    return cands, generated


def is_final(step: str) -> bool:
    return "\\boxed{" in step


def rollout(arm: str, problem: str, gold: str, backbone, tok, checker,
            args, rng: random.Random) -> dict:
    steps: list[str] = []
    chosen_scores: list[float] = []
    pool_scores: list[list[float]] = []
    n = 1 if arm == "plain" else args.n_candidates
    gen_tokens = 0
    for _ in range(args.max_steps):
        cands, used = sample_candidates(backbone, tok, problem, steps, n,
                                        args.temperature, args.top_p,
                                        args.max_new_tokens, args.device)
        gen_tokens += used
        cands = [c for c in cands if c] or [""]
        if arm == "guided":
            u = checker.score_steps(problem, steps, cands)
            k = int(np.argmin(u))
            pool_scores.append([round(x, 5) for x in u])
            chosen_scores.append(u[k])
        elif arm == "random":
            k = rng.randrange(len(cands))
        else:
            k = 0
        steps.append(cands[k])
        if is_final(cands[k]) or not cands[k]:
            break
    solution = STEP_SEP.join(steps)
    ok = bool(grade(solution, gold))
    return {"arm": arm, "steps": steps, "n_steps": len(steps),
            "correct": ok, "chosen_scores": chosen_scores,
            "pool_scores": pool_scores,
            # generation tokens actually sampled (discarded branches included),
            # and how many times the head ran, so cost can be reported per arm
            # and accuracy compared at a matched token budget
            "gen_tokens": gen_tokens,
            "checker_calls": len(pool_scores),
            "scored_candidates": sum(len(x) for x in pool_scores)}


# ---------------------------------------------------------------------------
# verification: does online scoring reproduce the offline scores?
# ---------------------------------------------------------------------------

def verify(checker: Checker, traces_path: Path, offline_scores: Path,
           n_traces: int, tol: float, max_tol: float) -> int:
    """Re-score stored steps through the live path and compare to the cell's own scores.

    Reports the whole distribution, not just the maximum. A systematic mismatch
    (wrong span, wrong layer, wrong rescaling) shifts every step and shows up in
    the median; float16 round-off shows up as a small median with a heavier tail.
    Telling those apart from a single max would be guesswork, and the response to
    each is different.
    """
    traces = {}
    for line in open(traces_path):
        r = json.loads(line)
        traces[r["id"]] = r
    diffs, n_traces_done = [], 0
    for line in open(offline_scores):
        r = json.loads(line)
        t = traces.get(r["id"])
        if t is None:
            continue
        steps = t["steps"] if isinstance(t["steps"], list) else eval(t["steps"])
        if len(steps) != len(r["scores"]):
            continue
        for i in range(min(len(steps), 6)):
            got = checker.score_steps(t["problem"], steps[:i], [steps[i]])[0]
            diffs.append(abs(got - r["scores"][i]))
        n_traces_done += 1
        if n_traces_done >= n_traces:
            break

    # Noise floor: the same step scored under two batch shapes. The encoder read
    # steps in padded batches; this scores them singly, and bf16 matmuls are not
    # invariant to batch composition. Any disagreement below this floor is not a
    # bug and cannot be fixed by changing the scoring code, so the gate tolerance
    # has to be set above it or it can never pass.
    floor = []
    for line in open(offline_scores):
        r = json.loads(line)
        t = traces.get(r["id"])
        if t is None:
            continue
        steps = t["steps"] if isinstance(t["steps"], list) else eval(t["steps"])
        if len(steps) != len(r["scores"]):
            continue
        for i in range(min(len(steps), 4)):
            alone = checker.score_steps(t["problem"], steps[:i], [steps[i]])[0]
            # Same step, but in a batch of 8 with widely varying lengths, which
            # is what the encoder actually did (BATCH_SIZE=8). A 3-element probe
            # understates the floor: the perturbation grows with how much padding
            # and length spread the batch carries.
            neigh = [steps[i]] + [steps[i] + " " + "x " * k
                                  for k in (1, 5, 20, 60, 120, 250, 400)]
            padded = checker.score_steps(t["problem"], steps[:i], neigh)[0]
            floor.append(abs(alone - padded))
        if len(floor) >= 40:
            break
    f = np.asarray(floor) if floor else np.zeros(1)
    print(f"[verify] batch-shape noise floor over {f.size} steps: "
          f"median {np.median(f):.6f}  p95 {np.percentile(f, 95):.6f}  max {f.max():.6f}")

    d = np.asarray(diffs)
    print(f"[verify] {d.size} steps over {n_traces_done} traces\n"
          f"[verify]   median {np.median(d):.6f}   p95 {np.percentile(d, 95):.6f}   "
          f"max {d.max():.6f}")

    # Two thresholds, because the two failure modes look different and only one
    # of them is a bug.
    #
    # A wrong span, layer or rescaling shifts EVERY step, so it shows in the
    # median. That is the real check, and it is tight.
    #
    # The tail is numerical. Changing nothing but batch composition, with
    # identical inputs and weights, already moves scores by up to ~0.0064 (the
    # floor printed above): a single sequence and a batch take different kernel
    # paths in bf16. A max tolerance below that floor cannot be met by any
    # implementation, so gating on the max alone meant gating on noise.
    ok = True
    if np.median(d) > tol:
        print(f"[verify] FAIL: median {np.median(d):.6f} exceeds {tol}. A shift "
              "this uniform is systematic, not numerical: check the span "
              "boundary, the hidden-state layer index, and the rescaling stats.")
        ok = False
    if d.max() > max_tol:
        print(f"[verify] FAIL: max {d.max():.6f} exceeds {max_tol}, which is well "
              "above the measured batch-shape floor, so this is not numerical.")
        ok = False
    if not ok:
        return 1
    print(f"[verify] OK: median within {tol} (no systematic offset) and max within "
          f"{max_tol}. The residual tail is batch-shape numerics, measured above.")
    return 0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cell_dir", required=True, type=Path)
    p.add_argument("--traces", required=True, type=Path,
                   help="judge traces jsonl: id, problem, gold, problem_id, steps")
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--layer", type=int, default=35)
    p.add_argument("--stats", type=Path, default=None,
                   help="rescaling stats json, if already fit; otherwise pass "
                        "--prm_store and they are refit exactly")
    p.add_argument("--prm_store", type=Path, default=None,
                   help="the store the cell trained on. Rescaling statistics are "
                        "not saved with a cell, so they are refit from here by the "
                        "same fingerprint-checked path scoring uses; a mismatch is "
                        "refused rather than silently rescaling by numbers the cell "
                        "never saw.")
    p.add_argument("--stats_cache", type=Path, default=None)
    p.add_argument("--train_stem", default=None)
    p.add_argument("--assume_rescale", default=None)
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    p.add_argument("--n_candidates", type=int, default=5)
    p.add_argument("--temperature", type=float, default=1.5)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_steps", type=int, default=16)
    p.add_argument("--max_new_tokens", type=int, default=160)
    p.add_argument("--max_problems", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--model_dtype", default="bfloat16")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--verify_against", type=Path, default=None,
                   help="pb_step_scores_*.jsonl from the same cell; verify and exit")
    p.add_argument("--verify_traces", type=int, default=20)
    p.add_argument("--verify_tol", type=float, default=2e-3,
                   help="MEDIAN tolerance: catches a systematic offset, which is "
                        "the failure that would invalidate results")
    p.add_argument("--verify_max_tol", type=float, default=0.02,
                   help="MAX tolerance: set above the measured batch-shape noise "
                        "floor (~0.0064), since no implementation can beat it")
    a = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[a.model_dtype]
    tok = AutoTokenizer.from_pretrained(a.model_name_or_path,
                                        local_files_only=a.local_files_only)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    backbone = AutoModelForCausalLM.from_pretrained(
        a.model_name_or_path, torch_dtype=dtype, local_files_only=a.local_files_only,
    ).to(a.device).eval()

    if a.stats:
        stats = json.loads(a.stats.read_text())
    elif a.prm_store:
        res = json.loads((a.cell_dir / "results.json").read_text())
        stats = cell_stats(res, a.prm_store, None, a.stats_cache, {},
                           train_stem_override=a.train_stem,
                           assume_rescale=a.assume_rescale)
    else:
        raise SystemExit(
            "pass --prm_store (to refit the cell's rescaling statistics) or "
            "--stats. Scoring without them applies the head to unscaled states "
            "and the numbers would be wrong without looking wrong.")
    checker = Checker(a.cell_dir, backbone, tok, a.layer, stats, a.device)

    if a.verify_against:
        sys.exit(verify(checker, a.traces, a.verify_against,
                        a.verify_traces, a.verify_tol, a.verify_max_tol))

    problems: dict[str, dict] = {}
    for line in open(a.traces):
        r = json.loads(line)
        problems.setdefault(r["problem_id"], r)
    keys = sorted(problems)[: a.max_problems]
    # Shard round-robin, not in contiguous blocks: the problem ids sort into
    # dataset order, so a block split would hand one GPU all the hard subset and
    # make per-shard timings useless for estimating the whole run.
    keys = keys[a.shard_idx :: a.num_shards]
    print(f"{len(keys)} problems, arms {a.arms}, N={a.n_candidates}, T={a.temperature}")

    a.out.parent.mkdir(parents=True, exist_ok=True) if a.out else None
    fh = open(a.out, "a") if a.out else None
    done = set()
    if a.out and a.out.exists():
        for line in open(a.out):
            try:
                r = json.loads(line)
                done.add((r["problem_id"], r["arm"]))
            except json.JSONDecodeError:
                pass
        print(f"resuming: {len(done)} rollouts already written")

    t0 = time.time()
    for i, pid in enumerate(keys):
        rec = problems[pid]
        for arm in a.arms:
            if (pid, arm) in done:
                continue
            rng = random.Random(f"{a.seed}:{pid}:{arm}")
            torch.manual_seed(abs(hash((a.seed, pid, arm))) % (2**31))
            r = rollout(arm, rec["problem"], rec["gold"], backbone, tok,
                        checker, a, rng)
            r["problem_id"] = pid
            if fh:
                fh.write(json.dumps(r) + "\n")
                fh.flush()
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(keys)} problems, {time.time()-t0:.0f}s")
    if fh:
        fh.close()
    print(f"[online] done in {time.time()-t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
