#!/usr/bin/env python3
"""Score saved trajectories from the states the sampler itself computed.

`score_traces_with_cell.py` rereads every step under the verifier template
("Problem: ... Previous reasoning: ... Current step:"), which is the context
the cells were fitted on. That costs one extra backbone pass per step, so the
verifier is not free: on tts_roster_v1 it took about six times the wall-clock of
a 7B PRM (jobs 465706 and 487162).

This is the version whose cost case is real. It rebuilds the exact context the
sampler had (`context_from_row`, the few-shot prompt byte for byte), runs one
teacher-forced pass over prompt + generated solution, and reads each step's
token states at the cell's layer. Under causal attention those are the states
the sampler computed while writing the solution, so in deployment the only
added cost is the head. The pass here is a reconstruction, not a cost the
method has to pay; the manifest times the head separately so both can be
reported.

**The head was not fitted on these states.** It was trained on verifier-template
states, and a step's states depend on its left context. Scores here therefore
measure transfer to the generation context, not the head's best case.

Step spans come from `src/onpolicy/spans.py`, the construction the
token-confidence encoder already validated on this pool (99.7 to 99.9% of
traces tile). No trace is dropped: a missing score becomes -inf quality in the
frontier and would lose every comparison, which is a bias, not a gap. A trace
whose spans do not tile is scored on its clipped spans, an empty span reads the
single token at its boundary, and both are flagged `span_ok: false`. The job
aborts if flagged traces exceed `1 - --min_span_coverage`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.encode_prm800k_hidden_states import git_commit, read_jsonl, write_jsonl  # noqa: E402
from scripts.generate_onpolicy_steps import split_into_steps  # noqa: E402
from scripts.onpolicy.online_bon import Checker, cell_stats  # noqa: E402
from src.onpolicy.prompts import context_from_row  # noqa: E402
from src.onpolicy.spans import step_token_spans, verify_spans_cover  # noqa: E402


def gather_step_states(h: torch.Tensor, n_prompt: int,
                       spans: list[tuple[int, int]], t_max: int) -> list[torch.Tensor]:
    """Each step's token states from one full-sequence pass.

    `h` is (T, d) over prompt + generated tokens; spans are half-open in
    generated coordinates, so step k lives at [n_prompt + a, n_prompt + b).
    Rounded through float16 because the head was fitted on a float16 store
    (see Checker.score_steps). Truncated to the first `t_max` tokens, as
    Checker does.
    """
    out = []
    for a, b in spans:
        if b <= a:  # empty span: read its boundary token, as Checker does
            a, b = max(0, a - 1), max(1, a)
        out.append(h[n_prompt + a: n_prompt + b].to(torch.float16).float()[:t_max])
    return out


@torch.no_grad()
def head_scores(checker: Checker, seqs: list[torch.Tensor]) -> list[float]:
    """P(step is wrong) for every step at once, through the cell's own head."""
    seqs = [checker._rescale(x) for x in seqs]
    t = max(x.shape[0] for x in seqs)
    batch = torch.zeros(len(seqs), t, checker.dim, device=checker.device)
    mask = torch.zeros(len(seqs), t, dtype=torch.bool, device=checker.device)
    for i, x in enumerate(seqs):
        batch[i, : x.shape[0]] = x
        mask[i, : x.shape[0]] = True
    logits = checker.model(batch, mask)
    if logits.ndim > 1:
        logits = logits.squeeze(-1)
    return torch.sigmoid(logits).float().cpu().tolist()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trajectories", type=Path, nargs="+", required=True)
    p.add_argument("--cell_dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--model_dtype", choices=["float16", "bfloat16", "float32"],
                   default="bfloat16")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--prm_store", type=Path, default=None)
    p.add_argument("--stats", type=Path, default=None)
    p.add_argument("--stats_cache", type=Path, default=None)
    p.add_argument("--train_stem", type=str, default=None)
    p.add_argument("--assume_rescale", choices=["none", "zscore", "whiten"],
                   default="zscore")
    p.add_argument("--min_span_coverage", type=float, default=0.95)
    p.add_argument("--max_traces", type=int, default=0)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--force", action="store_true")
    a = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    if a.out.exists() and not a.force:
        sys.exit(f"[gen] refusing to overwrite {a.out}. Pass --force.")
    if not (a.prm_store or a.stats):
        sys.exit("[gen] pass --prm_store or --stats: the head was fitted on rescaled states.")

    rows = [r for path in a.trajectories for r in read_jsonl(path)]
    rows = [r for r in rows if r.get("gradeable") and (r.get("solution") or "").strip()]
    if a.max_traces:
        rows = rows[: a.max_traces]
    mine = rows[a.shard_idx:: a.num_shards]
    print(f"[gen] shard {a.shard_idx}/{a.num_shards}: {len(mine)} of {len(rows)} traces, "
          f"cell={a.cell_dir.name}", flush=True)

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}[a.model_dtype]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model_name_or_path,
                                        local_files_only=a.local_files_only)
    backbone = AutoModelForCausalLM.from_pretrained(
        a.model_name_or_path, torch_dtype=dtype,
        local_files_only=a.local_files_only).to(device).eval()
    if a.stats:
        stats = json.loads(a.stats.read_text())
    else:
        res = json.loads((a.cell_dir / "results.json").read_text())
        stats = cell_stats(res, a.prm_store, None, a.stats_cache, {},
                           train_stem_override=a.train_stem,
                           assume_rescale=a.assume_rescale)
    checker = Checker(a.cell_dir, backbone, tok, a.layer, stats, device)

    def n_tokens(s: str) -> int:
        return len(tok(s, add_special_tokens=False)["input_ids"])

    out, bad_spans = [], 0
    t_backbone = t_head = 0.0
    t0 = time.perf_counter()
    for i, r in enumerate(mine):
        steps = split_into_steps(r["solution"])
        if not steps:
            continue
        # Same construction as scripts/onpolicy/encode_token_confidence.py.
        prompt = context_from_row(r)
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        s_ids = tok(r["solution"], add_special_tokens=False)["input_ids"]
        spans = step_token_spans(prompt, steps, n_tokens)
        spans = [(x, min(y, len(s_ids))) for x, y in spans]
        span_ok = verify_spans_cover(spans, len(s_ids), tol=3) and all(y > x for x, y in spans)
        bad_spans += int(not span_ok)

        if device == "cuda":
            torch.cuda.synchronize()
        tb = time.perf_counter()
        with torch.no_grad():
            ids = torch.tensor([p_ids + s_ids], device=device)
            h = backbone(input_ids=ids, output_hidden_states=True).hidden_states[a.layer][0]
        if device == "cuda":
            torch.cuda.synchronize()
        th = time.perf_counter()
        scores = head_scores(checker, gather_step_states(h, len(p_ids), spans, checker.t_max))
        if device == "cuda":
            torch.cuda.synchronize()
        t_backbone += th - tb
        t_head += time.perf_counter() - th

        out.append({"traj_uid": r["traj_uid"], "problem_id": r.get("fork_id"),
                    "dataset": r.get("dataset"), "split": r.get("split"),
                    "correct": bool(r["correct"]), "n_steps": len(steps),
                    "cell": f"{a.cell_dir.name}__gen", "span_ok": span_ok,
                    "scores": scores})
        if (i + 1) % 200 == 0 or i + 1 == len(mine):
            print(f"[gen] {i+1}/{len(mine)}  ({time.perf_counter()-t0:.0f}s)  "
                  f"bad_spans={bad_spans}", flush=True)

    coverage = 1 - bad_spans / max(1, len(mine))
    print(f"[gen] span coverage {coverage:.4f} ({bad_spans} traces flagged); "
          f"backbone {t_backbone:.1f}s, head {t_head:.1f}s", flush=True)
    if coverage < a.min_span_coverage:
        sys.exit(f"[gen] FAILED: span coverage {coverage:.4f} < {a.min_span_coverage}. "
                 "Nothing written.")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(a.out, out)
    (a.out.parent / f"{a.out.stem}_manifest.json").write_text(json.dumps({
        "cell": str(a.cell_dir), "layer": a.layer, "context": "generation",
        "n_traces": len(out), "flagged_bad_spans": bad_spans,
        "seconds_backbone": t_backbone, "seconds_head": t_head,
        "shard_idx": a.shard_idx, "num_shards": a.num_shards,
        "model": a.model_name_or_path, "assume_rescale": a.assume_rescale,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": git_commit()}, indent=1))
    print(f"[gen] wrote {a.out}  traces={len(out)}")


if __name__ == "__main__":
    main()
