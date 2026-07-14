"""transition_operator_v0 Stage 0: gates 1, 2, 4, 5 (docs/transition_operator_v0_plan.md).

Gate 1  suffix selection      gold rank-1 rate per candidate suffix on complete golden
                              trajectories; best suffix must reach >= 0.60
Gate 2  directional sanity    paired d_margin(correct) vs d_margin(wrong) on forks;
                              Wilcoxon p < 0.01 required
Gate 4  tokenization sanity   piecewise sep-joined ids round-trip + merge-rate report
Gate 5  boundary oracle       patch the TRUE post boundary state into the pre context
                              at each patch layer; KL recovery for Target A, belief
                              recovery for Target B; identity check of the patch path

Tokenization rule (consistency requirement): pieces (question, steps) are tokenized
separately and joined with the explicit separator id 198, and every context ends with
an appended 198. This guarantees (a) the readout token is identical everywhere and
(b) pre-context ids are a strict prefix of branch ids, so boundary states are shared
between passes. Gate 4 reports how often this differs from naive string tokenization.

Smoke (tiny model, CPU/MPS):
  python scripts/transition_operator/to_stage0.py --model_name_or_path Qwen/Qwen2.5-0.5B \
      --layers 14 18 20 --n_gate1 5 --n_gate2 5 --n_oracle 5 --n_oracle_b 3
Full (TamIA, H100):
  python scripts/transition_operator/to_stage0.py   # defaults: 7B, layers 20 24 26
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.analysis.transition_operator import (  # noqa: E402
    SEP_TOKEN_ID,
    belief_from_scores,
    build_candidates,
    candidate_mean_logprobs,
    forward_with_boundary_patch,
    gold_margin,
    recovery_from_logits,
    sep_join_ids,
    stable_seed,
)

SUFFIXES = ["\nThe answer is", "\n# Answer\n\n", "\nSo the final answer is"]


def cand_token_ids(tok, suffix: str, answers: list[str]) -> list[list[int]]:
    lead = "" if suffix[-1].isspace() else " "
    return [tok(lead + a, add_special_tokens=False)["input_ids"] for a in answers]


def load_jsonl(path: Path, n: int) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if len(rows) >= n:
                break
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", type=Path, default=Path("runs/transition_operator"))
    ap.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-7B")
    ap.add_argument("--layers", type=int, nargs="+", default=[20, 24, 26],
                    help="patch layers (hidden_states indices); final layer is "
                         "added automatically as a diagnostic, never as an arm")
    ap.add_argument("--n_gate1", type=int, default=300)
    ap.add_argument("--n_gate2", type=int, default=500)
    ap.add_argument("--n_oracle", type=int, default=500)
    ap.add_argument("--n_oracle_b", type=int, default=200)
    ap.add_argument("--k_candidates", type=int, default=8)
    ap.add_argument("--max_seq_len", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--local_files_only", action="store_true")
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float32 if device == "cpu" else torch.float16

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                        local_files_only=args.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, torch_dtype=dtype,
        local_files_only=args.local_files_only).to(device).eval()
    n_layers = model.config.num_hidden_layers
    print(f"[stage0] {args.model_name_or_path} on {device} ({dtype}), "
          f"{n_layers} blocks, patch layers {args.layers}", flush=True)

    forks = load_jsonl(args.run_dir / "forks.jsonl",
                       max(args.n_gate2, args.n_oracle))
    golden = load_jsonl(args.run_dir / "golden.jsonl", args.n_gate1)
    corpus_pool = sorted({fk["gt_answer"] for fk in forks}
                         | {tr["gt_answer"] for tr in golden})
    results: dict = {"created": datetime.now(timezone.utc).isoformat(),
                     "model": args.model_name_or_path, "device": device,
                     "layers": args.layers, "n_blocks": n_layers,
                     "seed": args.seed}

    def cands_for(gold: str, pre_gen=None, wrong_finals=(), seed_key: str = "") -> list[str]:
        return build_candidates(gold, pre_gen, wrong_finals, corpus_pool,
                                k=args.k_candidates,
                                seed=stable_seed(seed_key, args.seed))

    def fits(ids: list[int]) -> bool:
        return len(ids) <= args.max_seq_len - 48  # headroom for suffix + candidate

    # ---- gate 4: tokenization sanity ------------------------------------------
    n_check, n_merge_diff, n_roundtrip_bad = 0, 0, 0
    for fk in forks[:200]:
        pieces = [fk["question"], *fk["prefix_steps"]]
        ids = sep_join_ids(tok, pieces)
        n_check += 1
        if tok.decode(ids) != "\n".join(pieces) + "\n":
            n_roundtrip_bad += 1
        naive = tok("\n".join(pieces) + "\n", add_special_tokens=False)["input_ids"]
        if naive != ids:
            n_merge_diff += 1
    results["gate4"] = {"n_checked": n_check, "roundtrip_failures": n_roundtrip_bad,
                        "merge_rate_vs_string_tokenization": n_merge_diff / max(n_check, 1)}
    print(f"[gate4] roundtrip failures {n_roundtrip_bad}/{n_check}, "
          f"merge rate vs naive {n_merge_diff / max(n_check, 1):.3f}", flush=True)

    # ---- gate 1: suffix selection ----------------------------------------------
    suffix_ids = {s: tok(s, add_special_tokens=False)["input_ids"] for s in SUFFIXES}
    rank1 = {s: [] for s in SUFFIXES}
    for i, tr in enumerate(golden):
        ctx = sep_join_ids(tok, [tr["question"], *tr["steps"]])
        if not fits(ctx):
            continue
        cands = cands_for(tr["gt_answer"], seed_key=tr["trajectory_id"])
        for s in SUFFIXES:
            scores = candidate_mean_logprobs(
                model, ctx + suffix_ids[s], cand_token_ids(tok, s, cands),
                pad_id=tok.pad_token_id or 0, device=device)
            rank1[s].append(int(np.argmax(scores) == 0))
        if (i + 1) % 50 == 0:
            print(f"[gate1] {i + 1}/{len(golden)}", flush=True)
    gate1 = {s: float(np.mean(v)) if v else float("nan") for s, v in rank1.items()}
    best_suffix = max(gate1, key=lambda s: -1 if np.isnan(gate1[s]) else gate1[s])
    results["gate1"] = {"rank1_rates": gate1, "best_suffix": best_suffix,
                        "n_evaluated": len(rank1[best_suffix]),
                        "pass": bool(gate1[best_suffix] >= 0.60)}
    print(f"[gate1] rank-1 rates {gate1} -> best {best_suffix!r} "
          f"(pass={results['gate1']['pass']})", flush=True)

    # ---- gate 2: directional sanity --------------------------------------------
    sfx = suffix_ids[best_suffix]
    d_corr, d_wrong = [], []
    for i, fk in enumerate(forks[:args.n_gate2]):
        pre = sep_join_ids(tok, [fk["question"], *fk["prefix_steps"]])
        post_c = pre + tok(fk["correct"], add_special_tokens=False)["input_ids"] + [SEP_TOKEN_ID]
        post_w = pre + tok(fk["wrong"], add_special_tokens=False)["input_ids"] + [SEP_TOKEN_ID]
        if not (fits(post_c) and fits(post_w)):
            continue
        cands = cands_for(fk["gt_answer"], fk["pre_generated_answer"],
                          tuple(fk["wrong_finals"]), seed_key=fk["fork_id"])
        cids = cand_token_ids(tok, best_suffix, cands)
        pad = tok.pad_token_id or 0
        m = {}
        for name, ctx in (("pre", pre), ("post_c", post_c), ("post_w", post_w)):
            m[name] = gold_margin(candidate_mean_logprobs(
                model, ctx + sfx, cids, pad_id=pad, device=device))
        d_corr.append(m["post_c"] - m["pre"])
        d_wrong.append(m["post_w"] - m["pre"])
        if (i + 1) % 50 == 0:
            print(f"[gate2] {i + 1}/{args.n_gate2}", flush=True)
    from scipy.stats import wilcoxon
    diff = np.array(d_corr) - np.array(d_wrong)
    w = wilcoxon(diff) if len(diff) >= 10 and np.any(diff != 0) else None
    results["gate2"] = {
        "n": len(diff),
        "mean_d_margin_correct": float(np.mean(d_corr)) if d_corr else float("nan"),
        "mean_d_margin_wrong": float(np.mean(d_wrong)) if d_wrong else float("nan"),
        "mean_paired_diff": float(np.mean(diff)) if len(diff) else float("nan"),
        "wilcoxon_p": float(w.pvalue) if w else float("nan"),
        "pass": bool(w and np.mean(diff) > 0 and w.pvalue < 0.01),
    }
    print(f"[gate2] d_margin corr {results['gate2']['mean_d_margin_correct']:.4f} vs "
          f"wrong {results['gate2']['mean_d_margin_wrong']:.4f}, "
          f"p={results['gate2']['wilcoxon_p']:.2e} "
          f"(pass={results['gate2']['pass']})", flush=True)

    # ---- gate 5: boundary-sufficiency oracle ------------------------------------
    diag_layer = n_layers
    all_layers = list(args.layers) + [diag_layer]
    rec_a: dict[int, list[float]] = {L: [] for L in all_layers}
    rec_b: dict[int, list[float]] = {L: [] for L in args.layers}
    identity_maxdiff = None
    pad = tok.pad_token_id or 0
    for i, fk in enumerate(forks[:args.n_oracle]):
        pre = sep_join_ids(tok, [fk["question"], *fk["prefix_steps"]])
        post = pre + tok(fk["correct"], add_special_tokens=False)["input_ids"] + [SEP_TOKEN_ID]
        if not fits(post):
            continue
        b = len(pre) - 1  # boundary position in the pre context
        pre_t = torch.tensor([pre], device=device)
        post_t = torch.tensor([post], device=device)
        with torch.no_grad():
            out_post = model(input_ids=post_t, output_hidden_states=True)
            out_pre = model(input_ids=pre_t, output_hidden_states=True)
        p_actual = out_post.logits[0, -1]
        p_pre = out_pre.logits[0, -1]
        if identity_maxdiff is None:  # patch path identity check, once
            own = out_pre.hidden_states[args.layers[0]][:, b, :]
            lo = forward_with_boundary_patch(model, pre_t, hs_index=args.layers[0],
                                             boundary_pos=b, patched_state=own)
            identity_maxdiff = float((lo[0, -1] - p_pre).abs().max())
        for L in all_layers:
            donor = out_post.hidden_states[L][:, -1, :]
            lo = forward_with_boundary_patch(model, pre_t, hs_index=L,
                                             boundary_pos=b, patched_state=donor)
            rec_a[L].append(recovery_from_logits(p_actual, lo[0, -1], p_pre))
        if len(rec_b[args.layers[0]]) < args.n_oracle_b:
            cands = cands_for(fk["gt_answer"], fk["pre_generated_answer"],
                              tuple(fk["wrong_finals"]), seed_key=fk["fork_id"])
            cids = cand_token_ids(tok, best_suffix, cands)
            b_pre = belief_from_scores(candidate_mean_logprobs(
                model, pre + sfx, cids, pad_id=pad, device=device))
            b_act = belief_from_scores(candidate_mean_logprobs(
                model, post + sfx, cids, pad_id=pad, device=device))
            denom = float((b_act - b_pre).norm())
            for L in args.layers:
                donor = out_post.hidden_states[L][:, -1, :]
                b_orc = belief_from_scores(candidate_mean_logprobs(
                    model, pre + sfx, cids, pad_id=pad, device=device,
                    hs_index=L, boundary_pos=b, patched_state=donor))
                rec_b[L].append(1.0 - float((b_act - b_orc).norm()) / denom
                                if denom > 1e-6 else float("nan"))
        if (i + 1) % 25 == 0:
            print(f"[gate5] {i + 1}/{args.n_oracle}", flush=True)

    def med(v):
        v = [x for x in v if not np.isnan(x)]
        return float(np.median(v)) if v else float("nan")

    gate5 = {"identity_patch_max_logit_diff": identity_maxdiff,
             "recovery_a_median": {L: med(v) for L, v in rec_a.items()},
             "recovery_a_mean": {L: (float(np.nanmean(v)) if v else float("nan"))
                                 for L, v in rec_a.items()},
             "recovery_b_median": {L: med(v) for L, v in rec_b.items()},
             "n_oracle": len(rec_a[args.layers[0]]),
             "n_oracle_b": len(rec_b[args.layers[0]]),
             "diagnostic_layer": diag_layer}
    gate5["pass"] = bool(any(gate5["recovery_a_median"][L] >= 0.3
                             for L in args.layers))
    results["gate5"] = gate5
    print(f"[gate5] identity diff {identity_maxdiff:.2e}; "
          f"median recovery A {gate5['recovery_a_median']}; "
          f"median recovery B {gate5['recovery_b_median']} "
          f"(pass={gate5['pass']})", flush=True)

    # ---- write ------------------------------------------------------------------
    out_dir = args.run_dir / "stage0"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stage0_results.json").write_text(
        json.dumps(results, indent=2, default=str))
    lines = [
        "# transition_operator_v0 Stage 0 results", "",
        f"Model {args.model_name_or_path}, device {device}, "
        f"{datetime.now(timezone.utc).isoformat()}", "",
        f"- Gate 1 (suffix): best {best_suffix!r} rank-1 "
        f"{gate1[best_suffix]:.3f} -> {'PASS' if results['gate1']['pass'] else 'FAIL'}",
        f"- Gate 2 (direction): paired diff "
        f"{results['gate2']['mean_paired_diff']:.4f}, "
        f"p={results['gate2']['wilcoxon_p']:.2e} -> "
        f"{'PASS' if results['gate2']['pass'] else 'FAIL'}",
        f"- Gate 4 (tokenization): {n_roundtrip_bad} roundtrip failures, "
        f"merge rate {results['gate4']['merge_rate_vs_string_tokenization']:.3f}",
        f"- Gate 5 (oracle): identity {identity_maxdiff:.2e}; "
        f"median A recovery {gate5['recovery_a_median']}; "
        f"median B recovery {gate5['recovery_b_median']} -> "
        f"{'PASS' if gate5['pass'] else 'FAIL'}",
    ]
    (out_dir / "stage0_report.md").write_text("\n".join(lines) + "\n")
    print(f"[stage0] wrote {out_dir}/stage0_results.json and stage0_report.md")


if __name__ == "__main__":
    main()
