#!/usr/bin/env python3
"""cot_causal_graph_v0 Stage 1: teacher-forced interventions + node features (GPU).

Per trace, one base forward and one forward per intervention, downstream text kept
fixed (spec: docs/cot_causal_graph_v0_plan.md). Arm forks: swap the fork step with
{wrong sibling, +1 sibling when present, off-topic length-matched control}. Arm
onpolicy: {delete, cross-trace swap} at every step of the model's own graded
trajectories (from generate_onpolicy_steps.py).

Measurements:
  node features   probe logit (L28 deployed, L20 secondary), step mean NLL,
                  boundary entropy, base answer-margin profile m_0..m_T
  tf edges        d mean-logprob of every downstream step, d answer margin at
                  the final boundary (+ boundary profile for arm forks),
                  d probe logit downstream (DIAGNOSTIC only, S3 Stage-5 rule),
                  probe logit AT the swapped-in step (the detection quantity)

Sharding: in-node over GPUs (worker k of N takes trace_idx % N == k), one process
per CUDA_VISIBLE_DEVICES slice, then --merge on CPU concatenates shards and
computes gates G1 (margin validity) and G3-TF (null calibration) plus the
val-selected detection threshold.

Usage (one shard):
  CUDA_VISIBLE_DEVICES=0 python scripts/causal_graph/cg_stage1_tf.py \
    --run_dir runs/causal_graph --model_name_or_path Qwen/Qwen2.5-7B \
    --directions_npz .../directions_L28.npz .../directions_L20.npz \
    --arm forks --shard_id 0 --num_shards 4
Merge + gates (CPU):
  python scripts/causal_graph/cg_stage1_tf.py --run_dir runs/causal_graph --merge
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from scripts.encode_prm800k_hidden_states import git_commit, read_jsonl  # noqa: E402
from scripts.generate_onpolicy_steps import build_prompt, split_into_steps  # noqa: E402
from src.analysis.causal_graph import (  # noqa: E402
    ELICITATION_SUFFIX,
    assemble_ids,
    cand_token_ids,
    encode_pieces,
    entropy_at,
    length_matched_step,
    margin_at_boundary,
    per_span_mean_logprob,
    probe_logits_at,
)
from src.analysis.transition_operator import build_candidates, stable_seed  # noqa: E402


def load_probes(paths: list[Path]) -> dict[int, tuple[np.ndarray, float]]:
    """{layer_index: (w_score, b_score)} from build_steering_directions npz files."""
    probes = {}
    for p in paths:
        z = np.load(p, allow_pickle=True)
        probes[int(z["layer_index"])] = (z["w_score"].astype(np.float32),
                                         float(z["b_score"]))
    return probes


def forward(model, ids: list[int], device, layers: list[int]):
    """One teacher-forced pass; returns (logits (L,V) cpu-float unnecessary, kept on
    device), {layer: hidden (L,d)}). Hidden states are needed at two layers only."""
    import torch
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids], device=device),
                    output_hidden_states=True)
    hs = {l: out.hidden_states[l][0] for l in layers}
    return out.logits[0], hs


def trace_measurements(model, tok, device, pieces: list[str], probes,
                       suffix_ids, cand_ids, pad_id, margin_boundaries,
                       layers: list[int]):
    """Forward one piece sequence and read everything: per-piece ids/spans, logits,
    step NLLs, boundary entropies, probe logits at step-final tokens, margins at
    the requested boundary indices (piece indices)."""
    import torch as _t
    piece_ids = encode_pieces(tok, pieces)
    full, spans, bounds = assemble_ids(piece_ids)
    logits, hs = forward(model, full, device, layers)
    ids_tensor = _t.tensor(full, device=logits.device)
    nll = per_span_mean_logprob(logits, ids_tensor, spans)
    ent = entropy_at(logits, bounds)
    last_tok = [max(lo, hi - 1) for lo, hi in spans]
    probe = {l: probe_logits_at(hs[l], last_tok, _t.tensor(w), b)
             for l, (w, b) in probes.items()}
    margins = {}
    for p in margin_boundaries:
        margins[p] = margin_at_boundary(model, full[:bounds[p] + 1], suffix_ids,
                                        cand_ids, pad_id, device)
    return {"full": full, "spans": spans, "bounds": bounds, "step_nll": nll,
            "entropy": ent, "probe": probe, "margins": margins}


def run_forks(args, model, tok, device, probes, node_rows, edge_rows, skips):
    traces = read_jsonl(args.run_dir / "traces_forks.jsonl")
    suffix_ids = tok(ELICITATION_SUFFIX, add_special_tokens=False)["input_ids"]
    pad_id = tok.pad_token_id or tok.eos_token_id
    layers = sorted(probes.keys())
    t0 = time.perf_counter()
    mine = [tr for i, tr in enumerate(traces) if i % args.num_shards == args.shard_id]
    for ti, tr in enumerate(mine):
        pieces = [tr["question"]] + tr["steps"]
        n_tok = sum(len(x) for x in encode_pieces(tok, pieces)) + len(pieces)
        if n_tok > args.max_seq_len:
            skips.append({"trace_id": tr["trace_id"], "reason": "overlength",
                          "n_tok": n_tok})
            continue
        T = len(tr["steps"])
        t_fork = tr["fork_t"]
        cand_ids = cand_token_ids(tok, ELICITATION_SUFFIX, tr["candidates"])
        base = trace_measurements(model, tok, device, pieces, probes, suffix_ids,
                                  cand_ids, pad_id, list(range(T + 1)), layers)
        for i in range(T):
            node_rows.append({
                "arm": "forks", "trace_id": tr["trace_id"], "split": tr["split"],
                "step_idx": i, "is_fork_step": i == t_fork,
                "text": tr["steps"][i][:400],
                "probe_l28": base["probe"].get(28, [np.nan] * (T + 1))[i + 1],
                "probe_l20": base["probe"].get(20, [np.nan] * (T + 1))[i + 1],
                "step_logp": base["step_nll"][i + 1],
                "boundary_entropy": base["entropy"][i + 1],
                "margin": base["margins"][i + 1],
                "margin_pre": base["margins"][i],
            })
        swaps = [("swap_wrong", tr["wrong_step"]),
                 ("swap_xprob", tr["xprob_step"])]
        if tr.get("alt_pos_step"):
            swaps.append(("swap_pos", tr["alt_pos_step"]))
        for name, text in swaps:
            pieces2 = list(pieces)
            pieces2[t_fork + 1] = text
            mb = list(range(t_fork, T + 1))  # boundary before the swap unchanged
            iv = trace_measurements(model, tok, device, pieces2, probes,
                                    suffix_ids, cand_ids, pad_id, mb, layers)
            edge_rows.append({
                "arm": "forks", "trace_id": tr["trace_id"], "split": tr["split"],
                "interv": name, "t": t_fork, "swap_text": text[:400],
                "swap_len_delta": len(text.split()) - len(tr["steps"][t_fork].split()),
                "probe_at_interv_l28": iv["probe"].get(28, [np.nan] * (T + 1))[t_fork + 1],
                "probe_at_interv_l20": iv["probe"].get(20, [np.nan] * (T + 1))[t_fork + 1],
                "d_margin_final": iv["margins"][T] - base["margins"][T],
                "d_margin_profile": json.dumps(
                    {str(p): iv["margins"][p] - base["margins"][p] for p in mb}),
                "d_logp_steps": json.dumps(
                    {str(j): iv["step_nll"][j + 1] - base["step_nll"][j + 1]
                     for j in range(t_fork + 1, T)}),
                "d_probe_steps": json.dumps(
                    {str(j): iv["probe"][layers[-1]][j + 1] - base["probe"][layers[-1]][j + 1]
                     for j in range(t_fork + 1, T)}),
            })
        if (ti + 1) % 20 == 0:
            el = time.perf_counter() - t0
            print(f"[forks shard {args.shard_id}] {ti + 1}/{len(mine)} "
                  f"({el / (ti + 1):.1f}s/trace)", flush=True)


def run_onpolicy(args, model, tok, device, probes, node_rows, edge_rows, skips):
    trajs = read_jsonl(args.onpolicy_trajectories)
    trajs = [t for t in trajs if t.get("gradeable")]
    # balance correct/incorrect, cap, deterministic
    import random
    rng = random.Random(args.seed)
    rng.shuffle(trajs)
    cor = [t for t in trajs if t["correct"]][:args.max_onpolicy_traces // 2]
    inc = [t for t in trajs if not t["correct"]][:args.max_onpolicy_traces // 2]
    trajs = cor + inc
    suffix_ids = tok(ELICITATION_SUFFIX, add_special_tokens=False)["input_ids"]
    pad_id = tok.pad_token_id or tok.eos_token_id
    layers = sorted(probes.keys())
    golds = tuple(t["gold"] for t in trajs)
    pool = [(t["traj_uid"], s) for t in trajs for s in split_into_steps(t["solution"])]
    mine = [t for i, t in enumerate(trajs) if i % args.num_shards == args.shard_id]
    t0 = time.perf_counter()
    for ti, tr in enumerate(mine):
        steps = split_into_steps(tr["solution"])
        if len(steps) < 3:
            skips.append({"trace_id": tr["traj_uid"], "reason": "too_few_steps"})
            continue
        pieces = [build_prompt(tr["problem"])] + steps
        n_tok = sum(len(x) for x in encode_pieces(tok, pieces)) + len(pieces)
        if n_tok > args.max_seq_len:
            skips.append({"trace_id": tr["traj_uid"], "reason": "overlength",
                          "n_tok": n_tok})
            continue
        T = len(steps)
        cands = build_candidates(tr["gold"], None, (), golds, k=8,
                                 seed=stable_seed(tr["traj_uid"], args.seed))
        cand_ids = cand_token_ids(tok, ELICITATION_SUFFIX, cands)
        base = trace_measurements(model, tok, device, pieces, probes, suffix_ids,
                                  cand_ids, pad_id, list(range(T + 1)), layers)
        for i in range(T):
            node_rows.append({
                "arm": "onpolicy", "trace_id": tr["traj_uid"],
                "split": tr.get("split", "test"), "step_idx": i,
                "traj_correct": bool(tr["correct"]), "text": steps[i][:400],
                "probe_l28": base["probe"].get(28, [np.nan] * (T + 1))[i + 1],
                "probe_l20": base["probe"].get(20, [np.nan] * (T + 1))[i + 1],
                "step_logp": base["step_nll"][i + 1],
                "boundary_entropy": base["entropy"][i + 1],
                "margin": base["margins"][i + 1],
            })
        import random as _r
        r = _r.Random(stable_seed(tr["traj_uid"], args.seed + 1))
        for i in range(T):
            for name in ("delete", "swap_xtrace"):
                if name == "delete":
                    pieces2 = pieces[:i + 1] + pieces[i + 2:]
                    off = -1  # downstream piece index shift
                    swap_text = ""
                else:
                    swap_text = length_matched_step(
                        r, pool, target_words=len(steps[i].split()),
                        exclude_key=tr["traj_uid"])
                    pieces2 = list(pieces)
                    pieces2[i + 1] = swap_text
                    off = 0
                T2 = len(pieces2) - 1
                iv = trace_measurements(model, tok, device, pieces2, probes,
                                        suffix_ids, cand_ids, pad_id, [T2], layers)
                edge_rows.append({
                    "arm": "onpolicy", "trace_id": tr["traj_uid"],
                    "split": tr.get("split", "test"), "interv": name, "t": i,
                    "swap_text": swap_text[:400],
                    "swap_len_delta": (len(swap_text.split()) - len(steps[i].split())
                                       if name == "swap_xtrace"
                                       else -len(steps[i].split())),
                    "probe_at_interv_l28": (iv["probe"][28][i + 1]
                                            if name == "swap_xtrace" and 28 in iv["probe"]
                                            else np.nan),
                    "probe_at_interv_l20": (iv["probe"][20][i + 1]
                                            if name == "swap_xtrace" and 20 in iv["probe"]
                                            else np.nan),
                    "d_margin_final": iv["margins"][T2] - base["margins"][T],
                    "d_margin_profile": json.dumps({}),
                    "d_logp_steps": json.dumps(
                        {str(j): iv["step_nll"][j + 1 + off] - base["step_nll"][j + 1]
                         for j in range(i + 1, T)}),
                    "d_probe_steps": json.dumps(
                        {str(j): (iv["probe"][layers[-1]][j + 1 + off]
                                  - base["probe"][layers[-1]][j + 1])
                         for j in range(i + 1, T)}),
                })
        if (ti + 1) % 10 == 0:
            el = time.perf_counter() - t0
            print(f"[onpolicy shard {args.shard_id}] {ti + 1}/{len(mine)} "
                  f"({el / (ti + 1):.1f}s/trace)", flush=True)


def merge(args) -> None:
    """Concatenate shards, compute G1/G3-TF gates + val-selected detection threshold."""
    import pandas as pd
    from scipy import stats
    d = args.run_dir / "stage1"
    nodes = pd.concat([pd.read_parquet(p) for p in sorted(d.glob("node_features_shard*.parquet"))],
                      ignore_index=True)
    edges = pd.concat([pd.read_parquet(p) for p in sorted(d.glob("tf_edges_shard*.parquet"))],
                      ignore_index=True)
    nodes.to_parquet(d / "node_features.parquet")
    edges.to_parquet(d / "tf_edges.parquet")
    fk = edges[edges.arm == "forks"]
    piv = fk.pivot_table(index="trace_id", columns="interv",
                         values="d_margin_final", aggfunc="first")
    gates: dict = {"n_traces_forks": int(piv.shape[0])}

    def wilcox(a, b, name):
        both = piv[[a, b]].dropna()
        if len(both) < 20:
            gates[name] = {"n": int(len(both)), "skipped": True}
            return
        stat = stats.wilcoxon(both[a], both[b], alternative="less")
        gates[name] = {"n": int(len(both)),
                       "median_" + a: float(both[a].median()),
                       "median_" + b: float(both[b].median()),
                       "p": float(stat.pvalue),
                       "pass": bool(stat.pvalue < 0.01)}

    if "swap_pos" in piv:
        wilcox("swap_wrong", "swap_pos", "G1_wrong_vs_pos")
    wilcox("swap_wrong", "swap_xprob", "G1_wrong_vs_xprob")

    def site_auc(a, b, name):
        x = piv[a].dropna().abs().to_numpy()
        y = piv[b].dropna().abs().to_numpy()
        if len(x) < 20 or len(y) < 20:
            gates[name] = {"skipped": True}
            return
        u = stats.mannwhitneyu(x, y, alternative="greater")
        gates[name] = {"auc": float(u.statistic / (len(x) * len(y))),
                       "p": float(u.pvalue),
                       "pass": bool(u.statistic / (len(x) * len(y)) > 0.55)}

    site_auc("swap_wrong", "swap_xprob", "G3_tf_wrong_vs_xprob")
    if "swap_pos" in piv:
        site_auc("swap_wrong", "swap_pos", "G3_tf_wrong_vs_pos")

    # detection threshold: probe at the swapped wrong step (pos) vs the golden
    # fork step in the base pass (neg), val split, max balanced accuracy
    val_wrong = fk[(fk.interv == "swap_wrong") & (fk.split == "val")]
    val_gold = nodes[(nodes.arm == "forks") & (nodes.split == "val")
                     & nodes.is_fork_step]
    pos = val_wrong.probe_at_interv_l28.dropna().to_numpy()
    neg = val_gold.probe_l28.dropna().to_numpy()
    if len(pos) >= 30 and len(neg) >= 30:
        grid = np.quantile(np.concatenate([pos, neg]), np.linspace(0.02, 0.98, 97))
        bal = [((pos > c).mean() + (neg <= c).mean()) / 2 for c in grid]
        k = int(np.argmax(bal))
        gates["detection_threshold_l28"] = {
            "threshold": float(grid[k]), "val_balanced_acc": float(bal[k]),
            "n_pos": int(len(pos)), "n_neg": int(len(neg))}
    (d / "gates_stage1.json").write_text(json.dumps(gates, indent=2))
    print(json.dumps(gates, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", type=Path, default=Path("runs/causal_graph"))
    ap.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-7B")
    ap.add_argument("--directions_npz", type=Path, nargs="+", default=[])
    ap.add_argument("--arm", choices=["forks", "onpolicy", "both"], default="forks")
    ap.add_argument("--onpolicy_trajectories", type=Path,
                    default=Path("runs/causal_graph/onpolicy_trajectories.jsonl"))
    ap.add_argument("--max_onpolicy_traces", type=int, default=400)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--max_seq_len", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()

    out_dir = args.run_dir / "stage1"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.merge:
        merge(args)
        return

    import pandas as pd
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                        local_files_only=args.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, torch_dtype=torch.bfloat16,
        local_files_only=args.local_files_only).to(device).eval()
    probes = load_probes(args.directions_npz)
    if 28 not in probes:
        sys.exit("need --directions_npz including the L28 deployed probe")

    node_rows: list[dict] = []
    edge_rows: list[dict] = []
    skips: list[dict] = []
    if args.arm in ("forks", "both"):
        run_forks(args, model, tok, device, probes, node_rows, edge_rows, skips)
    if args.arm in ("onpolicy", "both"):
        run_onpolicy(args, model, tok, device, probes, node_rows, edge_rows, skips)

    pd.DataFrame(node_rows).to_parquet(out_dir / f"node_features_shard{args.shard_id}.parquet")
    pd.DataFrame(edge_rows).to_parquet(out_dir / f"tf_edges_shard{args.shard_id}.parquet")
    (out_dir / f"manifest_shard{args.shard_id}.json").write_text(json.dumps({
        "created": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(), "args": {k: str(v) for k, v in vars(args).items()},
        "n_node_rows": len(node_rows), "n_edge_rows": len(edge_rows),
        "skips": skips}, indent=2))
    print(f"[stage1 shard {args.shard_id}] nodes {len(node_rows)} "
          f"edges {len(edge_rows)} skips {len(skips)}", flush=True)


if __name__ == "__main__":
    main()
