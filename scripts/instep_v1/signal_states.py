#!/usr/bin/env python3
"""InStep paper, "where the signal lives": per-step states of the two other verifiers, on the
same steps as the stored InStep states (read_token_inline.py encode, $ST/qwen3_8b).

  prm      Qwen2.5-Math-PRM-7B on the problem and the trace's steps, as score_prm.py feeds it
           (one forward per trace, <extra_0> after each step). Kept per step, at that step's
           separator: the output of blocks --blocks (`prm_block<l>` = hidden_states[l + 1])
           and the final post-norm state the PRM head reads (`prm_final`).
  reprobe  ReProbe's released hidden-state head (DeepSeek labels) on Qwen3-8B, as
           reprobe_probe.py runs it (future visible, as released). Kept per step: the
           head's pooled step representation, the input of its classifier (`reprobe_rep`,
           512 dims).

Each output <out>/<name>.shard0.{npz,meta.jsonl} holds the uids of --keep <name>:<meta.jsonl>
in that order (the InStep states' meta), so every representation is aligned by row: npz keys
y (int8, from the items / trace labels), kept (bool), the states (n, 1, d) float16, and p
(float32, the method's own P(incorrect) from the same pass) for the gate. Traces without a
kept uid are not run.

Gate: p against the method's stored step scores (--gate_scores <dir> with <name>.jsonl
{"uid", "p"}): Spearman >= 0.999 on the kept steps, else exit 1.

    python scripts/instep_v1/signal_states.py prm --out $SS --gate_scores $OUT/scores/qwen3_8b/prm7b \\
        --sets math:$D/test/math.items.jsonl ... --keep math:$ST/math.shard0.meta.jsonl ...
    python scripts/instep_v1/signal_states.py reprobe --ckpt $V1/released/ds --out $SS \\
        --gate_scores $V1/runs/released/scores/rel_ds --sets math:$D/test/math.traces.jsonl ... --keep ...
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p) if l.strip()]


def parse_pairs(specs: list[str]) -> dict[str, Path]:
    return {s.split(":", 1)[0]: Path(s.split(":", 1)[1]) for s in specs}


def keep_uids(meta: Path) -> list[str]:
    return [r["uid"] for r in jl(meta)]


def trace_of(uid: str) -> str:
    return uid.rsplit("::", 1)[0]


def assemble(keep: list[str], rows: dict[str, tuple], keys: list[str]) -> dict:
    """npz arrays in keep order from rows {uid: (y, p, {key: vector})}; kept = uid present."""
    n = len(keep)
    first = next(iter(rows.values()))[2]
    out = {k: np.zeros((n, 1, len(first[k])), np.float16) for k in keys}
    y = np.full(n, -1, np.int8)
    p = np.full(n, np.nan, np.float32)
    kept = np.zeros(n, bool)
    for i, u in enumerate(keep):
        r = rows.get(u)
        if r is None:
            continue
        y[i], p[i], kept[i] = r[0], r[1], True
        for k in keys:
            out[k][i, 0] = r[2][k]
    return {"y": y, "p": p, "kept": kept, **out}


def write(out: Path, name: str, keep: list[str], arrays: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{name}.shard0.npz", **arrays)
    with open(out / f"{name}.shard0.meta.jsonl", "w") as f:
        for u in keep:
            f.write(json.dumps({"uid": u}) + "\n")


def gate(name: str, keep: list[str], arrays: dict, scores_dir: Path | None) -> dict | None:
    if scores_dir is None or not (scores_dir / f"{name}.jsonl").exists():
        return None
    from scipy.stats import spearmanr
    st = {r["uid"]: r["p"] for r in jl(scores_dir / f"{name}.jsonl")}
    m = [i for i, u in enumerate(keep) if arrays["kept"][i] and u in st]
    ours = arrays["p"][m]
    ref = np.array([st[keep[i]] for i in m])
    rho = float(spearmanr(ours, ref).correlation)
    return {"n": len(m), "spearman": rho, "max_abs_diff": float(np.abs(ours - ref).max()), "pass": rho >= 0.999}


# ---------------------------------------------------------------- PRM
def run_prm(a) -> dict:
    import torch
    from transformers import AutoModel, AutoTokenizer
    import transformers
    from scripts.instep_v1.score_prm import SEP, TAGS, batches, load_traces, messages
    if int(transformers.__version__.split(".")[0]) >= 5:
        raise SystemExit(f"the Qwen PRMs need transformers < 5 (found {transformers.__version__})")
    path = a.model_path or TAGS["prm7b"]
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model, info = AutoModel.from_pretrained(path, torch_dtype=torch.bfloat16, trust_remote_code=True,
                                            local_files_only=True, output_loading_info=True)
    if info.get("missing_keys"):
        raise SystemExit(f"PRM weights not loaded: {info['missing_keys'][:3]}")
    model = model.to("cuda").eval()
    sep_id = tok.encode(SEP)[0]
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    blocks = [int(x) for x in a.blocks.split(",") if x]
    keys = [f"prm_block{l}" for l in blocks] + ["prm_final"]
    keeps, report = parse_pairs(a.keep), {}
    for name, items in parse_pairs(a.sets).items():
        keep = keep_uids(keeps[name])
        want = set(keep)
        labels = {r["uid"]: int(r["label"]) for r in jl(items)}
        trs, _ = load_traces(items)
        trs = [t for t in trs if want.intersection(t["uids"])]
        if a.smoke:
            trs = trs[:a.smoke]
        ids = [tok.encode(tok.apply_chat_template(messages(t["problem"], t["steps"]), tokenize=False,
                                                  add_generation_prompt=False)) for t in trs]
        lens = [len(x) for x in ids]
        rows, t0 = {}, time.perf_counter()
        with torch.no_grad():
            for b in batches(lens, a.max_tokens):
                T = lens[b[0]]
                x = torch.full((len(b), T), pad_id, dtype=torch.long)
                att = torch.zeros((len(b), T), dtype=torch.long)
                for j, i in enumerate(b):
                    x[j, :lens[i]] = torch.tensor(ids[i])
                    att[j, :lens[i]] = 1
                x, att = x.cuda(), att.cuda()
                o = model.model(input_ids=x, attention_mask=att, use_cache=False, output_hidden_states=True)
                hs = o.hidden_states
                pc = torch.softmax(model.score(o.last_hidden_state).float(), -1)[..., 1]
                for j, i in enumerate(b):
                    pos = (x[j, :lens[i]] == sep_id).nonzero().squeeze(-1)
                    if len(pos) != len(trs[i]["steps"]):
                        raise SystemExit(f"{trs[i]['id']}: {len(pos)} separators for {len(trs[i]['steps'])} steps")
                    st = {f"prm_block{l}": hs[l + 1][j, pos].float().cpu().numpy() for l in blocks}
                    st["prm_final"] = o.last_hidden_state[j, pos].float().cpu().numpy()
                    p = (1 - pc[j, pos]).cpu().numpy()
                    for k, u in enumerate(trs[i]["uids"]):
                        if u in want:
                            rows[u] = (labels[u], p[k], {kk: v[k] for kk, v in st.items()})
        arrays = assemble(keep, rows, keys)
        write(a.out, name, keep, arrays)
        g = gate(name, keep, arrays, a.gate_scores)
        report[name] = {"traces": len(trs), "kept": int(arrays["kept"].sum()), "of": len(keep), "gate": g,
                        "seconds": time.perf_counter() - t0}
        print(f"[prm] {name}: {report[name]}", flush=True)
    return report


# ---------------------------------------------------------------- ReProbe
def run_reprobe(a) -> dict:
    import torch
    from scripts.reprobe_cmp.reprobe_probe import (claim_tensors, features, head_from_release, labels_of,
                                                   load_model, load_traces, micro_batches, run_head)
    torch.backends.cuda.enable_cudnn_sdp(False)  # as reprobe_probe.py
    dev = torch.device("cuda")
    model, tok, pad = load_model(a.model, dev)
    head = head_from_release(a.ckpt, dev, model.config.hidden_size)
    cap = []
    head.classifier.register_forward_hook(lambda m, inp, out: cap.append(inp[0].float()))
    keeps, report = parse_pairs(a.keep), {}
    for name, path in parse_pairs(a.sets).items():
        keep = keep_uids(keeps[name])
        want = set(keep)
        tids = {trace_of(u) for u in keep}
        trs = [t for t in load_traces(path, tok) if t["trace_id"] in tids]
        if a.smoke:
            trs = trs[:a.smoke]
        rows, t0 = {}, time.perf_counter()
        order = sorted(range(len(trs)), key=lambda i: len(trs[i]["ids"]))
        with torch.no_grad():
            for mb in micro_batches(trs, order, a.max_tokens, a.max_claim_tokens):
                batch = [trs[i] for i in mb]
                feats, valid = features(model, batch, dev, pad)
                claims, owner, where = claim_tensors(batch, feats.shape[1], dev)
                if claims is None:
                    continue
                cap.clear()
                p = torch.sigmoid(run_head(head, feats, valid, claims, owner, "visible")).cpu().numpy()
                rep = cap[0].cpu().numpy()
                for (b, k), pv, r in zip(where, p, rep):
                    u = f"{batch[b]['trace_id']}::{k}"
                    if u in want:
                        rows[u] = (labels_of(batch[b], a.label)[k], pv, {"reprobe_rep": r})
        arrays = assemble(keep, rows, ["reprobe_rep"])
        write(a.out, name, keep, arrays)
        g = gate(name, keep, arrays, a.gate_scores)
        report[name] = {"traces": len(trs), "kept": int(arrays["kept"].sum()), "of": len(keep), "gate": g,
                        "seconds": time.perf_counter() - t0}
        print(f"[reprobe] {name}: {report[name]}", flush=True)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("method", choices=["prm", "reprobe"])
    p.add_argument("--sets", nargs="+", required=True, help="name:path (prm: items.jsonl, reprobe: traces.jsonl)")
    p.add_argument("--keep", nargs="+", required=True, help="name:meta.jsonl (uids to keep, in order)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--gate_scores", type=Path, default=None)
    p.add_argument("--blocks", default="3,7,11,15,19,23", help="prm: block outputs kept (of 28)")
    p.add_argument("--model_path", default=None, help="prm: default Qwen/Qwen2.5-Math-PRM-7B")
    p.add_argument("--model", default="Qwen/Qwen3-8B", help="reprobe: backbone")
    p.add_argument("--ckpt", type=Path, help="reprobe: released dir (config.yaml + weights.pth)")
    p.add_argument("--label", default="ds", help="reprobe: label set of the trace files")
    p.add_argument("--max_tokens", type=int, default=24000)
    p.add_argument("--max_claim_tokens", type=int, default=160000)
    p.add_argument("--smoke", type=int, default=0, help="first N traces per set")
    a = p.parse_args()
    rep = run_prm(a) if a.method == "prm" else run_reprobe(a)
    a.out.mkdir(parents=True, exist_ok=True)
    tag = "_".join(sorted(parse_pairs(a.sets)))
    (a.out / f"report_{a.method}_{tag}.json").write_text(json.dumps(rep, indent=1))
    bad = [n for n, r in rep.items() if r["gate"] is not None and not r["gate"]["pass"]]
    if bad and not a.smoke:
        raise SystemExit(f"gate failed on {bad}")


if __name__ == "__main__":
    main()
