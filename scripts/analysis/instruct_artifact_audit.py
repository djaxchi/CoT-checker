#!/usr/bin/env python3
"""instruct_arm_v1 §4: the activation-artifact audit, one arm at a time.

Run once per backbone through identical code, then `--part compare` puts the two
side by side. The measurements themselves live in src/analysis/instruct_audit.py.

  --part probe      store + trained cell, no backbone. Outlier-dimension mass,
                    massive tokens, per-position occlusion of the probe, and
                    length/position residualisation, on PRM800K test_2k and on
                    every ProcessBench subset.
  --part attention  backbone forward passes over PRM800K test steps under the
                    verifier template, eager attention: where step tokens attend
                    (token 0, template, problem, prior steps, the step itself),
                    plus the token-0 norm ratio at the read layer.
  --part compare    Base against Instruct, as JSON and a markdown table.

The probe part reads the stores the cell was trained and evaluated on and never
refits anything: the scores are the cell's own, and every covariate is taken
from the store's meta.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.analysis import instruct_audit as ia  # noqa: E402

PB_SUBSETS = ("gsm8k", "math", "olympiadbench", "omnimath")


def git_commit() -> str:
    import subprocess
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# probe part
# ---------------------------------------------------------------------------


def load_cell(cell_dir: Path, assume_rescale: str | None, device):
    import torch
    from src.harness.learners import build_learner
    res = json.loads((cell_dir / "results.json").read_text())
    mode = res["protocol"].get("rescale") or assume_rescale
    if mode is None:
        sys.exit(f"[audit] {cell_dir} predates protocol.rescale; pass --assume_rescale "
                 "after checking its log, as scripts/onpolicy/score_cells_on_split.py does")
    if mode != "none":
        sys.exit(f"[audit] cell rescale={mode!r}; this audit only reads raw-state cells, "
                 "which is what both arms of instruct_arm_v1 are")
    if res["rep"] != "step_tokens":
        sys.exit(f"[audit] rep {res['rep']!r}: occlusion is defined for step_tokens only")
    t_max = int(res["protocol"].get("t_max", 512))
    model = build_learner(res["learner"], int(res["dim"]), t_max=t_max)
    model.load_state_dict(torch.load(cell_dir / "model.pt", map_location=device))
    return model.to(device).eval(), res, t_max


def split_arrays(store_dir: Path, t_max: int, device):
    from scripts.train_rep_learner_cell import build_handles
    from src.harness.spanloader import SpanLoader
    from src.repstore.store import ShardedRepSplit
    handles, meta = build_handles(ShardedRepSplit(store_dir))
    loader = SpanLoader(handles, t_max, device, preload=True)
    return loader, meta


def run_split(model, loader, meta, y, keep, device, max_occl: int,
              token_sample: int, batch: int = 256, seed: int = 0) -> dict:
    """Everything the probe part measures on one split."""
    import torch
    n = len(meta)
    logits = np.empty(n, dtype=np.float64)
    with torch.no_grad():
        for idx in loader.eval_batches(batch):
            xb, mb, _ = loader.collate(idx)
            logits[idx] = model(xb, mb).float().cpu().numpy()

    rng = np.random.default_rng(seed)
    norms: list[np.ndarray] = []
    rows: list[np.ndarray] = []
    n_rows = 0
    occl_items = set(rng.choice(np.where(keep)[0], size=min(max_occl, int(keep.sum())),
                                replace=False).tolist())
    deltas, occl_norms, occl_idx = [], [], []
    drop_first = np.full(n, np.nan); drop_maxn = np.full(n, np.nan)
    with torch.no_grad():
        for k in range(n):
            xb, mb, _ = loader.collate(np.array([k]))
            x = xb[0]                                   # (L, d) float32
            v = torch.linalg.vector_norm(x, dim=1).cpu().numpy()
            norms.append(v)
            if n_rows < token_sample:
                rows.append(x.cpu().numpy()); n_rows += x.shape[0]
            if k in occl_items and x.shape[0] >= 2:
                L = x.shape[0]
                d_k = np.empty(L)
                for a in range(0, L, 128):
                    b = min(a + 128, L)
                    m = torch.ones((b - a, L), device=device)
                    m[torch.arange(b - a), torch.arange(a, b)] = 0.0
                    out = model(x.unsqueeze(0).expand(b - a, -1, -1), m)
                    d_k[a:b] = logits[k] - out.float().cpu().numpy()
                deltas.append(d_k); occl_norms.append(v); occl_idx.append(k)
                drop_first[k] = logits[k] - d_k[0]
                drop_maxn[k] = logits[k] - d_k[int(np.argmax(v))]

    H = np.concatenate(rows)[:token_sample]
    cov = np.c_[np.log([max(len(v), 1) for v in norms]),
                [int(m["step_idx"]) for m in meta],
                np.log([max(int(m.get("orig_step_start_idx", 1)), 1) for m in meta])]
    oi = np.array(occl_idx, dtype=np.int64)
    out = {
        "n_steps": int(keep.sum()),
        "prevalence": float(y[keep].mean()),
        "auroc": ia.auroc(y[keep], logits[keep]),
        "outlier": {"token_topk": ia.topk_norm_share(H),
                    "dims": ia.dimension_concentration(H, k=10),
                    "dims_top1": ia.dimension_concentration(H, k=1),
                    "n_tokens": int(H.shape[0])},
        "massive": ia.massive_token_stats([norms[k] for k in np.where(keep)[0]]),
        "occlusion": ia.occlusion_summary(deltas, occl_norms),
        "occlusion_auroc": {
            "n": int(len(oi)),
            "full": ia.auroc(y[oi], logits[oi]) if len(oi) else float("nan"),
            "drop_first_token": ia.auroc(y[oi], drop_first[oi]) if len(oi) else float("nan"),
            "drop_maxnorm_token": ia.auroc(y[oi], drop_maxn[oi]) if len(oi) else float("nan"),
        },
        "residualisation": ia.residual_auroc(y[keep], logits[keep], cov[keep]),
        "covariates": ["log_step_tokens", "step_idx", "log_context_tokens"],
    }
    return out


def part_probe(a) -> dict:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, res, t_max = load_cell(a.cell_dir, a.assume_rescale, device)
    out = {"cell": str(a.cell_dir), "learner": res["learner"], "seed": res["seed"],
           "in_domain_auroc_recorded": res["in_domain"]["auroc"], "splits": {}}
    t0 = time.perf_counter()
    loader, meta = split_arrays(a.prm_store / a.test_stem, t_max, device)
    y = np.array([h[4] for h in loader.handles], dtype=np.int8)
    keep = np.ones(len(meta), dtype=bool)
    out["splits"]["prm800k_" + a.test_stem] = run_split(
        model, loader, meta, y, keep, device, a.max_occl, a.token_sample)
    r = out["splits"]["prm800k_" + a.test_stem]
    print(f"[probe] prm {a.test_stem}: auroc={r['auroc']:.4f} (recorded "
          f"{out['in_domain_auroc_recorded']:.4f})  {time.perf_counter()-t0:.0f}s", flush=True)
    for sub in PB_SUBSETS:
        d = a.pb_store / sub
        if not d.exists():
            print(f"[probe] skip pb/{sub}: missing", flush=True); continue
        loader, meta = split_arrays(d, t_max, device)
        y, keep = ia.pb_step_labels(meta)
        out["splits"]["pb_" + sub] = run_split(
            model, loader, meta, y, keep, device, a.max_occl_pb, a.token_sample // 4)
        r = out["splits"]["pb_" + sub]
        print(f"[probe] pb/{sub}: n={r['n_steps']} step auroc={r['auroc']:.4f}  "
              f"{time.perf_counter()-t0:.0f}s", flush=True)
    return out


# ---------------------------------------------------------------------------
# attention part
# ---------------------------------------------------------------------------


def part_attention(a) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.onpolicy.prompts import verifier_prefix
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model_name_or_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        a.model_name_or_path, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="eager").to(device).eval()
    rows = [json.loads(l) for l in a.data.read_text().splitlines() if l.strip()]
    rng = np.random.default_rng(0)
    pick = rng.choice(len(rows), size=min(a.n_attn, len(rows)), replace=False)
    layers = [int(x) for x in a.attn_layers.split(",")]
    n_layers = model.config.num_hidden_layers
    per_layer = {l: [] for l in layers}
    all_mean = []
    sink_ratio = []
    for c, i in enumerate(pick):
        r = rows[int(i)]
        ctx = verifier_prefix(r["problem"], r["prefix"])
        enc = tok(ctx, add_special_tokens=True, return_offsets_mapping=True)
        step_ids = tok(r["candidate_step"], add_special_tokens=False)["input_ids"]
        if not step_ids:
            continue
        ids = enc["input_ids"] + step_ids
        if len(ids) > a.max_len:
            continue
        cats = ia.token_categories(enc["offset_mapping"],
                                   ia.verifier_char_spans(r["problem"], r["prefix"]),
                                   len(step_ids))
        with torch.no_grad():
            o = model(input_ids=torch.tensor([ids], device=device),
                      output_attentions=True, output_hidden_states=True)
        for l in layers:
            A = o.attentions[l][0].float().mean(0).cpu().numpy()
            per_layer[l].append(ia.attention_mass(A, cats))
        acc = None
        for l in range(n_layers):
            m = ia.attention_mass(o.attentions[l][0].float().mean(0).cpu().numpy(), cats)
            acc = m if acc is None else {k: acc[k] + m[k] for k in acc}
        all_mean.append({k: v / n_layers for k, v in acc.items()})
        h = o.hidden_states[a.read_index][0].float()
        nv = torch.linalg.vector_norm(h, dim=1).cpu().numpy()
        sink_ratio.append(float(nv[0] / np.median(nv[1:])))
        if (c + 1) % 50 == 0:
            print(f"[attn] {c+1}/{len(pick)}", flush=True)

    def avg(ms):
        return {k: float(np.mean([m[k] for m in ms])) for k in ms[0]} if ms else {}
    return {"model": a.model_name_or_path, "n": len(all_mean),
            "by_layer": {str(l): avg(per_layer[l]) for l in layers},
            "all_layers_mean": avg(all_mean),
            "read_index": a.read_index,
            "token0_norm_over_median": {"median": float(np.median(sink_ratio)),
                                         "mean": float(np.mean(sink_ratio))}}


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


def part_compare(a) -> dict:
    def load(d: Path) -> dict:
        return {"probe": json.loads((d / "probe.json").read_text())["probe"],
                "attention": json.loads((d / "attention.json").read_text())["attention"]}
    base = load(a.base); inst = load(a.instruct)
    lines = ["| split | metric | Base | Instruct |", "|---|---|---|---|"]

    def row(split, name, fb, fi):
        lines.append(f"| {split} | {name} | {fb:.4f} | {fi:.4f} |")

    for split in base["probe"]["splits"]:
        b = base["probe"]["splits"][split]; i = inst["probe"]["splits"].get(split)
        if i is None:
            continue
        row(split, "step AUROC", b["auroc"], i["auroc"])
        row(split, "AUROC, length/position residualised",
            b["residualisation"]["residual"], i["residualisation"]["residual"])
        row(split, "AUROC, covariates only",
            b["residualisation"]["covariates_only"], i["residualisation"]["covariates_only"])
        row(split, "token top-1 dim share of norm^2",
            b["outlier"]["token_topk"]["top1"], i["outlier"]["token_topk"]["top1"])
        row(split, "token top-10 dim share of norm^2",
            b["outlier"]["token_topk"]["top10"], i["outlier"]["token_topk"]["top10"])
        row(split, "top-10 fixed dims share of mean-square",
            b["outlier"]["dims"]["share"], i["outlier"]["dims"]["share"])
        row(split, "massive-token rate (>5x median norm)",
            b["massive"]["rate"], i["massive"]["rate"])
        row(split, "occlusion share, first step token",
            b["occlusion"]["share_first"], i["occlusion"]["share_first"])
        row(split, "  (uniform baseline)",
            b["occlusion"]["share_first_uniform"], i["occlusion"]["share_first_uniform"])
        row(split, "occlusion share, max-norm token",
            b["occlusion"]["share_maxnorm"], i["occlusion"]["share_maxnorm"])
        row(split, "AUROC with first token masked",
            b["occlusion_auroc"]["drop_first_token"], i["occlusion_auroc"]["drop_first_token"])
    for key in ("all_layers_mean",):
        b = base["attention"][key]; i = inst["attention"][key]
        for c in b:
            row("attention (" + key + ")", f"step-query mass on {c}", b[c], i[c])
    for l in base["attention"]["by_layer"]:
        b = base["attention"]["by_layer"][l]; i = inst["attention"]["by_layer"].get(l, {})
        for c in b:
            if c in i:
                row(f"attention (block {l})", f"step-query mass on {c}", b[c], i[c])
    row("read layer", "token-0 norm / median norm",
        base["attention"]["token0_norm_over_median"]["median"],
        inst["attention"]["token0_norm_over_median"]["median"])
    md = "\n".join(lines)
    a.out.with_suffix(".md").write_text(md + "\n")
    print(md)
    return {"base": str(a.base), "instruct": str(a.instruct), "table_md": md}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--part", choices=["probe", "attention", "compare"], required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cell_dir", type=Path)
    p.add_argument("--prm_store", type=Path)
    p.add_argument("--pb_store", type=Path)
    p.add_argument("--test_stem", default="test_2k")
    p.add_argument("--assume_rescale", choices=["none"], default=None,
                   help="For a cell that predates protocol.rescale (the Base arm).")
    p.add_argument("--max_occl", type=int, default=2000)
    p.add_argument("--max_occl_pb", type=int, default=500)
    p.add_argument("--token_sample", type=int, default=200_000)
    p.add_argument("--model_name_or_path")
    p.add_argument("--data", type=Path, help="prm800k_test_2k.jsonl, for --part attention")
    p.add_argument("--n_attn", type=int, default=300)
    p.add_argument("--attn_layers", default="34",
                   help="Block indices; 34 writes the state the probe reads.")
    p.add_argument("--read_index", type=int, default=35,
                   help="hidden_states index the probe reads (block 34's output)")
    p.add_argument("--max_len", type=int, default=2048)
    p.add_argument("--base", type=Path, help="dir holding the Base arm's probe.json "
                   "and attention.json, for --part compare")
    p.add_argument("--instruct", type=Path, help="the same for the Instruct arm")
    a = p.parse_args()

    t0 = time.perf_counter()
    if a.part == "probe":
        body = {"probe": part_probe(a)}
    elif a.part == "attention":
        body = {"attention": part_attention(a)}
    else:
        body = part_compare(a)
    body["meta"] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": git_commit(), "wall_seconds": time.perf_counter() - t0}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(body, indent=2))
    print(f"[audit] wrote {a.out}")


if __name__ == "__main__":
    main()
