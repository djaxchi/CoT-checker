"""attention_routing_v0: eager vs sdpa numerical parity check.

The attention extraction requires attn_implementation="eager", while the S4
hidden-state arm ran the default backend (sdpa). This script quantifies, on
a fork subset, whether the two backends compute materially different
forwards, so that later attention-vs-hidden-state comparisons cannot be
confounded by the backend switch. Per (fork, role) it compares:

  - candidate-token log-probabilities (max and mean absolute difference)
  - mean candidate NLL under each backend
  - greedy argmax agreement at candidate positions
  - cosine similarity of the final-layer last-token hidden state

Writes parity.json (summary) + parity_rows.csv into --out_dir. Tiny
floating-point differences are expected; a systematic shift is what to look
for. This check gates interpretation, not the run itself.

Usage:
  python scripts/attention_routing/ar_backend_parity.py \
      --model_name_or_path Qwen/Qwen2.5-7B --local_files_only --limit 100
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_prm800k_prestudy import git_commit, read_jsonl  # noqa: E402
from src.analysis.attention_routing import (  # noqa: E402
    assign_token_regions,
    build_fork_segments,
    candidate_token_span,
)

ROLES = ("correct", "wrong")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--forks", type=Path,
                    default=Path("runs/contrib_cluster/forks.jsonl"))
    ap.add_argument("--out_dir", type=Path,
                    default=Path("runs/attention_routing/parity"))
    ap.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-7B")
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--model_dtype",
                    choices=["bfloat16", "float16", "float32"],
                    default="bfloat16",
                    help="bfloat16 by default: Qwen2.5-7B overflows float16 "
                         "in the last attention layer and at the lm_head")
    ap.add_argument("--max_seq_len", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=100)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                 "float32": torch.float32}
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                        local_files_only=args.local_files_only)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    models = {}
    for impl in ("eager", "sdpa"):
        m = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, local_files_only=args.local_files_only,
            dtype=dtype_map[args.model_dtype], attn_implementation=impl)
        models[impl] = m.to(device).eval()

    forks = read_jsonl(args.forks)[: args.limit]
    rows = []
    for fk in forks:
        for role in ROLES:
            text, segments = build_fork_segments(
                fk["question"], fk["prefix_steps"], fk[role])
            enc = tok(text, return_offsets_mapping=True, return_tensors="np")
            ids = enc["input_ids"][0]
            if ids.shape[0] > args.max_seq_len:
                continue
            regions = assign_token_regions(
                [tuple(o) for o in enc["offset_mapping"][0]], segments)
            c0, c1 = candidate_token_span(regions)
            input_ids = torch.tensor(ids[None], device=device)
            tgt = torch.tensor(ids[c0:c1], device=device, dtype=torch.long)

            per = {}
            for impl, model in models.items():
                with torch.no_grad():
                    out = model(input_ids, output_hidden_states=True,
                                use_cache=False)
                logits = out.logits[0].float()
                lp = torch.log_softmax(logits[c0 - 1:c1 - 1], dim=-1)
                per[impl] = {
                    "cand_lp": lp.gather(1, tgt[:, None])[:, 0].cpu().numpy(),
                    "argmax": logits[c0 - 1:c1 - 1].argmax(-1).cpu().numpy(),
                    "h_last": out.hidden_states[-1][0, -1].float()
                              .cpu().numpy(),
                }
                del out, logits, lp

            d_lp = per["eager"]["cand_lp"] - per["sdpa"]["cand_lp"]
            ha, hb = per["eager"]["h_last"], per["sdpa"]["h_last"]
            cos = float(np.dot(ha, hb)
                        / (np.linalg.norm(ha) * np.linalg.norm(hb)))
            rows.append({
                "fork_id": fk["fork_id"], "role": role,
                "cand_len": int(c1 - c0),
                "max_abs_dlp": float(np.abs(d_lp).max()),
                "mean_abs_dlp": float(np.abs(d_lp).mean()),
                "mean_nll_eager": float(-per["eager"]["cand_lp"].mean()),
                "mean_nll_sdpa": float(-per["sdpa"]["cand_lp"].mean()),
                "greedy_agree": float(
                    (per["eager"]["argmax"] == per["sdpa"]["argmax"]).mean()),
                "hlast_cosine": cos,
            })

    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "parity_rows.csv", index=False)
    d_nll = df["mean_nll_eager"] - df["mean_nll_sdpa"]
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "model": args.model_name_or_path,
        "dtype": args.model_dtype,
        "device": str(device),
        "n_rows": len(df),
        "max_abs_dlp_p50": float(df["max_abs_dlp"].median()),
        "max_abs_dlp_max": float(df["max_abs_dlp"].max()),
        "mean_abs_dlp_p50": float(df["mean_abs_dlp"].median()),
        "d_mean_nll_mean": float(d_nll.mean()),
        "d_mean_nll_max_abs": float(d_nll.abs().max()),
        "greedy_agree_mean": float(df["greedy_agree"].mean()),
        "greedy_agree_min": float(df["greedy_agree"].min()),
        "hlast_cosine_min": float(df["hlast_cosine"].min()),
        "hlast_cosine_p50": float(df["hlast_cosine"].median()),
    }
    with open(args.out_dir / "parity.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
