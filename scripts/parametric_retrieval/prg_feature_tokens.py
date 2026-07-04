"""parametric_retrieval_geometry_v0: per-token activations of the top SAE
retrieval features (stage B of the feature anatomy).

extract mode (TamIA, GPU): runs every direct-mode WikiProfile prompt (QA +
completion controls) through Qwen2.5-7B-Instruct, reads hidden_states[hs_idx]
at EVERY prompt token, and computes the selected latents from
feature_pack_layer{B}.npz (encoder rows only; the 3.5 GB ae.pt is not needed):

    z_f(t) = relu(W_enc_f @ (h_t - b_dec) + b_enc_f), thresholded

  python scripts/parametric_retrieval/prg_feature_tokens.py extract \
      --out_dir runs/parametric_retrieval_geometry_v0 --local_files_only

analyze mode (local, after syncing feature_tokens.jsonl.gz back): prints and
writes, per feature,
  - top activating (token, context) snippets across all prompts
  - relative-position profile of the activation peak
  - fraction of peaks that land inside the subject-entity span

  python scripts/parametric_retrieval/prg_feature_tokens.py analyze \
      --out_dir runs/parametric_retrieval_geometry_v0

Outputs under <out_dir>/sae/: feature_tokens.jsonl.gz (extract),
feature_token_contexts.md (analyze).
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.analysis.parametric_retrieval import build_user_message  # noqa: E402


def extract(args) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    pack = np.load(args.out_dir / "sae" / f"feature_pack_layer{args.block}.npz")
    hs_idx = int(pack["hs_idx"])
    feats = pack["features"].tolist()
    md = pd.read_parquet(args.out_dir / "metadata.parquet")
    if args.limit:
        md = md.iloc[: args.limit]

    tok = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                        local_files_only=args.local_files_only)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, local_files_only=args.local_files_only,
            dtype=torch.bfloat16)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, local_files_only=args.local_files_only,
            torch_dtype=torch.bfloat16)
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device).eval()
    W = torch.tensor(pack["W_enc"], device=device, dtype=torch.float32)
    b_enc = torch.tensor(pack["b_enc"], device=device, dtype=torch.float32)
    b_dec = torch.tensor(pack["b_dec"], device=device, dtype=torch.float32)
    thr = float(pack["threshold"])

    enc = []
    for r in md.itertuples():
        ids = tok.apply_chat_template(
            [{"role": "user", "content":
              build_user_message(r.question, r.family, "direct")}],
            add_generation_prompt=True, tokenize=True, return_dict=False)
        if not isinstance(ids, list):
            ids = ids["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        enc.append(list(ids))

    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    out_path = args.out_dir / "sae" / "feature_tokens.jsonl.gz"
    order = sorted(range(len(enc)), key=lambda i: len(enc[i]))
    n_done = 0
    with gzip.open(out_path, "wt") as fout:
        for start in range(0, len(order), args.batch_size):
            idxs = order[start:start + args.batch_size]
            ids_list = [enc[i] for i in idxs]
            mx = max(len(x) for x in ids_list)
            inp = torch.tensor([x + [pad] * (mx - len(x)) for x in ids_list],
                               dtype=torch.long, device=device)
            att = torch.tensor([[1] * len(x) + [0] * (mx - len(x))
                                for x in ids_list], dtype=torch.long,
                               device=device)
            with torch.no_grad():
                o = model(inp, attention_mask=att,
                          output_hidden_states=True, use_cache=False)
                h = o.hidden_states[hs_idx].to(torch.float32)
                z = torch.relu((h - b_dec) @ W.T + b_enc)
                z = z * (z > thr)
            for b, i in enumerate(idxs):
                T = len(ids_list[b])
                r = md.iloc[i]
                fout.write(json.dumps({
                    "question_id": r.question_id,
                    "tokens": tok.convert_ids_to_tokens(ids_list[b]),
                    "acts": np.round(z[b, :T].cpu().numpy(), 2).tolist(),
                }) + "\n")
            n_done += len(idxs)
            if (start // args.batch_size) % 10 == 0:
                print(f"[ftok] {n_done}/{len(enc)}", flush=True)
    manifest = {"features": feats, "hs_idx": hs_idx, "block": int(args.block),
                "model": args.model_name_or_path, "n_prompts": len(enc)}
    (args.out_dir / "sae" / "feature_tokens_manifest.json").write_text(
        json.dumps(manifest, indent=2))
    print(f"[ftok] wrote {out_path} ({len(enc)} prompts, "
          f"{len(feats)} features)", flush=True)


def analyze(args) -> None:
    sae_dir = args.out_dir / "sae"
    manifest = json.loads((sae_dir / "feature_tokens_manifest.json")
                          .read_text())
    feats = manifest["features"]
    md = pd.read_parquet(args.out_dir / "metadata.parquet") \
        .set_index("question_id")
    recs = [json.loads(ln) for ln in
            gzip.open(sae_dir / "feature_tokens.jsonl.gz", "rt")]
    print(f"[ftok] {len(recs)} prompts, {len(feats)} features")

    def clean(t):  # BPE glyphs -> readable
        return t.replace("Ġ", " ").replace("Ċ", "\\n")

    # the chat template head/tail is token-identical across prompts; the
    # variable "question span" is everything between the shared prefix/suffix
    tok_lists = [r["tokens"] for r in recs]
    prefix = 0
    while all(len(t) > prefix and t[prefix] == tok_lists[0][prefix]
              for t in tok_lists):
        prefix += 1
    suffix = 0
    while all(len(t) > prefix + suffix
              and t[-1 - suffix] == tok_lists[0][-1 - suffix]
              for t in tok_lists):
        suffix += 1
    print(f"[ftok] template: {prefix} shared prefix tokens, "
          f"{suffix} shared suffix tokens")

    lines = [f"# Per-token contexts (hs{manifest['hs_idx']}, "
             f"block {manifest['block']})",
             f"\ntemplate span excluded from question-span stats: first "
             f"{prefix} and last {suffix} tokens\n"]
    for j, f in enumerate(feats):
        scored, scored_q = [], []
        pos_frac, in_subj = [], []
        for r in recs:
            a = np.asarray([row[j] for row in r["acts"]], dtype=np.float32)
            t = int(a.argmax())
            scored.append((float(a[t]), r, t))
            aq = a[prefix:len(a) - suffix]
            if len(aq):
                tq = int(aq.argmax()) + prefix
                scored_q.append((float(a[tq]), r, tq))
                pos_frac.append((tq - prefix) / max(1, len(aq) - 1))
                row = md.loc[r["question_id"]]
                subj_toks = str(row.subject).lower().split()
                tok_clean = clean(r["tokens"][tq]).strip().lower()
                in_subj.append(any(tok_clean and tok_clean in s
                                   for s in subj_toks))
        scored.sort(key=lambda x: -x[0])
        scored_q.sort(key=lambda x: -x[0])
        lines.append(f"\n## feature {f}")
        lines.append(f"question-span peak position: mean {np.mean(pos_frac):.2f}"
                     f" (1.0 = end of question); peak on subject-entity token: "
                     f"{np.mean(in_subj):.2f}")

        def ctx_lines(items, title):
            lines.append(f"\n**{title}:**")
            for v, r, t in items[:15]:
                toks = [clean(x) for x in r["tokens"]]
                lo, hi = max(0, t - 8), min(len(toks), t + 6)
                ctx = "".join(toks[lo:t]) + f" >>{toks[t].strip()}<< " \
                    + "".join(toks[t + 1:hi])
                lines.append(f"- [{v:.1f}] ({r['question_id']}) …{ctx}…")

        ctx_lines(scored_q, "top activating QUESTION tokens in context")
        ctx_lines(scored, "top activating tokens overall (template included)")
    out = sae_dir / "feature_token_contexts.md"
    out.write_text("\n".join(lines))
    print(f"[ftok] wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["extract", "analyze"])
    ap.add_argument("--out_dir", type=Path,
                    default=Path("runs/parametric_retrieval_geometry_v0"))
    ap.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--block", type=int, default=23)
    ap.add_argument("--batch_size", type=int, default=48)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if args.mode == "extract":
        extract(args)
    else:
        analyze(args)


if __name__ == "__main__":
    main()
