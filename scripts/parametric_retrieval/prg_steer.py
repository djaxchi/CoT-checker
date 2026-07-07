"""parametric_retrieval_geometry_v0 exp 1: causal steering at the decision
point.

Adds alpha * direction to the block-23 residual stream (= hidden_states[24])
at the decision-point positions only (the final prompt token on prefill, then
each generated answer token), during greedy direct-answer generation, and
measures whether retrieval behavior moves.

Directions (from prg_build_steer_dirs.py): sae_dec_<feat> (inject the answer-
commitment feature), dense_diff (the non-sparse retrieval axis, control),
random (matched-norm perturbation, control). +alpha points retrieval-ward for
every arm by construction.

Read: a causal LEVER shows retrieval rate rising with +alpha on the failing
classes (non_retrieved / unstable / reasoning_unlocked) and falling with
-alpha on direct_retrieval, specifically for the SAE direction and NOT for
random. A GAUGE shows flat curves (matches the S3 Stage 5 probe-steering null).

  # cluster:
  python scripts/parametric_retrieval/prg_steer.py \
      --out_dir runs/parametric_retrieval_geometry_v0 \
      --model_name_or_path Qwen/Qwen2.5-7B-Instruct --local_files_only

Outputs sae/steer_results.csv (+ steer_examples.jsonl for a few flips).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.analysis.parametric_retrieval import (  # noqa: E402
    build_user_message,
    grade_answer,
)

CLASSES = ["direct_retrieval", "reasoning_unlocked", "unstable_retrieval",
           "non_retrieved"]


def first_line(t: str) -> str:
    for ln in t.strip().splitlines():
        if ln.strip():
            return ln.strip()
    return ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", type=Path,
                    default=Path("runs/parametric_retrieval_geometry_v0"))
    ap.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--dirs", type=Path, default=None,
                    help="steer_dirs npz; default sae/steer_dirs_layer23.npz")
    ap.add_argument("--block", type=int, default=23)
    ap.add_argument("--arms", nargs="+",
                    default=["sae_dec_58264", "sae_dec_88965", "dense_diff",
                             "random"])
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[-40, -20, -10, 10, 20, 40])
    ap.add_argument("--n_per_class", type=int, default=100)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke_dim", type=int, default=None,
                    help="ignore --dirs, use random unit dirs of this dim")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rng = np.random.default_rng(args.seed)

    # ---- steer set: stratified across classes ------------------------------
    grading = pd.DataFrame([json.loads(ln) for ln in
                            (args.out_dir / "grading.jsonl")
                            .read_text().splitlines() if ln.strip()])
    md = pd.read_parquet(args.out_dir / "metadata.parquet")
    g = grading[~grading.is_control].merge(
        md[["question_id", "question"]], on="question_id")
    parts = []
    for c in CLASSES:
        sub = g[g.retrieval_class == c]
        parts.append(sub.sample(min(args.n_per_class, len(sub)),
                                 random_state=args.seed))
    steer = pd.concat(parts).reset_index(drop=True)
    print(f"[steer] {len(steer)} prompts: "
          f"{dict(steer.retrieval_class.value_counts())}", flush=True)

    # ---- model ------------------------------------------------------------
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
    dim = int(model.config.hidden_size)

    if args.smoke_dim:
        names = args.arms
        mat = rng.standard_normal((len(names), args.smoke_dim)).astype(
            np.float32)
        mat /= np.linalg.norm(mat, axis=1, keepdims=True)
        dirs = dict(zip(names, mat))
    else:
        dpath = args.dirs or (args.out_dir / "sae"
                              / f"steer_dirs_layer{args.block}.npz")
        z = np.load(dpath, allow_pickle=True)
        dirs = dict(zip([str(n) for n in z["names"]], z["mat"]))
        assert z["mat"].shape[1] == dim, \
            f"dirs dim {z['mat'].shape[1]} != model hidden {dim}"
    dvecs = {n: torch.tensor(v, device=device, dtype=model.dtype)
             for n, v in dirs.items()}

    # ---- decision-point steering hook -------------------------------------
    state = {"vec": None}  # set to (alpha * dvec) tensor to enable

    def hook(_module, _inp, out):
        if state["vec"] is None:
            return out
        h = out[0] if isinstance(out, tuple) else out
        h[:, -1, :] = h[:, -1, :] + state["vec"]  # last pos: prompt final /
        return (h, *out[1:]) if isinstance(out, tuple) else h  # gen token

    handle = model.model.layers[args.block].register_forward_hook(hook)

    prompts = [build_user_message(q, f, "direct")
               for q, f in zip(steer.question, steer.family)]
    enc = []
    for p in prompts:
        ids = tok.apply_chat_template([{"role": "user", "content": p}],
                                      add_generation_prompt=True,
                                      tokenize=True, return_dict=False)
        if not isinstance(ids, list):
            ids = ids["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        enc.append(list(ids))
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    order = sorted(range(len(enc)), key=lambda i: len(enc[i]))

    def generate_all(vec) -> list[str]:
        state["vec"] = vec
        out_txt = [None] * len(enc)
        for s in range(0, len(order), args.batch_size):
            idxs = order[s:s + args.batch_size]
            ids_list = [enc[i] for i in idxs]
            mx = max(len(x) for x in ids_list)
            inp = torch.tensor([[pad] * (mx - len(x)) + x for x in ids_list],
                               device=device)
            att = torch.tensor([[0] * (mx - len(x)) + [1] * len(x)
                                for x in ids_list], device=device)
            with torch.no_grad():
                seq = model.generate(inp, attention_mask=att,
                                     max_new_tokens=args.max_new_tokens,
                                     do_sample=False, pad_token_id=pad)
            for j, i in enumerate(idxs):
                out_txt[i] = tok.decode(seq[j, mx:], skip_special_tokens=True)
        return out_txt

    # ---- run baseline + every (arm, alpha) --------------------------------
    def score(texts) -> pd.DataFrame:
        rows = []
        for i, r in steer.iterrows():
            ok = grade_answer(first_line(texts[i]), r.gold_answer)[0]
            rows.append({"start_class": r.retrieval_class, "correct": ok})
        return pd.DataFrame(rows)

    print("[steer] baseline (alpha=0)", flush=True)
    base = score(generate_all(None))
    results = []
    for cls, sub in base.groupby("start_class"):
        results.append({"arm": "baseline", "alpha": 0.0, "start_class": cls,
                        "n": len(sub), "retrieval_rate": sub.correct.mean()})
    examples = []
    for arm in args.arms:
        for a in args.alphas:
            vec = dvecs[arm] * float(a)
            texts = generate_all(vec)
            sc = score(texts)
            for cls, sub in sc.groupby("start_class"):
                results.append({"arm": arm, "alpha": float(a),
                                "start_class": cls, "n": len(sub),
                                "retrieval_rate": float(sub.correct.mean())})
            flips = sc[(sc.start_class == "non_retrieved") & sc.correct]
            for i in flips.index[:2]:
                examples.append({"arm": arm, "alpha": float(a),
                                 "question": steer.question.iloc[i],
                                 "gold": steer.gold_answer.iloc[i],
                                 "steered_answer": first_line(texts[i])})
            print(f"[steer] {arm:14s} a={a:+.0f}  "
                  + " ".join(f"{c[:2]}={sc[sc.start_class==c].correct.mean():.2f}"
                             for c in CLASSES), flush=True)
    handle.remove()

    res = pd.DataFrame(results)
    out_csv = args.out_dir / "sae" / "steer_results.csv"
    res.to_csv(out_csv, index=False)
    (args.out_dir / "sae" / "steer_examples.jsonl").write_text(
        "\n".join(json.dumps(e) for e in examples))
    print("\n==== retrieval rate by arm x alpha x start_class ====")
    piv = res.pivot_table(index=["arm", "alpha"], columns="start_class",
                          values="retrieval_rate")
    print(piv.round(3).to_string())
    print(f"\n[steer] wrote {out_csv}")


if __name__ == "__main__":
    main()
