"""parametric_retrieval_geometry_v0 exp 1c: activation patching, the strong
causal test of whether the reasoning_unlocked recall is transplantable.

Self-patch across conditions of the SAME reasoning_unlocked question (fact held
constant, so no cross-fact contamination):

  donor    = the question's greedy CoT run (which retrieves correctly, by the
             reasoning_unlocked label), block-B residual captured at the
             first_final_answer_token (answer onset: the model has, by then,
             done the reasoning and retrieved the fact).
  receiver = the question's DIRECT run (which fails). At the decision point
             (final prompt token) we OVERWRITE the receiver's block-B residual
             with the donor's answer-onset residual, then generate freely.

If transplanting the post-reasoning state makes the direct run retrieve, the
recall is present-as-a-vector and transplantable (revises "purely dynamic"
toward "dynamic to compute, static to carry"). If it does nothing beyond the
direct baseline, even a full-residual transplant fails -> the recall genuinely
needs the reasoning tokens in context (strong form of the dynamic conclusion).

Controls:
  mismatched donor  -- patch a DIFFERENT question's CoT residual; graded vs the
                       receiver's gold (should stay wrong) AND the donor's gold
                       (if it now says the donor's answer = pure copying).
  interpolate       -- alpha<1 partial patch, to see dose-response.

Modes patched at block(s) --blocks (default 23 = hidden_states[24]).

Every generation stored to sae/patch_generations.jsonl.

  python scripts/parametric_retrieval/prg_patch.py \
      --out_dir runs/parametric_retrieval_geometry_v0 \
      --model_name_or_path Qwen/Qwen2.5-7B-Instruct --local_files_only
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
    compute_positions,
    extract_cot_final_answer,
    grade_answer,
)


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
    ap.add_argument("--blocks", type=int, nargs="+", default=[23],
                    help="decoder blocks to patch (23 = hidden_states[24])")
    ap.add_argument("--alphas", type=float, nargs="+", default=[1.0, 0.5],
                    help="patch strength: 1.0 = full replace")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--direct_max_new_tokens", type=int, default=32)
    ap.add_argument("--cot_max_new_tokens", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=48)
    ap.add_argument("--fwd_batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    grading = pd.DataFrame([json.loads(ln) for ln in
                            (args.out_dir / "grading.jsonl")
                            .read_text().splitlines() if ln.strip()])
    md = pd.read_parquet(args.out_dir / "metadata.parquet")
    ru = grading[(~grading.is_control)
                 & (grading.retrieval_class == "reasoning_unlocked")]
    ru = ru.merge(md[["question_id", "question"]], on="question_id")
    ru = ru.sample(min(args.n, len(ru)), random_state=args.seed) \
        .reset_index(drop=True)
    print(f"[patch] {len(ru)} reasoning_unlocked questions", flush=True)

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
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    stop = {t for t in [tok.eos_token_id,
                        tok.convert_tokens_to_ids("<|im_end|>")]
            if t is not None and t >= 0}

    def render(q, fam, mode):
        ids = tok.apply_chat_template(
            [{"role": "user", "content": build_user_message(q, fam, mode)}],
            add_generation_prompt=True, tokenize=True, return_dict=False)
        if not isinstance(ids, list):
            ids = ids["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return list(ids)

    direct_p = [render(r.question, r.family, "direct") for r in ru.itertuples()]
    cot_p = [render(r.question, r.family, "cot") for r in ru.itertuples()]

    # ---- generation helpers (with optional per-row decision-point patch) ---
    state = {"vecs": None, "alpha": 1.0, "block": None}
    handles = []

    def make_hook(b):
        def hook(_m, _i, out):
            if state["vecs"] is None or state["block"] != b:
                return out
            h = out[0] if isinstance(out, tuple) else out
            if h.shape[1] > 1:  # prefill only: patch the decision-point token
                a = state["alpha"]
                h[:, -1, :] = (1 - a) * h[:, -1, :] + a * state["vecs"]
            return (h, *out[1:]) if isinstance(out, tuple) else h
        return hook

    for b in set(args.blocks):
        handles.append(model.model.layers[b].register_forward_hook(make_hook(b)))

    def generate(enc_ids, max_new, vecs=None, alpha=1.0, block=None):
        order = sorted(range(len(enc_ids)), key=lambda i: len(enc_ids[i]))
        txt = [None] * len(enc_ids)
        gen_ids = [None] * len(enc_ids)
        for s in range(0, len(order), args.batch_size):
            idxs = order[s:s + args.batch_size]
            ids_list = [enc_ids[i] for i in idxs]
            mx = max(len(x) for x in ids_list)
            inp = torch.tensor([[pad] * (mx - len(x)) + x for x in ids_list],
                               device=device)
            att = torch.tensor([[0] * (mx - len(x)) + [1] * len(x)
                                for x in ids_list], device=device)
            if vecs is not None:
                state.update({"vecs": torch.stack([vecs[i] for i in idxs]),
                              "alpha": alpha, "block": block})
            else:
                state["vecs"] = None
            with torch.no_grad():
                seq = model.generate(inp, attention_mask=att,
                                     max_new_tokens=max_new, do_sample=False,
                                     pad_token_id=pad)
            state["vecs"] = None
            for j, i in enumerate(idxs):
                g = seq[j, mx:].tolist()
                for c, t in enumerate(g):
                    if t in stop:
                        g = g[:c]
                        break
                gen_ids[i] = g
                txt[i] = tok.decode(g, skip_special_tokens=True)
        return txt, gen_ids

    # ---- baselines ---------------------------------------------------------
    d_txt, _ = generate(direct_p, args.direct_max_new_tokens)
    d_ok = [grade_answer(first_line(d_txt[i]), ru.gold_answer.iloc[i])[0]
            for i in range(len(ru))]
    c_txt, c_gen = generate(cot_p, args.cot_max_new_tokens)
    c_ans = [extract_cot_final_answer(c_txt[i])[0] for i in range(len(ru))]
    c_ok = [grade_answer(c_ans[i], ru.gold_answer.iloc[i])[0]
            for i in range(len(ru))]
    print(f"[patch] baselines: direct {np.mean(d_ok):.3f} / "
          f"CoT donor {np.mean(c_ok):.3f}", flush=True)

    # ---- capture donor residuals at CoT answer-onset -----------------------
    # locate first_final_answer_token per CoT run, then one teacher-forced
    # forward over prompt+gen reading block-B residual there.
    donor_pos, seqs = [], []
    for i in range(len(ru)):
        g = c_gen[i]
        offs, prev = [], 0
        for c in range(len(g)):
            cur = max(len(tok.decode(g[:c + 1], skip_special_tokens=True)), prev)
            offs.append(cur)
            prev = cur
        pos = {p["position_name"]: p["token_index"]
               for p in compute_positions(len(cot_p[i]), len(g), c_txt[i],
                                          offs, "cot")}
        ti = pos.get("first_final_answer_token", pos.get("final_answer_token",
                                                         len(cot_p[i]) + len(g) - 1))
        donor_pos.append(ti)
        seqs.append(cot_p[i] + list(g))

    donor = {b: torch.zeros(len(ru), model.config.hidden_size, device=device,
                            dtype=model.dtype) for b in args.blocks}
    state["vecs"] = None
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    for s in range(0, len(order), args.fwd_batch_size):
        idxs = order[s:s + args.fwd_batch_size]
        ids_list = [seqs[i] for i in idxs]
        mx = max(len(x) for x in ids_list)
        inp = torch.tensor([x + [pad] * (mx - len(x)) for x in ids_list],
                           device=device)
        att = torch.tensor([[1] * len(x) + [0] * (mx - len(x))
                            for x in ids_list], device=device)
        with torch.no_grad():
            o = model(inp, attention_mask=att, output_hidden_states=True,
                      use_cache=False)
        for j, i in enumerate(idxs):
            ti = min(donor_pos[i], len(seqs[i]) - 1)
            for b in args.blocks:
                donor[b][i] = o.hidden_states[b + 1][j, ti]

    # ---- patch conditions --------------------------------------------------
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(ru))
    while np.any(perm == np.arange(len(ru))):  # derangement
        perm = rng.permutation(len(ru))

    records, metrics = [], []
    for b in args.blocks:
        for alpha in args.alphas:
            for donor_kind in ["matched", "mismatched"]:
                vecs = donor[b] if donor_kind == "matched" \
                    else donor[b][torch.tensor(perm, device=device)]
                p_txt, _ = generate(direct_p, args.direct_max_new_tokens,
                                    vecs=[vecs[i] for i in range(len(ru))],
                                    alpha=alpha, block=b)
                for i in range(len(ru)):
                    ans = first_line(p_txt[i])
                    ok_recv = grade_answer(ans, ru.gold_answer.iloc[i])[0]
                    donor_gold = ru.gold_answer.iloc[
                        perm[i] if donor_kind == "mismatched" else i]
                    ok_donor = grade_answer(ans, donor_gold)[0]
                    records.append({
                        "block": b, "alpha": alpha, "donor": donor_kind,
                        "question_id": ru.question_id.iloc[i],
                        "question": ru.question.iloc[i],
                        "gold": ru.gold_answer.iloc[i],
                        "donor_gold": donor_gold,
                        "direct_baseline": first_line(d_txt[i]),
                        "direct_correct": bool(d_ok[i]),
                        "cot_donor_answer": c_ans[i],
                        "cot_correct": bool(c_ok[i]),
                        "patched_answer": ans,
                        "patched_correct_recv": bool(ok_recv),
                        "patched_correct_donor": bool(ok_donor)})
                rec_rate = np.mean([grade_answer(first_line(p_txt[i]),
                                    ru.gold_answer.iloc[i])[0]
                                    for i in range(len(ru))])
                copy_rate = np.mean([
                    grade_answer(first_line(p_txt[i]),
                                 ru.gold_answer.iloc[perm[i]])[0]
                    for i in range(len(ru))]) if donor_kind == "mismatched" \
                    else float("nan")
                metrics.append({
                    "block": b, "alpha": alpha, "donor": donor_kind,
                    "n": len(ru),
                    "direct_baseline": float(np.mean(d_ok)),
                    "cot_donor": float(np.mean(c_ok)),
                    "patched_retrieval_recv": float(rec_rate),
                    "patched_copies_donor": float(copy_rate)})
                print(f"[patch] b{b} a={alpha} {donor_kind:10s} "
                      f"recv={rec_rate:.3f} copy_donor={copy_rate}",
                      flush=True)
    for h in handles:
        h.remove()

    sae = args.out_dir / "sae"
    sae.mkdir(parents=True, exist_ok=True)
    with open(sae / "patch_generations.jsonl", "w") as fo:
        for r in records:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    met = pd.DataFrame(metrics)
    met.to_csv(sae / "patch_metrics.csv", index=False)
    print("\n==== activation patching ====")
    print(met.to_string(index=False))
    print(f"\n[patch] wrote {sae}/patch_metrics.csv + patch_generations.jsonl")


if __name__ == "__main__":
    main()
