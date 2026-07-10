"""parametric_retrieval_geometry_v0 exp 1b: TARGETED causal tests of the
answer-commitment feature, with every generation stored for inspection.

Two clean tests, each in the population where it is informative, each against
a matched-norm random-direction control:

  NECESSITY  population = direct_retrieval (model answers correctly).
             Clamp the feature DOWN (ablate to 0 / to non_retrieved baseline).
             Does the correct answer break? If it breaks more than a random
             edit of the same norm, the feature is causally necessary.

  SUFFICIENCY population = reasoning_unlocked (fails direct, CoT unlocks: the
             fact IS accessible). Clamp the feature UP (boost) on the DIRECT
             (no-CoT) prompt. Does it now retrieve without reasoning? A CoT arm
             additionally measures tokens-before-final-answer (does boosting
             let it answer in fewer steps?).

Clamp = read z_f at the decision point and set it to target via the SAE
decoder direction (see prg_build_clamp_pack.py); the random/dense controls
apply an edit of the SAME per-token magnitude along a different direction.

Every generation is written to sae/steer_targeted_generations.jsonl with the
baseline answer, the steered answer, correctness of each, the pre-edit feature
activation, and the induced activation, so flips can be visualised
(prg_steer_view.py builds an HTML page).

  python scripts/parametric_retrieval/prg_steer_targeted.py \
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
    extract_cot_final_answer,
    grade_answer,
)


def first_line(t: str) -> str:
    for ln in t.strip().splitlines():
        if ln.strip():
            return ln.strip()
    return ""


# (population, prompt_mode, feature_target_preset, direction) plan.
def build_plan(features):
    plan = []
    for f in features:
        # necessity: break a known answer
        for tgt in ["ablate", "baseline"]:
            plan.append(("known", "direct", f, tgt, "feature"))
            plan.append(("known", "direct", f, tgt, "random"))
        # sufficiency: retrieve an unlocked fact without CoT
        for tgt in ["boost_hi", "boost_xl"]:
            plan.append(("unlocked", "direct", f, tgt, "feature"))
            plan.append(("unlocked", "direct", f, tgt, "random"))
    return plan


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", type=Path,
                    default=Path("runs/parametric_retrieval_geometry_v0"))
    ap.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--block", type=int, default=23)
    ap.add_argument("--features", type=int, nargs="+", default=[58264, 88965])
    ap.add_argument("--n_per_pop", type=int, default=150)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--cot_boost_arm", action="store_true",
                    help="also run reasoning_unlocked CoT with/without boost "
                         "and record tokens-before-final-answer")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke_dim", type=int, default=None)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    grading = pd.DataFrame([json.loads(ln) for ln in
                            (args.out_dir / "grading.jsonl")
                            .read_text().splitlines() if ln.strip()])
    md = pd.read_parquet(args.out_dir / "metadata.parquet")
    g = grading[~grading.is_control].merge(
        md[["question_id", "question"]], on="question_id")
    pops = {
        "known": g[g.retrieval_class == "direct_retrieval"],
        "unlocked": g[g.retrieval_class == "reasoning_unlocked"],
    }
    sel = {}
    for name, df in pops.items():
        sel[name] = df.sample(min(args.n_per_pop, len(df)),
                              random_state=args.seed).reset_index(drop=True)
        print(f"[tsteer] {name}: {len(sel[name])} prompts", flush=True)

    # ---- clamp pack --------------------------------------------------------
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
        rng = np.random.default_rng(0)
        feats = args.features
        enc = {f: torch.tensor(rng.standard_normal(args.smoke_dim) * 0.02,
                               device=device, dtype=model.dtype)
               for f in feats}
        decc = {f: torch.nn.functional.normalize(
            torch.tensor(rng.standard_normal(args.smoke_dim), device=device,
                         dtype=model.dtype), dim=0) for f in feats}
        b_dec = torch.zeros(args.smoke_dim, device=device, dtype=model.dtype)
        b_enc = {f: 0.0 for f in feats}
        thr = 0.0
        targets = {f: {"ablate": 0.0, "baseline": 2.0, "boost_hi": 8.0,
                       "boost_xl": 20.0} for f in feats}
        randd = torch.nn.functional.normalize(
            torch.tensor(rng.standard_normal(args.smoke_dim), device=device,
                         dtype=model.dtype), dim=0)
    else:
        pk = np.load(args.out_dir / "sae" / f"clamp_pack_layer{args.block}.npz",
                     allow_pickle=True)
        assert pk["dec_cols"].shape[1] == dim
        feats = [int(x) for x in pk["features"]]
        enc = {f: torch.tensor(pk["enc_rows"][i], device=device,
                               dtype=model.dtype)
               for i, f in enumerate(feats)}
        decc = {f: torch.tensor(pk["dec_cols"][i], device=device,
                                dtype=model.dtype)
                for i, f in enumerate(feats)}
        b_enc_np = None  # b_enc folded via enc pack? recompute from ae not here
        b_dec = torch.tensor(pk["b_dec"], device=device, dtype=model.dtype)
        thr = float(pk["threshold"])
        targets = json.loads(str(pk["targets"]))
        targets = {int(k): v for k, v in targets.items()}
        b_enc = {f: 0.0 for f in feats}  # bias folded below via measured pack
        randd = torch.tensor(pk["random"], device=device, dtype=model.dtype)
        # encoder bias lives in the pack implicitly? recompute z uses enc row +
        # bias; we stored only enc rows, so read bias from ae is unavailable
        # here. Instead: bias was included when targets were derived; for the
        # hook we approximate z with enc row only + a per-feature bias read
        # from the pack if present.
        if "b_enc" in pk.files:
            b_enc = {f: float(pk["b_enc"][i]) for i, f in enumerate(feats)}

    def zf(hlast, f):  # (batch, dim) -> (batch,) feature activation
        pre = (hlast - b_dec) @ enc[f] + b_enc[f]
        z = torch.relu(pre)
        return torch.where(z > thr, z, torch.zeros_like(z))

    state = {"f": None, "target": None, "dir": None, "mode": None}

    def hook(_m, _i, out):
        if state["f"] is None:
            return out
        h = out[0] if isinstance(out, tuple) else out
        last = h[:, -1, :]
        z = zf(last, state["f"])
        delta = state["target"] - z              # (batch,)
        d = decc[state["f"]] if state["mode"] == "feature" else state["dir"]
        h[:, -1, :] = last + delta[:, None] * d[None, :]
        return (h, *out[1:]) if isinstance(out, tuple) else h

    handle = model.model.layers[args.block].register_forward_hook(hook)

    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def encode(df, mode):
        out = []
        for r in df.itertuples():
            ids = tok.apply_chat_template(
                [{"role": "user",
                  "content": build_user_message(r.question, r.family, mode)}],
                add_generation_prompt=True, tokenize=True, return_dict=False)
            if not isinstance(ids, list):
                ids = ids["input_ids"]
            if ids and isinstance(ids[0], list):
                ids = ids[0]
            out.append(list(ids))
        return out

    def generate(enc_ids, max_new):
        order = sorted(range(len(enc_ids)), key=lambda i: len(enc_ids[i]))
        txt = [None] * len(enc_ids)
        for s in range(0, len(order), args.batch_size):
            idxs = order[s:s + args.batch_size]
            ids_list = [enc_ids[i] for i in idxs]
            mx = max(len(x) for x in ids_list)
            inp = torch.tensor([[pad] * (mx - len(x)) + x for x in ids_list],
                               device=device)
            att = torch.tensor([[0] * (mx - len(x)) + [1] * len(x)
                                for x in ids_list], device=device)
            with torch.no_grad():
                seq = model.generate(inp, attention_mask=att,
                                     max_new_tokens=max_new, do_sample=False,
                                     pad_token_id=pad)
            for j, i in enumerate(idxs):
                txt[i] = tok.decode(seq[j, mx:], skip_special_tokens=True)
        return txt

    def measure_zf(enc_ids, f):
        """pre-edit feature activation at the final prompt token."""
        state["f"] = None
        vals = np.zeros(len(enc_ids), np.float32)
        for s in range(0, len(enc_ids), args.batch_size):
            ids_list = enc_ids[s:s + args.batch_size]
            mx = max(len(x) for x in ids_list)
            inp = torch.tensor([[pad] * (mx - len(x)) + x for x in ids_list],
                               device=device)
            att = torch.tensor([[0] * (mx - len(x)) + [1] * len(x)
                                for x in ids_list], device=device)
            with torch.no_grad():
                o = model(inp, attention_mask=att, output_hidden_states=True,
                          use_cache=False)
                h = o.hidden_states[args.block + 1][:, -1, :]
                vals[s:s + len(ids_list)] = zf(h, f).float().cpu().numpy()
        return vals

    # ---- baselines (no edit) ----------------------------------------------
    records, metrics = [], []
    enc_cache, base_txt, base_ok, base_z = {}, {}, {}, {}
    for pop in ["known", "unlocked"]:
        df = sel[pop]
        eids = encode(df, "direct")
        enc_cache[pop] = eids
        state["f"] = None
        bt = generate(eids, args.max_new_tokens)
        bo = [grade_answer(first_line(bt[i]), df.gold_answer.iloc[i])[0]
              for i in range(len(df))]
        base_txt[pop], base_ok[pop] = bt, bo
        base_z[pop] = {f: measure_zf(eids, f) for f in feats}
        print(f"[tsteer] baseline {pop}: retrieval "
              f"{np.mean(bo):.3f}", flush=True)

    # ---- interventions -----------------------------------------------------
    for pop, mode_p, f, tgt, mdir in build_plan(feats):
        df = sel[pop]
        eids = enc_cache[pop]
        state.update({"f": f, "target": float(targets[f][tgt]),
                      "mode": mdir,
                      "dir": (decc[f] if mdir == "feature" else randd)})
        txt = generate(eids, args.max_new_tokens)
        state["f"] = None
        ok = [grade_answer(first_line(txt[i]), df.gold_answer.iloc[i])[0]
              for i in range(len(df))]
        bz = base_z[pop][f]
        edit_norm = np.abs(float(targets[f][tgt]) - bz)
        for i in range(len(df)):
            records.append({
                "population": pop, "feature": f, "target_preset": tgt,
                "target_value": float(targets[f][tgt]), "direction": mdir,
                "question_id": df.question_id.iloc[i],
                "question": df.question.iloc[i],
                "gold": df.gold_answer.iloc[i],
                "baseline_answer": first_line(base_txt[pop][i]),
                "baseline_correct": bool(base_ok[pop][i]),
                "baseline_zf": float(bz[i]),
                "steered_answer": first_line(txt[i]),
                "steered_correct": bool(ok[i]),
                "edit_norm": float(edit_norm[i]),
            })
        # flip direction depends on population
        if pop == "known":
            flip = np.mean([(not ok[i]) and base_ok[pop][i]
                            for i in range(len(df))])  # broke
            flip_name = "broke_correct"
        else:
            flip = np.mean([ok[i] and (not base_ok[pop][i])
                            for i in range(len(df))])  # rescued
            flip_name = "rescued_incorrect"
        metrics.append({
            "population": pop, "feature": f, "target_preset": tgt,
            "direction": mdir, "n": len(df),
            "baseline_retrieval": float(np.mean(base_ok[pop])),
            "steered_retrieval": float(np.mean(ok)),
            flip_name: float(flip),
            "mean_edit_norm": float(edit_norm.mean())})
        print(f"[tsteer] {pop:8s} f{f} {tgt:9s} {mdir:7s} "
              f"base={np.mean(base_ok[pop]):.2f} steer={np.mean(ok):.2f} "
              f"{flip_name}={flip:.3f} |edit|~{edit_norm.mean():.0f}",
              flush=True)

    # ---- optional CoT-steps arm (sufficiency, fewer steps) ----------------
    if args.cot_boost_arm:
        df = sel["unlocked"]
        ceids = encode(df, "cot")
        for label, setf in [("cot_baseline", None),
                            ("cot_boost", feats[0])]:
            if setf is None:
                state["f"] = None
            else:
                state.update({"f": setf, "mode": "feature",
                              "target": float(targets[setf]["boost_hi"]),
                              "dir": decc[setf]})
            txt = generate(ceids, 256)
            state["f"] = None
            for i in range(len(df)):
                ans, _ = extract_cot_final_answer(txt[i])
                ok = grade_answer(ans, df.gold_answer.iloc[i])[0]
                n_tok = len(tok(txt[i].split("Final answer")[0])["input_ids"])
                records.append({
                    "population": "unlocked_cot", "feature": setf or -1,
                    "target_preset": label, "direction": "feature",
                    "question_id": df.question_id.iloc[i],
                    "question": df.question.iloc[i],
                    "gold": df.gold_answer.iloc[i], "baseline_answer": "",
                    "baseline_correct": False, "baseline_zf": 0.0,
                    "steered_answer": ans, "steered_correct": bool(ok),
                    "edit_norm": 0.0, "cot_tokens_to_answer": int(n_tok)})
            print(f"[tsteer] {label}: acc {np.mean([grade_answer(extract_cot_final_answer(txt[i])[0], df.gold_answer.iloc[i])[0] for i in range(len(df))]):.2f}",
                  flush=True)
    handle.remove()

    sae = args.out_dir / "sae"
    with open(sae / "steer_targeted_generations.jsonl", "w") as fo:
        for r in records:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    met = pd.DataFrame(metrics)
    met.to_csv(sae / "steer_targeted_metrics.csv", index=False)
    print("\n==== targeted steering metrics ====")
    print(met.to_string(index=False))
    print(f"\n[tsteer] wrote {sae}/steer_targeted_metrics.csv + "
          f"steer_targeted_generations.jsonl ({len(records)} rows)")


if __name__ == "__main__":
    main()
