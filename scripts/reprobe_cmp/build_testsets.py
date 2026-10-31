#!/usr/bin/env python3
"""ReProbe's eight step-level test sets, from its released managers (reprobe_comparison_v1
and v2, S0), for Qwen3-8B (Table 1), Qwen3-32B (Table 5) or Phi-4 (Table 9).

For each set, downloads `ue_manager.pth` (login node; or --local <dir> holding
<set>/ue_manager.pth), unpickles it with an allowlist (src/data/reprobe_steps.py),
re-splits every reply into steps with the ported `StepsExtractor`, and writes:

  <out>/test/<set>.traces.jsonl  one trace per line (prompt, reply, reply token ids,
                                 steps with token alignment, labels, stored scores),
                                 read by scripts/reprobe_cmp/reprobe_probe.py
  <out>/test/<set>.items.jsonl   one step per line for the read token
                                 (scripts/read_token_inline.py --layout reprobe):
                                 uid, id, step_idx, problem, prefix, candidate_step, label
  <out>/test/gate_t.json         gate T: steps == labels, and the stored probe scores
                                 reproduce the paper's table (AP to 0.0005; sets
                                 whose stored probe is not in the paper are not gated)

Labels: 1 incorrect, 0 correct, -1 unlabeled (dropped from every metric).

    python scripts/reprobe_cmp/build_testsets.py --out $R/data [--backbone qwen3_32b]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.reprobe_steps import RP_AL_KEY, load_manager, manager_traces, step_item  # noqa: E402
from src.onpolicy.prompts import reprobe_prompt, reprobe_prompt_phi4  # noqa: E402

SUFFIX = "layer1_dim512_head16_e5_lr5e-4_pos3"
P4 = f"rediska0123/ue_manager_Phi-4_qwen3_anno_phi4_{SUFFIX}_on_"
Q32 = "JingweiNi/ue_manager_Qwen3-32B_hs_5e_on_"
# backbone -> tokenizer, prompt builder, the chat turn that ends the question, and
# set -> (manager repo, the paper's PR-AUC of the probe whose scores the manager stores,
# or None when the manager's probe is not in the paper)
BACKBONES = {
    # Table 1, "ReProbe, Attn+Logit, DeepSeek-anno"
    "qwen3_8b": ("Qwen/Qwen3-8B", reprobe_prompt, "<|im_end|>\n<|im_start|>assistant\n", {
        "math": (f"JingweiNi/ue_manager_math_no_prm800k_uhead_claim_Qwen3-8B_fixed_prm_{SUFFIX}", 0.465),
        "gsm8k": (f"rediska0123/ue_manager_gsm8k_uhead_claim_Qwen3-8B_fixed_prm_{SUFFIX}", 0.616),
        "proofnet": (f"rediska0123/ue_manager_proofnet_uhead_claim_Qwen3-8B_fixed_prm_{SUFFIX}", 0.243),
        "trips": (f"JingweiNi/ue_manager_fixed_prm_{SUFFIX}_Qwen3-8B_on_trip_plan", 0.740),
        "meetings": (f"JingweiNi/ue_manager_fixed_prm_{SUFFIX}_Qwen3-8B_on_meeting_plan", 0.802),
        "calendar": (f"JingweiNi/ue_manager_fixed_prm_{SUFFIX}_Qwen3-8B_on_calendar_plan", 0.786),
        "strategyqa": (f"JingweiNi/ue_manager_fixed_prm_{SUFFIX}_on_st_qa_cr", 0.395),
        "scienceqa": (f"JingweiNi/ue_manager_fixed_prm_{SUFFIX}_on_sci_qa_fixed", 0.361),
    }),
    # Table 5, "ReProbe, Hidden States, GPT-OSS-anno". The table prints the meeting
    # manager's value under Calendar and the calendar one under Meetings, and swaps
    # StrQA and SciQA the same way; the values here follow the managers' contents.
    "qwen3_32b": ("Qwen/Qwen3-32B", reprobe_prompt, "<|im_end|>\n<|im_start|>assistant\n", {
        "math": ("JingweiNi/ue_manager_Qwen3-32B_hs_4e_on_math", 0.676),
        "gsm8k": (Q32 + "gsm8k", 0.535),
        "proofnet": (Q32 + "proofnet", 0.682),
        "trips": (Q32 + "trip", 0.752),
        "meetings": (Q32 + "meeting", 0.698),
        "calendar": (Q32 + "calendar", 0.734),
        "strategyqa": (Q32 + "stqa", 0.303),
        "scienceqa": (Q32 + "sciqa", 0.302),
    }),
    # Table 9, "ReProbe, Attn+Logit, Qwen3-8B-anno". The Attn+Logit planning managers
    # carry no labels; the planning sets come from the hidden-state run's managers,
    # whose traces are a different sample and whose probe the paper does not report.
    "phi4": ("microsoft/phi-4", reprobe_prompt_phi4, "<|im_end|><|im_start|>assistant<|im_sep|>", {
        "math": (P4 + "math", 0.404),
        "gsm8k": (P4 + "gsm8k", 0.340),
        "proofnet": (P4 + "proofnet", 0.155),
        "trips": ("rediska0123/ue_manager_Phi-4_qwen3_anno_phi4_hs_5e_on_trip", None),
        "meetings": ("rediska0123/ue_manager_Phi-4_qwen3_anno_phi4_hs_5e_on_meeting", None),
        "calendar": ("rediska0123/ue_manager_Phi-4_qwen3_anno_phi4_hs_5e_on_calendar", None),
        "strategyqa": (P4 + "st_qa", 0.347),
        "scienceqa": (P4 + "sci_qa", 0.347),
    }),
}
SETS = BACKBONES["qwen3_8b"][3]


def ap(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, s))


def items_of(t: dict) -> list[dict]:
    out = []
    for k, (st, y) in enumerate(zip(t["steps"], t["labels"])):
        prefix, body = step_item(t["reply"], st)
        out.append({"uid": f"{t['trace_id']}::{k}", "id": t["trace_id"], "step_idx": k,
                    "problem": t["question"], "prefix": prefix, "candidate_step": body, "label": y})
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--local", type=Path, help="dir with <set>/ue_manager.pth instead of downloading")
    p.add_argument("--backbone", choices=sorted(BACKBONES), default="qwen3_8b")
    p.add_argument("--sets", default=",".join(SETS))
    a = p.parse_args()
    from transformers import AutoTokenizer
    tok_name, prompt_of, turn_end, sets = BACKBONES[a.backbone]
    tok = AutoTokenizer.from_pretrained(tok_name)
    question = re.compile(r"<Question>: (.*)" + re.escape(turn_end), re.S)
    d = a.out / "test"
    d.mkdir(parents=True, exist_ok=True)
    gate = {}
    for name in a.sets.split(","):
        repo, want = sets[name]
        if a.local is not None:
            path = a.local / name / "ue_manager.pth"
        else:
            from huggingface_hub import hf_hub_download
            path = Path(hf_hub_download(repo, "ue_manager.pth"))
        traces = manager_traces(load_manager(path), tok.decode, name)
        for t in traces:
            q = question.search(t["prompt_text"])
            if q is None or prompt_of(q.group(1)) != t["prompt_text"]:
                raise SystemExit(f"{t['trace_id']}: prompt is not {prompt_of.__name__}(question)")
            t["question"] = q.group(1)
        y = np.array([v for t in traces for v in t["labels"]])
        s = np.array([v for t in traces for v in t["scores"][RP_AL_KEY[1]]])
        lab = y >= 0
        got = ap(y[lab], s[lab])
        gate[name] = {"repo": repo, "traces": len(traces), "steps": int(len(y)), "labeled": int(lab.sum()),
                      "incorrect_rate": float(y[lab].mean()), "stored_ap": round(got, 4), "paper": want,
                      "pass": want is None or abs(got - want) <= 0.0005,
                      "stored_estimators": sorted(traces[0]["scores"])}
        with open(d / f"{name}.traces.jsonl", "w") as f:
            for t in traces:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
        with open(d / f"{name}.items.jsonl", "w") as f:
            for t in traces:
                for it in items_of(t):
                    f.write(json.dumps(it, ensure_ascii=False) + "\n")
        print(f"[testsets] {name}: {len(traces)} traces, {len(y)} steps ({int(lab.sum())} labeled, "
              f"{y[lab].mean():.3f} incorrect), stored probe AP {got:.4f} vs paper "
              f"{'-' if want is None else f'{want:.3f}'}", flush=True)
    (d / "gate_t.json").write_text(json.dumps(gate, indent=1))
    if not all(g["pass"] for g in gate.values()):
        raise SystemExit("gate T failed: " + ", ".join(k for k, g in gate.items() if not g["pass"]))
    print("[testsets] gate T passed", flush=True)


if __name__ == "__main__":
    main()
