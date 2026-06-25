"""Print the step text for the 7B dense-probe's most-confident held-out calls.

The dense linear probe (Qwen2.5-7B, L28/last token) scores each PRM800K
held-out step as P(incorrect) = sigmoid(w.h + b). This script takes the 20
steps the probe was most confident about -- the top-10 highest-score INCORRECT
steps (easiest errors to flag) and the top-10 lowest-score CORRECT steps
(easiest to clear) -- and joins them by uid to their text in the encoder-input
jsonl so we can read what kind of steps are easy to detect.

The (uid, score, rating, step_idx, n_tokens) rows are baked in: they were
computed locally from
  runs/s1_model_size_dense/qwen2_5_7b/{linear_probe.pt, prm_multitoken/...}
and reproduce results/prm800k_heldout_eval/7B.json exactly
(mean score 0.6248 incorrect / 0.3039 correct, AUC 0.811).

Run on TAMIA where the source jsonl lives:
  python scripts/inspect_heldout_easy_steps.py
  HELDOUT_JSONL=/path/to/prm800k_heldout_test.jsonl python scripts/inspect_heldout_easy_steps.py
"""

import json
import os
import statistics
import textwrap
from pathlib import Path

# group, uid, score=P(incorrect), rating, step_idx, n_tokens (prompt+step)
ROWS = [
    ["inc", "prm800k::p1_03fc51f6::s1::18::0", 0.9983, -1, 18, 717],
    ["inc", "prm800k::p524_9c03abed::s524::5::4", 0.9982, -1, 5, 440],
    ["inc", "prm800k::p438_550b1a4c::s438::10::3", 0.9966, -1, 10, 792],
    ["inc", "prm800k::p346_550b1a4c::s346::8::2", 0.9964, -1, 8, 679],
    ["inc", "prm800k::p1_03fc51f6::s1::18::3", 0.9958, -1, 18, 784],
    ["inc", "prm800k::p811_5a258a23::s811::2::1", 0.9943, -1, 2, 385],
    ["inc", "prm800k::p2499_9d135f32::s2499::8::4", 0.9940, -1, 8, 353],
    ["inc", "prm800k::p524_9c03abed::s524::5::2", 0.9939, -1, 5, 285],
    ["inc", "prm800k::p325_34c7f09e::s325::2::0", 0.9938, -1, 2, 332],
    ["inc", "prm800k::p1279_21077807::s1279::2::1", 0.9933, -1, 2, 168],
    ["cor", "prm800k::p1295_59930f06::s1295::0::0", 0.0008, 1, 0, 67],
    ["cor", "prm800k::p2041_2c3bc600::s2041::1::0", 0.0014, 1, 1, 104],
    ["cor", "prm800k::p2544_eb3187c3::s2544::0::1", 0.0023, 1, 0, 77],
    ["cor", "prm800k::p184_b787f6e0::s184::1::0", 0.0023, 1, 1, 128],
    ["cor", "prm800k::p2261_b4bdce0b::s2261::2::0", 0.0029, 1, 2, 97],
    ["cor", "prm800k::p1897_6a73c9a2::s1897::1::0", 0.0030, 1, 1, 69],
    ["cor", "prm800k::p189_b787f6e0::s189::6::0", 0.0032, 1, 6, 203],
    ["cor", "prm800k::p1778_d41930bf::s1778::0::0", 0.0034, 1, 0, 99],
    ["cor", "prm800k::p1096_2cd77149::s1096::1::0", 0.0036, 1, 1, 162],
    ["cor", "prm800k::p189_b787f6e0::s189::2::0", 0.0038, 1, 2, 118],
]

DEFAULT_SRC = "$SCRATCH/cot_mech/prestudy_v1/data/prm800k_heldout_test.jsonl"


def main() -> None:
    src = Path(os.path.expandvars(os.environ.get("HELDOUT_JSONL", DEFAULT_SRC)))
    if not src.exists():
        raise SystemExit(f"not found: {src}  (set HELDOUT_JSONL=/path/to.jsonl)")

    want = {r[1] for r in ROWS}
    found: dict[str, dict] = {}
    for line in src.open():
        if not line.strip():
            continue
        o = json.loads(line)
        if o.get("uid") in want:
            found[o["uid"]] = o
    print(f"matched {len(found)}/{len(ROWS)} uids   (source: {src})")
    missing = [u for u in want if u not in found]
    if missing:
        print("MISSING:", *missing, sep="\n  ")

    def show(title: str, group: str) -> None:
        print("\n" + "=" * 90 + f"\n{title}\n" + "=" * 90)
        for _, uid, score, rating, step_idx, n_tok in [r for r in ROWS if r[0] == group]:
            o = found.get(uid, {})
            step = (o.get("candidate_step") or "<text not found>").strip()
            problem = (o.get("problem") or "").strip()
            prefix = (o.get("prefix") or "").strip()
            print(f"\n[score={score:.4f} rating={rating} step_idx={step_idx} "
                  f"ntok={n_tok}] {uid}")
            if problem:
                print("  PROBLEM:", textwrap.shorten(problem, 300))
            if prefix:
                print("  PREFIX (tail):", "..." + prefix[-240:].replace("\n", " / "))
            print("  >>> STEP:", "\n           ".join(textwrap.wrap(step, 108)[:16]))

    show("TOP 10 INCORRECT  (highest P(incorrect) -- easiest errors to flag)", "inc")
    show("TOP 10 CORRECT  (lowest P(incorrect) -- easiest to clear)", "cor")

    print("\n" + "#" * 90)
    print("STRUCTURE CHECK (content vs. position/length)")
    print("#" * 90)
    for group, label in [("inc", "easy-INCORRECT"), ("cor", "easy-CORRECT")]:
        sidx = [r[4] for r in ROWS if r[0] == group]
        ntok = [r[5] for r in ROWS if r[0] == group]
        chars = [len((found.get(r[1], {}).get("candidate_step") or "")) for r in ROWS if r[0] == group]
        print(f"{label:15s} step_idx med={statistics.median(sidx):>4} "
              f"[{min(sidx)}-{max(sidx)}]   prompt_ntok med={statistics.median(ntok):>5} "
              f"[{min(ntok)}-{max(ntok)}]   step_chars med={statistics.median(chars):.0f}")


if __name__ == "__main__":
    main()
