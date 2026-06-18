"""Build a human-verification slice of the failure-mode labels.

Oversamples the headline cells (arithmetic_error, unsupported_premise) plus a
spread of the other common modes, and emits:
  - verify_slice.html  readable cards with MathJax (problem, prior steps, the
                       marked error step, the predicted label + rationale)
  - verify_slice.csv   editable: fill `your_verdict` (ok/wrong) and, if wrong,
                       `correct_mode`; I read it back to correct the labels.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

import numpy as np

from src.eval.failure_taxonomy import FAILURE_MODES

ROOT = Path("results/s3_first_error")

# per-mode quota for the slice (capped by availability)
QUOTA = {
    "arithmetic_error": 10,
    "unsupported_premise": 6,
    "logical_inference_error": 6,
    "variable_or_entity_binding_error": 4,
    "goal_drift": 3,
    "post_hoc_reasoning": 2,
    "algebraic_transformation_error": 2,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="failure_labels_opus.jsonl",
                    help="label file under results/s3_first_error/ to verify")
    ap.add_argument("--compare", default="failure_labels.jsonl",
                    help="other label file, to flag contested cases (or '' to skip)")
    args = ap.parse_args()

    rng = np.random.default_rng(42)
    labels = {json.loads(l)["sample_id"]: json.loads(l)
              for l in (ROOT / args.labels).read_text().splitlines() if l}
    other = {}
    if args.compare and (ROOT / args.compare).exists():
        other = {json.loads(l)["sample_id"]: json.loads(l)["failure_mode"]
                 for l in (ROOT / args.compare).read_text().splitlines() if l}
    sample = {json.loads(l)["sample_id"]: json.loads(l)
              for l in (ROOT / "first_error_sample.jsonl").read_text().splitlines() if l}

    chosen: list[str] = []
    for mode, q in QUOTA.items():
        ids = [sid for sid, r in labels.items() if r["failure_mode"] == mode]
        # within a mode, prefer a mix of detected and missed
        det = [s for s in ids if sample[s]["detected"]]
        mis = [s for s in ids if not sample[s]["detected"]]
        rng.shuffle(det); rng.shuffle(mis)
        pick = (mis[: (q + 1) // 2] + det[: q // 2])[:q]
        chosen.extend(pick)

    def contested(sid: str) -> str:
        return "yes" if other and other.get(sid) and other[sid] != labels[sid]["failure_mode"] else ""

    # ---- editable CSV ----
    with (ROOT / "verify_slice.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "subset", "detected", "predicted_mode", "contested",
                    "your_verdict (ok/wrong)", "correct_mode (if wrong)", "rationale"])
        for sid in chosen:
            r = labels[sid]; s = sample[sid]
            w.writerow([sid, s["subset"], s["detected"], r["failure_mode"],
                        contested(sid), "", "", r["rationale"]])

    # ---- readable HTML with MathJax ----
    def esc(s: str) -> str:
        return html.escape(s or "")

    cards = []
    for i, sid in enumerate(chosen, 1):
        r = labels[sid]; s = sample[sid]
        prior = "".join(
            f"<div class=prior><b>Step {j}.</b> {esc(t)}</div>"
            for j, t in enumerate(s["prior_steps"])
        ) or "<div class=prior><i>(no prior steps)</i></div>"
        cards.append(f"""
<div class=card>
  <div class=hdr>#{i} &middot; <code>{sid}</code> &middot; {s['subset']}
    &middot; probe {'DETECTED' if s['detected'] else 'MISSED'} (score {s['probe_score']})
    {'&middot; <b style=color:#b00>CONTESTED</b>' if contested(sid) else ''}</div>
  <div class=label>predicted: <b>{r['failure_mode']}</b> ({r.get('confidence','')})
    &mdash; {esc(r.get('rationale',''))}</div>
  <div class=problem><b>Problem.</b> {esc(s['problem'])}</div>
  {prior}
  <div class=err><b>First-error step (Step {s['step_idx']}).</b> {esc(s['error_step'])}</div>
</div>""")

    html_doc = f"""<!doctype html><html><head><meta charset=utf-8>
<title>S3 failure-label verification ({len(chosen)} steps)</title>
<script>window.MathJax={{tex:{{inlineMath:[['$','$'],['\\\\(','\\\\)']],
 displayMath:[['$$','$$'],['\\\\[','\\\\]']]}}}};</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>
body{{font:15px/1.5 -apple-system,system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;color:#222}}
.card{{border:1px solid #ddd;border-radius:8px;padding:1rem;margin:1rem 0;background:#fafafa}}
.hdr{{font-size:13px;color:#666;margin-bottom:.5rem}}
.label{{background:#fff3cd;padding:.4rem .6rem;border-radius:6px;margin-bottom:.6rem}}
.problem{{margin:.4rem 0}}
.prior{{color:#555;font-size:14px;margin:.2rem 0 .2rem 1rem}}
.err{{background:#f8d7da;padding:.5rem .6rem;border-radius:6px;margin-top:.5rem}}
code{{background:#eee;padding:0 4px;border-radius:3px}}
</style></head><body>
<h2>S3 failure-label verification &mdash; {len(chosen)} steps</h2>
<p>Predicted label (yellow) is the Haiku judge's call. Error step is red. Mark your
verdict in <code>verify_slice.csv</code>.</p>
{''.join(cards)}
</body></html>"""
    (ROOT / "verify_slice.html").write_text(html_doc)

    # composition report
    from collections import Counter
    comp = Counter(labels[s]["failure_mode"] for s in chosen)
    print(f"[verify] slice = {len(chosen)} steps")
    for m in FAILURE_MODES:
        if comp[m]:
            print(f"  {m:34s} {comp[m]}")
    print(f"[verify] open {ROOT}/verify_slice.html ; edit {ROOT}/verify_slice.csv")


if __name__ == "__main__":
    main()
