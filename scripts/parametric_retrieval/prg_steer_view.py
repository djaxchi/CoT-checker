"""parametric_retrieval_geometry_v0: build an HTML viewer for the targeted
steering generations, so the impact of the intervention is inspectable.

Groups the stored generations by (population, feature, target, direction) and
renders, for each condition, the baseline vs steered answer per question with
correctness colouring and the flips highlighted. A summary table at the top
gives baseline/steered retrieval and flip rate per condition.

  python scripts/parametric_retrieval/prg_steer_view.py \
      --out_dir runs/parametric_retrieval_geometry_v0
Writes sae/steer_view.html (open locally).
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import pandas as pd


def esc(s):
    return html.escape(str(s))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", type=Path,
                    default=Path("runs/parametric_retrieval_geometry_v0"))
    args = ap.parse_args()
    sae = args.out_dir / "sae"
    recs = [json.loads(ln) for ln in
            (sae / "steer_targeted_generations.jsonl").read_text().splitlines()
            if ln.strip()]
    df = pd.DataFrame(recs)
    met = pd.read_csv(sae / "steer_targeted_metrics.csv")

    css = """
    body{background:#1a1a19;color:#e8e6e1;font:13px/1.5 -apple-system,Segoe UI,
    Roboto,sans-serif;margin:0;padding:20px 26px}
    h1{font-size:17px}h2{font-size:14px;margin:22px 0 6px;color:#b0afab}
    table{border-collapse:collapse;font-size:12px;margin:8px 0 18px}
    th,td{padding:4px 10px;border-bottom:1px solid #3a3a37;text-align:left}
    th{color:#6f6e69}
    .ok{color:#199e70}.bad{color:#e34948}
    .flip{background:#3a2a12}.rescue{background:#123a24}
    .q{color:#b0afab;max-width:420px}
    .ans{font-family:ui-monospace,monospace}
    details{margin:6px 0}summary{cursor:pointer;color:#9ec5f4}
    .pill{display:inline-block;padding:1px 7px;border-radius:9px;font-size:11px;
    border:1px solid #3a3a37;margin-right:6px}
    """
    out = [f"<!doctype html><meta charset=utf-8><style>{css}</style>",
           "<h1>Targeted steering: baseline vs steered generations</h1>"]

    # ---- summary ----------------------------------------------------------
    out.append("<h2>Summary (feature vs random control)</h2><table>")
    out.append("<tr><th>population</th><th>feature</th><th>target</th>"
               "<th>dir</th><th>baseline</th><th>steered</th><th>flip</th>"
               "<th>|edit|</th></tr>")
    flipcol = "broke_correct" if "broke_correct" in met else None
    for _, r in met.iterrows():
        flip = (r.get("broke_correct") if not pd.isna(r.get("broke_correct",
                float("nan"))) else r.get("rescued_incorrect"))
        out.append(
            f"<tr><td>{esc(r.population)}</td><td>{r.feature}</td>"
            f"<td>{esc(r.target_preset)}</td><td>{esc(r.direction)}</td>"
            f"<td>{r.baseline_retrieval:.2f}</td>"
            f"<td>{r.steered_retrieval:.2f}</td>"
            f"<td>{flip:.3f}</td><td>{r.mean_edit_norm:.0f}</td></tr>")
    out.append("</table>")

    # ---- per-condition generation lists -----------------------------------
    for (pop, f, tgt, mdir), grp in df.groupby(
            ["population", "feature", "target_preset", "direction"]):
        if pop.endswith("_cot"):
            continue
        broke = grp[(grp.baseline_correct) & (~grp.steered_correct)]
        rescued = grp[(~grp.baseline_correct) & (grp.steered_correct)]
        head = (f"{pop} · feat {f} · {tgt} · {mdir} "
                f"&nbsp;<span class=pill>broke {len(broke)}</span>"
                f"<span class=pill>rescued {len(rescued)}</span>")
        out.append(f"<details><summary>{head}</summary><table>")
        out.append("<tr><th>question</th><th>gold</th><th>baseline</th>"
                   "<th>steered</th><th>z_f</th></tr>")
        show = pd.concat([broke, rescued,
                          grp.drop(broke.index).drop(rescued.index).head(20)])
        for _, r in show.iterrows():
            cls = ("flip" if r.baseline_correct and not r.steered_correct
                   else "rescue" if not r.baseline_correct and r.steered_correct
                   else "")
            bc = "ok" if r.baseline_correct else "bad"
            sc = "ok" if r.steered_correct else "bad"
            out.append(
                f"<tr class={cls}><td class=q>{esc(r.question)}</td>"
                f"<td>{esc(r.gold)}</td>"
                f"<td class='ans {bc}'>{esc(r.baseline_answer)}</td>"
                f"<td class='ans {sc}'>{esc(r.steered_answer)}</td>"
                f"<td>{r.baseline_zf:.1f}</td></tr>")
        out.append("</table></details>")

    (sae / "steer_view.html").write_text("\n".join(out))
    print(f"[view] wrote {sae}/steer_view.html")


if __name__ == "__main__":
    main()
