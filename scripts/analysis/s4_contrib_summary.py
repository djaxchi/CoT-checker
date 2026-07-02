"""S4 contrib-cluster stage 6: aggregate all repr/layer results into the
combined deliverables and a report skeleton with every quantitative table
filled in (qualitative sections are TODO markers for manual inspection).

Outputs (under --run_dir):
  cluster_summary_all.csv     all per-cluster summaries, stacked
  tag_enrichment_all.csv      all enrichment tables, stacked
  report.md                   final report (auto-filled metrics + TODO sections)

Usage:
  python scripts/analysis/s4_contrib_summary.py --run_dir runs/contrib_cluster
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SURFACE_COLS = ["char_len", "token_count", "n_digits", "n_equals", "n_math_ops",
                "step_index", "relative_step_index"]


def fmt(v, nd=3):
    if v is None or v != v:
        return "-"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", type=Path, default=Path("runs/contrib_cluster"))
    args = ap.parse_args()
    cdir = args.run_dir / "clusters"

    summaries, enrichments, metrics = [], [], []
    for p in sorted(cdir.glob("cluster_summary_*.csv")):
        summaries.append(pd.read_csv(p))
    for p in sorted(cdir.glob("tag_enrichment_*.csv")):
        df = pd.read_csv(p)
        stem = p.stem.replace("tag_enrichment_", "")
        name, li = stem.rsplit("_layer_", 1)
        df.insert(0, "repr", name)
        df.insert(1, "layer", int(li))
        enrichments.append(df)
    for p in sorted(cdir.glob("metrics_*.json")):
        metrics.append(json.loads(p.read_text()))
    if not metrics:
        raise SystemExit(f"no metrics_*.json under {cdir}; run s4_contrib_cluster.py first")

    pd.concat(summaries, ignore_index=True).to_csv(
        args.run_dir / "cluster_summary_all.csv", index=False)
    pd.concat(enrichments, ignore_index=True).to_csv(
        args.run_dir / "tag_enrichment_all.csv", index=False)

    audit = json.loads((args.run_dir / "audit.json").read_text()) \
        if (args.run_dir / "audit.json").exists() else {}
    ex_manifest_p = args.run_dir / "hidden_states" / "extract_manifest.json"
    if not ex_manifest_p.exists():
        cands = sorted((args.run_dir / "hidden_states").glob("extract_manifest*.json"))
        ex_manifest_p = cands[0] if cands else None
    ex = json.loads(ex_manifest_p.read_text()) if ex_manifest_p else {}
    rp = json.loads((args.run_dir / "reprs" / "reprs_manifest.json").read_text()) \
        if (args.run_dir / "reprs" / "reprs_manifest.json").exists() else {}
    tg = json.loads((args.run_dir / "tags_manifest.json").read_text()) \
        if (args.run_dir / "tags_manifest.json").exists() else {}

    metrics.sort(key=lambda m: (m["layer"], m["repr"]))
    header = ("| repr | layer | algo | clusters | noise | silhouette | DB | "
              "wtd tag-entropy (bits) | wtd max-enrichment | "
              + " | ".join(f"eta2 {c}" for c in SURFACE_COLS) + " |")
    sep = "|" + "---|" * (9 + len(SURFACE_COLS))
    comp_rows = [header, sep]
    for m in metrics:
        comp_rows.append(
            f"| {m['repr']} | {m['layer']} | {m['algo']} | {m['n_clusters']} "
            f"| {m['noise_ratio']:.1%} | {fmt(m['silhouette'])} "
            f"| {fmt(m['davies_bouldin'], 2)} | {fmt(m['weighted_tag_entropy'], 2)} "
            f"| {fmt(m['weighted_max_enrichment'], 2)} | "
            + " | ".join(fmt(m["surface_eta2"].get(c), 2) for c in SURFACE_COLS)
            + " |")

    sampling = audit.get("sampling", {})
    md = [
        "# S4 contrib-cluster report: do hidden-state step representations form interpretable clusters?",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat()}. Exploratory only: "
        "no correctness labels or probes were used anywhere in this pipeline.",
        "",
        "## Setup",
        "",
        f"- Model: `{ex.get('model_name_or_path', '?')}` "
        f"({ex.get('num_hidden_layers', '?')} layers, hidden {ex.get('hidden_size', '?')}), "
        f"layers extracted: {ex.get('layers', '?')} (hidden_states index = residual after that layer)",
        f"- Trajectories: {sampling.get('n_sampled', '?')} sampled "
        f"(seed {audit.get('seed', '?')}), steps: {rp.get('n_steps', '?')} "
        f"(cap {sampling.get('max_steps_cap', '?')} steps/trajectory, "
        f"min {sampling.get('min_steps', '?')})",
        f"- Untagged steps: {fmt(tg.get('untagged_fraction'))} of corpus",
        "- Prefixes: p_0 = question; p_i = question + \"\\n\" + step_1..step_i "
        "(no chat template); h_i = last non-padding token.",
        "",
        "## Representation definitions",
        "",
        "| name | definition |",
        "|---|---|",
        "| state | h_i |",
        "| qres | h_i - h_0 |",
        "| contribution | h_i - h_{i-1} |",
        "",
        "`contribution` (the main representation) is the closed form of the "
        "recursion c_1 = h_1 - h_0, c_i = h_i - (h_0 + sum_{k<i} c_k), which "
        "telescopes exactly to the local finite difference "
        "(pinned in tests/analysis/test_contrib_cluster.py).",
        "",
        "## Clustering method",
        "",
        "L2-normalize -> PCA-50 -> "
        f"{metrics[0]['algo']}; regex tags used only post-hoc; UMAP for "
        "visualization only. Seed 42 throughout.",
        "",
        "## Cluster quality comparison (all reprs x layers)",
        "",
        *comp_rows,
        "",
        "Reading guide: wtd max-enrichment high + wtd tag-entropy low = tag-coherent "
        "clusters; eta2 near 1 on char_len/step_index = clustering explained by a "
        "trivial surface feature.",
        "",
        "## Top clusters per representation",
        "",
        "See `cluster_summary_all.csv` and per-combo `cluster_cards_*.md`.",
        "",
        "## Qualitative interpretation",
        "",
        "TODO(manual): inspect cluster cards; note clusters that read as "
        "reasoning operations vs surface artifacts.",
        "",
        "## Surface-feature caveats",
        "",
        "TODO(manual): list clusters whose eta2 / surface means flag them as "
        "length-, LaTeX-, digit- or position-driven.",
        "",
        "## Recommendation for the next experiment",
        "",
        "TODO(manual).",
        "",
    ]
    (args.run_dir / "report.md").write_text("\n".join(md))
    print(f"[summary] wrote cluster_summary_all.csv, tag_enrichment_all.csv, "
          f"report.md under {args.run_dir}/")


if __name__ == "__main__":
    main()
