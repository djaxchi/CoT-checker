"""Merge latent_memory_v0 oracle shards and plot the capacity curve: median recovery of
the full-CoT answer-belief margin vs latent width m, per injection layer and method.

    python -m scripts.latent_memory.lm_summary --run_dir runs/latent_memory_v0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median


def load_rows(run_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for p in sorted(run_dir.glob("oracle_shard*.jsonl")):
        rows += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return rows


def summarize(rows: list[dict]):
    """median recovery keyed by (layer, m, method), finite recoveries only."""
    buckets: dict[tuple, list[float]] = {}
    for r in rows:
        if r["method"] in ("teacher_full", "no_cot"):
            continue
        rec = r.get("recovery")
        if rec is None or rec != rec:  # NaN
            continue
        buckets.setdefault((r["layer"], r["m"], r["method"]), []).append(rec)
    return {k: (median(v), len(v)) for k, v in buckets.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=Path, default=Path("runs/latent_memory_v0"))
    args = ap.parse_args()

    rows = load_rows(args.run_dir)
    if not rows:
        print(f"no oracle_shard*.jsonl under {args.run_dir}"); return
    n_traces = len({r["trace_id"] for r in rows})
    full = [r["margin"] for r in rows if r["method"] == "teacher_full"]
    no = [r["margin"] for r in rows if r["method"] == "no_cot"]
    print(f"traces={n_traces}  median margin_full={median(full):.3f}  "
          f"median margin_no={median(no):.3f}")

    summ = summarize(rows)
    layers = sorted({k[0] for k in summ})
    ms = sorted({k[1] for k in summ})
    methods = sorted({k[2] for k in summ})
    print("\nmedian recovery R (1.0 = full-CoT margin, 0.0 = no-CoT):")
    for layer in layers:
        print(f"\n-- layer {layer} --")
        header = "  m   " + "".join(f"{mth:>14}" for mth in methods)
        print(header)
        for m in ms:
            cells = []
            for mth in methods:
                v = summ.get((layer, m, mth))
                cells.append(f"{v[0]:>14.3f}" if v else f"{'-':>14}")
            print(f"{m:>4}  " + "".join(cells))

    out_json = args.run_dir / "capacity_summary.json"
    out_json.write_text(json.dumps(
        {f"{l}|{m}|{mth}": {"median_recovery": v[0], "n": v[1]}
         for (l, m, mth), v in summ.items()}, indent=2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(layers), figsize=(6 * len(layers), 4.2),
                                 squeeze=False)
        for ax, layer in zip(axes[0], layers):
            for mth in methods:
                ys = [summ.get((layer, m, mth), (float("nan"),))[0] for m in ms]
                ax.plot(ms, ys, marker="o", label=mth)
            ax.axhline(1.0, ls="--", c="k", lw=0.8, label="full CoT")
            ax.axhline(0.0, ls=":", c="grey", lw=0.8, label="no CoT")
            ax.set_xscale("log", base=2)
            ax.set_xticks(ms); ax.set_xticklabels(ms)
            ax.set_xlabel("latent width m")
            ax.set_ylabel("median recovery R")
            ax.set_title(f"injection layer {layer}"
                         + (" (soft input tokens)" if layer == 0 else ""))
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.suptitle(f"latent_memory_v0 capacity oracle (n={n_traces} traces)")
        fig.tight_layout()
        out_png = args.run_dir / "capacity_curve.png"
        fig.savefig(out_png, dpi=140)
        print(f"\nwrote {out_png}\nwrote {out_json}")
    except Exception as e:  # plotting is optional on headless nodes
        print(f"[plot skipped] {e}\nwrote {out_json}")


if __name__ == "__main__":
    main()
