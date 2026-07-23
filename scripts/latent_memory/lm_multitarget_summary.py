"""Merge multi-target oracle shards and plot reasoning-state capacity: median recovery vs
latent width m, for the answer query and the mean intermediate query, per injection layer.

    python -m scripts.latent_memory.lm_multitarget_summary --run_dir runs/latent_memory_v0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median


def med(xs):
    xs = [x for x in xs if x == x]
    return median(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=Path, default=Path("runs/latent_memory_v0"))
    args = ap.parse_args()

    rows = []
    for p in sorted(args.run_dir.glob("multitarget_shard*.jsonl")):
        rows += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    if not rows:
        print(f"no multitarget_shard*.jsonl under {args.run_dir}"); return
    n_traces = len({r["trace_id"] for r in rows})
    layers = sorted({r["layer"] for r in rows})
    ms = sorted({r["m"] for r in rows})
    print(f"traces={n_traces}  layers={layers}  m={ms}")

    summ = {}
    for layer in layers:
        print(f"\n-- layer {layer} --")
        print(f"  m   answer_rec  inter_mean  inter_min")
        for m in ms:
            L = [r for r in rows if r["layer"] == layer and r["m"] == m]
            a = med([r["answer_rec"] for r in L])
            im = med([r["inter_rec_mean"] for r in L])
            imn = med([r["inter_rec_min"] for r in L])
            summ[f"{layer}|{m}"] = {"answer_rec": a, "inter_rec_mean": im,
                                    "inter_rec_min": imn, "n": len(L)}
            print(f"  {m:>3}  {a:>10.3f}  {im:>10.3f}  {imn:>9.3f}")

    (args.run_dir / "multitarget_summary.json").write_text(json.dumps(summ, indent=2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(layers), figsize=(6 * len(layers), 4.2),
                                 squeeze=False)
        for ax, layer in zip(axes[0], layers):
            a = [summ[f"{layer}|{m}"]["answer_rec"] for m in ms]
            im = [summ[f"{layer}|{m}"]["inter_rec_mean"] for m in ms]
            imn = [summ[f"{layer}|{m}"]["inter_rec_min"] for m in ms]
            ax.plot(ms, a, marker="o", label="answer query")
            ax.plot(ms, im, marker="s", label="intermediate (mean)")
            ax.plot(ms, imn, marker="^", ls="--", label="intermediate (worst)")
            ax.axhline(1.0, ls="--", c="k", lw=0.8)
            ax.axhline(0.0, ls=":", c="grey", lw=0.8)
            ax.set_xscale("log", base=2); ax.set_xticks(ms); ax.set_xticklabels(ms)
            ax.set_xlabel("latent width m"); ax.set_ylabel("median recovery R")
            ax.set_title(f"injection layer {layer}")
            ax.grid(alpha=0.3); ax.legend(fontsize=8)
        fig.suptitle(f"latent_memory_v0 multi-target reasoning-state oracle (n={n_traces})")
        fig.tight_layout()
        out_png = args.run_dir / "multitarget_curve.png"
        fig.savefig(out_png, dpi=140)
        print(f"\nwrote {out_png}")
    except Exception as e:
        print(f"[plot skipped] {e}")
    print(f"wrote {args.run_dir / 'multitarget_summary.json'}")


if __name__ == "__main__":
    main()
