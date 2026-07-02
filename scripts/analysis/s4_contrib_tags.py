"""S4 contrib-cluster stage 4: weak regex tags + surface features per step.

Reads reprs/step_metadata.parquet and writes tags.parquet with one row per
step: the multi-label regex tags (interpretation only, never clustering
input), a single display `top_tag` (rarest matching tag), and the trivial
surface features used as controls.

Usage:
  python scripts/analysis/s4_contrib_tags.py --run_dir runs/contrib_cluster
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.analysis.contrib_cluster import (  # noqa: E402
    TAG_NAMES,
    assign_top_tag,
    surface_features,
    tag_step,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", type=Path, default=Path("runs/contrib_cluster"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_path = args.run_dir / "tags.parquet"
    if out_path.exists() and not args.force:
        sys.exit(f"refusing to overwrite {out_path}; pass --force")

    meta = pd.read_parquet(args.run_dir / "reprs" / "step_metadata.parquet")
    texts = meta["step_text"].tolist()

    tag_matrix = np.array([[tag_step(t)[name] for name in TAG_NAMES] for t in texts],
                          dtype=bool)
    surf = pd.DataFrame([surface_features(t) for t in texts])

    df = meta[["row_id", "trajectory_id", "step_index", "num_steps_in_trajectory",
               "relative_step_index", "token_count"]].copy()
    for j, name in enumerate(TAG_NAMES):
        df[f"tag_{name}"] = tag_matrix[:, j]
    df["n_tags"] = tag_matrix.sum(axis=1)
    df["top_tag"] = assign_top_tag(tag_matrix)
    for col in surf.columns:
        df[col] = surf[col].to_numpy()
    df.to_parquet(out_path, index=False)

    freq = {name: float(tag_matrix[:, j].mean()) for j, name in enumerate(TAG_NAMES)}
    (args.run_dir / "tags_manifest.json").write_text(json.dumps({
        "created": datetime.now(timezone.utc).isoformat(),
        "n_steps": len(df),
        "tag_frequency": freq,
        "untagged_fraction": float((tag_matrix.sum(axis=1) == 0).mean()),
    }, indent=2))
    print(f"[tags] wrote {out_path} ({len(df)} steps)")
    for name, f in sorted(freq.items(), key=lambda kv: -kv[1]):
        print(f"[tags]   {name:<24} {f:6.1%}")


if __name__ == "__main__":
    main()
