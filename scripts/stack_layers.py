#!/usr/bin/env python3
"""Concatenate the same pooling read at two layers into one representation.

The only thing that ever bought a large gain in this project's earlier
representation search was reading more than one layer: stacking lifted AUC by
about 0.05 to 0.85 on Qwen2.5, where every compression scheme lost. That was
never retested on Qwen3, and the grid so far reads a single layer.

The reason to expect it is concrete rather than stylistic. A late layer has
already committed to what the step says; a middle layer still carries the
intermediate quantities that a wrong step gets wrong. Reading one of them forces
the probe to choose.

Stacking is only meaningful if the two files describe the SAME steps in the SAME
order. The poolings are sampled from the store with a seeded RNG, so two runs
over two layers of the same split line up by construction, but "by construction"
is exactly the kind of claim that silently stops being true when a shard is
re-encoded. So this refuses to write unless the labels and the token counts agree
elementwise, which is a genuine fingerprint of the rows: two independently
sampled subsets would not reproduce 60,000 identical step lengths.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def check_aligned(a: dict, b: dict, keys) -> None:
    for k in keys:
        if k not in a or k not in b:
            raise SystemExit(f"missing {k} in one of the inputs")
        if a[k].shape != b[k].shape or not np.array_equal(a[k], b[k]):
            raise SystemExit(
                f"{k} differs between layers: the two files are not the same rows "
                f"in the same order, so concatenating them would pair each step "
                f"with a different step's activations")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", nargs="+", required=True, type=Path,
                   help="Two or more npz of the same pooling at different layers.")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    if len(args.npz) < 2:
        raise SystemExit("stacking needs at least two layers")

    zs = [dict(np.load(p_)) for p_ in args.npz]
    subs = sorted({k[5:] for k in zs[0] if k.startswith("pb_x_")})
    align = ["y_train", "y_val"] + [f"pb_y_{s}" for s in subs]
    if "len_train" in zs[0]:
        align += ["len_train", "len_val"] + [f"pb_len_{s}" for s in subs]
    for z in zs[1:]:
        check_aligned(zs[0], z, align)

    out = {k: zs[0][k] for k in align}
    for k in ["x_train", "x_val"] + [f"pb_x_{s}" for s in subs]:
        out[k] = np.concatenate([z[k].astype(np.float32) for z in zs], 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)
    print(f"[stack] {' + '.join(p_.stem for p_ in args.npz)} -> dim "
          f"{out['x_train'].shape[1]} over {len(out['y_train']):,} steps "
          f"(alignment verified on {len(align)} label and length arrays)")
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
