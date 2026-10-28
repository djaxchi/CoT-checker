#!/usr/bin/env python3
"""M6: RTH trained with offsets in some layers only (read_token_v1 mechanistic study).

M2 switched trained offsets off after training; this trains with offsets allowed only
in the listed layers, so the token can adapt to where it is allowed to act. Everything
else is RTH's recipe (scripts/read_token_probe.py train): the offsets outside the
window get a zero gradient, and since they start at zero and AdamW has no weight decay
on them they stay exactly zero; the saved checkpoint is an ordinary deep read token
(encode, score and attention dumps read it unchanged), with `deep_layers` recorded.

The layer window is passed as --layers (e.g. 0-17 or 9-17,27-35); every other
argument goes to read_token_probe.py train.

    python scripts/analysis/rth_mech_window_train.py --layers 0-17 -- --model Qwen/Qwen3-8B --m 1 \
        --init text --init_text " correct" --deep --head linear --n_train 100000 --seed 0 \
        --soft_attn no_cudnn --train <train.jsonl> --out <dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import scripts.read_token_probe as rtp  # noqa: E402


def parse_layers(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return sorted(set(out))


def windowed_optimizer(layers: list[int], make=rtp.make_optimizer):
    """make_optimizer, plus a gradient mask that keeps rt.deep zero outside `layers`."""
    def build(rt, lr, freeze_soft):
        if rt.deep is None:
            raise SystemExit("--layers needs --deep")
        if max(layers) >= rt.deep.shape[0]:
            raise SystemExit(f"layer {max(layers)} out of range ({rt.deep.shape[0]} layers)")
        mask = torch.zeros(rt.deep.shape[0], 1, 1, device=rt.deep.device)
        mask[layers] = 1.0
        with torch.no_grad():
            rt.deep.mul_(mask)
        rt.deep.register_hook(lambda g: g * mask.to(g.dtype))
        return make(rt, lr, freeze_soft)
    return build


def main() -> None:
    argv = sys.argv[1:]
    if "--layers" not in argv or "--" not in argv:
        raise SystemExit("usage: rth_mech_window_train.py --layers A-B[,C-D] -- <read_token_probe train args>")
    layers = parse_layers(argv[argv.index("--layers") + 1])
    rest = argv[argv.index("--") + 1:]
    out = Path(rest[rest.index("--out") + 1])
    rtp.make_optimizer = windowed_optimizer(layers)
    sys.argv = ["read_token_probe.py", "train", *rest]
    print(f"[m6] offsets trained in layers {layers}", flush=True)
    rtp.main()
    ck = torch.load(out / "read_token.pt", map_location="cpu", weights_only=False)
    off = [l for l in range(ck["deep"].shape[0]) if l not in layers]
    if off and ck["deep"][off].abs().max() != 0:
        raise SystemExit("offsets outside the window are not zero")
    ck["deep_layers"] = layers
    torch.save(ck, out / "read_token.pt")
    norms = ck["deep"].float().flatten(1).norm(dim=1)
    print(f"[m6] saved; offset norms in window {norms[layers].mean():.3f}, outside "
          f"{norms[off].max() if off else 0:.3f}", flush=True)


if __name__ == "__main__":
    main()
