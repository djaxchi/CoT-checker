"""Stage 2 NaN bisection, part 2: run REAL A-arm training steps (encoder + D +
backward + AdamW) and print where finiteness dies, across dtype x mask-value.

The forward decode was proven healthy (to_debug_decode); prime suspect is the
sdpa backward with an additive finfo.min mask in half precision.

  python scripts/transition_operator/to_debug_train.py --run_dir runs/transition_operator
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.analysis.transition_operator import sep_join_ids  # noqa: E402
from src.analysis.transition_operator_train import (  # noqa: E402
    TransitionEncoder,
    UpperDecoder,
    kl_to_actual,
)


def fin(t: torch.Tensor) -> str:
    return "OK" if torch.isfinite(t).all() else "NONFINITE"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=Path, default=Path("runs/transition_operator"))
    ap.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-7B")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--n_forks", type=int, default=32)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--local_files_only", action="store_true")
    args = ap.parse_args()

    a = args.run_dir / "stage2" / "arrays"
    fork_rows = json.loads((a / "fork_rows.json").read_text())[:args.n_forks]
    trans_rows = json.loads((a / "trans_rows.json").read_text())
    tid = {}
    for i, r in enumerate(trans_rows):
        tid.setdefault(r["fork_id"], {})[r["branch"]] = i
    t_idx = np.array([tid[r["fork_id"]]["correct"] for r in fork_rows]
                     + [tid[r["fork_id"]]["wrong"] for r in fork_rows])
    with np.load(a / "fork_arrays.npz") as z:
        S_prev = np.asarray(z["S_prev"][:args.n_forks], np.float32)
    with np.load(a / "trans_arrays.npz") as z:
        H = np.asarray(z["H_steps"][t_idx], np.float32)
        n_steps = np.asarray(z["n_steps"][t_idx])
        post_logits = np.asarray(z["post_logits"][t_idx], np.float32)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                        local_files_only=args.local_files_only)
    forks_meta = {}
    with open(args.run_dir / "forks.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["fork_id"] in tid:
                forks_meta[r["fork_id"]] = r
    rows = [sep_join_ids(tok, [forks_meta[r["fork_id"]]["question"],
                               *forks_meta[r["fork_id"]]["prefix_steps"]])[:-1]
            for r in fork_rows]
    lens = torch.tensor([len(r) for r in rows])
    width = int(lens.max())
    ids = torch.zeros(len(rows), width, dtype=torch.long)
    am = torch.zeros(len(rows), width, dtype=torch.long)
    for i, r in enumerate(rows):
        ids[i, :len(r)], am[i, :len(r)] = torch.tensor(r), 1

    dev = "cuda"
    sp = torch.tensor(S_prev, device=dev)
    sp2 = torch.cat([sp, sp])
    Ht = torch.tensor(H, device=dev)
    mask = (torch.arange(Ht.shape[1], device=dev)[None]
            < torch.tensor(n_steps, device=dev)[:, None])
    actual = torch.tensor(post_logits, device=dev)

    for dtype in (torch.bfloat16, torch.float16):
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, torch_dtype=dtype,
            local_files_only=args.local_files_only).to(dev).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        for mask_value in (None, -1e4):
            torch.manual_seed(0)
            enc = TransitionEncoder(hidden=sp.shape[1], d_z=64).to(dev)
            D = torch.nn.Linear(64, sp.shape[1]).to(dev)
            torch.nn.init.zeros_(D.bias)
            with torch.no_grad():
                D.weight.mul_(0.01)
            ud = UpperDecoder(model, args.layer, mask_value=mask_value)
            opt = torch.optim.AdamW(list(enc.parameters()) + list(D.parameters()),
                                    lr=3e-4, weight_decay=0.01)
            print(f"\n===== dtype {dtype} mask_value {mask_value} =====", flush=True)
            for step in range(args.steps):
                cache = ud.prefill(ids.to(dev), am.to(dev))
                base_len = ud.cache_len(cache)
                z = enc(sp2, Ht, mask)
                preds = []
                F = len(fork_rows)
                for half in (slice(0, F), slice(F, 2 * F)):
                    preds.append(ud.decode_boundary(cache, sp + D(z[half]), lens))
                    ud.crop(cache, base_len)
                pred = torch.cat(preds)
                loss = kl_to_actual(actual, pred)
                opt.zero_grad()
                loss.backward()
                gd = D.weight.grad
                ge = next(enc.parameters()).grad
                print(f"step {step}: z {fin(z)} pred {fin(pred)} "
                      f"loss {float(loss.detach()):.4f} "
                      f"gradD {fin(gd)} |gradD| {float(gd.norm()):.2e} "
                      f"gradEnc {fin(ge)}", flush=True)
                opt.step()
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
