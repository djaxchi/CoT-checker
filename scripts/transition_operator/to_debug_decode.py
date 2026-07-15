"""Compute-node debug for the Stage 2 NaN: exercise the exact A-arm decode path
on CUDA against the stored arrays, in bf16 and fp16.

Checks, per dtype:
  1. prefill finiteness (cache stats)
  2. identity decode: h_hat = S_prev exactly -> logits must be finite and close
     to the stored pre_logits (end-to-end CUDA-path sanity)
  3. perturbed decode: h_hat = S_prev + 0.01*randn -> finite
  4. kl_to_actual(post_logits, pred) finiteness

  python scripts/transition_operator/to_debug_decode.py --run_dir runs/transition_operator
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
from src.analysis.transition_operator_train import UpperDecoder, kl_to_actual  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=Path, default=Path("runs/transition_operator"))
    ap.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-7B")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--n_forks", type=int, default=8)
    ap.add_argument("--local_files_only", action="store_true")
    args = ap.parse_args()

    a = args.run_dir / "stage2" / "arrays"
    fork_rows = json.loads((a / "fork_rows.json").read_text())[:args.n_forks]
    with np.load(a / "fork_arrays.npz") as z:
        S_prev = np.asarray(z["S_prev"][:args.n_forks], np.float32)
        pre_logits = np.asarray(z["pre_logits"][:args.n_forks], np.float32)
    forks_meta = {}
    with open(args.run_dir / "forks.jsonl") as f:
        for line in f:
            r = json.loads(line)
            forks_meta[r["fork_id"]] = r

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                        local_files_only=args.local_files_only)

    rows = []
    for r in fork_rows:
        m = forks_meta[r["fork_id"]]
        rows.append(sep_join_ids(tok, [m["question"], *m["prefix_steps"]])[:-1])
    lens = torch.tensor([len(r) for r in rows])
    width = int(lens.max())
    ids = torch.zeros(len(rows), width, dtype=torch.long)
    am = torch.zeros(len(rows), width, dtype=torch.long)
    for i, r in enumerate(rows):
        ids[i, :len(r)], am[i, :len(r)] = torch.tensor(r), 1

    for dtype in (torch.bfloat16, torch.float16):
        print(f"\n===== dtype {dtype} =====", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, torch_dtype=dtype,
            local_files_only=args.local_files_only).to("cuda").eval()
        print("attn_implementation:", model.config._attn_implementation, flush=True)
        ud = UpperDecoder(model, args.layer)
        cache = ud.prefill(ids.cuda(), am.cuda())
        k = cache.layers[args.layer].keys
        print(f"cache L{args.layer} keys shape {tuple(k.shape)} "
              f"finite {bool(torch.isfinite(k).all())} "
              f"absmax {float(k.abs().max()):.1f}", flush=True)
        base_len = ud.cache_len(cache)
        sp = torch.tensor(S_prev, device="cuda")
        for name, h_hat in (("identity", sp),
                            ("perturbed", sp + 0.01 * torch.randn_like(sp))):
            logits = ud.decode_boundary(cache, h_hat, lens)
            ud.crop(cache, base_len)
            fin = bool(torch.isfinite(logits).all())
            stored = torch.tensor(pre_logits, device="cuda")
            diff = (logits - stored).abs().max() if fin else float("nan")
            agree = (logits.argmax(-1) == stored.argmax(-1)).float().mean()
            kl = kl_to_actual(stored, logits)
            print(f"{name:9s} finite {fin} max|pred-stored| {float(diff):.3f} "
                  f"argmax agree {float(agree):.2f} KL(stored||pred) {float(kl):.4f}",
                  flush=True)
        del model, cache
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
