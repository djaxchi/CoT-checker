"""Encode cached Gemma-2-9B residuals through GemmaScope (JumpReLU) SAEs.

GemmaScope arm of the public-SAE audit. Reads merged dense residuals
(heldout_{L20,L31}_h.npy) from extract_instruct_residuals.py and, per readout,
loads the matching GemmaScope canonical residual SAE (params.npz, JumpReLU) and
computes z = encode(h), h_hat = decode(z), r = h - h_hat.

GemmaScope is trained on the FULL activation distribution over all positions, so
unlike the Qwen2.5-Instruct SAE it should reconstruct our last-token deep-layer
readout well; the recon-FVU gate is the empirical check (compare against the
repo's eval / the Qwen arm's FVU).

JumpReLU inference (official GemmaScope convention):
    pre = h @ W_enc + b_enc ;  z = (pre > threshold) * relu(pre)
    h_hat = z @ W_dec + b_dec
params.npz keys: W_enc (d_model,d_sae), W_dec (d_sae,d_model), threshold (d_sae,),
b_enc (d_sae,), b_dec (d_model,).

Outputs match the probe/plot contract exactly (so those scripts are reused as-is):
  {L}_t{t}_z.npz / _hhat.npy / _resid.npy / _feat_count.npy / _encode_manifest.json
  {L}_t{t}_decoder.npy   (d_model,d_sae) for the SAE->h direction map (fig F)
Use --trainer 0 (a dummy slot label; GemmaScope has no trainer variants).

SAE path pattern: <sae_root>/<sae_folder>/params.npz, default
  L20 -> layer_20/width_16k/canonical ;  L31 -> layer_31/width_16k/canonical
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from encode_public_sae import encode_layer  # noqa: E402  (shared CSR/FVU loop)


class JumpReLUSAE(torch.nn.Module):
    """Minimal GemmaScope JumpReLU inference module (loads params.npz directly)."""

    def __init__(self, W_enc, W_dec, b_enc, b_dec, threshold):
        super().__init__()
        self.activation_dim, self.dict_size = W_enc.shape
        self.register_buffer("W_enc", W_enc)
        self.register_buffer("W_dec", W_dec)
        self.register_buffer("b_enc", b_enc)
        self.register_buffer("b_dec", b_dec)
        self.register_buffer("threshold", threshold)

    @classmethod
    def from_npz(cls, path: Path, device) -> "JumpReLUSAE":
        p = np.load(path)
        need = {"W_enc", "W_dec", "b_enc", "b_dec", "threshold"}
        if need - set(p.files):
            sys.exit(f"[gsae] FATAL {path} missing keys {need - set(p.files)} (have {p.files})")
        t = lambda k: torch.from_numpy(p[k]).float()  # noqa: E731
        ae = cls(t("W_enc"), t("W_dec"), t("b_enc"), t("b_dec"), t("threshold"))
        return ae.to(device).eval()

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = x @ self.W_enc + self.b_enc
        return (pre > self.threshold) * torch.relu(pre)

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec + self.b_dec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--enc_dir", type=Path, required=True)
    ap.add_argument("--sae_root", type=Path, required=True,
                    help="local google/gemma-scope-9b-pt-res-canonical dir")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--layers", nargs="+", default=["L20", "L31"])
    ap.add_argument("--sae_layer_map", nargs="+",
                    default=["L20:layer_20/width_16k/canonical",
                             "L31:layer_31/width_16k/canonical"],
                    help="readout:gemma-scope-subpath pairs (folder holding params.npz)")
    ap.add_argument("--trainer", type=int, default=0, help="dummy slot label for filenames")
    ap.add_argument("--batch_size", type=int, default=256)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folder_of = dict(s.split(":", 1) for s in args.sae_layer_map)

    for L in args.layers:
        folder = folder_of[L]
        npz = args.sae_root / folder / "params.npz"
        if not npz.exists():
            sys.exit(f"[gsae] FATAL missing {npz} (pre-fetch this layer/width; "
                     "check width_16k vs width_131k and that 'canonical' exists)")
        h = np.load(args.enc_dir / f"heldout_{L}_h.npy").astype(np.float32)
        print(f"[gsae] {L} <- {folder}  h={h.shape}", flush=True)
        ae = JumpReLUSAE.from_npz(npz, device)
        if ae.activation_dim != h.shape[1]:
            sys.exit(f"[gsae] FATAL dim mismatch SAE {ae.activation_dim} vs h {h.shape[1]}")

        data, idx, indptr, hhat, fvu, active, fcount = encode_layer(h, ae, device, args.batch_size)
        n, t = h.shape[0], args.trainer
        stem = f"{L}_t{t}"
        np.savez(args.out_dir / f"{stem}_z.npz", data=data, indices=idx, indptr=indptr,
                 shape=np.array([n, ae.dict_size], dtype=np.int64), active_per_row=active)
        np.save(args.out_dir / f"{stem}_hhat.npy", hhat)
        np.save(args.out_dir / f"{stem}_resid.npy", (h - hhat.astype(np.float32)).astype(np.float16))
        np.save(args.out_dir / f"{stem}_feat_count.npy", fcount.astype(np.int64))
        # decoder for the SAE->h direction map (fig F): (d_model, d_sae) = W_dec.T
        np.save(args.out_dir / f"{stem}_decoder.npy",
                ae.W_dec.t().contiguous().cpu().numpy().astype(np.float32))

        n_used = int((fcount > 0).sum())
        gate = "OK" if fvu < 0.5 else ("HIGH (mild OOD?)" if fvu < 0.9 else
                                       "*** SUSPECT: likely wrong layer/hook ***")
        (args.out_dir / f"{stem}_encode_manifest.json").write_text(json.dumps({
            "layer": L, "sae_folder": folder, "trainer": t, "sae_id": folder,
            "k": int(round(float(active.mean()))), "threshold": "per-feature (JumpReLU)",
            "dict_size": int(ae.dict_size), "activation_dim": int(ae.activation_dim),
            "n": n, "recon_fvu": round(fvu, 4), "fvu_gate": gate,
            "active_per_row_mean": float(active.mean()),
            "active_per_row_median": float(np.median(active)),
            "active_per_row_p95": float(np.percentile(active, 95)),
            "n_features_ever_active": n_used, "frac_dict_used": round(n_used / ae.dict_size, 4),
            "sae_arch": "JumpReLU (GemmaScope)", "params_npz": str(npz),
        }, indent=2))
        print(f"[gsae] {L} FVU={fvu:.3f} [{gate}]  active/row mean={active.mean():.1f} "
              f"median={np.median(active):.0f}  feats_used={n_used}", flush=True)


if __name__ == "__main__":
    main()
