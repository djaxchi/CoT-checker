"""Encode cached Instruct residuals through a public BatchTopK SAE.

Reads merged dense residuals (heldout_{L20,L28}_h.npy) produced by
extract_instruct_residuals.py and, for each requested layer, loads the matching
public SAE (`andyrdt/saes-qwen2.5-7b-instruct`) and computes:

  z     = SAE.encode(h)          (sparse; only above-threshold features kept)
  h_hat = SAE.decode(z)          (reconstruction, dense f16)
  r     = h - h_hat              (reconstruction residual, dense f16)

The SAE is loaded by reading ae.pt (a plain state_dict) directly, so this has NO
dependency on the andyrdt/dictionary_learning package. The inference path matches
dictionary_learning's BatchTopKSAE.encode(use_threshold=True):
    pre = relu(encoder(h - b_dec)); z = pre * (pre > threshold)

Layer mapping (see download_public_sae.md §4):
    L20 -> resid_post_layer_19 ;  L28 -> resid_post_layer_27

A reconstruction-FVU sanity gate is computed and stored: a wrong layer pairing
yields FVU ~>= 1.0 and the script warns loudly.

Outputs (per layer L, trainer t), under <out_dir>/:
  {L}_t{t}_z.npz        sparse z: data,indices,indptr (CSR), shape, plus
                        active_per_row (n,) and feature freq stats
  {L}_t{t}_hhat.npy     (n,3584) f16
  {L}_t{t}_resid.npy    (n,3584) f16
  {L}_t{t}_encode_manifest.json   fvu, active stats, k, threshold, paths, mapping
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# L<readout> -> SAE residual-post folder (output of block = hidden_states idx - 1)
LAYER_TO_SAE_FOLDER = {"L20": "resid_post_layer_19", "L28": "resid_post_layer_27"}


class BatchTopKSAE(torch.nn.Module):
    """Minimal re-impl of dictionary_learning BatchTopKSAE inference path."""

    def __init__(self, activation_dim: int, dict_size: int, k: int):
        super().__init__()
        self.activation_dim, self.dict_size = activation_dim, dict_size
        self.register_buffer("k", torch.tensor(k, dtype=torch.int))
        self.register_buffer("threshold", torch.tensor(-1.0, dtype=torch.float32))
        self.encoder = torch.nn.Linear(activation_dim, dict_size)
        self.decoder = torch.nn.Linear(dict_size, activation_dim, bias=False)
        self.b_dec = torch.nn.Parameter(torch.zeros(activation_dim))

    @classmethod
    def from_ae_pt(cls, path: Path, device) -> "BatchTopKSAE":
        sd = torch.load(path, map_location="cpu")
        dict_size, activation_dim = sd["encoder.weight"].shape
        k = int(sd["k"].item()) if "k" in sd else -1
        ae = cls(activation_dim, dict_size, k)
        missing, unexpected = ae.load_state_dict(sd, strict=False)
        crit = {"encoder.weight", "encoder.bias", "decoder.weight", "b_dec", "threshold"}
        if crit - set(sd.keys()):
            sys.exit(f"[sae] FATAL ae.pt missing critical keys: {crit - set(sd.keys())}")
        if unexpected:
            print(f"[sae] note: unexpected state_dict keys ignored: {unexpected}", flush=True)
        return ae.to(device).eval()

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = torch.relu(self.encoder(x - self.b_dec))
        return pre * (pre > self.threshold)

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z) + self.b_dec


def encode_layer(h: np.ndarray, ae: BatchTopKSAE, device, batch_size: int):
    """Returns (csr_data, csr_indices, csr_indptr, hhat f16, fvu, active_per_row)."""
    n = h.shape[0]
    data, indices, indptr = [], [], [0]
    hhat = np.zeros_like(h, dtype=np.float16)
    sse, sst_acc = 0.0, []
    active = np.zeros(n, dtype=np.int32)
    feat_count = torch.zeros(ae.dict_size, dtype=torch.float64, device=device)
    hbar = torch.tensor(h.mean(0), dtype=torch.float32, device=device)
    t0 = time.perf_counter()
    for i in range(0, n, batch_size):
        xb = torch.tensor(h[i:i + batch_size], dtype=torch.float32, device=device)
        z = ae.encode(xb)
        xhat = ae.decode(z)
        hhat[i:i + batch_size] = xhat.to(torch.float16).cpu().numpy()
        sse += float(((xb - xhat) ** 2).sum().item())
        sst_acc.append(((xb - hbar) ** 2).sum().item())
        nz = z.nonzero(as_tuple=False)
        active_b = (z > 0).sum(1).to(torch.int32).cpu().numpy()
        active[i:i + xb.shape[0]] = active_b
        feat_count += (z > 0).sum(0).to(torch.float64)
        # build CSR rows
        zc = z.cpu()
        for r in range(xb.shape[0]):
            idx = torch.nonzero(zc[r], as_tuple=False).flatten()
            data.extend(zc[r, idx].tolist())
            indices.extend(idx.tolist())
            indptr.append(len(indices))
        if (i // batch_size) % 8 == 0:
            print(f"[sae]   {i}/{n} ({time.perf_counter()-t0:.0f}s)", flush=True)
    sst = float(sum(sst_acc))
    fvu = sse / sst if sst > 0 else float("nan")
    return (np.asarray(data, np.float32), np.asarray(indices, np.int32),
            np.asarray(indptr, np.int64), hhat, fvu, active, feat_count.cpu().numpy())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--enc_dir", type=Path, required=True,
                    help="dir with heldout_{L20,L28}_h.npy + heldout_y.npy")
    ap.add_argument("--sae_root", type=Path, required=True,
                    help="local andyrdt/saes-qwen2.5-7b-instruct dir")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--layers", nargs="+", default=["L20", "L28"])
    ap.add_argument("--trainer", type=int, default=1, help="trainer idx (1 => k=64)")
    ap.add_argument("--batch_size", type=int, default=256)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y = np.load(args.enc_dir / "heldout_y.npy")

    for L in args.layers:
        folder = LAYER_TO_SAE_FOLDER[L]
        ae_pt = args.sae_root / folder / f"trainer_{args.trainer}" / "ae.pt"
        cfg_p = args.sae_root / folder / f"trainer_{args.trainer}" / "config.json"
        if not ae_pt.exists():
            sys.exit(f"[sae] FATAL missing {ae_pt} (pre-fetch this layer/trainer)")
        cfg = json.loads(cfg_p.read_text()) if cfg_p.exists() else {}
        h = np.load(args.enc_dir / f"heldout_{L}_h.npy").astype(np.float32)
        print(f"[sae] {L} <- {folder}/trainer_{args.trainer}  h={h.shape}", flush=True)
        ae = BatchTopKSAE.from_ae_pt(ae_pt, device)
        if ae.activation_dim != h.shape[1]:
            sys.exit(f"[sae] FATAL dim mismatch SAE {ae.activation_dim} vs h {h.shape[1]}")

        data, idx, indptr, hhat, fvu, active, fcount = encode_layer(
            h, ae, device, args.batch_size)
        n = h.shape[0]
        stem = f"{L}_t{args.trainer}"
        np.savez(args.out_dir / f"{stem}_z.npz", data=data, indices=idx, indptr=indptr,
                 shape=np.array([n, ae.dict_size], dtype=np.int64), active_per_row=active)
        np.save(args.out_dir / f"{stem}_hhat.npy", hhat)
        np.save(args.out_dir / f"{stem}_resid.npy", (h - hhat.astype(np.float32)).astype(np.float16))

        n_feats_used = int((fcount > 0).sum())
        gate = "OK" if fvu < 0.5 else ("HIGH (mild OOD?)" if fvu < 0.9 else
                                       "*** SUSPECT: likely wrong layer/hook ***")
        manifest = {
            "layer": L, "sae_folder": folder, "trainer": args.trainer,
            "sae_id": f"{folder}/trainer_{args.trainer}",
            "k": int(ae.k.item()), "threshold": float(ae.threshold.item()),
            "dict_size": int(ae.dict_size), "activation_dim": int(ae.activation_dim),
            "n": n, "recon_fvu": round(fvu, 4), "fvu_gate": gate,
            "active_per_row_mean": float(active.mean()),
            "active_per_row_median": float(np.median(active)),
            "active_per_row_p95": float(np.percentile(active, 95)),
            "n_features_ever_active": n_feats_used,
            "frac_dict_used": round(n_feats_used / ae.dict_size, 4),
            "config": cfg, "ae_pt": str(ae_pt),
        }
        (args.out_dir / f"{stem}_encode_manifest.json").write_text(json.dumps(manifest, indent=2))
        np.save(args.out_dir / f"{stem}_feat_count.npy", fcount.astype(np.int64))
        print(f"[sae] {L} FVU={fvu:.3f} [{gate}]  active/row mean={active.mean():.1f} "
              f"median={np.median(active):.0f}  feats_used={n_feats_used}", flush=True)


if __name__ == "__main__":
    main()
