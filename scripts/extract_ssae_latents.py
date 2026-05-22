"""Extract SSAE latents for the standard benchmark JSONL files.

For each JSONL (probe_train, val, processbench), tokenize each row, run the
trained QwenSSAE encoder + autoencoder, L2-normalize, and save:

    <name>_z.npy      (n_examples, n_latents) float32
    <name>_y.npy      (n_examples,) int64    -- only when labels are present
    <name>_meta.jsonl one row per example, preserving id/step_idx/label/n_steps

Plus a latent_manifest.json with method/model/ckpt provenance.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ssae.dataset import (  # noqa: E402
    SSAECollator,
    SSAEJsonlDataset,
    add_step_sep_token,
)
from src.ssae.model_qwen_ssae import QwenSSAE  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, type=Path,
                   help="Path to ssae_model.pt (state_dict).")
    p.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-1.5B")
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--sparsity_factor", type=int, default=1)
    p.add_argument("--contrastive_ckpt", action="store_true",
                   help="Set when the checkpoint was trained with aux head "
                        "(method=ssae_contrastive) so we instantiate the "
                        "matching architecture before load_state_dict.")
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--probe_train_jsonl", required=True, type=Path)
    p.add_argument("--val_jsonl", required=True, type=Path)
    p.add_argument("--pb_jsonl", required=True, type=Path)
    p.add_argument("--limit", type=int, default=-1,
                   help="Limit rows per split (smoke mode). -1 = no limit.")
    return p.parse_args()


def extract_one(model, dataset, collator, batch_size: int, num_workers: int,
                device: str) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collator, num_workers=num_workers, pin_memory=True,
    )
    zs: list[np.ndarray] = []
    ys: list[int] = []
    metas: list[dict] = []
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            if device != "cpu" and torch.cuda.is_available():
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                    z = model.encode_latents(input_ids, attention_mask)
            else:
                z = model.encode_latents(input_ids, attention_mask)
            zs.append(z.float().cpu().numpy())
            ys.extend(batch["labels"].tolist())
            metas.extend(batch["meta"])
    z_arr = np.concatenate(zs, axis=0).astype(np.float32)
    y_arr = np.array(ys, dtype=np.int64)
    return z_arr, y_arr, metas


def write_split(name: str, out_dir: Path, z: np.ndarray, y: np.ndarray,
                metas: list[dict], save_labels: bool) -> None:
    np.save(out_dir / f"{name}_z.npy", z)
    if save_labels:
        np.save(out_dir / f"{name}_y.npy", y)
    with (out_dir / f"{name}_meta.jsonl").open("w") as f:
        for m in metas:
            f.write(json.dumps(m) + "\n")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, local_files_only=args.local_files_only
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    add_step_sep_token(tokenizer)
    collator = SSAECollator(pad_token_id=int(tokenizer.pad_token_id))

    model = QwenSSAE(
        tokenizer=tokenizer,
        model_name_or_path=args.model_name_or_path,
        sparsity_factor=args.sparsity_factor,
        phase=1,
        local_files_only=args.local_files_only,
        contrastive=args.contrastive_ckpt,
    ).to(device)
    state = torch.load(args.ckpt, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Only the aux_head parameters are allowed to differ between contrastive
    # and non-contrastive checkpoints. Anything else is a real corruption
    # (renamed module, partial save) and must abort -- silently loading a
    # partially-initialized SSAE would produce garbage latents and a
    # meaningless leaderboard.
    allowed = {"aux_head.weight", "aux_head.bias"}
    bad_missing = [k for k in missing if k not in allowed]
    bad_unexpected = [k for k in unexpected if k not in allowed]
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            "Unexpected key mismatch when loading SSAE checkpoint:\n"
            f"  missing (not in {{aux_head.weight, aux_head.bias}}): {bad_missing}\n"
            f"  unexpected (not in {{aux_head.weight, aux_head.bias}}): {bad_unexpected}"
        )
    if missing or unexpected:
        print(
            f"[extract_ssae_latents] aux_head delta on load "
            f"(expected for contrastive_ckpt={args.contrastive_ckpt}): "
            f"missing={missing}, unexpected={unexpected}"
        )
    model.eval()

    limit = None if args.limit <= 0 else args.limit

    splits = [
        ("probe_train_40k", args.probe_train_jsonl, True),
        ("val_1k", args.val_jsonl, True),
        ("pb_gsm8k_step", args.pb_jsonl, False),
    ]
    times: dict[str, float] = {}
    for name, path, save_labels in splits:
        ds = SSAEJsonlDataset(path, tokenizer, max_seq_len=args.max_seq_len, limit=limit)
        # Strict no-truncation check before extraction.
        ds.length_audit(raise_on_violation=True)
        t0 = time.time()
        z, y, metas = extract_one(
            model, ds, collator, args.batch_size, args.num_workers, device
        )
        times[name] = time.time() - t0
        write_split(name, args.out_dir, z, y, metas, save_labels=save_labels)
        print(f"{name}: z={z.shape} time={times[name]:.1f}s")

    manifest = {
        "ckpt": str(args.ckpt),
        "model_name_or_path": args.model_name_or_path,
        "sparsity_factor": args.sparsity_factor,
        "contrastive_ckpt": args.contrastive_ckpt,
        "n_latents": model.n_latents,
        "n_inputs": model.n_inputs,
        "splits": {name: str(path) for name, path, _ in splits},
        "extraction_time_sec": times,
    }
    (args.out_dir / "latent_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote latents + manifest to {args.out_dir}")


if __name__ == "__main__":
    main()
