"""Convert a Miaow-Lab/SSAE pretrained checkpoint into a QwenSSAE state_dict.

The Miaow-Lab/SSAE-Checkpoints HuggingFace repo ships full ``MyModel`` phase-1
checkpoints whose state_dict carries phase-2/3 modules (``mean_mlp``,
``var_mlp``) that our ``QwenSSAE`` (phase-1 only) does not have. This script
loads such a checkpoint, drops the phase-2/3 keys, and saves a clean
``ssae_model.pt`` that ``scripts/extract_ssae_latents.py`` and
``scripts/extract_ssae_pb_all.py`` can load via their existing
``strict=False`` + allowlist path.

Cluster note: TamIA compute nodes are offline, so this script does NOT call
``huggingface_hub`` directly. The user (or a login-node prep step) must have
downloaded the checkpoint to a local path beforehand. The slurm wrapper
documents the required env var.

Usage:
  python scripts/audit_load_paper_ssae_checkpoint.py \
      --raw_ckpt /path/to/paper_ckpt.pt \
      --out_dir  $RUN_ROOT/runs/ssae_original_paper_ckpt_qwen0p5b \
      --model_name_or_path Qwen/Qwen2.5-0.5B \
      --sparsity_factor 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ssae.dataset import add_step_sep_token  # noqa: E402
from src.ssae.model_qwen_ssae import QwenSSAE  # noqa: E402


# Keys present in the paper's MyModel(phase=1) state_dict that our QwenSSAE
# does NOT have. They belong to phase 2/3 modules.
DROP_PREFIXES = ("mean_mlp.", "var_mlp.")
# Keys our QwenSSAE has but the paper's checkpoint may not (aux_head exists
# only in our contrastive variant). The extract scripts already tolerate
# this delta via the {aux_head.weight, aux_head.bias} allowlist.
ALLOWED_MISSING = {"aux_head.weight", "aux_head.bias"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw_ckpt", required=True, type=Path,
                   help="Path to the downloaded Miaow-Lab/SSAE checkpoint "
                        "(a .pt or .bin file, or directory containing exactly "
                        "one such file).")
    p.add_argument("--out_dir", required=True, type=Path,
                   help="Method run directory; will receive ssae_model.pt + "
                        "load_manifest.json.")
    p.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-0.5B",
                   help="HF model id the paper checkpoint was trained from. "
                        "Defaults to Qwen2.5-0.5B (the paper's setting).")
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--sparsity_factor", type=int, default=1)
    p.add_argument("--latent_norm_eps", type=float, default=1e-8)
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing ssae_model.pt in --out_dir.")
    return p.parse_args()


def resolve_ckpt_file(p: Path) -> Path:
    if p.is_file():
        return p
    if p.is_dir():
        candidates = sorted(
            list(p.glob("*.pt")) + list(p.glob("*.bin")) + list(p.glob("*.safetensors"))
        )
        if len(candidates) == 0:
            sys.exit(f"[load_paper_ssae] no .pt/.bin/.safetensors under {p}")
        if len(candidates) > 1:
            sys.exit(
                f"[load_paper_ssae] expected exactly 1 checkpoint file under "
                f"{p}, found {len(candidates)}: {candidates}. Pass a file "
                f"path directly via --raw_ckpt."
            )
        return candidates[0]
    sys.exit(f"[load_paper_ssae] --raw_ckpt path does not exist: {p}")


def normalize_state_dict(raw: dict) -> dict:
    """Strip the paper's phase-2/3 module weights so the result is loadable
    into our phase-1-only QwenSSAE.
    """
    if "model" in raw and isinstance(raw["model"], dict):
        sd = raw["model"]
    elif "state_dict" in raw and isinstance(raw["state_dict"], dict):
        sd = raw["state_dict"]
    elif "model_state" in raw and isinstance(raw["model_state"], dict):
        sd = raw["model_state"]
    else:
        sd = raw  # already a state_dict
    if not isinstance(sd, dict):
        sys.exit(f"[load_paper_ssae] unexpected checkpoint structure: {type(sd)}")
    dropped: list[str] = []
    kept: dict = {}
    for k, v in sd.items():
        if any(k.startswith(pref) for pref in DROP_PREFIXES):
            dropped.append(k)
            continue
        # The paper's DDP wrapper sometimes prefixes keys with "module."
        ck = k[len("module."):] if k.startswith("module.") else k
        kept[ck] = v
    return {"state_dict": kept, "dropped_keys": dropped}


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_ckpt = args.out_dir / "ssae_model.pt"
    if out_ckpt.exists() and not args.force:
        sys.exit(
            f"[load_paper_ssae] {out_ckpt} already exists; pass --force to overwrite."
        )

    ckpt_file = resolve_ckpt_file(args.raw_ckpt)
    print(f"[load_paper_ssae] loading raw checkpoint from {ckpt_file}", flush=True)
    raw = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    norm = normalize_state_dict(raw)
    sd = norm["state_dict"]
    dropped = norm["dropped_keys"]
    print(f"[load_paper_ssae] kept {len(sd)} keys, dropped {len(dropped)} "
          f"phase-2/3 keys", flush=True)

    # Instantiate the target QwenSSAE skeleton on CPU. Tokenizer is built the
    # same way the train/extract scripts build it, so the resized embedding
    # vocab size matches what the paper checkpoint expects (Qwen base + 1
    # extra special token).
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, local_files_only=args.local_files_only
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    sep_id = add_step_sep_token(tokenizer)
    print(f"[load_paper_ssae] tokenizer vocab_size={len(tokenizer)} sep_id={sep_id}",
          flush=True)

    model = QwenSSAE(
        tokenizer=tokenizer,
        model_name_or_path=args.model_name_or_path,
        sparsity_factor=args.sparsity_factor,
        phase=1,
        local_files_only=args.local_files_only,
        contrastive=False,
        latent_norm_eps=args.latent_norm_eps,
    )

    missing, unexpected = model.load_state_dict(sd, strict=False)
    bad_missing = [k for k in missing if k not in ALLOWED_MISSING]
    bad_unexpected = [k for k in unexpected if k not in ALLOWED_MISSING]
    if bad_missing or bad_unexpected:
        sys.exit(
            "[load_paper_ssae] unexpected key delta when loading paper "
            "checkpoint into QwenSSAE:\n"
            f"  bad_missing    = {bad_missing[:20]}{' ...' if len(bad_missing)>20 else ''}\n"
            f"  bad_unexpected = {bad_unexpected[:20]}{' ...' if len(bad_unexpected)>20 else ''}\n"
            f"Total missing={len(missing)} unexpected={len(unexpected)}."
        )

    # Quick shape sanity: encoder.embed_tokens row count must equal
    # len(tokenizer); autoencoder.encoder must be square (n_inputs == n_latents
    # at sparsity_factor=1).
    emb_rows = int(model.encoder.get_input_embeddings().weight.shape[0])
    if emb_rows != len(tokenizer):
        sys.exit(
            f"[load_paper_ssae] vocab mismatch after load: "
            f"encoder.embed_tokens has {emb_rows} rows but tokenizer "
            f"has {len(tokenizer)}. The paper checkpoint may have used a "
            f"different sep-token vocab size."
        )
    ae_w = model.autoencoder.encoder.weight
    print(f"[load_paper_ssae] n_inputs={model.n_inputs} n_latents={model.n_latents} "
          f"autoencoder.encoder.weight={tuple(ae_w.shape)}", flush=True)

    torch.save(model.state_dict(), out_ckpt)
    manifest = {
        "raw_ckpt": str(ckpt_file),
        "model_name_or_path": args.model_name_or_path,
        "sparsity_factor": args.sparsity_factor,
        "n_inputs": int(model.n_inputs),
        "n_latents": int(model.n_latents),
        "tokenizer_vocab_size": int(len(tokenizer)),
        "sep_token_id": int(sep_id),
        "kept_keys": len(sd),
        "dropped_keys": dropped,
        "missing_after_load": missing,
        "unexpected_after_load": unexpected,
        "notes": (
            "This run uses the original Miaow-Lab/SSAE paper checkpoint. "
            "It is Qwen2.5-0.5B based (hidden=896), NOT directly apples-to-"
            "apples with the Qwen2.5-1.5B dense_linear baseline."
        ),
    }
    (args.out_dir / "load_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[load_paper_ssae] wrote {out_ckpt} + load_manifest.json", flush=True)


if __name__ == "__main__":
    main()
