"""Orchestrate one SSAE method end-to-end.

Pipeline (per spec section 16.1):
  1. SSAE representation training (launched via torchrun --nproc_per_node=N).
  2. Latent extraction for probe_train_40k / val_1k / processbench_gsm8k.
  3. Fresh linear probe + threshold selection + ProcessBench evaluation.

This script is the single entrypoint for one method. It always invokes
`torchrun` for step 1 (with --nproc_per_node=1 for smoke/single-GPU and
=4 for production), then runs steps 2-3 in the current process.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

METHOD_TRAIN_JSONL = {
    "ssae_positive": "prm800k_pos_base_20k.jsonl",
    "ssae_mixed": "prm800k_mixed_train_40k.jsonl",
    "ssae_contrastive": "prm800k_mixed_train_40k.jsonl",
}
METHOD_USES_LABELS = {
    "ssae_positive": False,
    "ssae_mixed": False,
    "ssae_contrastive": True,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=list(METHOD_TRAIN_JSONL))
    p.add_argument("--data_dir", required=True, type=Path)
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-1.5B")
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--phase", type=int, default=1)
    p.add_argument("--sparsity_factor", type=int, default=1)
    p.add_argument("--l1_weight", type=float, default=1e-4)
    p.add_argument("--bce_weight", type=float, default=0.1)
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--grad_accum_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=1e-6)
    p.add_argument("--min_lr", type=float, default=1e-7)
    p.add_argument("--warmup_iters", type=int, default=2)
    p.add_argument("--max_iters", type=int, default=30)
    p.add_argument("--lr_decay_iters", type=int, default=-1)
    p.add_argument("--train_attn_mask_ratio", type=float, default=0.1,
                   help="Official phase-1 step-token masking ratio.")
    p.add_argument("--gradient_checkpointing", action="store_true",
                   help="Enable HF gradient checkpointing on encoder+decoder.")
    p.add_argument("--ce_chunk_size", type=int, default=2048,
                   help="Active-token CE chunk size (memory knob).")
    p.add_argument("--debug_attn_mask", action="store_true",
                   help="Pass --debug_attn_mask to the trainer (verbose).")
    p.add_argument("--debug_grad_check", action="store_true",
                   help="Fail-fast on any NaN/Inf grad or param.")
    p.add_argument("--max_grad_norm", type=float, default=1.0,
                   help="Gradient clip norm threshold (canonical).")
    p.add_argument("--attn_implementation", type=str, default="eager",
                   choices=["eager", "sdpa", "flash_attention_2", "default"],
                   help="HF attention backend (eager bypasses SDPA quirks).")
    p.add_argument("--latent_norm_eps", type=float, default=1e-8,
                   help="Normalize eps (audit D1).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--nproc_per_node", type=int, default=1,
                   help="Number of GPUs for torchrun. Use 4 in production, 1 for smoke.")
    p.add_argument("--extract_batch_size", type=int, default=8)
    p.add_argument("--epochs_probe", type=int, default=50)
    p.add_argument("--probe_batch_size", type=int, default=512)
    p.add_argument("--lr_probe", type=float, default=1e-3)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke_train_n", type=int, default=128)
    p.add_argument("--smoke_val_n", type=int, default=32)
    p.add_argument("--skip_train", action="store_true")
    p.add_argument("--skip_extract", action="store_true")
    p.add_argument("--skip_probe", action="store_true")
    return p.parse_args()


def require_jsonl(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required JSONL missing: {path}. "
            "Do not silently fall back to cached .npy. Materialize this file "
            "before launching SSAE training (see spec section 1)."
        )


def child_env_clean() -> dict[str, str]:
    """Return os.environ without DDP env vars, for non-torchrun children."""
    env = os.environ.copy()
    for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR",
              "MASTER_PORT", "TORCHELASTIC_RUN_ID"):
        env.pop(k, None)
    return env


def run_cmd(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("[run_ssae_method] $", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_jsonl = args.data_dir / METHOD_TRAIN_JSONL[args.method]
    val_jsonl = args.data_dir / "prm800k_val_1k.jsonl"
    probe_train_jsonl = args.data_dir / "prm800k_probe_train_40k.jsonl"
    pb_jsonl = args.data_dir / "processbench_gsm8k.jsonl"
    for p in (train_jsonl, val_jsonl, probe_train_jsonl, pb_jsonl):
        require_jsonl(p)

    # ------------------------------------------------------------- step 1
    if not args.skip_train:
        torchrun = shutil.which("torchrun") or "torchrun"
        train_script = str(ROOT / "scripts" / "train_ssae_official.py")
        cmd = [
            torchrun, "--standalone",
            f"--nproc_per_node={args.nproc_per_node}",
            train_script,
            "--method", args.method,
            "--train_jsonl", str(train_jsonl),
            "--val_jsonl", str(val_jsonl),
            "--out_dir", str(args.out_dir),
            "--model_name_or_path", args.model_name_or_path,
            "--phase", str(args.phase),
            "--sparsity_factor", str(args.sparsity_factor),
            "--l1_weight", str(args.l1_weight),
            "--bce_weight", str(args.bce_weight),
            "--max_seq_len", str(args.max_seq_len),
            "--batch_size", str(args.batch_size),
            "--grad_accum_steps", str(args.grad_accum_steps),
            "--learning_rate", str(args.learning_rate),
            "--min_lr", str(args.min_lr),
            "--warmup_iters", str(args.warmup_iters),
            "--max_iters", str(args.max_iters),
            "--lr_decay_iters", str(args.lr_decay_iters),
            "--train_attn_mask_ratio", str(args.train_attn_mask_ratio),
            "--ce_chunk_size", str(args.ce_chunk_size),
            "--max_grad_norm", str(args.max_grad_norm),
            "--attn_implementation", args.attn_implementation,
            "--latent_norm_eps", str(args.latent_norm_eps),
            "--seed", str(args.seed),
        ]
        if args.local_files_only:
            cmd.append("--local_files_only")
        if args.gradient_checkpointing:
            cmd.append("--gradient_checkpointing")
        if args.debug_attn_mask:
            cmd.append("--debug_attn_mask")
        if args.debug_grad_check:
            cmd.append("--debug_grad_check")
        if args.smoke:
            cmd.extend([
                "--smoke",
                "--smoke_train_n", str(args.smoke_train_n),
                "--smoke_val_n", str(args.smoke_val_n),
            ])
        # torchrun sets RANK/WORLD_SIZE itself; pass a clean parent env.
        run_cmd(cmd, env=child_env_clean())
    else:
        print("[run_ssae_method] skipping SSAE training (--skip_train)")

    ckpt = args.out_dir / "ssae_model.pt"
    latents_dir = args.out_dir / "latents"
    latents_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------- step 2
    if not args.skip_extract:
        if not ckpt.exists():
            raise FileNotFoundError(f"Expected SSAE checkpoint missing: {ckpt}")
        cmd = [
            sys.executable, str(ROOT / "scripts" / "extract_ssae_latents.py"),
            "--ckpt", str(ckpt),
            "--model_name_or_path", args.model_name_or_path,
            "--sparsity_factor", str(args.sparsity_factor),
            "--max_seq_len", str(args.max_seq_len),
            "--batch_size", str(args.extract_batch_size),
            "--out_dir", str(latents_dir),
            "--probe_train_jsonl", str(probe_train_jsonl),
            "--val_jsonl", str(val_jsonl),
            "--pb_jsonl", str(pb_jsonl),
        ]
        if args.local_files_only:
            cmd.append("--local_files_only")
        if METHOD_USES_LABELS[args.method]:
            cmd.append("--contrastive_ckpt")
        if args.smoke:
            cmd.extend(["--limit", "64"])
        run_cmd(cmd, env=child_env_clean())
    else:
        print("[run_ssae_method] skipping latent extraction (--skip_extract)")

    # ------------------------------------------------------------- step 3
    if not args.skip_probe:
        cmd = [
            sys.executable, str(ROOT / "scripts" / "train_eval_ssae_probe.py"),
            "--method", args.method,
            "--latents_dir", str(latents_dir),
            "--out_dir", str(args.out_dir),
            "--seed", str(args.seed),
            "--epochs_probe", str(args.epochs_probe),
            "--batch_size", str(args.probe_batch_size),
            "--lr_probe", str(args.lr_probe),
        ]
        if args.smoke:
            cmd.append("--smoke")
        run_cmd(cmd, env=child_env_clean())
    else:
        print("[run_ssae_method] skipping probe step (--skip_probe)")


if __name__ == "__main__":
    main()
