"""Extract SSAE latents for every available ProcessBench subset using a single
trained QwenSSAE checkpoint.

For each subset:
  - flatten raw ProcessBench JSONL (id, problem, steps, label) into a
    step-level JSONL with the schema the SSAE dataset expects
    (id, step_idx, label, n_steps, problem, prefix, candidate_step,
     pb_subset);
  - run the trained QwenSSAE encoder over every step and save
    L2-normalized latents:
        <out_root>/<subset>/pb_step_z.npy
        <out_root>/<subset>/pb_step_meta.jsonl
  - build a combined view that concatenates subsets (ids prefixed with
    "<subset>::").

A single ``latent_manifest_pb.json`` is written under --out_root capturing
the checkpoint, model, subset list, and timing.

Score / probe convention is identical to extract_ssae_latents.py so the
existing linear probe in the SSAE run directory can be reused as-is.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
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


SUBSET_RE = re.compile(r"^processbench[_-](?P<name>[A-Za-z0-9._-]+)\.(?:jsonl|json)$",
                       re.IGNORECASE)


def infer_subset_name(path: Path) -> str | None:
    m = SUBSET_RE.match(path.name)
    return m.group("name").lower() if m else None


def discover(pb_root: Path) -> list[tuple[str, Path]]:
    found: dict[str, Path] = {}
    for p in sorted(list(pb_root.glob("*.jsonl")) + list(pb_root.glob("*.json"))):
        name = infer_subset_name(p)
        if not name:
            continue
        existing = found.get(name)
        if existing is None or (existing.suffix == ".json" and p.suffix == ".jsonl"):
            found[name] = p
    return [(n, found[n]) for n in sorted(found.keys())]


def parse_pb_files(items: list[str]) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for s in items:
        if ":" not in s:
            sys.exit(f"--pb_files entry must be name:path, got {s!r}")
        n, p = s.split(":", 1)
        out.append((n.strip().lower(), Path(p)))
    return out


def load_traces(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    s = text.lstrip()
    if s.startswith("["):
        return json.loads(text)
    out: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def flatten_traces_to_step_jsonl(
    traces: list[dict], subset: str, out_jsonl: Path,
    shard_idx: int = 0, num_shards: int = 1,
) -> int:
    """Flatten traces into step-level rows, assigning a deterministic
    global_step_index. When num_shards > 1, only rows whose global index
    satisfies (gi % num_shards == shard_idx) are written.
    """
    required = {"id", "problem", "steps", "label"}
    # First pass: build the full ordered list so the global index is
    # identical across workers regardless of shard.
    all_rows: list[dict] = []
    for t in traces:
        missing = required - set(t.keys())
        if missing:
            raise ValueError(
                f"[{subset}] trace missing fields {missing}: id={t.get('id')}"
            )
        problem = t["problem"]
        steps = t["steps"]
        trace_label = int(t["label"])
        n_steps = len(steps)
        for k, step_text in enumerate(steps):
            prefix = "\n\n".join(steps[:k])
            all_rows.append({
                "id": t["id"],
                "step_idx": k,
                "label": trace_label,
                "n_steps": n_steps,
                "problem": problem,
                "prefix": prefix,
                "candidate_step": step_text,
                "pb_subset": subset,
            })
    for gi, r in enumerate(all_rows):
        r["global_step_index"] = gi

    n_rows = 0
    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in all_rows:
            if num_shards > 1 and (r["global_step_index"] % num_shards) != shard_idx:
                continue
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_rows += 1
    return n_rows


def extract_one(model, dataset, collator, batch_size: int,
                num_workers: int, device: str) -> tuple[np.ndarray, list[dict]]:
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collator, num_workers=num_workers, pin_memory=True,
    )
    zs: list[np.ndarray] = []
    metas: list[dict] = []
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attn = batch["attention_mask"].to(device, non_blocking=True)
            if device != "cpu" and torch.cuda.is_available():
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                    z = model.encode_latents(input_ids, attn)
            else:
                z = model.encode_latents(input_ids, attn)
            zs.append(z.float().cpu().numpy())
            metas.extend(batch["meta"])
    z_arr = np.concatenate(zs, axis=0).astype(np.float32)
    return z_arr, metas


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, required=True,
                   help="Trained QwenSSAE state_dict (ssae_model.pt).")
    p.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-1.5B")
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--sparsity_factor", type=int, default=1)
    p.add_argument("--contrastive_ckpt", action="store_true",
                   help="Pass when checkpoint was trained with aux head.")
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--out_root", type=Path, required=True,
                   help="Run-dir-relative output root (typically <run_dir>/latents_full_pb).")

    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--pb_root", type=Path,
                     help="Directory of processbench_<subset>.jsonl files.")
    grp.add_argument("--pb_files", nargs="+",
                     help="Explicit 'name:path' pairs.")

    p.add_argument("--force", action="store_true",
                   help="Overwrite existing per-subset z/meta.")
    p.add_argument("--keep_intermediate_jsonl", action="store_true",
                   help="Keep the flattened per-subset step JSONLs on disk.")
    p.add_argument("--shard_idx", type=int, default=0,
                   help="Worker shard index in [0, num_shards).")
    p.add_argument("--num_shards", type=int, default=1,
                   help="Total number of shards (e.g. 4 for 4-GPU extraction).")
    p.add_argument("--shard_subdir", action="store_true",
                   help="When set, write outputs under <out_root>/<subset>/"
                        "shards/shard_NN/ and SKIP the combined-view build "
                        "(merging is the orchestrator's job).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    if args.pb_root is not None:
        pairs = discover(args.pb_root)
        if not pairs:
            sys.exit(f"[extract_ssae_pb_all] No PB files under {args.pb_root}")
    else:
        pairs = parse_pb_files(args.pb_files)
        missing = [p for _, p in pairs if not p.exists()]
        if missing:
            sys.exit(f"[extract_ssae_pb_all] Missing PB files: {missing}")

    print(f"[extract_ssae_pb_all] {len(pairs)} subset(s):")
    for n, p in pairs:
        print(f"    {n:<14} {p}")

    if args.num_shards < 1 or not (0 <= args.shard_idx < args.num_shards):
        sys.exit(
            f"[extract_ssae_pb_all] bad shard config: "
            f"shard_idx={args.shard_idx} num_shards={args.num_shards}"
        )
    print(
        f"[extract_ssae_pb_all] shard config: shard_idx={args.shard_idx} "
        f"num_shards={args.num_shards} shard_subdir={args.shard_subdir}",
        flush=True,
    )

    def subset_out_dir(name: str) -> Path:
        if args.shard_subdir:
            return args.out_root / name / "shards" / f"shard_{args.shard_idx:02d}"
        return args.out_root / name

    if not args.force:
        for name, _ in pairs:
            z_path = subset_out_dir(name) / "pb_step_z.npy"
            if z_path.exists():
                sys.exit(
                    f"[extract_ssae_pb_all] Refusing to overwrite {z_path}. "
                    "Pass --force to re-extract."
                )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, local_files_only=args.local_files_only,
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
    missing_keys, unexpected_keys = model.load_state_dict(state, strict=False)
    allowed = {"aux_head.weight", "aux_head.bias"}
    bad_missing = [k for k in missing_keys if k not in allowed]
    bad_unexpected = [k for k in unexpected_keys if k not in allowed]
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            "Unexpected SSAE checkpoint mismatch:\n"
            f"  missing={bad_missing}\n  unexpected={bad_unexpected}"
        )
    if missing_keys or unexpected_keys:
        print(
            f"[extract_ssae_pb_all] aux_head delta on load "
            f"(contrastive_ckpt={args.contrastive_ckpt}): "
            f"missing={missing_keys} unexpected={unexpected_keys}"
        )
    model.eval()

    per_subset_info: list[dict] = []
    t_all = time.perf_counter()
    for name, src in pairs:
        sub_dir = subset_out_dir(name)
        sub_dir.mkdir(parents=True, exist_ok=True)
        step_jsonl = sub_dir / "_pb_step.jsonl"  # intermediate

        traces = load_traces(src)
        n_rows = flatten_traces_to_step_jsonl(
            traces, name, step_jsonl,
            shard_idx=args.shard_idx, num_shards=args.num_shards,
        )
        print(f"[extract_ssae_pb_all] [{name}] shard={args.shard_idx}/"
              f"{args.num_shards} flattened {n_rows} steps from "
              f"{len(traces)} traces -> {step_jsonl}", flush=True)

        # (id, step_idx) -> global_step_index, used to enrich the meta
        # rows the SSAE dataset emits (it does not pass arbitrary fields
        # through).
        gi_lookup: dict[tuple[str, int], int] = {}
        for line in step_jsonl.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            gi_lookup[(str(r["id"]), int(r["step_idx"]))] = int(r["global_step_index"])

        ds = SSAEJsonlDataset(step_jsonl, tokenizer,
                              max_seq_len=args.max_seq_len, limit=None)
        ds.length_audit(raise_on_violation=True)

        t0 = time.perf_counter()
        z, metas = extract_one(
            model, ds, collator, args.batch_size, args.num_workers, device,
        )
        dt = time.perf_counter() - t0

        z_path = sub_dir / "pb_step_z.npy"
        meta_path = sub_dir / "pb_step_meta.jsonl"
        np.save(z_path, z)
        with meta_path.open("w") as f:
            for m in metas:
                m.setdefault("pb_subset", name)
                key = (str(m.get("id")), int(m.get("step_idx", -1)))
                gi = gi_lookup.get(key)
                if gi is None:
                    sys.exit(
                        f"[extract_ssae_pb_all] missing global_step_index for "
                        f"id={key[0]} step_idx={key[1]} in subset {name}"
                    )
                m["global_step_index"] = gi
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        print(f"[extract_ssae_pb_all] [{name}] z.shape={z.shape} "
              f"time={dt:.1f}s -> {z_path}", flush=True)

        if not args.keep_intermediate_jsonl:
            step_jsonl.unlink(missing_ok=True)

        per_subset_info.append({
            "name": name,
            "src": str(src),
            "z_path": str(z_path),
            "meta_path": str(meta_path),
            "n_rows": int(z.shape[0]),
            "n_traces": len(traces),
            "extract_time_sec": dt,
            "sha256_z": sha256_file(z_path),
        })

    if args.shard_subdir:
        # Sharded workers do NOT build the combined view; the merge step
        # in the orchestrator script is responsible for assembling final
        # per-subset and combined outputs.
        manifest_shard = {
            "ckpt": str(args.ckpt),
            "model_name_or_path": args.model_name_or_path,
            "sparsity_factor": args.sparsity_factor,
            "contrastive_ckpt": args.contrastive_ckpt,
            "shard_idx": args.shard_idx,
            "num_shards": args.num_shards,
            "subsets": per_subset_info,
            "total_wall_sec": time.perf_counter() - t_all,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        manifest_path = (
            args.out_root / f"latent_manifest_pb_shard{args.shard_idx:02d}.json"
        )
        manifest_path.write_text(json.dumps(manifest_shard, indent=2))
        print(f"[extract_ssae_pb_all] shard {args.shard_idx} done. "
              f"manifest -> {manifest_path}")
        return

    # ---- Combined view ----------------------------------------------------
    combined_dir = args.out_root / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    combined_z_path = combined_dir / "pb_step_z.npy"
    combined_meta_path = combined_dir / "pb_step_meta.jsonl"
    if combined_z_path.exists() and not args.force:
        sys.exit(f"[extract_ssae_pb_all] Refusing to overwrite {combined_z_path}; --force")

    zs: list[np.ndarray] = []
    all_meta: list[dict] = []
    for info in per_subset_info:
        z = np.load(info["z_path"])
        zs.append(z)
        sub_meta = [json.loads(l) for l in Path(info["meta_path"]).read_text().splitlines() if l.strip()]
        for row in sub_meta:
            row["pb_subset"] = row.get("pb_subset", info["name"])
            row["id"] = f"{info['name']}::{row['id']}"
        all_meta.extend(sub_meta)
        if z.shape[0] != len(sub_meta):
            sys.exit(
                f"[extract_ssae_pb_all] {info['name']}: z rows ({z.shape[0]}) "
                f"!= meta rows ({len(sub_meta)})"
            )
    combined_z = np.concatenate(zs, axis=0) if zs else np.zeros((0,))
    np.save(combined_z_path, combined_z)
    with combined_meta_path.open("w") as f:
        for row in all_meta:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "ckpt": str(args.ckpt),
        "model_name_or_path": args.model_name_or_path,
        "sparsity_factor": args.sparsity_factor,
        "contrastive_ckpt": args.contrastive_ckpt,
        "n_latents": int(getattr(model, "n_latents", -1)),
        "n_inputs": int(getattr(model, "n_inputs", -1)),
        "subsets": per_subset_info,
        "combined": {
            "z_path": str(combined_z_path),
            "meta_path": str(combined_meta_path),
            "n_rows": int(combined_z.shape[0]),
            "sha256_z": sha256_file(combined_z_path),
        },
        "id_namespacing": "combined ids are prefixed '<subset>::<id>'",
        "total_wall_sec": time.perf_counter() - t_all,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (args.out_root / "latent_manifest_pb.json").write_text(json.dumps(manifest, indent=2))
    print(f"[extract_ssae_pb_all] Done. combined z={combined_z.shape}. "
          f"manifest -> {args.out_root / 'latent_manifest_pb.json'}")


if __name__ == "__main__":
    main()
