#!/usr/bin/env python3
"""Write model_config.json for one ablation backbone (no weights loaded).

Records the architecture metadata the leaderboard needs, read straight from the
HF config so nothing is hardcoded. Runs on CPU.

Usage:
    python scripts/s1ms_dump_model_config.py \\
        --model_name_or_path Qwen/Qwen2.5-7B \\
        --params_label 7B \\
        --out_json <model_dir>/model_config.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--params_label", type=str, required=True)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--out_json", type=Path, required=True)
    args = p.parse_args()

    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(args.model_name_or_path, local_files_only=args.local_files_only)

    def g(name: str):
        return getattr(cfg, name, None)

    payload = {
        "model_name": args.model_name_or_path,
        "params_label": args.params_label,
        "model_type": g("model_type"),
        "hidden_size": g("hidden_size"),
        "num_hidden_layers": g("num_hidden_layers"),
        "num_attention_heads": g("num_attention_heads"),
        "num_key_value_heads": g("num_key_value_heads"),
        "intermediate_size": g("intermediate_size"),
        "vocab_size": g("vocab_size"),
        "max_position_embeddings": g("max_position_embeddings"),
        "torch_dtype": str(g("torch_dtype")),
        "rope_theta": g("rope_theta"),
        "tie_word_embeddings": g("tie_word_embeddings"),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2))
    print(f"[model_config] {args.model_name_or_path}: "
          f"hidden_size={payload['hidden_size']} layers={payload['num_hidden_layers']} "
          f"heads={payload['num_attention_heads']} kv_heads={payload['num_key_value_heads']} "
          f"-> {args.out_json}")


if __name__ == "__main__":
    main()
