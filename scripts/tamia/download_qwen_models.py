#!/usr/bin/env python3
"""Download Qwen2.5 model weights to the HF cache from a login node.

Usage (from project root on a login node):
    python scripts/tamia/download_qwen_models.py
"""
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = [
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-7B",
]

for model_id in MODELS:
    print(f"\nDownloading {model_id} ...")
    AutoTokenizer.from_pretrained(model_id)
    AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="float16")
    print(f"  done: {model_id}")

print("\nAll models downloaded.")
