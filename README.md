# CoT-Checker: Step-Level CoT Verification via Sparse Autoencoders

Reproduction of step-level correctness probing from
**"Step-Level Sparse Autoencoders for Interpretable Chain-of-Thought Verification"** (arXiv:2603.03031).

**Research question:** Can SSAE sparse latent vectors predict whether a reasoning step in a chain-of-thought trace is correct?

**Result:** 77.50% validation accuracy on 1,000 steps vs the paper's 78.58% (−1.08 pp, using 0.26% of the data).

---

## Setup

Requires Python 3.10+ and [uv](https://github.com/astral-sh/uv).

```bash
uv sync
```

Download the SSAE checkpoint from HuggingFace (Miaow-Lab/SSAE-Checkpoints):

```bash
python - <<'EOF'
from huggingface_hub import hf_hub_download
hf_hub_download("Miaow-Lab/SSAE-Checkpoints", "gsm8k-385k_Qwen2.5-0.5b_spar-10.pt", local_dir=".")
EOF
```

Set your Telegram credentials (optional — the pipeline sends a notification when done):

```bash
export TELEGRAM_BOT_TOKEN="your_bot_token"
export TELEGRAM_CHAT_ID="your_chat_id"
```

---

## Running the Pipeline

```bash
# Full pipeline: encode 1000 Math-Shepherd steps → train probe → report
bash scripts/run_pipeline.sh 1000 mps gsm8k-385k_Qwen2.5-0.5b_spar-10.pt

# Arguments: <max_steps> <device: cpu|cuda|mps> <checkpoint>
```

Or run each step manually:

```bash
# Step 1 — encode Math-Shepherd steps with SSAE
uv run python scripts/generate_probe_data.py \
    --checkpoint gsm8k-385k_Qwen2.5-0.5b_spar-10.pt \
    --output results/probe_data/math_shepherd_1000.npz \
    --max-steps 1000 \
    --device mps

# Step 2 — train correctness probe
uv run python scripts/train_probe.py \
    --data results/probe_data/math_shepherd_1000.npz \
    --output results/probes/correctness_probe_1000.pt \
    --epochs 30 \
    --device mps
```

---

## Results

![Probe training curve](results/probe_training_curve.png)

| Metric | This repo | Paper (SSAE-Qwen, GSM8K) |
|--------|-----------|--------------------------|
| Best val accuracy | **77.50%** (epoch 7) | **78.58%** |
| Majority baseline | 72.2% | 70.49% |
| Steps used | 1,000 | ~385,000 |

See [REPORT.md](REPORT.md) for full experimental setup and discussion.

---

## Repository Structure

```
src/
  saes/          SSAE model (Qwen2.5-0.5B backbone, sparse autoencoder)
  probes/        MLP probe (896 → 256 → 64 → 2)
  data/          Data loading utilities
scripts/
  generate_probe_data.py   Encode Math-Shepherd steps → .npz latents
  train_probe.py           Train MLP on latents, plot curve
  run_pipeline.sh          End-to-end pipeline wrapper
tests/                     Unit and integration tests (pytest)
results/
  probe_training_curve.png Training curve figure
```

---

## Tests

```bash
uv run pytest tests/ -m "not slow"
```

---

## Citation

If you use this code, please also cite the original paper:

```
@article{miaow2026ssae,
  title={Step-Level Sparse Autoencoders for Interpretable Chain-of-Thought Verification},
  author={Miaow-Lab},
  journal={arXiv:2603.03031},
  year={2026}
}
```
