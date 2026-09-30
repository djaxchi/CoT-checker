"""Checkers that score a freshly written step during generation (online_reject_v2).

`online_bon.Checker` re-reads a step under the verifier template, one step_tokens
cell at a time. These read what a deployment would have:

  GenStateChecker  the policy's own states for the step, from one causal pass over
                   the sampler's context plus the solution so far. Every cell in
                   `cells` reads the same pass, so scoring two probes costs one
                   backbone forward. Readouts are those of score_gen_states_multi,
                   which a test pins against the training derivation.
  PRMChecker       Qwen2.5-Math-PRM-7B on the problem and the steps so far; the
                   candidate's suspicion is 1 - P(correct) at its separator.
  PanelChecker     scores each candidate with every checker (so every draft of
                   every arm carries all scores for inspection), and decides with
                   the `active` one. Keeps per-checker wall time apart from
                   generation.

All return SUSPICION: higher means more likely wrong, as the offline score files
store it, so a rejection threshold is a quantile of the offline pool's scores.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.onpolicy.score_gen_states_multi import Cell, step_blocks  # noqa: E402
from src.onpolicy.prompts import context, verifier_prefix  # noqa: E402
from src.onpolicy.spans import step_token_spans  # noqa: E402

STEP_SEP = "\n\n"


def _sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


class GenStateChecker:
    def __init__(self, cells: list[Path], backbone, tok, layer: int, device: str,
                 prompt_style: str):
        self.cells = [Cell(d, device) for d in cells]
        self.backbone, self.tok, self.layer = backbone, tok, layer
        self.device, self.style = device, prompt_style
        self.seconds_backbone = 0.0

    def _n(self, s: str) -> int:
        return len(self.tok(s, add_special_tokens=False)["input_ids"])

    @torch.no_grad()
    def score_all(self, problem: str, prior: list[str], cand: str,
                  dataset: str = "", n_shot: int = 4) -> dict[str, float]:
        """Suspicion of `cand` as the next step, for every cell, from one pass."""
        steps = prior + [cand]
        prompt = context(self.style, problem, "", dataset, n_shot)
        p_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        s_ids = self.tok(STEP_SEP.join(steps), add_special_tokens=False)["input_ids"]
        spans = [(x, min(y, len(s_ids))) for x, y in step_token_spans(prompt, steps, self._n)]
        _sync(self.device); t0 = time.perf_counter()
        h = self.backbone(input_ids=torch.tensor([p_ids + s_ids], device=self.device),
                          output_hidden_states=True).hidden_states[self.layer][0]
        h = h.to(torch.float16).cpu().numpy()
        _sync(self.device); self.seconds_backbone += time.perf_counter() - t0
        block = step_blocks(h, len(p_ids), spans)[-1]
        return {c.name: float(c.score([block])[0]) for c in self.cells}


class TemplateChecker(GenStateChecker):
    """The probes' training context, exactly: the step re-read under the verifier
    template ("Problem: ... Previous reasoning: ... Current step:"), prefix
    tokenised with special tokens and the step without, as the PRM800K and
    ProcessBench encoders did. The block is [last prefix token; step tokens],
    the span-store layout. Costs one extra backbone pass per draft."""

    @torch.no_grad()
    def score_all(self, problem: str, prior: list[str], cand: str,
                  dataset: str = "", n_shot: int = 4) -> dict[str, float]:
        p_ids = self.tok(verifier_prefix(problem, STEP_SEP.join(prior)),
                         add_special_tokens=True)["input_ids"]
        s_ids = self.tok(cand, add_special_tokens=False)["input_ids"] or p_ids[-1:]
        _sync(self.device); t0 = time.perf_counter()
        h = self.backbone(input_ids=torch.tensor([p_ids + s_ids], device=self.device),
                          output_hidden_states=True).hidden_states[self.layer][0]
        block = h[len(p_ids) - 1:].to(torch.float16).cpu().numpy()
        _sync(self.device); self.seconds_backbone += time.perf_counter() - t0
        return {c.name: float(c.score([block])[0]) for c in self.cells}


class PRMChecker:
    name = "prm_qwen25_math_7b"

    def __init__(self, prm_name_or_path: str, device: str, local_files_only: bool = True,
                 bf16_rewards: bool = False):
        from transformers import AutoModel, AutoTokenizer
        from scripts.onpolicy.score_traces_with_prm import score_one
        self._score_one = score_one
        self.tok = AutoTokenizer.from_pretrained(prm_name_or_path,
                                                 local_files_only=local_files_only)
        model, info = AutoModel.from_pretrained(
            prm_name_or_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
            local_files_only=local_files_only, output_loading_info=True)
        if info.get("missing_keys"):
            raise SystemExit(f"[prm] {len(info['missing_keys'])} weights not loaded")
        self.model = model.to(device).eval()
        self.device, self.seconds = device, 0.0
        # Pools scored before 4b4731c used a bfloat16 softmax, so their PRM
        # scores sit on bfloat16's grid (0, 1/256, 1/128 ...). Thresholds taken
        # from such a pool need the live reward rounded to the same grid
        # (bf16_rewards=True); otherwise a quantile of exactly 0 would reject any
        # step with a float32 reward below 1. Pools scored since then are float32
        # and compare directly.
        self.bf16_rewards = bf16_rewards

    @torch.no_grad()
    def score(self, problem: str, prior: list[str], cand: str) -> float:
        _sync(self.device); t0 = time.perf_counter()
        rewards = self._score_one(self.model, self.tok, problem, prior + [cand or "."],
                                  self.device)
        _sync(self.device); self.seconds += time.perf_counter() - t0
        r = float(rewards[-1])
        if self.bf16_rewards:
            r = float(torch.tensor(r, dtype=torch.float32).to(torch.bfloat16).float())
        return 1.0 - r


class PanelChecker:
    """Every checker scores every candidate; `active` decides."""

    def __init__(self, gen: GenStateChecker | None, prm: PRMChecker | None, active: str):
        self.gen, self.prm, self.active = gen, prm, active
        names = ([c.name for c in gen.cells] if gen else []) + ([prm.name] if prm else [])
        if active not in names and active != "none":
            raise ValueError(f"active checker {active!r} is not among {names}")
        self.names = names
        self.last: list[dict[str, float]] = []
        self.calls = 0

    def score_steps(self, problem: str, prior: list[str], candidates: list[str],
                    dataset: str = "", n_shot: int = 4) -> list[float]:
        self.last = []
        for c in candidates:
            s = self.gen.score_all(problem, prior, c, dataset, n_shot) if self.gen else {}
            if self.prm:
                s[self.prm.name] = self.prm.score(problem, prior, c)
            self.last.append(s)
            self.calls += 1
        return [s.get(self.active, 0.0) for s in self.last]

    def timing(self) -> dict:
        return {"seconds_genstate_backbone": self.gen.seconds_backbone if self.gen else 0.0,
                "seconds_heads": {c.name: c.seconds for c in self.gen.cells} if self.gen else {},
                "seconds_prm": self.prm.seconds if self.prm else 0.0}
