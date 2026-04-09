"""
Pass 2 screening: LLM scores each paper abstract for relevance (0-3).

Uses OpenAI-compatible API so it works with OpenRouter (free models),
local Ollama, or any other OpenAI-compatible endpoint.
Set OPENROUTER_API_KEY (or OPENAI_API_KEY) and optionally OPENROUTER_BASE_URL in .env.

Score meaning:
  3 — directly relevant (new method, result, or dataset in the core research area)
  2 — clearly related (adjacent technique, relevant baseline, useful prior work)
  1 — tangentially related (mentions a key concept but outside the core scope)
  0 — not relevant

Returns a Pass2Result with score, matched profile names, and a one-sentence reason.
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import requests
import yaml

from models import Paper

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a research assistant for a project on mechanistic interpretability and \
chain-of-thought (CoT) verification. Your job is to score papers for relevance to \
this research area.

Research profiles:
1. method — Step-level sparse autoencoder (SAE/SSAE) probing to detect CoT errors
2. task — CoT verification, process reward models (PRM), first-error step detection
3. labels — Process supervision datasets with step-level labels (Math-Shepherd, PRM800K)
4. robustness — SAE artifacts, falsification, token injection, seed sensitivity

Scoring rubric:
  3 = directly relevant: new method, result, or dataset in the core area
  2 = clearly related: adjacent technique, useful baseline, or relevant prior work
  1 = tangentially related: mentions a key concept but outside the core scope
  0 = not relevant

Respond with ONLY a JSON object. No prose, no markdown, no code fences.
Format: {"score": <int>, "profiles": [<str>, ...], "reason": "<one sentence>"}
"""


@dataclass
class Pass2Result:
    score: int
    profiles: list[str] = field(default_factory=list)
    reason: str = ""


def _load_config() -> dict:
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def _call_openai_compatible(model: str, base_url: str, api_key: str, max_tokens: int, user_msg: str) -> str:
    """
    Call any OpenAI-compatible chat completions endpoint.
    Retries on 429 with exponential backoff (15s, 30s, 60s).
    """
    import time

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/djaxchi/CoT-checker",  # required by OpenRouter
    }
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    }
    url = f"{base_url}/chat/completions"
    for attempt in range(4):
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        if resp.status_code == 429:
            wait = 15 * (2 ** attempt)  # 15s, 30s, 60s, 120s
            logger.warning(f"[pass2] rate limited (429), waiting {wait}s (attempt {attempt + 1}/4)")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    resp.raise_for_status()  # raises after exhausting retries
    return ""  # unreachable


def pass2_score(paper: Paper, client=None, config: dict | None = None) -> Pass2Result:
    """
    Score a paper's relevance via an OpenAI-compatible API (OpenRouter, Ollama, etc.).
    Credentials are read from environment variables:
      OPENROUTER_API_KEY  — API key (required)
      OPENROUTER_BASE_URL — base URL (default: https://openrouter.ai/api/v1)
    Returns Pass2Result(score=0, ...) on any error.
    """
    if config is None:
        config = _load_config()

    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    model = config["screening"]["pass2_model"]
    max_tokens = config["screening"]["pass2_max_tokens"]

    if not api_key:
        logger.warning("[pass2] no API key found — skipping scoring")
        return Pass2Result(score=0, reason="no api key")

    abstract_excerpt = paper.abstract[:1500] if paper.abstract else "(no abstract)"
    user_msg = f"Title: {paper.title}\n\nAbstract: {abstract_excerpt}"

    try:
        raw_text = _call_openai_compatible(model, base_url, api_key, max_tokens, user_msg)
        # Extract the first {...} JSON object from the response.
        # Handles: plain JSON, markdown code fences, inline reasoning before JSON.
        match = re.search(r"\{[^{}]*\}", raw_text, re.DOTALL)
        if not match:
            raise json.JSONDecodeError("no JSON object found", raw_text, 0)
        data = json.loads(match.group())
        return Pass2Result(
            score=int(data.get("score", 0)),
            profiles=data.get("profiles") or [],
            reason=data.get("reason") or "",
        )
    except json.JSONDecodeError as e:
        logger.warning(f"[pass2] JSON parse error for '{paper.title}': {e}")
        return Pass2Result(score=0, reason="parse error")
    except Exception as e:
        logger.warning(f"[pass2] API error for '{paper.title}': {e}")
        return Pass2Result(score=0, reason=f"api error: {e}")
