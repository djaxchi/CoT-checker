"""
Send the daily paper digest to Telegram via the Bot API.

Handles the 4096-character message limit by splitting into multiple messages.
Requires TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in the environment (via .env).
"""

import logging
import os
from datetime import date

import requests

logger = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"
MAX_MSG_LEN = 4000  # conservative margin below the 4096 hard limit


def _send(token: str, chat_id: str, text: str) -> None:
    url = TELEGRAM_API.format(token=token)
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    resp = requests.post(url, json=payload, timeout=15)
    if not resp.ok:
        logger.error(f"[telegram] send failed: {resp.status_code} {resp.text[:200]}")
    resp.raise_for_status()


def _split_messages(text: str, max_len: int = MAX_MSG_LEN) -> list[str]:
    """Split text into chunks that fit within Telegram's message size limit."""
    if len(text) <= max_len:
        return [text]

    chunks = []
    current = ""
    for line in text.splitlines(keepends=True):
        if len(current) + len(line) > max_len:
            if current:
                chunks.append(current.rstrip())
            current = line
        else:
            current += line
    if current:
        chunks.append(current.rstrip())
    return chunks


def _build_message(target_date: date, papers: list[dict], stats: dict, max_papers: int) -> str:
    high = [p for p in papers if p["score"] >= 2][:max_papers]
    header = (
        f"<b>Paper Digest — {target_date}</b>\n"
        f"Fetched: {stats['total']} | Pass 1: {stats['pass1']} | High-relevance: {stats['pass2_high']}\n"
    )

    if not high:
        return header + "\nNo high-relevance papers today."

    lines = [header]
    for i, p in enumerate(high, 1):
        profiles = ", ".join(p["profiles"]) if p["profiles"] else "—"
        title_trunc = p["title"][:120] + ("..." if len(p["title"]) > 120 else "")
        lines.append(
            f"\n{i}. [{p['score']}/3] <b>{title_trunc}</b>\n"
            f"   <a href=\"{p['url']}\">{p['url']}</a>\n"
            f"   Profiles: {profiles}\n"
            f"   {p['reason']}"
        )
    return "\n".join(lines)


def send_telegram_digest(
    target_date: date,
    papers: list[dict],
    stats: dict,
    max_papers: int = 15,
) -> None:
    """
    Send the daily digest to the Telegram chat configured in environment variables.
    Skips silently if credentials are not set.
    """
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")

    if not token or not chat_id:
        logger.warning("[telegram] TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set — skipping")
        return

    message = _build_message(target_date, papers, stats, max_papers)
    chunks = _split_messages(message)

    for i, chunk in enumerate(chunks):
        try:
            _send(token, chat_id, chunk)
            logger.info(f"[telegram] sent message {i + 1}/{len(chunks)}")
        except requests.RequestException as e:
            logger.error(f"[telegram] failed to send chunk {i + 1}: {e}")
            break
