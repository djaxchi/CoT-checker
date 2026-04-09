#!/usr/bin/env python3
"""
Daily paper surveillance scraper for CoT-checker research.

Usage:
  python scraper.py                          # scrape yesterday's papers
  python scraper.py --date 2026-04-07        # scrape a specific date
  python scraper.py --dry-run                # fetch + screen, no DB write, no Telegram
  python scraper.py --date 2026-04-07 --dry-run

Cron (runs daily at 7am):
  0 7 * * * cd /path/to/Paper-Scrapper && python scraper.py >> logs/scraper.log 2>&1
"""

import argparse
import logging
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Load .env from the same directory as this script
load_dotenv(Path(__file__).parent / ".env")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("scraper")

# ---------------------------------------------------------------------------
# Imports from local modules (after path is set)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))

from models import ScreenedPaper
from sources import fetch_arxiv, fetch_semantic_scholar
from screening import pass1_filter, pass2_score
from storage import init_db, is_seen, save_paper, get_digest_papers
from storage.db import get_daily_stats
from notify import send_telegram_digest, write_digest_file


def _load_config() -> dict:
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def run(target_date: date, dry_run: bool = False) -> None:
    config = _load_config()
    threshold = config["screening"]["pass2_threshold"]
    max_papers = config["telegram"]["max_papers_in_message"]

    logger.info(f"=== Paper scraper starting — target date: {target_date} (dry_run={dry_run}) ===")

    # ------------------------------------------------------------------
    # 1. Init DB
    # ------------------------------------------------------------------
    if not dry_run:
        init_db()

    # ------------------------------------------------------------------
    # 2. Fetch papers from all sources
    # ------------------------------------------------------------------
    logger.info("Fetching from arXiv...")
    arxiv_papers = fetch_arxiv(target_date)

    logger.info("Fetching from Semantic Scholar...")
    s2_papers = fetch_semantic_scholar(target_date)

    all_papers = arxiv_papers + s2_papers
    logger.info(f"Total fetched: {len(all_papers)} ({len(arxiv_papers)} arXiv + {len(s2_papers)} S2)")

    # ------------------------------------------------------------------
    # 3. Dedup against DB
    # ------------------------------------------------------------------
    if not dry_run:
        new_papers = [p for p in all_papers if not is_seen(p.id)]
        logger.info(f"New (not yet in DB): {len(new_papers)}")
    else:
        new_papers = all_papers
        logger.info(f"[dry-run] skipping dedup, processing all {len(new_papers)} papers")

    # ------------------------------------------------------------------
    # 4. Pass 1 — keyword filter
    # ------------------------------------------------------------------
    pass1_hits = [p for p in new_papers if pass1_filter(p, config)]
    logger.info(f"Pass 1 hits: {len(pass1_hits)} / {len(new_papers)}")

    # ------------------------------------------------------------------
    # 5. Pass 2 — LLM relevance scoring (OpenRouter / any OpenAI-compatible API)
    # ------------------------------------------------------------------
    has_api_key = bool(os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY"))
    if not has_api_key:
        logger.warning("No API key found (OPENROUTER_API_KEY / OPENAI_API_KEY) — Pass 2 scoring will be skipped")

    screened: list[ScreenedPaper] = []

    # Papers that didn't pass Pass 1 — store with pass1_hit=0, no Pass 2
    for paper in new_papers:
        if paper not in pass1_hits:
            screened.append(ScreenedPaper(paper=paper, pass1_hit=False))

    # Papers that passed Pass 1 — run Pass 2
    for i, paper in enumerate(pass1_hits):
        if has_api_key:
            logger.info(f"[pass2] {i + 1}/{len(pass1_hits)}: {paper.title[:80]!r}")
            result = pass2_score(paper, config=config)
            sp = ScreenedPaper(
                paper=paper,
                pass1_hit=True,
                pass2_score=result.score,
                pass2_reason=result.reason,
                profiles_matched=result.profiles,
            )
            logger.info(f"[pass2] score={result.score} profiles={result.profiles}")
        else:
            sp = ScreenedPaper(paper=paper, pass1_hit=True)
        screened.append(sp)

    high_count = sum(1 for s in screened if s.pass2_score is not None and s.pass2_score >= threshold)
    logger.info(f"Pass 2 high-relevance (>= {threshold}): {high_count}")

    # ------------------------------------------------------------------
    # 6. Persist
    # ------------------------------------------------------------------
    if not dry_run:
        for sp in screened:
            save_paper(sp, fetched_date=target_date)
        logger.info(f"Saved {len(screened)} papers to DB")
    else:
        logger.info("[dry-run] skipping DB writes")
        # Print top papers to stdout for inspection
        top = sorted(
            [s for s in screened if s.pass2_score is not None],
            key=lambda s: s.pass2_score,
            reverse=True,
        )[:10]
        if top:
            print("\n--- Top papers (dry-run) ---")
            for s in top:
                print(f"[{s.pass2_score}/3] {s.paper.title}")
                print(f"   {s.paper.url}")
                print(f"   profiles: {s.profiles_matched}")
                print(f"   {s.pass2_reason}\n")
        return

    # ------------------------------------------------------------------
    # 7. Build digest and notify
    # ------------------------------------------------------------------
    digest_papers = get_digest_papers(target_date, threshold=threshold)
    stats = get_daily_stats(target_date)

    logger.info(f"Writing digest file...")
    digest_path = write_digest_file(target_date, digest_papers, stats)
    logger.info(f"Digest written to {digest_path}")

    logger.info("Sending Telegram notification...")
    send_telegram_digest(target_date, digest_papers, stats, max_papers=max_papers)

    logger.info("=== Done ===")


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily paper surveillance scraper")
    parser.add_argument(
        "--date",
        type=date.fromisoformat,
        default=date.today() - timedelta(days=1),
        help="Target date in YYYY-MM-DD format (default: yesterday)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and screen papers without writing to DB or sending Telegram",
    )
    args = parser.parse_args()
    run(target_date=args.date, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
