"""
Fetch papers from the Semantic Scholar API for a given date.

Uses the public S2 Graph API (no API key required, rate-limited to ~1 req/sec).
Runs each query from config.yaml and deduplicates by paperId.
"""

import logging
import time
from datetime import date

import requests
import yaml
from pathlib import Path

from models import Paper

logger = logging.getLogger(__name__)

S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_FIELDS = "paperId,title,abstract,authors,url,year,publicationDate,externalIds"


def _load_config() -> dict:
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def _s2_paper_to_model(raw: dict) -> Paper | None:
    """Convert a S2 API result dict to a Paper. Returns None if missing required fields."""
    paper_id = raw.get("paperId")
    title = raw.get("title") or ""
    abstract = raw.get("abstract") or ""

    if not paper_id or not title:
        return None

    authors = [a.get("name", "") for a in (raw.get("authors") or [])]
    pub_date = raw.get("publicationDate") or str(raw.get("year") or "")

    # Prefer arXiv URL if available
    external = raw.get("externalIds") or {}
    arxiv_id = external.get("ArXiv")
    if arxiv_id:
        url = f"https://arxiv.org/abs/{arxiv_id}"
        paper_id = f"s2:{paper_id}"  # namespace to avoid collision with arxiv IDs
    else:
        url = raw.get("url") or f"https://www.semanticscholar.org/paper/{paper_id}"
        paper_id = f"s2:{paper_id}"

    return Paper(
        id=paper_id,
        source="semantic_scholar",
        title=title,
        abstract=abstract,
        authors=authors,
        url=url,
        published_date=pub_date,
    )


def fetch_semantic_scholar(target_date: date) -> list[Paper]:
    """
    Run each configured query against S2 and return papers published on target_date.
    Results are deduplicated by paperId.
    """
    config = _load_config()
    s2_cfg = config["sources"]["semantic_scholar"]
    queries = s2_cfg["queries"]
    max_results = s2_cfg["max_results_per_query"]
    delay = s2_cfg["rate_limit_delay"]

    date_str = target_date.strftime("%Y-%m-%d")
    seen_ids: set[str] = set()
    papers: list[Paper] = []

    for query in queries:
        logger.info(f"[S2] querying: {query!r}")
        params = {
            "query": query,
            "fields": S2_FIELDS,
            "publicationDateOrYear": date_str,
            "limit": max_results,
        }
        try:
            for attempt in range(3):
                resp = requests.get(S2_SEARCH_URL, params=params, timeout=30)
                if resp.status_code == 429:
                    wait = delay * (2 ** attempt) + 2
                    logger.warning(f"[S2] rate limited, retrying in {wait:.1f}s (attempt {attempt + 1}/3)")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                break
            else:
                logger.warning(f"[S2] gave up after 3 attempts for query {query!r}")
                time.sleep(delay)
                continue
            data = resp.json()
        except requests.RequestException as e:
            logger.warning(f"[S2] request failed for query {query!r}: {e}")
            time.sleep(delay)
            continue

        for raw in data.get("data", []):
            paper = _s2_paper_to_model(raw)
            if paper and paper.id not in seen_ids:
                seen_ids.add(paper.id)
                papers.append(paper)

        logger.info(f"[S2] got {len(data.get('data', []))} results (total unique so far: {len(papers)})")
        time.sleep(delay)

    logger.info(f"[S2] fetched {len(papers)} unique papers for {target_date}")
    return papers
