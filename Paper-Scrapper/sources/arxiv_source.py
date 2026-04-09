"""
Fetch papers submitted to arXiv on a given date across cs.AI, cs.LG, cs.CL.

Uses the `arxiv` Python package (v2.x) which wraps the arXiv API.
The query uses a submittedDate range so we only pull papers from the target day.
"""

import logging
from datetime import date

import arxiv
import yaml
from pathlib import Path

from models import Paper

logger = logging.getLogger(__name__)


def _load_config() -> dict:
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def fetch_arxiv(target_date: date) -> list[Paper]:
    """
    Return all papers submitted on target_date in the configured arXiv categories.
    Date filtering uses the arXiv API submittedDate field.
    """
    config = _load_config()
    categories = config["sources"]["arxiv"]["categories"]
    max_results = config["sources"]["arxiv"]["max_results"]

    date_from = target_date.strftime("%Y%m%d") + "0000"
    date_to = target_date.strftime("%Y%m%d") + "2359"

    cat_filter = " OR ".join(f"cat:{c}" for c in categories)
    query = f"submittedDate:[{date_from} TO {date_to}] AND ({cat_filter})"

    logger.info(f"[arXiv] querying: {query}")

    client = arxiv.Client(num_retries=3, delay_seconds=3.0)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    papers: list[Paper] = []
    for result in client.results(search):
        arxiv_id = result.entry_id.split("/abs/")[-1]
        papers.append(
            Paper(
                id=arxiv_id,
                source="arxiv",
                title=result.title,
                abstract=result.summary,
                authors=[a.name for a in result.authors],
                url=result.entry_id,
                published_date=result.published.strftime("%Y-%m-%d"),
            )
        )

    logger.info(f"[arXiv] fetched {len(papers)} papers for {target_date}")
    return papers
