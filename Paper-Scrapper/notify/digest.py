"""
Write a local Markdown digest file for archival purposes.
Saved to Paper-Scrapper/digests/YYYY-MM-DD.md.
"""

import logging
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)

DIGESTS_DIR = Path(__file__).parent.parent / "digests"


def write_digest_file(target_date: date, papers: list[dict], stats: dict) -> Path:
    """
    Write a markdown file summarising the day's relevant papers.
    Returns the path of the written file.
    """
    DIGESTS_DIR.mkdir(exist_ok=True)
    out_path = DIGESTS_DIR / f"{target_date}.md"

    high = [p for p in papers if p["score"] >= 2]

    lines = [
        f"# Paper Digest — {target_date}",
        "",
        f"**Stats:** fetched {stats['total']} | pass 1: {stats['pass1']} | high-relevance (≥2): {stats['pass2_high']}",
        "",
    ]

    if not high:
        lines.append("_No high-relevance papers today._")
    else:
        lines.append("## High-Relevance Papers\n")
        for i, p in enumerate(high, 1):
            profiles = ", ".join(p["profiles"]) if p["profiles"] else "—"
            lines += [
                f"### {i}. [{p['score']}/3] {p['title']}",
                f"- **Source:** {p['source']}",
                f"- **URL:** {p['url']}",
                f"- **Profiles:** {profiles}",
                f"- **Reason:** {p['reason']}",
                "",
            ]

    if len(papers) > len(high):
        mid = [p for p in papers if p["score"] == 1]
        if mid:
            lines.append("## Tangentially Related (score 1)\n")
            for p in mid:
                lines.append(f"- [{p['title']}]({p['url']})")
            lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"[digest] written to {out_path}")
    return out_path
