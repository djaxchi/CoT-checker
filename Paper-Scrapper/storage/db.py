"""
SQLite persistence layer for the paper scraper.

Schema
------
papers(
    id TEXT PRIMARY KEY,
    source TEXT,
    title TEXT,
    abstract TEXT,
    authors TEXT,        -- JSON-encoded list
    url TEXT,
    published_date TEXT,
    fetched_date TEXT,
    pass1_hit INTEGER,   -- 0 or 1
    pass2_score INTEGER, -- 0-3, NULL if not scored
    pass2_reason TEXT,
    profiles_matched TEXT  -- JSON-encoded list
)
"""

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import date
from pathlib import Path

from models import Paper, ScreenedPaper

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "papers.db"

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS papers (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    title TEXT NOT NULL,
    abstract TEXT,
    authors TEXT,
    url TEXT,
    published_date TEXT,
    fetched_date TEXT NOT NULL,
    pass1_hit INTEGER NOT NULL DEFAULT 0,
    pass2_score INTEGER,
    pass2_reason TEXT,
    profiles_matched TEXT
);
"""

INSERT_SQL = """
INSERT OR IGNORE INTO papers
    (id, source, title, abstract, authors, url, published_date, fetched_date,
     pass1_hit, pass2_score, pass2_reason, profiles_matched)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""


@contextmanager
def _connect(db_path: Path = DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: Path = DB_PATH) -> None:
    """Create the papers table if it doesn't already exist."""
    with _connect(db_path) as conn:
        conn.execute(CREATE_TABLE_SQL)
    logger.debug(f"[db] initialized at {db_path}")


def is_seen(paper_id: str, db_path: Path = DB_PATH) -> bool:
    """Return True if this paper ID is already in the database."""
    with _connect(db_path) as conn:
        row = conn.execute("SELECT 1 FROM papers WHERE id = ?", (paper_id,)).fetchone()
    return row is not None


def save_paper(screened: ScreenedPaper, fetched_date: date, db_path: Path = DB_PATH) -> None:
    """Insert a ScreenedPaper. Silently ignores duplicates (INSERT OR IGNORE)."""
    p = screened.paper
    with _connect(db_path) as conn:
        conn.execute(
            INSERT_SQL,
            (
                p.id,
                p.source,
                p.title,
                p.abstract,
                json.dumps(p.authors),
                p.url,
                p.published_date,
                fetched_date.isoformat(),
                int(screened.pass1_hit),
                screened.pass2_score,
                screened.pass2_reason,
                json.dumps(screened.profiles_matched),
            ),
        )


def get_digest_papers(target_date: date, threshold: int = 2, db_path: Path = DB_PATH) -> list[dict]:
    """
    Return papers fetched on target_date with pass2_score >= threshold,
    sorted by score descending.
    """
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, source, title, url, pass2_score, pass2_reason, profiles_matched
            FROM papers
            WHERE fetched_date = ? AND pass2_score >= ?
            ORDER BY pass2_score DESC
            """,
            (target_date.isoformat(), threshold),
        ).fetchall()

    results = []
    for row in rows:
        results.append({
            "id": row["id"],
            "source": row["source"],
            "title": row["title"],
            "url": row["url"],
            "score": row["pass2_score"],
            "reason": row["pass2_reason"] or "",
            "profiles": json.loads(row["profiles_matched"] or "[]"),
        })
    return results


def get_daily_stats(target_date: date, db_path: Path = DB_PATH) -> dict:
    """Return counts for the daily digest summary."""
    with _connect(db_path) as conn:
        total = conn.execute(
            "SELECT COUNT(*) FROM papers WHERE fetched_date = ?",
            (target_date.isoformat(),),
        ).fetchone()[0]
        pass1 = conn.execute(
            "SELECT COUNT(*) FROM papers WHERE fetched_date = ? AND pass1_hit = 1",
            (target_date.isoformat(),),
        ).fetchone()[0]
        pass2_high = conn.execute(
            "SELECT COUNT(*) FROM papers WHERE fetched_date = ? AND pass2_score >= 2",
            (target_date.isoformat(),),
        ).fetchone()[0]

    return {"total": total, "pass1": pass1, "pass2_high": pass2_high}
