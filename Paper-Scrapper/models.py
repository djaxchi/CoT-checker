from dataclasses import dataclass, field


@dataclass
class Paper:
    id: str            # arXiv ID (e.g. "2604.01234") or S2 corpus ID
    source: str        # "arxiv" | "semantic_scholar"
    title: str
    abstract: str
    authors: list[str]
    url: str
    published_date: str  # ISO format: "YYYY-MM-DD"


@dataclass
class ScreenedPaper:
    paper: Paper
    pass1_hit: bool
    pass2_score: int | None = None     # 0-3; None if not scored
    pass2_reason: str | None = None
    profiles_matched: list[str] = field(default_factory=list)
