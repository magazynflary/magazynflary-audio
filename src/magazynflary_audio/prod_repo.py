"""Access to the magazynflary/prod repo, where articles and authors are published"""

import contextlib
import dataclasses
import datetime
import logging
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path

from magazynflary_audio.markdown_frontmatter import split_front_matter
from magazynflary_audio.types import ParserError

__all__ = (
    "PROD_REPO_ARTICLES_SUBDIR",
    "PROD_REPO_AUTHORS_SUBDIR",
    "FoundArticle",
    "FoundArticles",
    "clone_prod_repo",
    "find_articles",
)

LOG = logging.getLogger(__name__)

PROD_REPO_URL = "https://github.com/magazynflary/prod.git"
PROD_REPO_ARTICLES_SUBDIR = Path("content") / "artykuly"
PROD_REPO_AUTHORS_SUBDIR = Path("content") / "autorzy"


@contextlib.contextmanager
def clone_prod_repo() -> Iterator[Path]:
    """Shallow-clones the prod repo into a temporary directory, removed on exit"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        LOG.info("Cloning %s", PROD_REPO_URL)
        subprocess.run(["git", "clone", "--quiet", "--depth", "1", PROD_REPO_URL, tmp_dir], check=True)
        yield Path(tmp_dir)


@dataclasses.dataclass(frozen=True, slots=True)
class FoundArticle:
    path: Path
    publication_date: datetime.date
    """Front matter `date`, without the time of day"""


@dataclasses.dataclass(frozen=True, slots=True)
class FoundArticles:
    articles: list[FoundArticle]
    """Articles with a publication date in the requested range, oldest first"""

    invalid: dict[Path, ParserError]
    """Articles whose front matter could not be read, so it is unknown whether they are in the range"""


def find_articles(
    repo_dir: Path, *, since_date: datetime.date | None, to_date: datetime.date | None, skip_drafts: bool
) -> FoundArticles:
    """Finds articles with a publication date in [since_date, to_date]

    None for since_date or to_date means no bound on that side. An article with invalid front matter does not stop
    the search, it is reported in FoundArticles.invalid instead.
    """
    found: list[tuple[datetime.date, Path]] = []
    invalid: dict[Path, ParserError] = {}
    for path in sorted((repo_dir / PROD_REPO_ARTICLES_SUBDIR).glob("*.md")):
        try:
            front_matter, _ = split_front_matter(path.read_text(encoding="utf-8"))
            is_draft = _is_draft(front_matter.get("draft", False))
            date = _publication_date(front_matter.get("date"))
        except ParserError as e:
            invalid[path] = e
            continue
        if skip_drafts and is_draft:
            continue
        if (since_date is None or since_date <= date) and (to_date is None or date <= to_date):
            found.append((date, path))
    return FoundArticles(
        articles=[FoundArticle(path=path, publication_date=date) for date, path in sorted(found)], invalid=invalid
    )


def _is_draft(value: object) -> bool:
    # Strict, so that e.g. a quoted "false" is not taken as a draft just because it is a non-empty string
    if isinstance(value, bool):
        return value
    raise ParserError(f"Front matter field 'draft' is not a boolean: {value!r}")


def _publication_date(value: object) -> datetime.date:
    # YAML parses unquoted dates and datetimes itself; datetime is a subclass of date
    if isinstance(value, datetime.datetime):
        return value.date()
    if isinstance(value, datetime.date):
        return value
    raise ParserError(f"Front matter field 'date' is missing or not a date: {value!r}")
