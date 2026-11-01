"""Access to the magazynflary/prod repo, where articles and authors are published"""

import contextlib
import datetime
import logging
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path

from magazynflary_audio.markdown_frontmatter import split_front_matter
from magazynflary_audio.types import ParserError

__all__ = ("PROD_REPO_ARTICLES_SUBDIR", "PROD_REPO_AUTHORS_SUBDIR", "clone_prod_repo", "find_articles")

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


def find_articles(
    repo_dir: Path, *, since_date: datetime.date | None, to_date: datetime.date | None, skip_drafts: bool
) -> list[Path]:
    """Returns articles with a publication date in [since_date, to_date], oldest first

    None for since_date or to_date means no bound on that side.
    """
    found: list[tuple[datetime.date, Path]] = []
    for path in sorted((repo_dir / PROD_REPO_ARTICLES_SUBDIR).glob("*.md")):
        front_matter, _ = split_front_matter(path.read_text(encoding="utf-8"))
        if skip_drafts and front_matter.get("draft", False):
            continue
        date = _publication_date(front_matter.get("date"), path)
        if (since_date is None or since_date <= date) and (to_date is None or date <= to_date):
            found.append((date, path))
    return [path for _, path in sorted(found)]


def _publication_date(value: object, path: Path) -> datetime.date:
    # YAML parses unquoted dates and datetimes itself; datetime is a subclass of date
    if isinstance(value, datetime.datetime):
        return value.date()
    if isinstance(value, datetime.date):
        return value
    raise ParserError(f"Front matter field 'date' is missing or not a date in {path.name}")
