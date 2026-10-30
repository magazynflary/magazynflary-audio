"""Manages authors data - mapping author slugs to names"""

import logging
import subprocess
import tempfile
from pathlib import Path

import yaml

from magazynflary_audio.data import DATA_DIR
from magazynflary_audio.logging import setup_logging
from magazynflary_audio.markdown_frontmatter import required_string, split_front_matter

__all__ = ("get_author_name",)

LOG = logging.getLogger(__name__)

type AuthorSlug = str
type AuthorName = str

AUTHORS_FILE = DATA_DIR / "authors.yaml"
PROD_REPO_URL = "https://github.com/magazynflary/prod.git"
PROD_REPO_AUTHORS_SUBDIR = Path("content") / "autorzy"

PREDEFINED_AUTHORS: dict[AuthorSlug, AuthorName] = {
    "adam-sensowny": "Adam Sensowny",
}


def get_author_name(slug: str) -> str:
    result = yaml.safe_load(AUTHORS_FILE.read_text())[slug]
    assert isinstance(result, str)
    return result


def _fetch_authors() -> dict[AuthorSlug, AuthorName]:
    """Clones the prod repo and reads author names from the front matter of its author files"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        LOG.info("Cloning %s", PROD_REPO_URL)
        subprocess.run(["git", "clone", "--quiet", "--depth", "1", PROD_REPO_URL, tmp_dir], check=True)
        authors: dict[str, str] = {}
        for path in sorted((Path(tmp_dir) / PROD_REPO_AUTHORS_SUBDIR).glob("*.md")):
            front_matter, _ = split_front_matter(path.read_text())
            authors[path.stem] = required_string(front_matter, "name")
        return authors


def _update_authors_file(
    *, fetched_values: dict[AuthorSlug, AuthorName], authors_file_path: Path = AUTHORS_FILE
) -> None:
    LOG.info("Fetched authors count: %s", len(fetched_values))
    current_values = yaml.safe_load(authors_file_path.read_text()) if authors_file_path.exists() else PREDEFINED_AUTHORS
    LOG.info("Current authors count: %s", len(current_values))
    for slug in fetched_values:
        if slug in current_values:
            if current_values[slug] != fetched_values[slug]:
                raise RuntimeError("Name change", slug, current_values[slug], fetched_values[slug])
        else:
            LOG.info("New author: %s -> %s", slug, fetched_values[slug])
    updated_values = {**current_values, **fetched_values}

    LOG.info("Updating authors file")
    with authors_file_path.open("w") as f:
        yaml.safe_dump(updated_values, f, allow_unicode=True)


def main() -> None:
    setup_logging()
    fetched_authors = _fetch_authors()
    _update_authors_file(fetched_values=fetched_authors)


if __name__ == "__main__":
    main()
