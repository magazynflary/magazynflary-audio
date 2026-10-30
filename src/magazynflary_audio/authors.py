"""Manages authors data - mapping author slugs to names and genders"""

import dataclasses
import logging
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

import yaml

from magazynflary_audio.data import DATA_DIR
from magazynflary_audio.logging import setup_logging
from magazynflary_audio.markdown_frontmatter import required_string, split_front_matter

__all__ = ("AuthorInfo", "get_author_info")

LOG = logging.getLogger(__name__)

type AuthorSlug = str
type AuthorName = str

AUTHORS_FILE = DATA_DIR / "authors.yaml"
PROD_REPO_URL = "https://github.com/magazynflary/prod.git"
PROD_REPO_AUTHORS_SUBDIR = Path("content") / "autorzy"


@dataclasses.dataclass(frozen=True, slots=True)
class AuthorInfo:
    name: AuthorName
    is_female: bool


PREDEFINED_AUTHORS: dict[AuthorSlug, AuthorInfo] = {
    "adam-sensowny": AuthorInfo(name="Adam Sensowny", is_female=False),
}


def _load_authors(authors_file_path: Path) -> dict[AuthorSlug, AuthorInfo]:
    return {slug: AuthorInfo(**values) for slug, values in yaml.safe_load(authors_file_path.read_text()).items()}


def get_author_info(slug: str) -> AuthorInfo:
    return _load_authors(AUTHORS_FILE)[slug]


def _guess_is_female(name: AuthorName) -> bool:
    """Guesses gender from the first name - Polish female first names usually end with -a"""
    return name.split()[0].lower().endswith("a")


def _ask_is_female(name: AuthorName) -> bool:
    guess = _guess_is_female(name)
    prompt = f"Is {name} female? [{'Y/n' if guess else 'y/N'}] "
    while True:
        answer = input(prompt).strip().lower()
        if not answer:
            return guess
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False


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
    *,
    fetched_values: dict[AuthorSlug, AuthorName],
    authors_file_path: Path = AUTHORS_FILE,
    ask_is_female: Callable[[AuthorName], bool] = _ask_is_female,
) -> None:
    LOG.info("Fetched authors count: %s", len(fetched_values))
    current_values = _load_authors(authors_file_path) if authors_file_path.exists() else PREDEFINED_AUTHORS
    LOG.info("Current authors count: %s", len(current_values))
    updated_values = dict(current_values)
    for slug, name in fetched_values.items():
        if slug in current_values:
            if current_values[slug].name != name:
                raise RuntimeError("Name change", slug, current_values[slug].name, name)
        else:
            LOG.info("New author: %s -> %s", slug, name)
            updated_values[slug] = AuthorInfo(name=name, is_female=ask_is_female(name))

    LOG.info("Updating authors file")
    with authors_file_path.open("w") as f:
        yaml.safe_dump(
            {slug: dataclasses.asdict(author) for slug, author in updated_values.items()}, f, allow_unicode=True
        )


def main() -> None:
    setup_logging()
    fetched_authors = _fetch_authors()
    _update_authors_file(fetched_values=fetched_authors)


if __name__ == "__main__":
    main()
