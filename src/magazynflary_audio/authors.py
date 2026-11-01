"""Manages authors data - mapping author slugs to names and genders"""

import dataclasses
import logging
from collections.abc import Callable
from pathlib import Path

import yaml

from magazynflary_audio.data import DATA_DIR
from magazynflary_audio.logging import setup_logging
from magazynflary_audio.markdown_frontmatter import required_string, split_front_matter
from magazynflary_audio.prod_repo import PROD_REPO_AUTHORS_SUBDIR, clone_prod_repo

__all__ = ("UnknownAuthorError", "get_author_info")

from magazynflary_audio.types import AuthorInfo

LOG = logging.getLogger(__name__)

type AuthorSlug = str
type AuthorName = str

AUTHORS_FILE = DATA_DIR / "authors.yaml"


class UnknownAuthorError(KeyError):
    def __init__(self, slug: str) -> None:
        super().__init__(slug)
        self.slug = slug


def get_author_info(slug: str) -> AuthorInfo:
    """Returns the author's info or raises UnknownAuthorError if the author is not in the authors file"""
    try:
        return _load_authors(AUTHORS_FILE)[slug]
    except KeyError:
        raise UnknownAuthorError(slug) from None


def _load_authors(authors_file_path: Path) -> dict[AuthorSlug, AuthorInfo]:
    return {slug: AuthorInfo(**values) for slug, values in yaml.safe_load(authors_file_path.read_text()).items()}


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
    with clone_prod_repo() as repo_dir:
        authors: dict[str, str] = {}
        for path in sorted((repo_dir / PROD_REPO_AUTHORS_SUBDIR).glob("*.md")):
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
    current_values = _load_authors(authors_file_path) if authors_file_path.exists() else {}
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
