import datetime
from pathlib import Path

import pytest

from magazynflary_audio.prod_repo import PROD_REPO_ARTICLES_SUBDIR, FoundArticle, find_articles


def _write_article(repo_dir: Path, stem: str, front_matter: str) -> Path:
    articles_dir = repo_dir / PROD_REPO_ARTICLES_SUBDIR
    articles_dir.mkdir(parents=True, exist_ok=True)
    path = articles_dir / f"{stem}.md"
    path.write_text(f"---\n{front_matter}\n---\nBody\n", encoding="utf-8")
    return path


@pytest.fixture
def repo_dir(tmp_path: Path) -> Path:
    # File names sort in a different order than publication dates, to check sorting by date
    _write_article(tmp_path, "a-newest", "date: 2024-03-01T20:00:00")
    _write_article(tmp_path, "b-oldest", "date: 2024-01-01")
    _write_article(tmp_path, "c-middle", "date: 2024-02-01T08:00:00")
    _write_article(tmp_path, "d-draft", "date: 2024-02-15\ndraft: true")
    _write_article(tmp_path, "e-not-draft", "date: 2024-02-20\ndraft: false")
    return tmp_path


def _stems(articles: list[FoundArticle]) -> list[str]:
    return [article.path.stem for article in articles]


def test_find_articles_unbounded_sorted_by_date(repo_dir: Path) -> None:
    found = find_articles(repo_dir, since_date=None, to_date=None, skip_drafts=False)
    assert _stems(found.articles) == ["b-oldest", "c-middle", "d-draft", "e-not-draft", "a-newest"]
    assert found.invalid == {}


def test_find_articles_publication_dates_drop_the_time_of_day(repo_dir: Path) -> None:
    found = find_articles(repo_dir, since_date=None, to_date=datetime.date(2024, 2, 1), skip_drafts=False)
    assert [article.publication_date for article in found.articles] == [
        datetime.date(2024, 1, 1),
        datetime.date(2024, 2, 1),
    ]


def test_find_articles_skip_drafts(repo_dir: Path) -> None:
    found = find_articles(repo_dir, since_date=None, to_date=None, skip_drafts=True)
    assert _stems(found.articles) == ["b-oldest", "c-middle", "e-not-draft", "a-newest"]


def test_find_articles_bounds_are_inclusive(repo_dir: Path) -> None:
    found = find_articles(
        repo_dir, since_date=datetime.date(2024, 2, 1), to_date=datetime.date(2024, 2, 20), skip_drafts=False
    )
    assert _stems(found.articles) == ["c-middle", "d-draft", "e-not-draft"]


def test_find_articles_only_since_date(repo_dir: Path) -> None:
    found = find_articles(repo_dir, since_date=datetime.date(2024, 2, 16), to_date=None, skip_drafts=False)
    assert _stems(found.articles) == ["e-not-draft", "a-newest"]


def test_find_articles_only_to_date(repo_dir: Path) -> None:
    found = find_articles(repo_dir, since_date=None, to_date=datetime.date(2024, 1, 31), skip_drafts=False)
    assert _stems(found.articles) == ["b-oldest"]


def test_find_articles_empty_repo(tmp_path: Path) -> None:
    found = find_articles(tmp_path, since_date=None, to_date=None, skip_drafts=False)
    assert found.articles == []
    assert found.invalid == {}


INVALID_FRONT_MATTER = {
    "missing-date": ("title: No date", "'date'"),
    "date-not-a-date": ('date: "not a date"', "'date'"),
    "draft-quoted-string": ('date: 2024-02-10\ndraft: "false"', "'draft'"),
    "draft-not-a-boolean": ("date: 2024-02-10\ndraft: 0", "'draft'"),
    "invalid-yaml": ("date: [2024-02-10", "YAML"),
}


@pytest.mark.parametrize(
    ("front_matter", "error_match"), INVALID_FRONT_MATTER.values(), ids=INVALID_FRONT_MATTER.keys()
)
@pytest.mark.parametrize("skip_drafts", [False, True])
def test_find_articles_invalid_front_matter_does_not_stop_search(
    repo_dir: Path, front_matter: str, error_match: str, skip_drafts: bool
) -> None:
    bad_path = _write_article(repo_dir, "bad", front_matter)

    found = find_articles(repo_dir, since_date=None, to_date=None, skip_drafts=skip_drafts)

    assert "bad" not in _stems(found.articles)
    assert "a-newest" in _stems(found.articles)
    assert list(found.invalid) == [bad_path]
    assert error_match in str(found.invalid[bad_path])


def test_find_articles_invalid_front_matter_reported_regardless_of_date_range(repo_dir: Path) -> None:
    # Its date is unknown, so it cannot be excluded by the range
    bad_path = _write_article(repo_dir, "bad", "title: No date")

    found = find_articles(repo_dir, since_date=datetime.date(2030, 1, 1), to_date=None, skip_drafts=False)

    assert found.articles == []
    assert list(found.invalid) == [bad_path]


def test_find_articles_no_front_matter(repo_dir: Path) -> None:
    bad_path = repo_dir / PROD_REPO_ARTICLES_SUBDIR / "bad.md"
    bad_path.write_text("Just a body\n", encoding="utf-8")

    found = find_articles(repo_dir, since_date=None, to_date=None, skip_drafts=False)

    assert list(found.invalid) == [bad_path]
