import datetime
from pathlib import Path

from magazynflary_audio.cli import _group_by_output_path
from magazynflary_audio.prod_repo import FoundArticle

OUT = Path("out")


def _found(stem: str, date: datetime.date = datetime.date(2026, 5, 7)) -> FoundArticle:
    return FoundArticle(path=Path("content/artykuly") / f"{stem}.md", publication_date=date)


def test_group_by_output_path() -> None:
    a, b = _found("samotni-wśród-ludzi"), _found("inny")
    groups, errors = _group_by_output_path([a, b], output_dir=OUT, output_format="mp3")
    assert groups == {
        OUT / "2026-05-07-samotni-wsrod-ludzi.mp3": [a],
        OUT / "2026-05-07-inny.mp3": [b],
    }
    assert errors == []


def test_conflicting_articles_are_grouped_together() -> None:
    # Equal after transliteration and on the same day: one output path, both articles (the caller reads neither).
    a, b = _found("samotni-wśród-ludzi"), _found("samotni-wsrod-ludzi")
    other_day = _found("samotni-wsrod-ludzi", datetime.date(2026, 5, 8))
    groups, _ = _group_by_output_path([a, b, other_day], output_dir=OUT, output_format="mp3")
    assert groups[OUT / "2026-05-07-samotni-wsrod-ludzi.mp3"] == [a, b]
    assert groups[OUT / "2026-05-08-samotni-wsrod-ludzi.mp3"] == [other_day]


def test_naming_errors_are_reported_separately() -> None:
    bad, good = _found("!?"), _found("inny")
    groups, errors = _group_by_output_path([bad, good], output_dir=OUT, output_format="wav")
    assert list(groups) == [OUT / "2026-05-07-inny.wav"]
    assert [article for article, _ in errors] == [bad]
    assert "no letters or digits" in str(errors[0][1])
