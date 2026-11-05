import datetime

import pytest

from magazynflary_audio.audio_naming import PUBLISHED_AUDIO_STEM_RE, audio_file_stem, slugify

SLUGS = {
    "already-a-slug": ("samotni-wsrod-ludzi", "samotni-wsrod-ludzi"),
    "polish-letters": ("samotni-wśród-ludzi", "samotni-wsrod-ludzi"),
    "all-polish-letters": ("ąćęłńóśźż", "acelnoszz"),
    "polish-uppercase": ("ĄĆĘŁŃÓŚŹŻ", "acelnoszz"),
    "other-accents": ("café-über-naïve", "cafe-uber-naive"),
    "spaces-and-punctuation": ("  Żółć, gęś & (jaźń)!  ", "zolc-ges-jazn"),
    "underscores-and-dots": ("rok_2024.podsumowanie", "rok-2024-podsumowanie"),
    "repeated-dashes": ("a--b---c", "a-b-c"),
    "digits": ("10-lat-flary", "10-lat-flary"),
}


@pytest.mark.parametrize(("text", "expected"), SLUGS.values(), ids=SLUGS.keys())
def test_slugify(text: str, expected: str) -> None:
    assert slugify(text) == expected


@pytest.mark.parametrize("text", ["", "---", "!?", "日本"])
def test_slugify_nothing_left(text: str) -> None:
    with pytest.raises(ValueError, match="no letters or digits"):
        slugify(text)


def test_audio_file_stem() -> None:
    assert audio_file_stem(datetime.date(2026, 5, 7), "samotni-wśród-ludzi") == "2026-05-07-samotni-wsrod-ludzi"


@pytest.mark.parametrize("article_stem", [text for text, _ in SLUGS.values()])
def test_audio_file_stem_matches_the_server_rule(article_stem: str) -> None:
    assert PUBLISHED_AUDIO_STEM_RE.fullmatch(audio_file_stem(datetime.date(2024, 1, 31), article_stem))


@pytest.mark.parametrize(
    "stem",
    [
        "samotni-wsrod-ludzi",  # no date
        "2026-05-07-Samotni",  # uppercase
        "2026-05-07-samotni-wśród-ludzi",  # Polish letters
        "2026-05-07-samotni_wsrod",  # underscore
        "2026-05-07-samotni-",  # trailing dash
        "2026-05-07--samotni",  # double dash
    ],
)
def test_server_rule_rejects(stem: str) -> None:
    assert not PUBLISHED_AUDIO_STEM_RE.fullmatch(stem)
