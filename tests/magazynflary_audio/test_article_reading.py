import dataclasses
import typing

import numpy as np
import pytest

from magazynflary_audio import audio
from magazynflary_audio.article_reading import (
    BRAND_NAME,
    MAX_TRUE_PEAK,
    OUTPUT_SAMPLE_RATE,
    VOICE_DISCLOSURE,
    ArticleReaderImpl,
    JingleMix,
    chunks_from_paragraphs,
    compose,
    filter_out_bibliography,
)
from magazynflary_audio.types import (
    Audio,
    AuthorInfo,
    Language,
    ParsedArticle,
    ParsedArticleSection,
    TextToSpeechSystem,
)


@dataclasses.dataclass(frozen=True, slots=True)
class FixedLengthTextToSpeech(TextToSpeechSystem[None]):
    max_chunk_length: int

    def cache_key(self) -> str:
        return f"FixedLengthTextToSpeech({self.max_chunk_length})"

    def get_max_chunk_length(self, chunk: str) -> int:
        return self.max_chunk_length

    def text_to_speech(
        self,
        chunks: list[str],
        *,
        language: Language,
        speech_context: None = None,
    ) -> tuple[list[Audio], None]:
        raise NotImplementedError


class ChunkingTestCase(typing.NamedTuple):
    paragraphs: list[str]
    header: str | None
    max_chunk_length: int
    expected: list[str]


TEST_CASES = {
    "single-paragraph": ChunkingTestCase(
        paragraphs=["aaaa"],
        header=None,
        max_chunk_length=10,
        expected=["aaaa"],
    ),
    "all-fit-in-one-chunk": ChunkingTestCase(
        paragraphs=["aa", "bb", "cc"],
        header=None,
        max_chunk_length=100,
        expected=["aa\n\nbb\n\ncc"],
    ),
    "exactly-max-length": ChunkingTestCase(
        paragraphs=["aaa", "bbb"],
        header=None,
        max_chunk_length=len("aaa\n\nbbb"),
        expected=["aaa\n\nbbb"],
    ),
    "one-over-max-length": ChunkingTestCase(
        paragraphs=["aaa", "bbb"],
        header=None,
        max_chunk_length=len("aaa\n\nbbb") - 1,
        expected=["aaa", "bbb"],
    ),
    "each-paragraph-own-chunk": ChunkingTestCase(
        paragraphs=["aaaa", "bbbb", "cccc"],
        header=None,
        max_chunk_length=5,
        expected=["aaaa", "bbbb", "cccc"],
    ),
    "greedy-grouping": ChunkingTestCase(
        paragraphs=["aa", "bb", "cc", "dd", "ee"],
        header=None,
        max_chunk_length=len("aa\n\nbb\n\ncc"),
        expected=["aa\n\nbb\n\ncc", "dd\n\nee"],
    ),
    "header-merged-with-paragraphs": ChunkingTestCase(
        paragraphs=["aa", "bb"],
        header="Header",
        max_chunk_length=100,
        expected=["Header\n\naa\n\nbb"],
    ),
    "header-only": ChunkingTestCase(
        paragraphs=[],
        header="Header",
        max_chunk_length=100,
        expected=["Header"],
    ),
    "header-in-separate-chunk": ChunkingTestCase(
        paragraphs=["aaaa", "bbbb"],
        header="Header",
        max_chunk_length=len("aaaa\n\nbbbb"),
        expected=["Header", "aaaa\n\nbbbb"],
    ),
    "empty-header-ignored": ChunkingTestCase(
        paragraphs=["aa", "bb"],
        header="",
        max_chunk_length=100,
        expected=["aa\n\nbb"],
    ),
}


@pytest.mark.parametrize("test_case", TEST_CASES.values(), ids=TEST_CASES.keys())
def test_chunks_from_paragraphs(test_case: ChunkingTestCase) -> None:
    text_to_speech = FixedLengthTextToSpeech(max_chunk_length=test_case.max_chunk_length)
    chunks = chunks_from_paragraphs(test_case.paragraphs, header=test_case.header, text_to_speech=text_to_speech)
    assert chunks == test_case.expected


@pytest.mark.parametrize("test_case", TEST_CASES.values(), ids=TEST_CASES.keys())
def test_chunks_from_paragraphs_preserves_text(test_case: ChunkingTestCase) -> None:
    text_to_speech = FixedLengthTextToSpeech(max_chunk_length=test_case.max_chunk_length)
    chunks = chunks_from_paragraphs(test_case.paragraphs, header=test_case.header, text_to_speech=text_to_speech)
    expected_parts = ([test_case.header] if test_case.header else []) + test_case.paragraphs
    assert "\n\n".join(chunks) == "\n\n".join(expected_parts)
    assert all(len(chunk) <= test_case.max_chunk_length for chunk in chunks)


def _section(header: str) -> ParsedArticleSection:
    return ParsedArticleSection(header=header, paragraphs=["Lorem ipsum."])


@pytest.mark.parametrize(
    "header",
    ["Bibliografia", "Bibliografia:", "bibliografia", "BIBLIOGRAFIA:", "  Bibliografia:  "],
)
def test_filter_out_bibliography_removes_last_section(header: str) -> None:
    sections = [_section("Wstęp"), _section("Rozwinięcie"), _section(header)]
    assert list(filter_out_bibliography(sections)) == sections[:2]


@pytest.mark.parametrize("header", ["Bibliografia", "Bibliografia:"])
def test_filter_out_bibliography_keeps_bibliography_not_at_end(header: str) -> None:
    sections = [_section("Wstęp"), _section(header), _section("Zakończenie")]
    assert list(filter_out_bibliography(sections)) == sections


def test_filter_out_bibliography_keeps_other_last_section() -> None:
    sections = [_section("Wstęp"), _section("Zakończenie")]
    assert list(filter_out_bibliography(sections)) == sections


@pytest.mark.parametrize("header", ["Bibliografia", "Bibliografia:"])
def test_filter_out_bibliography_only_section(header: str) -> None:
    assert list(filter_out_bibliography([_section(header)])) == []


def test_filter_out_bibliography_empty() -> None:
    assert list(filter_out_bibliography([])) == []


@dataclasses.dataclass
class RecordingTextToSpeech(TextToSpeechSystem[None]):
    """Returns one second of a quiet tone per chunk and records the chunks of every call"""

    calls: list[list[str]] = dataclasses.field(default_factory=list)
    sample_rate: int = 24000

    def cache_key(self) -> str:
        return "RecordingTextToSpeech()"

    def get_max_chunk_length(self, chunk: str) -> int:
        return 4000

    def text_to_speech(
        self,
        chunks: list[str],
        *,
        language: Language,
        speech_context: None = None,
    ) -> tuple[list[Audio], None]:
        self.calls.append(list(chunks))
        t = np.arange(self.sample_rate, dtype=np.float32) / self.sample_rate
        tone = (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        return [Audio(samples=tone, sample_rate=self.sample_rate) for _ in chunks], None


def test_read_article_reads_brand_then_title_author_and_disclosure() -> None:
    tts = RecordingTextToSpeech()
    article = ParsedArticle(
        author_slug="zaneta-zulawska",
        title="Tytuł",
        intro_paragraphs=["Wstęp."],
        sections=[ParsedArticleSection(header="Sekcja", paragraphs=["Treść."])],
    )
    result = ArticleReaderImpl(tts=tts, tts_cache_enabled=False).read_article(
        article, language="pl", author_info=AuthorInfo(name="Żaneta Żuławska", is_female=True)
    )
    # The brand name alone (the same audio in every episode), then the intro, then the sections.
    assert tts.calls == [
        [BRAND_NAME],
        ["Artykuł pod tytułem: Tytuł", "Autorka: Żaneta Żuławska", VOICE_DISCLOSURE],
        ["Wstęp."],
        ["Sekcja.\n\nTreść."],
    ]
    assert VOICE_DISCLOSURE == "Czyta głos syntetyczny."
    assert result.sample_rate == OUTPUT_SAMPLE_RATE
    assert result.samples.ndim == 2 and result.samples.shape[1] == 2


def _tone(seconds: float, *, sample_rate: int = 24000, frequency: float = 220.0, channels: int = 1) -> Audio:
    t = np.arange(round(seconds * sample_rate), dtype=np.float32) / sample_rate
    tone = (0.1 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
    return Audio(samples=tone if channels == 1 else np.repeat(tone[:, None], channels, 1), sample_rate=sample_rate)


def _compose(jingle_seconds: float, mix: JingleMix) -> Audio:
    return compose(
        sample_rate=OUTPUT_SAMPLE_RATE,
        target_loudness=-16.0,
        max_true_peak=MAX_TRUE_PEAK,
        max_silently_clipped_fraction=0.001,
        mix=mix,
        jingle_audio=_tone(jingle_seconds, sample_rate=44100, frequency=440.0, channels=2),
        brand_audio=_tone(1.0),
        title_audio=_tone(2.0),
        author_audio=_tone(1.5),
        disclosure_audio=_tone(1.0),
        section_audios=[_tone(3.0), _tone(4.0)],
    )


def test_compose_layout_with_a_long_tail() -> None:
    # Jingle 16 s, voice from 7.5 s: intro voice = 1 + 0.5 + 2 + 0.5 + 1.5 + 0.5 + 1 = 7 s, ends at 14.5 s;
    # the tail fades out over 1.5 s -> intro 16 s (capped at the jingle's end).
    mix = JingleMix(voice_offset_seconds=7.5, intro_fade_seconds=1.5, pause_seconds=0.5)
    result = _compose(16.0, mix)
    intro, article, outro = 16.0, 3.0 + 0.5 + 4.0, 16.0  # the outro rings out to the jingle's end
    assert audio.duration_seconds(result) == pytest.approx(intro + 0.5 + article + 0.5 + outro, abs=1e-3)


def test_compose_layout_with_a_short_jingle() -> None:
    # Jingle 10 s, voice from 7.5 s to 14.5 s: no tail left to fade, the intro ends with the voice;
    # the outro (brand name 7.5-8.5 s) rings out to the jingle's end, 10 s.
    mix = JingleMix(voice_offset_seconds=7.5, intro_fade_seconds=1.5, pause_seconds=0.5)
    result = _compose(10.0, mix)
    assert audio.duration_seconds(result) == pytest.approx(14.5 + 0.5 + 7.5 + 0.5 + 10.0, abs=1e-3)


def test_compose_meets_loudness_and_true_peak() -> None:
    result = _compose(16.0, JingleMix())
    assert audio.measure_loudness(result) == pytest.approx(-16.0, abs=0.5)
    assert audio.true_peak_dbfs(result.samples) <= MAX_TRUE_PEAK + 0.05
    assert result.sample_rate == OUTPUT_SAMPLE_RATE and result.samples.shape[1] == 2
