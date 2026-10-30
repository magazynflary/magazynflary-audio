import dataclasses
import typing

import pytest

from magazynflary_audio.article_reading import chunks_from_paragraphs
from magazynflary_audio.types import Audio, Language, TextToSpeechSystem


@dataclasses.dataclass(frozen=True, slots=True)
class FixedLengthTextToSpeech(TextToSpeechSystem[None]):
    max_chunk_length: int

    def get_max_chunk_length(self, chunk: str) -> int:
        return self.max_chunk_length

    def text_to_speech(
        self,
        chunks: list[str],
        *,
        language: Language,
        on_chunk: typing.Callable[[int], None] | None = None,
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
