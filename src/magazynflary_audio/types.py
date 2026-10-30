import dataclasses
import typing

import numpy as np
import numpy.typing as npt

type MarkdownText = str
type CleanText = str
"""Clean text without markup, can be read out loud"""


@dataclasses.dataclass(frozen=True, slots=True)
class ParsedArticleSection:
    header: CleanText
    paragraphs: list[CleanText]


@dataclasses.dataclass(frozen=True, slots=True)
class ParsedArticle:
    author_slug: str
    """Slug of the author, like adam-something"""

    title: CleanText

    intro_paragraphs: list[CleanText]
    """Paragraphs before the first section, including the lead if there is one"""

    sections: list[ParsedArticleSection]


class ParserError(ValueError):
    pass


class ArticleParser(typing.Protocol):
    def parse_article_markdown(self, article_markdown: MarkdownText) -> ParsedArticle:
        """Parses article markdown into an Article object or raises ParserError"""


@dataclasses.dataclass(frozen=True, slots=True)
class Audio:
    """Audio with float samples in the range [-1, 1]"""

    samples: npt.NDArray[np.float32]
    sample_rate: int


type Language = typing.Literal["pl"]


class ChunkTooLongError(ValueError):
    def __init__(self, chunk_index: int) -> None:
        super().__init__(f"chunk {chunk_index} is too long")


class TextToSpeechSystem[SpeechContext](typing.Protocol):
    def get_max_chunk_length(self, chunk: str) -> int:
        """Returns the maximum length of a chunk that can be synthesized

        Given a chunk of text, returns the maximum number of characters that can be synthesized in one call.
        """

    def text_to_speech(
        self,
        chunks: list[str],
        *,
        language: Language,
        on_chunk: typing.Callable[[int], None] | None = None,
        speech_context: SpeechContext | None = None,
    ) -> tuple[list[Audio], SpeechContext]:
        """Converts text to speech

        Raises ChunkTooLongError if any chunk is too long
        """


class ArticleReader(typing.Protocol):
    def read_article(self, article: ParsedArticle, *, language: Language) -> Audio:
        """Reads the article"""
