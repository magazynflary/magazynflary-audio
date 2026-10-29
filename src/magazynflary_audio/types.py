import dataclasses
import typing

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
