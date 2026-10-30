import dataclasses
import enum
import re
import typing
from collections.abc import Sequence

import yaml
from markdown_it import MarkdownIt
from markdown_it.token import Token

from .types import ArticleParser, CleanText, MarkdownText, ParsedArticle, ParsedArticleSection, ParserError

FRONT_MATTER_DELIMITER = "---"
PULLQUOTE_OPEN_RE = re.compile(r"^\s*\{\{<\s*pullquote\s*>\}\}\s*$")
PULLQUOTE_CLOSE_RE = re.compile(r"^\s*\{\{<\s*/pullquote\s*>\}\}\s*$")
SECTION_HEADER_TAG = "h2"
TEXT_INLINE_TOKENS = {"text"}
SPACE_INLINE_TOKENS = {"softbreak", "hardbreak"}
IGNORED_INLINE_TOKENS = {"strong_open", "strong_close", "em_open", "em_close", "link_open", "link_close"}


class BlockKind(enum.Enum):
    """Kind of a top-level markdown block, the value is markdown-it token type prefix"""

    PARAGRAPH = "paragraph"
    HEADING = "heading"


@dataclasses.dataclass(frozen=True, slots=True)
class Block:
    """Top-level markdown block"""

    kind: BlockKind
    inline: Token
    """Inline token holding the block's content"""


@dataclasses.dataclass(frozen=True, slots=True)
class Body:
    intro_paragraphs: list[CleanText]
    sections: list[ParsedArticleSection]


class ArticleParserImpl(ArticleParser):
    def __init__(self) -> None:
        self._markdown = MarkdownIt("commonmark")

    def parse_article_markdown(self, article_markdown: MarkdownText) -> ParsedArticle:
        front_matter, body_markdown = _split_front_matter(article_markdown)
        tokens = self._markdown.parse(_remove_pullquote_markers(body_markdown))
        body = _parse_body(tokens)
        return ParsedArticle(
            author_slug=_required_string(front_matter, "author"),
            title=_clean(_required_string(front_matter, "title")),
            intro_paragraphs=body.intro_paragraphs,
            sections=body.sections,
        )


def _split_front_matter(article_markdown: MarkdownText) -> tuple[dict[str, object], MarkdownText]:
    lines = article_markdown.lstrip().splitlines()
    if not lines or lines[0].rstrip() != FRONT_MATTER_DELIMITER:
        raise ParserError("Article does not start with front matter")
    try:
        end = next(i for i, line in enumerate(lines[1:], 1) if line.rstrip() == FRONT_MATTER_DELIMITER)
    except StopIteration:
        raise ParserError("Front matter is not closed") from None
    try:
        front_matter = yaml.safe_load("\n".join(lines[1:end]))
    except yaml.YAMLError as e:
        raise ParserError(f"Invalid front matter YAML: {e}") from e
    if not isinstance(front_matter, dict):
        raise ParserError("Front matter is not a mapping")
    return front_matter, "\n".join(lines[end + 1 :])


def _required_string(front_matter: dict[str, object], key: str) -> str:
    value = front_matter.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ParserError(f"Front matter field {key!r} is missing or not a non-empty string")
    return value.strip()


def _remove_pullquote_markers(body: MarkdownText) -> MarkdownText:
    """Removes pullquote markers; without blank lines around it, the pullquote joins the surrounding paragraph"""
    in_lines = body.splitlines()
    out_lines = []
    in_pullquote = False
    for i, line in enumerate(in_lines):
        if PULLQUOTE_OPEN_RE.match(line):
            if in_pullquote:
                raise ParserError(f"Nested pullquote in line {i + 1} of the body")
            in_pullquote = True
        elif PULLQUOTE_CLOSE_RE.match(line):
            if not in_pullquote:
                raise ParserError(f"Unopened pullquote closed in line {i + 1} of the body")
            in_pullquote = False
        elif "{{" in line or "}}" in line:
            raise ParserError(f"Unsupported shortcode in line {i + 1} of the body: {line!r}")
        else:
            out_lines.append(line)
    if in_pullquote:
        raise ParserError("Pullquote is not closed")
    return "\n".join(out_lines)


def _parse_body(tokens: Sequence[Token]) -> Body:
    blocks = _blocks(tokens)
    if not blocks:
        raise ParserError("Article body is empty")

    intro_paragraphs: list[CleanText] = []
    sections: list[tuple[CleanText, list[CleanText]]] = []
    for block in blocks:
        clean_text = _inline_text(block.inline)
        match block.kind:
            case BlockKind.HEADING:
                sections.append((clean_text, []))
            case BlockKind.PARAGRAPH:
                paragraphs = sections[-1][1] if sections else intro_paragraphs
                paragraphs.append(clean_text)
            case _:
                assert typing.assert_never(block.kind)

    for header, paragraphs in sections:
        if not paragraphs:
            raise ParserError(f"Section {header!r} has no paragraphs")
    return Body(
        intro_paragraphs=intro_paragraphs,
        sections=[ParsedArticleSection(header, paragraphs) for header, paragraphs in sections],
    )


def _blocks(tokens: Sequence[Token]) -> list[Block]:
    """Returns each top-level block, raising on unsupported blocks; blockquote paragraphs are normal paragraphs"""
    blocks: list[Block] = []
    in_blockquote = False
    i = 0
    while i < len(tokens):
        opening = tokens[i]
        line = opening.map[0] + 1 if opening.map else "?"
        if opening.type == "blockquote_open" and not in_blockquote:
            in_blockquote = True
            i += 1
            continue
        elif opening.type == "blockquote_close" and in_blockquote:
            in_blockquote = False
            i += 1
            continue
        elif opening.type == "paragraph_open":
            kind = BlockKind.PARAGRAPH
        elif opening.type == "heading_open" and opening.tag == SECTION_HEADER_TAG and not in_blockquote:
            kind = BlockKind.HEADING
        else:
            where = " inside blockquote" if in_blockquote else ""
            raise ParserError(f"Unsupported element {opening.type!r} ({opening.tag}){where} in line {line} of the body")
        if i + 2 >= len(tokens) or tokens[i + 1].type != "inline" or tokens[i + 2].type != f"{kind.value}_close":
            raise ParserError(f"Unexpected token structure of {kind.value}")
        blocks.append(Block(kind=kind, inline=tokens[i + 1]))
        i += 3
    if in_blockquote:
        raise ParserError("Blockquote is not closed")
    return blocks


def _inline_text(inline: Token) -> CleanText:
    parts: list[str] = []
    for child in inline.children or []:
        if child.type in TEXT_INLINE_TOKENS:
            parts.append(child.content)
        elif child.type in SPACE_INLINE_TOKENS:
            parts.append(" ")
        elif child.type not in IGNORED_INLINE_TOKENS:
            raise ParserError(f"Unsupported inline markdown element {child.type!r} in {inline.content!r}")
    text = _clean("".join(parts))
    if not text:
        raise ParserError(f"Empty text in {inline.content!r}")
    return text


def _clean(text: str) -> CleanText:
    return " ".join(text.split())
