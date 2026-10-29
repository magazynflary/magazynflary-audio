from typing import NamedTuple

import pytest

from magazynflary_audio.article_parsing import ArticleParserImpl
from magazynflary_audio.types import ArticleParser, MarkdownText, ParsedArticle, ParsedArticleSection, ParserError


class ParsingTestCase(NamedTuple):
    markdown: MarkdownText
    expected: ParsedArticle


TEST_CASE_LEAD = ParsingTestCase(
    markdown="""
---
edition: 2020-2
author: żelisław-wój
title: "Lorem Ipsum"
date: 2020-09-01T20:00:00
draft: false
tags:
  - foo
  - bar
image: images/uploads/lorem_ipsum.jpg
image_focus_banner: 50% 50%
image_focus_card: 52% 50%
image_credit_author: ewa-uważna
description: Co jeżeli
  Lorem Ipsum ma sens
noindex: false
---

**Lorem ipsum dolor sit amet consectetur adipiscing elit. Quisque faucibus ex sapien vitae pellentesque sem placerat.**

In id cursus mi pretium tellus duis convallis. Tempus leo eu aenean sed diam urna tempor.
Pulvinar vivamus fringilla lacus nec metus bibendum egestas.

Iaculis massa nisl malesuada lacinia integer nunc posuere. Ut hendrerit semper vel class aptent taciti sociosqu.

## Ad litora torquent per conubia nostra inceptos himenaeos.

Maecenas vitae vehicula velit, a *bibendum metus*. Suspendisse mattis dui ac metus consectetur iaculis.
Nunc vulputate vel odio ut accumsan. Maecenas quis lorem sed purus placerat luctus.

{{< pullquote >}}
Curabitur at urna dictum, dictum odio eget, tempus arcu. Maecenas luctus risus non sapien gravida,
eget aliquet elit congue. Aliquam in velit id enim vestibulum maximus. Aliquam pellentesque accumsan vestibulum.
Curabitur auctor fermentum magna, nec tempus dolor venenatis eu.
{{< /pullquote >}}

Aenean ac lorem ex. Donec mattis tortor quis posuere suscipit. Curabitur **semper** tincidunt diam a vestibulum.
Nullam rutrum nisl ut pharetra convallis.

Fusce facilisis, nunc rhoncus tristique porta, ex dolor elementum nunc, et elementum arcu ligula quis ante.
Morbi mauris urna, maximus sed ligula eu, fringilla elementum odio. Nulla facilisi.

## Mauris non nunc blandit, commodo mauris sit amet, blandit odio.

Cras a dolor justo. Ut interdum pulvinar nibh ac porta.
Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia curae;
Praesent pretium molestie mauris [in interdum](https://lipsum.com/).
""",
    expected=ParsedArticle(
        author_slug="żelisław-wój",
        title="Lorem Ipsum",
        intro_paragraphs=[
            (
                "Lorem ipsum dolor sit amet consectetur adipiscing elit. "
                "Quisque faucibus ex sapien vitae pellentesque sem placerat."
            ),
            (
                "In id cursus mi pretium tellus duis convallis. "
                "Tempus leo eu aenean sed diam urna tempor. "
                "Pulvinar vivamus fringilla lacus nec metus bibendum egestas."
            ),
            (
                "Iaculis massa nisl malesuada lacinia integer nunc posuere. "
                "Ut hendrerit semper vel class aptent taciti sociosqu."
            ),
        ],
        sections=[
            ParsedArticleSection(
                header="Ad litora torquent per conubia nostra inceptos himenaeos.",
                paragraphs=[
                    (
                        "Maecenas vitae vehicula velit, a bibendum metus. "
                        "Suspendisse mattis dui ac metus consectetur iaculis. "
                        "Nunc vulputate vel odio ut accumsan. "
                        "Maecenas quis lorem sed purus placerat luctus."
                    ),
                    (
                        "Curabitur at urna dictum, dictum odio eget, tempus arcu. "
                        "Maecenas luctus risus non sapien gravida, eget aliquet elit congue. "
                        "Aliquam in velit id enim vestibulum maximus. "
                        "Aliquam pellentesque accumsan vestibulum. "
                        "Curabitur auctor fermentum magna, nec tempus dolor venenatis eu."
                    ),
                    (
                        "Aenean ac lorem ex. Donec mattis tortor quis posuere suscipit. "
                        "Curabitur semper tincidunt diam a vestibulum. "
                        "Nullam rutrum nisl ut pharetra convallis."
                    ),
                    (
                        "Fusce facilisis, nunc rhoncus tristique porta, "
                        "ex dolor elementum nunc, et elementum arcu ligula quis ante. "
                        "Morbi mauris urna, maximus sed ligula eu, fringilla elementum odio. "
                        "Nulla facilisi."
                    ),
                ],
            ),
            ParsedArticleSection(
                header="Mauris non nunc blandit, commodo mauris sit amet, blandit odio.",
                paragraphs=[
                    (
                        "Cras a dolor justo. Ut interdum pulvinar nibh ac porta. "
                        "Vestibulum ante ipsum primis in faucibus orci luctus et ultrices "
                        "posuere cubilia curae; Praesent pretium molestie mauris in interdum."
                    ),
                ],
            ),
        ],
    ),
)

TEST_CASE_NO_LEAD = ParsingTestCase(
    markdown="""
---
author: żelisław-wój
title: "Dolor Sit Amet"
---

Lorem ipsum dolor sit amet consectetur adipiscing elit.
Quisque faucibus ex sapien vitae pellentesque sem placerat.

In id cursus mi pretium tellus duis convallis. Tempus leo eu aenean sed diam urna tempor.

## Ad litora torquent per conubia nostra inceptos himenaeos.

Maecenas vitae vehicula velit, a *bibendum metus*. Suspendisse mattis dui ac metus consectetur iaculis.

{{< pullquote >}}
Curabitur at urna dictum, dictum odio eget, tempus arcu.
{{< /pullquote >}}

Aenean ac lorem ex. Donec mattis tortor quis posuere suscipit.
""",
    expected=ParsedArticle(
        author_slug="żelisław-wój",
        title="Dolor Sit Amet",
        intro_paragraphs=[
            (
                "Lorem ipsum dolor sit amet consectetur adipiscing elit. "
                "Quisque faucibus ex sapien vitae pellentesque sem placerat."
            ),
            ("In id cursus mi pretium tellus duis convallis. Tempus leo eu aenean sed diam urna tempor."),
        ],
        sections=[
            ParsedArticleSection(
                header="Ad litora torquent per conubia nostra inceptos himenaeos.",
                paragraphs=[
                    (
                        "Maecenas vitae vehicula velit, a bibendum metus. "
                        "Suspendisse mattis dui ac metus consectetur iaculis."
                    ),
                    ("Curabitur at urna dictum, dictum odio eget, tempus arcu."),
                    ("Aenean ac lorem ex. Donec mattis tortor quis posuere suscipit."),
                ],
            ),
        ],
    ),
)

TEST_CASE_HEADER_FIRST = ParsingTestCase(
    markdown="""
---
author: ewa-uważna
title: Consectetur Adipiscing
---

## Ad litora torquent per conubia nostra inceptos himenaeos.

Maecenas vitae vehicula velit, a bibendum metus.
Nunc vulputate vel odio ut accumsan.

## Mauris non nunc blandit, commodo mauris sit amet, blandit odio.

Cras a dolor justo. Ut interdum pulvinar nibh ac porta.

Praesent pretium molestie mauris [in interdum](https://lipsum.com/).
""",
    expected=ParsedArticle(
        author_slug="ewa-uważna",
        title="Consectetur Adipiscing",
        intro_paragraphs=[],
        sections=[
            ParsedArticleSection(
                header="Ad litora torquent per conubia nostra inceptos himenaeos.",
                paragraphs=[
                    ("Maecenas vitae vehicula velit, a bibendum metus. Nunc vulputate vel odio ut accumsan."),
                ],
            ),
            ParsedArticleSection(
                header="Mauris non nunc blandit, commodo mauris sit amet, blandit odio.",
                paragraphs=[
                    ("Cras a dolor justo. Ut interdum pulvinar nibh ac porta."),
                    ("Praesent pretium molestie mauris in interdum."),
                ],
            ),
        ],
    ),
)

TEST_CASE_BLOCKQUOTE = ParsingTestCase(
    markdown="""
---
author: żelisław-wój
title: "Quisque Faucibus"
---

Lorem ipsum dolor sit amet consectetur adipiscing elit.

> **Lorem:** Quisque faucibus ex sapien vitae pellentesque. \\[...] Sem placerat
> in id cursus mi pretium tellus duis convallis.

Tempus leo eu aenean sed diam urna tempor.

## Ad litora torquent per conubia nostra inceptos himenaeos.

> Maecenas vitae vehicula velit, a *bibendum metus*.
>
> Suspendisse mattis dui ac metus consectetur iaculis.

(Lorem, 1987, s. 277)
""",
    expected=ParsedArticle(
        author_slug="żelisław-wój",
        title="Quisque Faucibus",
        intro_paragraphs=[
            "Lorem ipsum dolor sit amet consectetur adipiscing elit.",
            (
                "Lorem: Quisque faucibus ex sapien vitae pellentesque. [...] Sem placerat "
                "in id cursus mi pretium tellus duis convallis."
            ),
            "Tempus leo eu aenean sed diam urna tempor.",
        ],
        sections=[
            ParsedArticleSection(
                header="Ad litora torquent per conubia nostra inceptos himenaeos.",
                paragraphs=[
                    "Maecenas vitae vehicula velit, a bibendum metus.",
                    "Suspendisse mattis dui ac metus consectetur iaculis.",
                    "(Lorem, 1987, s. 277)",
                ],
            ),
        ],
    ),
)


TEST_CASE_PULLQUOTE_MID_SENTENCE = ParsingTestCase(
    markdown="""
---
author: żelisław-wój
title: "Pellentesque Sem"
---

Lorem ipsum dolor sit amet consectetur adipiscing elit.

{{< pullquote >}}
Quisque faucibus ex sapien vitae pellentesque sem placerat
{{< /pullquote >}}
in id cursus mi pretium tellus duis convallis.

## Ad litora torquent per conubia nostra inceptos himenaeos.

Maecenas vitae vehicula velit,
{{< pullquote >}}
a bibendum metus suspendisse mattis dui
{{< /pullquote >}}
ac metus consectetur iaculis.
""",
    expected=ParsedArticle(
        author_slug="żelisław-wój",
        title="Pellentesque Sem",
        intro_paragraphs=[
            "Lorem ipsum dolor sit amet consectetur adipiscing elit.",
            (
                "Quisque faucibus ex sapien vitae pellentesque sem placerat "
                "in id cursus mi pretium tellus duis convallis."
            ),
        ],
        sections=[
            ParsedArticleSection(
                header="Ad litora torquent per conubia nostra inceptos himenaeos.",
                paragraphs=[
                    (
                        "Maecenas vitae vehicula velit, "
                        "a bibendum metus suspendisse mattis dui "
                        "ac metus consectetur iaculis."
                    ),
                ],
            ),
        ],
    ),
)

TEST_CASES = {
    "lead": TEST_CASE_LEAD,
    "no-lead": TEST_CASE_NO_LEAD,
    "header-first": TEST_CASE_HEADER_FIRST,
    "blockquote": TEST_CASE_BLOCKQUOTE,
    "pullquote-mid-sentence": TEST_CASE_PULLQUOTE_MID_SENTENCE,
}

ARTICLE_PARSER_IMPLEMENTATIONS: list[type[ArticleParser]] = [ArticleParserImpl]


@pytest.fixture(params=ARTICLE_PARSER_IMPLEMENTATIONS, ids=lambda cls: cls.__name__)
def parser(request: pytest.FixtureRequest) -> ArticleParser:
    parser: ArticleParser = request.param()
    return parser


@pytest.mark.parametrize("test_case", TEST_CASES.values(), ids=TEST_CASES.keys())
def test_parse_article(parser: ArticleParser, test_case: ParsingTestCase) -> None:
    assert parser.parse_article_markdown(test_case.markdown) == test_case.expected


@pytest.mark.parametrize(
    "article_markdown",
    [
        pytest.param("", id="empty"),
        pytest.param(TEST_CASE_LEAD.markdown.split("---", 2)[2], id="no-front-matter"),
        pytest.param(TEST_CASE_LEAD.markdown.rsplit("---", 1)[0] + "---\n", id="empty-body"),
        pytest.param(TEST_CASE_LEAD.markdown.replace("\n---\n\n", "\n\n", 1), id="unclosed-front-matter"),
        pytest.param(TEST_CASE_LEAD.markdown.replace('title: "Lorem Ipsum"\n', ""), id="no-title"),
        pytest.param(TEST_CASE_LEAD.markdown.replace("author: żelisław-wój\n", ""), id="no-author"),
        pytest.param(TEST_CASE_LEAD.markdown + "\n> ## Lorem\n", id="header-in-blockquote"),
        pytest.param(TEST_CASE_LEAD.markdown + "\n> > Lorem\n", id="nested-blockquote"),
        pytest.param(TEST_CASE_LEAD.markdown + "\n> - Lorem\n", id="list-in-blockquote"),
    ],
)
def test_parse_article_invalid(parser: ArticleParser, article_markdown: str) -> None:
    with pytest.raises(ParserError):
        parser.parse_article_markdown(article_markdown)
