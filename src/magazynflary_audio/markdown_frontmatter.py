import yaml

from .types import MarkdownText, ParserError

FRONT_MATTER_DELIMITER = "---"

type FrontMatter = dict[str, object]


def split_front_matter(markdown: MarkdownText) -> tuple[FrontMatter, MarkdownText]:
    lines = markdown.lstrip().splitlines()
    if not lines or lines[0].rstrip() != FRONT_MATTER_DELIMITER:
        raise ParserError("Markdown does not start with front matter")
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


def required_string(front_matter: FrontMatter, key: str) -> str:
    value = front_matter.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ParserError(f"Front matter field {key!r} is missing or not a non-empty string")
    return value.strip()
