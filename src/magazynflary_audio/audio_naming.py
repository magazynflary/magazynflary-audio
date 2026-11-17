"""Names of the audio files this tool saves

`YYYY-MM-DD-slug`: the article's publication date (front matter `date`) followed by a slug of the article's file name,
using only `a-z`, `0-9` and `-`. This is the name format required for publishing on the audio server.
"""

import datetime
import re
import unicodedata

__all__ = ("PUBLISHED_AUDIO_STEM_RE", "AudioNamingError", "audio_file_stem", "slugify")

PUBLISHED_AUDIO_STEM_RE = re.compile(r"\d{4}-\d{2}-\d{2}-[a-z0-9]+(?:-[a-z0-9]+)*")
"""Stem (name without the extension) in the name format required for publishing"""

# Polish letters, spelled out even though most decompose under NFKD anyway: "ł" does not, and an explicit table
# documents the intent better than relying on Unicode decomposition.
_POLISH_TRANSLITERATION = str.maketrans("ąćęłńóśźż", "acelnoszz")


class AudioNamingError(ValueError):
    """The text gives no usable name, e.g. an empty string or only punctuation"""


def slugify(text: str) -> str:
    """Lowercase ASCII slug: Polish (and other accented) letters transliterated, any other run of characters -> "-"

    Raises AudioNamingError if nothing usable is left, e.g. for an empty string or only punctuation.
    """
    transliterated = text.lower().translate(_POLISH_TRANSLITERATION)
    ascii_text = unicodedata.normalize("NFKD", transliterated).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_text).strip("-")
    if not slug:
        raise AudioNamingError(f"Cannot make a slug of {text!r}: no letters or digits left")
    return slug


def audio_file_stem(publication_date: datetime.date, article_stem: str) -> str:
    """`YYYY-MM-DD-slug` for an article published on publication_date, whose file is named article_stem.md

    Raises AudioNamingError if article_stem gives no slug.
    """
    return f"{publication_date.isoformat()}-{slugify(article_stem)}"
