"""Names of published audio files

The audio server (audio.magazynflary.pl) publishes only files named `YYYY-MM-DD-slug.mp3`: the article's publication
date (front matter `date`) followed by a slug of the article's file name, using only `a-z`, `0-9` and `-`. Files named
otherwise are skipped by the server, never renamed. A corrected recording gets a new name (e.g. a `-v2` suffix) instead
of replacing a published file, because podcast apps and caches keep the old one under the old name.
"""

import datetime
import re
import unicodedata

__all__ = ("PUBLISHED_AUDIO_STEM_RE", "audio_file_stem", "slugify")

PUBLISHED_AUDIO_STEM_RE = re.compile(r"\d{4}-\d{2}-\d{2}-[a-z0-9]+(?:-[a-z0-9]+)*")
"""Stem (name without the extension) of a published audio file, the same rule as on the audio server"""

# Polish letters, spelled out even though most decompose under NFKD anyway: "ł" does not, and an explicit table
# documents the intent better than relying on Unicode decomposition.
_POLISH_TRANSLITERATION = str.maketrans("ąćęłńóśźż", "acelnoszz")


def slugify(text: str) -> str:
    """Lowercase ASCII slug: Polish (and other accented) letters transliterated, any other run of characters -> "-"

    Raises ValueError if nothing usable is left, e.g. for an empty string or only punctuation.
    """
    transliterated = text.lower().translate(_POLISH_TRANSLITERATION)
    ascii_text = unicodedata.normalize("NFKD", transliterated).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_text).strip("-")
    if not slug:
        raise ValueError(f"Cannot make a slug of {text!r}: no letters or digits left")
    return slug


def audio_file_stem(publication_date: datetime.date, article_stem: str) -> str:
    """`YYYY-MM-DD-slug` for an article published on publication_date, whose file is named article_stem.md"""
    return f"{publication_date.isoformat()}-{slugify(article_stem)}"
