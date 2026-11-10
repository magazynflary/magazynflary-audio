import dataclasses
import itertools
import logging
import re
import typing

from magazynflary_audio import audio
from magazynflary_audio.data import DATA_DIR, cache
from magazynflary_audio.types import (
    ArticleReader,
    Audio,
    AuthorInfo,
    Language,
    ParsedArticle,
    ParsedArticleSection,
    TextToSpeechSystem,
)

__all__ = ("ArticleReaderImpl",)

LOG = logging.getLogger(__name__)

DEFAULT_TARGET_LOUDNESS: float = -16.0


@dataclasses.dataclass(frozen=True, slots=True)
class ArticleReaderImpl[SpeechContext](ArticleReader):
    tts: TextToSpeechSystem[SpeechContext]
    tts_cache_enabled: bool
    target_loudness: float = DEFAULT_TARGET_LOUDNESS

    def read_article(self, article: ParsedArticle, *, language: Language, author_info: AuthorInfo) -> Audio:
        gendered_author = "Autorka" if author_info.is_female else "Autor"
        (title_audio, author_audio), speech_context = self._text_to_speech(
            chunks=[
                f"Artykuł pod tytułem: {article.title}",
                f"{gendered_author}: {author_info.name}",
            ],
            language=language,
        )
        sample_rate = title_audio.sample_rate

        chunks_per_section: list[list[str]] = []
        if article.intro_paragraphs:
            chunks_per_section.append(chunks_from_paragraphs(article.intro_paragraphs, text_to_speech=self.tts))
        for section in filter_out_bibliography(article.sections):
            raw_header = section.header.strip()
            header_with_sentence_ending = raw_header if raw_header[-1] in ".?!" else raw_header + "."  # for TTS
            chunks_per_section.append(
                chunks_from_paragraphs(section.paragraphs, text_to_speech=self.tts, header=header_with_sentence_ending)
            )
        audio_per_section = []
        for chunks in chunks_per_section:
            section_audio, speech_context = self._text_to_speech(
                chunks=chunks,
                language=language,
                speech_context=speech_context,
            )
            audio_per_section.append(audio.concat(section_audio))

        intersection_pause = audio.mono_silence(sample_rate, duration_seconds=0.5)
        content_audio = audio.concat(list(itertools.chain(*[[a, intersection_pause] for a in audio_per_section]))[:-1])
        full_audio = compose(
            sample_rate=sample_rate,
            target_loudness=self.target_loudness,
            max_silently_clipped_fraction=0.001,
            max_true_peak=1,
            intersection_pause=intersection_pause,
            brand_audio=audio.read_mp3(DATA_DIR / "jingle/v0/magazyn-flary.mp3"),
            jingle_audio=audio.read_mp3(DATA_DIR / "jingle/v1/flary_v2_wiecej-flar.mpeg"),
            title_audio=title_audio,
            author_audio=author_audio,
            content_audio=content_audio,
        )
        return full_audio

    def _text_to_speech(
        self,
        *,
        chunks: list[str],
        language: Language,
        speech_context: SpeechContext | None = None,
    ) -> tuple[list[Audio], SpeechContext]:
        if self.tts_cache_enabled:
            return _cached_text_to_speech(
                tts=self.tts,
                tts_cache_key=self.tts.cache_key(),
                chunks=chunks,
                language=language,
                speech_context=speech_context,
            )
        else:
            return self.tts.text_to_speech(chunks, language=language, speech_context=speech_context)


@cache(ignore=["tts"])
def _cached_text_to_speech[SpeechContext](
    *,
    tts: TextToSpeechSystem[SpeechContext],
    tts_cache_key: str,
    chunks: list[str],
    language: Language,
    speech_context: SpeechContext | None = None,
) -> tuple[list[Audio], SpeechContext]:
    """Caches by the TTS configuration (tts_cache_key stands in for tts itself), text, language and speech context

    The speech context is part of the key because with request stitching the same text sounds different depending
    on what was read before it. As long as everything before is cached too, the cached speech contexts are replayed
    and match; once a chunk is synthesized live, its new speech context makes every later chunk of the same run miss
    the cache. So editing a section re-synthesizes that section and the ones after it, but not the ones before.
    """
    return tts.text_to_speech(chunks, language=language, speech_context=speech_context)


def filter_out_bibliography(sections: list[ParsedArticleSection]) -> typing.Generator[ParsedArticleSection]:
    for i, section in enumerate(sections):
        if i == len(sections) - 1 and re.match(r"bibliografia:?", section.header.strip().lower()):
            LOG.warning("Article has bibliography section at the end, skipping")
            continue
        yield section


def chunks_from_paragraphs(
    paragraphs: list[str], *, header: str | None = None, text_to_speech: TextToSpeechSystem[typing.Any]
) -> list[str]:
    chunks = []
    iter_paragraphs = iter(paragraphs)
    current_chunk = header if header else next(iter_paragraphs)
    for paragraph in iter_paragraphs:
        extended_chunk = current_chunk + "\n\n" + paragraph
        if text_to_speech.get_max_chunk_length(extended_chunk) < len(extended_chunk):
            assert text_to_speech.get_max_chunk_length(current_chunk) >= len(current_chunk)
            chunks.append(current_chunk)
            current_chunk = paragraph
        else:
            current_chunk = extended_chunk
    chunks.append(current_chunk)
    return chunks


def compose(
    *,
    sample_rate: int,
    target_loudness: float,
    max_silently_clipped_fraction: float,
    max_true_peak: float,
    intersection_pause: Audio,
    brand_audio: Audio,
    jingle_audio: Audio,
    title_audio: Audio,
    author_audio: Audio,
    content_audio: Audio,
) -> Audio:
    raw_parts: dict[str, Audio] = {
        "brand": brand_audio,
        "jingle": jingle_audio,
        "title": title_audio,
        "author": author_audio,
        "pause": intersection_pause,
        "content": content_audio,
    }
    normalized_parts = {
        name: normalize_audio(
            a,
            sample_rate=sample_rate,
            target_loudness=target_loudness,
            max_silently_clipped_fraction=max_silently_clipped_fraction,
            max_true_peak=max_true_peak,
        )
        for name, a in raw_parts.items()
    }
    full_audio = audio.concat(
        [
            normalized_parts[k]
            for k in [
                "brand",
                "jingle",
                "pause",
                "title",
                "pause",
                "author",
                "pause",
                "content",
                "pause",
                "jingle",
                "brand",
            ]
        ]
    )
    return full_audio


def normalize_audio(
    a: Audio,
    *,
    sample_rate: int,
    target_loudness: float,
    max_silently_clipped_fraction: float,
    max_true_peak: float,
) -> Audio:
    r = a
    r = audio.resample(r, sample_rate)
    r = audio.ensure_stereo(r)
    r = audio.normalize_loudness(
        r,
        target_loudness=target_loudness,
        max_silently_clipped_fraction=max_silently_clipped_fraction,
        max_true_peak=max_true_peak,
    )
    return r
