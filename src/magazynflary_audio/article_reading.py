import dataclasses
import itertools
import logging
import re
import typing
from pathlib import Path

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
"""Integrated loudness of the whole recording and of each voice part (LUFS); -16 is the common podcast target"""

MAX_TRUE_PEAK: float = -1.0
"""dBTP; headroom for the inter-sample peaks that MP3 encoding adds"""

SPEECH_MAX_LIMITING_DB: float = 3.0
"""How far the limiter may bring down the voice's rare peaks (plosives) to reach the target loudness. ElevenLabs speech
peaks 14-18 dB over its loudness, so with the gain alone a whole article lands at -17 to -19 LUFS, decided by its single
highest peak. The jingle is never limited."""

MIX_MAX_LIMITING_DB: float = 1.5
"""For the final pass over the whole recording: where the voice (already at the true peak limit) lies over the jingle,
the sum can peak a little higher; without limiting those few spots would lower the whole recording."""

OUTPUT_SAMPLE_RATE = 44100
"""The TTS voice (24 kHz) is resampled up; the jingle keeps its full band"""

VOICE_DISCLOSURE = "Czyta głos syntetyczny."
"""Read after the author in every recording: Apple Podcasts (guideline 1.11) requires disclosing a synthetic voice in
the audio itself."""


@dataclasses.dataclass(frozen=True, slots=True)
class JingleMix:
    """The intro over the jingle (brand name, title, author, disclosure) and the outro (the brand name alone)

    Times are seconds from the start of the jingle file; levels are relative to the jingle normalized like the voice.
    """

    path: Path = DATA_DIR / "jingle/v2/fortepian.flac"
    brand_path: Path = DATA_DIR / "jingle/v2/magazyn-flary.flac"
    """"Magazyn Flary", a recorded voice (not the TTS voice): over the jingle's tail in the intro, and alone as the
    outro; how it was made: README"""
    voice_offset_seconds: float = 8.24
    """Where the voice starts: in the decaying tail after the last note (6.5 s), where the jingle's REAPER mix has it"""
    level_db: float = -3.0
    """The jingle's loudness relative to the voice (a matter of balance: piano attacks sound louder than speech of the
    same LUFS). The tail is quiet enough to stay under the voice as it is - no ducking, no fade."""
    brand_pause_seconds: float = 1.2
    """In the intro, between "Magazyn Flary" and the article's title"""
    outro_pause_seconds: float = 2.0
    """Between the end of the article and "Magazyn Flary" closing the recording"""
    pause_seconds: float = 0.5
    """Between the voice parts, between the article's sections, and after the intro"""


@dataclasses.dataclass(frozen=True, slots=True)
class ArticleReaderImpl[SpeechContext](ArticleReader):
    tts: TextToSpeechSystem[SpeechContext]
    tts_cache_enabled: bool
    target_loudness: float = DEFAULT_TARGET_LOUDNESS
    jingle_mix: JingleMix = JingleMix()

    def read_article(self, article: ParsedArticle, *, language: Language, author_info: AuthorInfo) -> Audio:
        gendered_author = "Autorka" if author_info.is_female else "Autor"
        (title_audio, author_audio, disclosure_audio), speech_context = self._text_to_speech(
            chunks=[
                f"Artykuł pod tytułem: {article.title}",
                f"{gendered_author}: {author_info.name}",
                VOICE_DISCLOSURE,
            ],
            language=language,
        )

        chunks_per_section: list[list[str]] = []
        if article.intro_paragraphs:
            chunks_per_section.append(chunks_from_paragraphs(article.intro_paragraphs, text_to_speech=self.tts))
        for section in filter_out_bibliography(article.sections):
            raw_header = section.header.strip()
            header_with_sentence_ending = raw_header if raw_header[-1] in ".?!" else raw_header + "."  # for TTS
            chunks_per_section.append(
                chunks_from_paragraphs(section.paragraphs, text_to_speech=self.tts, header=header_with_sentence_ending)
            )
        section_audios = []
        for chunks in chunks_per_section:
            section_audio, speech_context = self._text_to_speech(
                chunks=chunks,
                language=language,
                speech_context=speech_context,
            )
            section_audios.append(audio.concat(section_audio))

        return compose(
            sample_rate=OUTPUT_SAMPLE_RATE,
            target_loudness=self.target_loudness,
            max_true_peak=MAX_TRUE_PEAK,
            max_limiting_db=SPEECH_MAX_LIMITING_DB,
            max_silently_clipped_fraction=0.001,
            mix=self.jingle_mix,
            jingle_audio=audio.read_audio(self.jingle_mix.path),
            brand_audio=audio.read_audio(self.jingle_mix.brand_path),
            title_audio=title_audio,
            author_audio=author_audio,
            disclosure_audio=disclosure_audio,
            section_audios=section_audios,
        )

    def _text_to_speech(
        self,
        *,
        chunks: list[str],
        language: Language,
        speech_context: SpeechContext | None = None,
    ) -> tuple[list[Audio], SpeechContext]:
        cleaned_chunks = [self._clean_text(chunk) for chunk in chunks]
        if self.tts_cache_enabled:
            return _cached_text_to_speech(
                tts=self.tts,
                tts_cache_key=self.tts.cache_key(),
                chunks=cleaned_chunks,
                language=language,
                speech_context=speech_context,
            )
        else:
            return self.tts.text_to_speech(cleaned_chunks, language=language, speech_context=speech_context)

    @staticmethod
    def _clean_text(text: str) -> str:
        transformed_text = text

        # Square brackets are used for audio tags and may be misinterpreted by TTS, eg. [...] was beeped out
        transformed_text = transformed_text.replace(" [...]", "")
        transformed_text = transformed_text.replace("[...]", "")
        transformed_text = transformed_text.replace("[", "(")
        transformed_text = transformed_text.replace("]", ")")

        if transformed_text != text:
            LOG.warning("Text content had to be cleaned")
        return transformed_text


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
    max_true_peak: float,
    max_limiting_db: float,
    max_silently_clipped_fraction: float,
    mix: JingleMix,
    jingle_audio: Audio,
    brand_audio: Audio,
    title_audio: Audio,
    author_audio: Audio,
    disclosure_audio: Audio,
    section_audios: list[Audio],
) -> Audio:
    """The whole recording, stereo at sample_rate:

    intro   jingle; over its tail: brand name, title, author, disclosure (pauses between); then the article starts
            when both the voice and the jingle have ended
    article sections with pauses between
    outro   brand name alone, mix.outro_pause_seconds after the article; no jingle

    Every part is normalized to target_loudness on its own - the voice parts with up to max_limiting_db of peak
    limiting, the jingle to target_loudness + mix.level_db with the gain alone - then the whole recording once more
    (one constant gain, so the balance stays; peaks where the voice lies over the jingle limited by at most
    MIX_MAX_LIMITING_DB) to meet target_loudness and max_true_peak after mixing.
    """

    def normalized(a: Audio) -> Audio:
        return normalize_audio(
            a,
            sample_rate=sample_rate,
            target_loudness=target_loudness,
            max_silently_clipped_fraction=max_silently_clipped_fraction,
            max_true_peak=max_true_peak,
            max_limiting_db=max_limiting_db,
        )

    pause = audio.silence(sample_rate, duration_seconds=mix.pause_seconds)
    jingle = normalize_audio(
        jingle_audio,
        sample_rate=sample_rate,
        target_loudness=target_loudness + mix.level_db,
        max_silently_clipped_fraction=max_silently_clipped_fraction,
        max_true_peak=max_true_peak,
    )
    brand = normalized(brand_audio)

    intro_voice = audio.concat(
        [
            brand,
            audio.silence(sample_rate, duration_seconds=mix.brand_pause_seconds),
            normalized(title_audio),
            pause,
            normalized(author_audio),
            pause,
            normalized(disclosure_audio),
        ]
    )
    intro = audio.overlay(jingle, intro_voice, offset_seconds=mix.voice_offset_seconds)
    article = audio.concat(list(itertools.chain(*[[normalized(a), pause] for a in section_audios]))[:-1])
    outro_pause = audio.silence(sample_rate, duration_seconds=mix.outro_pause_seconds)

    full = audio.concat([intro, pause, article, outro_pause, brand])
    return audio.normalize_loudness(
        full,
        target_loudness=target_loudness,
        max_true_peak=max_true_peak,
        max_silently_clipped_fraction=max_silently_clipped_fraction,
        max_limiting_db=min(max_limiting_db, MIX_MAX_LIMITING_DB),
    )


def normalize_audio(
    a: Audio,
    *,
    sample_rate: int,
    target_loudness: float,
    max_silently_clipped_fraction: float,
    max_true_peak: float,
    max_limiting_db: float = 0,
) -> Audio:
    r = a
    r = audio.resample(r, sample_rate)
    r = audio.ensure_stereo(r)
    r = audio.normalize_loudness(
        r,
        target_loudness=target_loudness,
        max_silently_clipped_fraction=max_silently_clipped_fraction,
        max_true_peak=max_true_peak,
        max_limiting_db=max_limiting_db,
    )
    return r
