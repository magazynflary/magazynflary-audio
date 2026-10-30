import dataclasses
import itertools
import typing

from magazynflary_audio import audio
from magazynflary_audio.authors import get_author_name
from magazynflary_audio.data import DATA_DIR, cache
from magazynflary_audio.types import ArticleReader, Audio, Language, ParsedArticle, TextToSpeechSystem

__all__ = ("ArticleReaderImpl",)


@dataclasses.dataclass(frozen=True, slots=True)
class ArticleReaderImpl[SpeechContext](ArticleReader):
    tts: TextToSpeechSystem[SpeechContext]
    tts_cache_enabled: bool

    def read_article(self, article: ParsedArticle, *, language: Language) -> Audio:
        (title_audio, author_audio), speech_context = self._text_to_speech(
            chunks=[f"Artykuł pod tytułem: {article.title}", f"Autor: {get_author_name(article.author_slug)}"],
            language=language,
        )
        sample_rate = title_audio.sample_rate

        chunks_per_section: list[list[str]] = []
        if article.intro_paragraphs:
            chunks_per_section.append(chunks_from_paragraphs(article.intro_paragraphs, text_to_speech=self.tts))
        for section in article.sections:
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
        intersection_pause = audio.silence(sample_rate, duration_seconds=0.5)
        content_audio = audio.concat(list(itertools.chain(*[[a, intersection_pause] for a in audio_per_section]))[:-1])

        return audio.concat(
            [
                audio.resample(
                    audio.read_mp3(DATA_DIR / "jingle" / "v0" / "magazyn-flary-jingle.mp3"),
                    sample_rate=sample_rate,
                ),
                intersection_pause,
                title_audio,
                intersection_pause,
                author_audio,
                intersection_pause,
                content_audio,
                intersection_pause,
                audio.resample(
                    audio.read_mp3(DATA_DIR / "jingle" / "v0" / "jingle-magazyn-flary.mp3"),
                    sample_rate=sample_rate,
                ),
            ]
        )

    def _text_to_speech(
        self,
        *,
        chunks: list[str],
        language: Language,
        speech_context: SpeechContext | None = None,
    ) -> tuple[list[Audio], SpeechContext]:
        if self.tts_cache_enabled:
            return _cached_text_to_speech(tts=self.tts, chunks=chunks, language=language, speech_context=speech_context)
        else:
            return self.tts.text_to_speech(chunks, language=language, speech_context=speech_context)


@cache(ignore=["tts", "speech_context"])
def _cached_text_to_speech[SpeechContext](
    *,
    tts: TextToSpeechSystem[SpeechContext],
    chunks: list[str],
    language: Language,
    speech_context: SpeechContext | None = None,
) -> tuple[list[Audio], SpeechContext]:
    return tts.text_to_speech(chunks, language=language, speech_context=speech_context)


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
