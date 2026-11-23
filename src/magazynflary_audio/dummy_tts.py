"""Dummy text-to-speech: 1 s of silence per chunk, for trying out the pipeline without calling a real TTS."""

from typing import override

import numpy as np

from magazynflary_audio.types import Audio, ChunkTooLongError, Language, TextToSpeechSystem

SAMPLE_RATE = 24000
"""The rate ElevenLabs returns"""

MAX_CHUNK_CHARS = 4000
"""As in ElevenLabsConfig, so the article is chunked the same way"""


class DummyTTS(TextToSpeechSystem[None]):
    @override
    def cache_key(self) -> str:
        return f"{type(self).__name__}()"

    @override
    def get_max_chunk_length(self, chunk: str) -> int:
        return MAX_CHUNK_CHARS

    @override
    def text_to_speech(
        self,
        chunks: list[str],
        *,
        language: Language,
        speech_context: None = None,
    ) -> tuple[list[Audio], None]:
        audios = []
        for i, chunk in enumerate(chunks):
            if len(chunk) > self.get_max_chunk_length(chunk):
                raise ChunkTooLongError(i)
            audios.append(Audio(samples=np.zeros(SAMPLE_RATE, dtype=np.float32), sample_rate=SAMPLE_RATE))
        return audios, None
