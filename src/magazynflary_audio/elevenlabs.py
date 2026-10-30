"""ElevenLabs text-to-speech client."""

import dataclasses
import os
import time
from collections.abc import Callable
from typing import Any, Self, override

import httpx
import numpy as np

from magazynflary_audio.types import Audio, ChunkTooLongError, Language, TextToSpeechSystem


@dataclasses.dataclass(frozen=True, slots=True)
class ElevenLabsConfig:
    api_key: str
    voices: dict[Language, str]
    model_id: str = "eleven_multilingual_v2"
    max_chunk_chars: int = 4000
    send_language_code: bool = False
    stitching: bool = True


API_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
RETRY_STATUSES = {429, 500, 502, 503, 504}


type RequestId = str
type ElevenLabsSpeechContext = list[RequestId]


class TTSError(Exception):
    pass


class ElevenLabs(TextToSpeechSystem[ElevenLabsSpeechContext]):
    def __init__(
        self,
        config: ElevenLabsConfig,
        client: httpx.Client,
        attempts: int,
    ):
        self.config = config
        self.client = client
        self.attempts = attempts

    @classmethod
    def from_env(cls) -> Self:
        api_key = os.environ["ELEVENLABS_API_KEY"]
        assert api_key
        return cls(
            config=ElevenLabsConfig(
                api_key=api_key,
                voices={"pl": "B9cNwbQXN3s6l3nU6fqz"},
            ),
            client=httpx.Client(timeout=httpx.Timeout(30, read=600)),
            attempts=5,
        )

    @override
    def get_max_chunk_length(self, chunk: str) -> int:
        return self.config.max_chunk_chars

    @override
    def text_to_speech(
        self,
        chunks: list[str],
        *,
        language: Language,
        on_chunk: Callable[[int], None] | None = None,
        speech_context: ElevenLabsSpeechContext | None = None,
    ) -> tuple[list[Audio], ElevenLabsSpeechContext]:
        sample_rate = 24000
        voice_id = self.config.voices[language]
        request_ids: list[RequestId] = speech_context or []
        chunk_outputs: list[bytes] = []
        for i, chunk in enumerate(chunks):
            if len(chunk) > self.get_max_chunk_length(chunk):
                raise ChunkTooLongError(i)
            chunk_output, request_id = self._synthesize_chunk(
                chunk,
                voice_id=voice_id,
                language=language,
                previous_request_ids=request_ids,
                output_format=f"pcm_{sample_rate}",
            )
            chunk_outputs.append(chunk_output)
            if request_id:
                request_ids.append(request_id)
            if on_chunk:
                on_chunk(i)
        int16_little_endian_dtype = np.dtype("<i2")
        # Absolute value of the int16 minimum (one more than the max), so scaled samples stay in [-1, 1]
        int16_abs_max_value = np.float32(-np.iinfo(int16_little_endian_dtype).min)
        audio = [
            Audio(
                samples=(
                    np.frombuffer(chunk_output, int16_little_endian_dtype).astype(np.float32) / int16_abs_max_value
                ),
                sample_rate=sample_rate,
            )
            for chunk_output in chunk_outputs
        ]
        return audio, request_ids

    def _synthesize_chunk(
        self,
        text: str,
        *,
        voice_id: str,
        language: Language,
        previous_request_ids: list[RequestId],
        output_format: str,
    ) -> tuple[bytes, RequestId | None]:
        body: dict[str, Any] = {"text": text, "model_id": self.config.model_id}
        if self.config.send_language_code:
            body["language_code"] = language
        if self.config.stitching and previous_request_ids:
            # ElevenLabs accepts at most three previous requests for continuity.
            body["previous_request_ids"] = previous_request_ids[-3:]

        for attempt in range(self.attempts):
            last_attempt = attempt == self.attempts - 1
            try:
                response = self.client.post(
                    API_URL.format(voice_id=voice_id),
                    params={"output_format": output_format},
                    headers={"xi-api-key": self.config.api_key},
                    json=body,
                )
            except httpx.RequestError as e:
                # A timeout or dropped connection; losing the chunks already paid for is
                # worse than the small chance of paying for this one twice.
                if last_attempt:
                    raise TTSError(f"could not reach ElevenLabs: {e}") from e
                time.sleep(2 ** (attempt + 1))
                continue
            if response.status_code in RETRY_STATUSES and not last_attempt:
                time.sleep(2 ** (attempt + 1))
                continue
            if response.status_code != 200:
                raise TTSError(f"ElevenLabs returned {response.status_code}: {response.text}")
            return response.content, response.headers.get("request-id")
        raise AssertionError("unreachable")
