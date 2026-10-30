import math
from pathlib import Path

import numpy as np
import scipy.signal
import soundfile

from magazynflary_audio.types import Audio


def concat(audio_list: list[Audio]) -> Audio:
    assert len(audio_list) > 0
    sample_rate = audio_list[0].sample_rate
    assert all(audio.sample_rate == sample_rate for audio in audio_list)
    return Audio(
        samples=np.concatenate([audio.samples for audio in audio_list]),
        sample_rate=sample_rate,
    )


def silence(sample_rate: int, *, duration_seconds: float) -> Audio:
    return Audio(
        samples=np.zeros(int(sample_rate * duration_seconds), dtype=np.float32),
        sample_rate=sample_rate,
    )


def read_mp3(path: str | Path) -> Audio:
    """Reads an mp3 file, mixing multiple channels down to mono"""
    samples, sample_rate = soundfile.read(path, dtype="float32", always_2d=True)
    return Audio(
        samples=samples.mean(axis=1, dtype=np.float32),
        sample_rate=sample_rate,
    )


def resample(audio: Audio, sample_rate: int) -> Audio:
    if audio.sample_rate == sample_rate:
        return audio
    gcd = math.gcd(audio.sample_rate, sample_rate)
    samples = scipy.signal.resample_poly(audio.samples, up=sample_rate // gcd, down=audio.sample_rate // gcd)
    return Audio(
        samples=samples.astype(np.float32, copy=False),
        sample_rate=sample_rate,
    )
