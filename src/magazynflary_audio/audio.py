import logging
import math
import warnings
from pathlib import Path

import numpy as np
import pyloudnorm
import scipy.signal
import soundfile

from magazynflary_audio.types import Audio

LOG = logging.getLogger(__name__)


def concat(audio_list: list[Audio]) -> Audio:
    assert len(audio_list) > 0
    sample_rate = audio_list[0].sample_rate
    assert all(audio.sample_rate == sample_rate for audio in audio_list)
    return Audio(
        samples=np.concatenate([audio.samples for audio in audio_list]),
        sample_rate=sample_rate,
    )


def mono_silence(sample_rate: int, *, duration_seconds: float) -> Audio:
    return Audio(
        samples=np.zeros(int(sample_rate * duration_seconds), dtype=np.float32),
        sample_rate=sample_rate,
    )


def read_mp3(path: str | Path) -> Audio:
    """Reads an mp3 file"""
    samples, sample_rate = soundfile.read(path, dtype="float32", always_2d=True)
    return Audio(samples=samples, sample_rate=sample_rate)


def get_channel_count(audio: Audio) -> int:
    """Returns the number of channels: 1 for samples of shape (n_samples,), n_channels for (n_samples, n_channels)"""
    match audio.samples.ndim:
        case 1:
            return 1
        case 2:
            return int(audio.samples.shape[1])
        case _:
            raise ValueError(f"expected samples with 1 or 2 dimensions, got shape {audio.samples.shape}")


def ensure_stereo(audio: Audio) -> Audio:
    """Returns audio with samples of shape (n_samples, 2), duplicating a single channel into both

    Raises ValueError for audio with more than two channels, as there is no single right way to mix them down.
    Note that duplicating mono into stereo raises its integrated loudness (ITU-R BS.1770) by about 3 LU.
    """
    match get_channel_count(audio):
        case 2:
            return audio
        case 1:
            return Audio(
                samples=np.repeat(audio.samples.reshape(-1, 1), 2, axis=1),
                sample_rate=audio.sample_rate,
            )
        case n_channels:
            raise ValueError(f"cannot convert {n_channels} channels to stereo")


def resample(audio: Audio, sample_rate: int) -> Audio:
    if audio.sample_rate == sample_rate:
        return audio
    gcd = math.gcd(audio.sample_rate, sample_rate)
    samples = scipy.signal.resample_poly(audio.samples, up=sample_rate // gcd, down=audio.sample_rate // gcd)
    return Audio(
        samples=samples.astype(np.float32, copy=False),
        sample_rate=sample_rate,
    )


def measure_loudness(audio: Audio) -> float:
    """Integrated loudness per ITU-R BS.1770-4, in LUFS; -inf for silence

    Raises ValueError for audio shorter than one 400 ms gating block.
    """
    return float(pyloudnorm.Meter(audio.sample_rate).integrated_loudness(audio.samples))


def normalize_loudness(
    audio: Audio,
    *,
    target_loudness: float,
    max_silently_clipped_fraction: float = 0,
) -> Audio:
    """Scales audio to a target integrated loudness (ITU-R BS.1770)

    Audio is scaled to target_loudness (LUFS). Peaks that would exceed [-1, 1] are hard-clipped, and a warning
    is logged if more than max_silently_clipped_fraction of the samples are clipped (by default, any
    clipping is logged). Silent audio is returned unchanged.
    Raises ValueError if any audio is shorter than 400 ms.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Possible clipped samples in output", category=UserWarning)
        input_loudness = measure_loudness(audio)
        if not math.isfinite(input_loudness):
            return audio
        samples = pyloudnorm.normalize.loudness(
            data=audio.samples,
            input_loudness=input_loudness,
            target_loudness=target_loudness,
        )
        clipped_fraction = float(np.mean(np.abs(samples) > 1.0))
        if clipped_fraction > max_silently_clipped_fraction:
            LOG.warning(
                "Loudness normalization of audio to %s LUFS clipped %.2f%% of samples",
                target_loudness,
                clipped_fraction * 100,
            )
        return Audio(
            samples=np.clip(samples, -1.0, 1.0).astype(np.float32, copy=False),
            sample_rate=audio.sample_rate,
        )
