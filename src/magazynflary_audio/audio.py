import logging
import math
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


def read_audio(path: str | Path) -> Audio:
    """Reads an audio file in any format libsndfile reads (FLAC, WAV, MP3, ...), always as (n_samples, n_channels)"""
    samples, sample_rate = soundfile.read(path, dtype="float32", always_2d=True)
    return Audio(samples=samples, sample_rate=sample_rate)


def silence(sample_rate: int, *, duration_seconds: float, channels: int = 2) -> Audio:
    """Silence of shape (n_samples, channels)"""
    return Audio(
        samples=np.zeros((int(sample_rate * duration_seconds), channels), dtype=np.float32),
        sample_rate=sample_rate,
    )


def seconds_to_samples(audio: Audio, seconds: float) -> int:
    return round(seconds * audio.sample_rate)


def duration_seconds(audio: Audio) -> float:
    return len(audio.samples) / audio.sample_rate


def overlay(base: Audio, top: Audio, *, offset_seconds: float) -> Audio:
    """Mixes top into base, starting offset_seconds into base; the result is as long as the longer of the two ends

    Both must have the same sample rate and channel count. Samples are summed, not limited: keep the levels such that
    the sum stays in range (or normalize afterwards).
    """
    assert base.sample_rate == top.sample_rate, (base.sample_rate, top.sample_rate)
    assert base.samples.shape[1:] == top.samples.shape[1:], (base.samples.shape, top.samples.shape)
    assert offset_seconds >= 0
    start = seconds_to_samples(base, offset_seconds)
    length = max(len(base.samples), start + len(top.samples))
    samples = np.zeros((length, *base.samples.shape[1:]), dtype=np.float32)
    samples[: len(base.samples)] += base.samples
    samples[start : start + len(top.samples)] += top.samples
    return Audio(samples=samples, sample_rate=base.sample_rate)


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


def true_peak_dbfs(samples: np.ndarray, oversample: int = 4) -> float:
    """Approximates true peak (ITU-R BS.1770 Annex 2) in dBFS via 4x oversampling.

    Inter-sample peaks can exceed the highest sample value, and they are what
    clips after lossy encoding, so they are checked instead of raw sample peaks.
    """
    upsampled = scipy.signal.resample_poly(samples, oversample, 1, axis=0)
    peak = float(np.max(np.abs(upsampled)))
    return 20.0 * math.log10(peak) if peak > 0 else -math.inf


def normalize_loudness(
    audio: Audio,
    *,
    target_loudness: float,
    max_true_peak: float = -1,
    max_silently_clipped_fraction: float = 0,
) -> Audio:
    """Scales audio toward a target integrated loudness (ITU-R BS.1770) without clipping.

    One constant gain is applied: the gain needed to reach target_loudness (LUFS),
    capped so the true peak stays at or below max_true_peak (dBTP). When the cap
    applies, the result is quieter than the target; this matches how Spotify treats
    quiet, dynamic masters, so no limiter or clipping is involved.

    If the input's true peak already exceeds max_true_peak, the gain is negative,
    so the audio may come out quieter than it went in.

    Hard clipping to [-1, 1] remains only as a safety net and logs a warning if
    more than max_silently_clipped_fraction of samples are affected.
    Silent audio is returned unchanged.
    Raises ValueError if any audio is shorter than 400 ms.
    """
    input_loudness = measure_loudness(audio)
    if not math.isfinite(input_loudness):
        return audio

    wanted_gain_db = target_loudness - input_loudness
    allowed_gain_db = max_true_peak - true_peak_dbfs(audio.samples)
    gain_db = min(wanted_gain_db, allowed_gain_db)

    if gain_db < wanted_gain_db:
        LOG.info(
            "Gain capped by true peak: reaching %.1f LUFS instead of %.1f LUFS",
            input_loudness + gain_db,
            target_loudness,
        )

    samples = audio.samples * (10.0 ** (gain_db / 20.0))

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
