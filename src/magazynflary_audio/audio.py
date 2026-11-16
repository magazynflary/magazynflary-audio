import logging
import math
from pathlib import Path

import numpy as np
import pyloudnorm
import scipy.ndimage
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


def true_peak_envelope(samples: np.ndarray, oversample: int = 4, block: int = 1 << 18) -> np.ndarray:
    """Per-sample true peak magnitude: the highest absolute value over the channels and the oversampled points of
    each sample (ITU-R BS.1770 Annex 2, 4x oversampling). Computed in blocks, so long recordings fit in memory."""
    x = samples if samples.ndim == 2 else samples[:, None]
    n = len(x)
    margin = 64  # more than the resampling filter's reach, so block edges match a whole-signal computation
    envelope = np.empty(n, dtype=np.float32)
    for start in range(0, n, block):
        end = min(start + block, n)
        lo, hi = max(0, start - margin), min(n, end + margin)
        upsampled = np.abs(scipy.signal.resample_poly(x[lo:hi], oversample, 1, axis=0)).max(axis=1)
        per_sample = upsampled[: (hi - lo) * oversample].reshape(hi - lo, oversample).max(axis=1)
        envelope[start:end] = per_sample[start - lo : end - lo]
    return envelope


def true_peak_dbfs(samples: np.ndarray, oversample: int = 4) -> float:
    """Approximates true peak (ITU-R BS.1770 Annex 2) in dBFS via 4x oversampling.

    Inter-sample peaks can exceed the highest sample value, and they are what
    clips after lossy encoding, so they are checked instead of raw sample peaks.
    """
    peak = float(true_peak_envelope(samples, oversample).max()) if len(samples) else 0.0
    return 20.0 * math.log10(peak) if peak > 0 else -math.inf


def limit_true_peak(
    audio: Audio,
    *,
    ceiling_db: float,
    attack_seconds: float = 0.005,
    hold_seconds: float = 0.05,
    release_seconds: float = 0.2,
) -> Audio:
    """Look-ahead peak limiter: lowers the gain around the moments the true peak would exceed ceiling_db

    The gain is fully down attack_seconds before a peak (ramping over half of that), stays down hold_seconds after
    it, then recovers exponentially (release_seconds time constant). Everywhere else the audio is untouched. Meant
    for rare, short peaks in speech (a few dB of reduction), not as a loudness maximizer.
    """
    sr = audio.sample_rate
    ceiling = 10.0 ** (ceiling_db / 20.0)
    wanted = np.minimum(1.0, ceiling / np.maximum(true_peak_envelope(audio.samples), 1e-12))
    attack = max(2, round(attack_seconds * sr))
    hold = max(attack, round(hold_seconds * sr))
    # held[n] = min(wanted[n - hold .. n + attack]): the window reaches attack samples ahead.
    size = attack + hold + 1
    held = scipy.ndimage.minimum_filter1d(wanted, size=size, mode="nearest", origin=(hold - attack) // 2)
    # A box no wider than the look-ahead turns the step into a ramp that is complete before the peak; every value
    # it averages already includes the peak's own gain, so the result never exceeds what the peak needs.
    smoothed = scipy.ndimage.uniform_filter1d(held, size=attack | 1, mode="nearest")
    reduction = 1.0 - smoothed
    coefficient = math.exp(-1.0 / (release_seconds * sr))
    released = scipy.signal.lfilter([1.0 - coefficient], [1.0, -coefficient], reduction)
    gain = (1.0 - np.maximum(reduction, released)).astype(np.float32)
    samples = audio.samples * (gain[:, None] if audio.samples.ndim == 2 else gain)
    return Audio(samples=samples.astype(np.float32, copy=False), sample_rate=sr)


def normalize_loudness(
    audio: Audio,
    *,
    target_loudness: float,
    max_true_peak: float = -1,
    max_silently_clipped_fraction: float = 0,
    max_limiting_db: float = 0,
) -> Audio:
    """Scales audio toward a target integrated loudness (ITU-R BS.1770) without clipping.

    One constant gain is applied: the gain needed to reach target_loudness (LUFS). Peaks that this gain would push
    above max_true_peak (dBTP) are brought down by the look-ahead limiter (limit_true_peak), by at most
    max_limiting_db; if more would be needed, the gain is lowered instead and the result is quieter than the target.
    With max_limiting_db = 0 (the default) nothing is limited: the gain alone is capped by the true peak.

    If the input's true peak already exceeds max_true_peak, the gain is negative,
    so the audio may come out quieter than it went in.

    Hard clipping to [-1, 1] remains only as a safety net and logs a warning if
    more than max_silently_clipped_fraction of samples are affected.
    Silent audio is returned unchanged.
    Raises ValueError if any audio is shorter than 400 ms.
    """
    assert max_limiting_db >= 0
    input_loudness = measure_loudness(audio)
    if not math.isfinite(input_loudness):
        return audio

    input_true_peak = true_peak_dbfs(audio.samples)
    wanted_gain_db = target_loudness - input_loudness
    allowed_gain_db = max_true_peak + max_limiting_db - input_true_peak
    gain_db = min(wanted_gain_db, allowed_gain_db)

    if gain_db < wanted_gain_db:
        LOG.info(
            "Gain capped by true peak%s: reaching %.1f LUFS instead of %.1f LUFS",
            f" (after {max_limiting_db:.1f} dB of limiting)" if max_limiting_db else "",
            input_loudness + gain_db,
            target_loudness,
        )

    def gained(db: float) -> Audio:
        return Audio(samples=audio.samples * np.float32(10.0 ** (db / 20.0)), sample_rate=audio.sample_rate)

    result = gained(gain_db)
    if max_limiting_db and input_true_peak + gain_db > max_true_peak:
        result = limit_true_peak(result, ceiling_db=max_true_peak)
        # Limiting lowers the loudness a little; make up for it, as far as max_limiting_db allows.
        for _ in range(2):
            shortfall_db = target_loudness - measure_loudness(result)
            if shortfall_db < 0.05 or gain_db >= allowed_gain_db:
                break
            gain_db = min(gain_db + shortfall_db, allowed_gain_db)
            result = limit_true_peak(gained(gain_db), ceiling_db=max_true_peak)
    samples = result.samples

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
