import numpy as np
import pytest

from magazynflary_audio import audio
from magazynflary_audio.types import Audio

SR = 1000


def _ones(seconds: float, channels: int = 2) -> Audio:
    return Audio(samples=np.ones((round(seconds * SR), channels), dtype=np.float32), sample_rate=SR)


def test_overlay_extends_and_sums() -> None:
    result = audio.overlay(_ones(2.0), _ones(1.5), offset_seconds=1.0)
    assert len(result.samples) == round(2.5 * SR)
    assert result.samples[500, 0] == 1.0  # base only
    assert result.samples[1500, 0] == 2.0  # both
    assert result.samples[2200, 0] == 1.0  # top only, after the base ended


def test_overlay_shorter_top_keeps_base_length() -> None:
    result = audio.overlay(_ones(3.0), _ones(0.5), offset_seconds=1.0)
    assert len(result.samples) == 3 * SR


def test_overlay_requires_same_shape() -> None:
    with pytest.raises(AssertionError):
        audio.overlay(_ones(1.0, channels=2), _ones(1.0, channels=1), offset_seconds=0)


def test_silence_is_stereo() -> None:
    result = audio.silence(SR, duration_seconds=0.5)
    assert result.samples.shape == (500, 2) and not result.samples.any()


def _speech_like(seconds: float, *, sample_rate: int = 44100, spikes_at: tuple[float, ...] = (3.0,)) -> Audio:
    """A quiet modulated tone (crest about 4 dB) with short loud spikes, like plosives in speech"""
    t = np.arange(round(seconds * sample_rate)) / sample_rate
    x = 0.05 * np.sin(2 * np.pi * 180 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))
    for at in spikes_at:
        i = round(at * sample_rate)
        x[i : i + 40] += 0.6 * np.hanning(40)
    x = x.astype(np.float32)
    return Audio(samples=np.stack([x, x], axis=1), sample_rate=sample_rate)


def test_true_peak_envelope_blocks_match_the_whole_signal() -> None:
    a = _speech_like(2.0, spikes_at=(1.0,))
    whole = audio.true_peak_envelope(a.samples, block=len(a.samples))
    blocks = audio.true_peak_envelope(a.samples, block=1000)
    np.testing.assert_allclose(blocks, whole, rtol=1e-5, atol=1e-7)


def test_limiter_brings_peaks_to_the_ceiling_and_leaves_the_rest() -> None:
    a = _speech_like(6.0)
    limited = audio.limit_true_peak(a, ceiling_db=-20.0)
    assert audio.true_peak_dbfs(limited.samples) <= -20.0 + 0.1
    sr = a.sample_rate
    # Untouched well before the spike (look-ahead only 5 ms) and long after it (hold + release).
    np.testing.assert_array_equal(limited.samples[: 2 * sr], a.samples[: 2 * sr])
    np.testing.assert_allclose(limited.samples[5 * sr :], a.samples[5 * sr :], rtol=1e-3)


def test_normalize_with_limiting_reaches_the_target() -> None:
    a = _speech_like(10.0, spikes_at=(2.0, 5.0, 8.0))
    plain = audio.normalize_loudness(a, target_loudness=-16.0, max_true_peak=-1.0)
    limited = audio.normalize_loudness(a, target_loudness=-16.0, max_true_peak=-1.0, max_limiting_db=12.0)
    assert audio.measure_loudness(plain) < -17.0  # capped by the spikes' true peak
    assert audio.measure_loudness(limited) == pytest.approx(-16.0, abs=0.3)
    assert audio.true_peak_dbfs(limited.samples) <= -1.0 + 0.1


def test_normalize_limits_at_most_max_limiting_db() -> None:
    a = _speech_like(10.0, spikes_at=(2.0, 5.0, 8.0))
    input_peak = audio.true_peak_dbfs(a.samples)
    result = audio.normalize_loudness(a, target_loudness=-16.0, max_true_peak=-1.0, max_limiting_db=3.0)
    # The spikes need far more than 3 dB: the gain stops where 3 dB of limiting is enough.
    gain_db = -1.0 + 3.0 - input_peak
    assert audio.measure_loudness(result) == pytest.approx(audio.measure_loudness(a) + gain_db, abs=0.3)
    assert audio.true_peak_dbfs(result.samples) <= -1.0 + 0.1
