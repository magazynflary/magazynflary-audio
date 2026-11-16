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


def test_duck_lowers_between_ramps() -> None:
    result = audio.duck(_ones(5.0), start_seconds=2.0, end_seconds=3.0, gain_db=-20.0, ramp_seconds=0.5)
    low = 10 ** (-20 / 20)
    gain = result.samples[:, 0]
    assert gain[1000] == pytest.approx(1.0)  # before the ramp down (1.5 s)
    assert gain[1750] == pytest.approx((1 + low) / 2, abs=1e-3)  # halfway down
    assert gain[2500] == pytest.approx(low)  # under the voice
    assert gain[3250] == pytest.approx((1 + low) / 2, abs=1e-3)  # halfway up
    assert gain[4000] == pytest.approx(1.0)  # back up after 3.5 s


def test_fade_out_and_cut() -> None:
    result = audio.fade_out_and_cut(_ones(5.0), end_seconds=3.0, fade_seconds=1.0)
    assert len(result.samples) == 3 * SR
    assert result.samples[1500, 0] == pytest.approx(1.0)
    assert result.samples[2500, 0] == pytest.approx(0.5, abs=1e-3)
    assert result.samples[-1, 0] == pytest.approx(0.0, abs=2e-3)


def test_silence_is_stereo() -> None:
    result = audio.silence(SR, duration_seconds=0.5)
    assert result.samples.shape == (500, 2) and not result.samples.any()
