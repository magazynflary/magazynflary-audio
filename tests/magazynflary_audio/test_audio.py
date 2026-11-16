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
