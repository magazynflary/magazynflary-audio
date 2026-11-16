import datetime
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile

from magazynflary_audio.cli import _group_by_output_path, _save
from magazynflary_audio.prod_repo import FoundArticle
from magazynflary_audio.types import Audio

OUT = Path("out")


def _found(stem: str, date: datetime.date = datetime.date(2026, 5, 7)) -> FoundArticle:
    return FoundArticle(path=Path("content/artykuly") / f"{stem}.md", publication_date=date)


def test_group_by_output_path() -> None:
    a, b = _found("samotni-wśród-ludzi"), _found("inny")
    groups, errors = _group_by_output_path([a, b], output_dir=OUT, output_format="mp3")
    assert groups == {
        OUT / "2026-05-07-samotni-wsrod-ludzi.mp3": [a],
        OUT / "2026-05-07-inny.mp3": [b],
    }
    assert errors == []


def test_conflicting_articles_are_grouped_together() -> None:
    # Equal after transliteration and on the same day: one output path, both articles (the caller reads neither).
    a, b = _found("samotni-wśród-ludzi"), _found("samotni-wsrod-ludzi")
    other_day = _found("samotni-wsrod-ludzi", datetime.date(2026, 5, 8))
    groups, _ = _group_by_output_path([a, b, other_day], output_dir=OUT, output_format="mp3")
    assert groups[OUT / "2026-05-07-samotni-wsrod-ludzi.mp3"] == [a, b]
    assert groups[OUT / "2026-05-08-samotni-wsrod-ludzi.mp3"] == [other_day]


def test_naming_errors_are_reported_separately() -> None:
    bad, good = _found("!?"), _found("inny")
    groups, errors = _group_by_output_path([bad, good], output_dir=OUT, output_format="wav")
    assert list(groups) == [OUT / "2026-05-07-inny.wav"]
    assert [article for article, _ in errors] == [bad]
    assert "no letters or digits" in str(errors[0][1])


def test_mp3_is_constant_128_kbps_at_44_1_khz_stereo(tmp_path: Path) -> None:
    sample_rate = 44100
    t = np.arange(10 * sample_rate, dtype=np.float32) / sample_rate
    tone = (0.2 * np.sin(2 * np.pi * 220 * t) + 0.01 * np.random.default_rng(0).standard_normal(len(t))).astype(
        np.float32
    )
    path = tmp_path / "out.mp3"
    _save(Audio(samples=np.stack([tone, tone], axis=1), sample_rate=sample_rate), path)
    info = soundfile.info(path)
    assert (info.samplerate, info.channels) == (sample_rate, 2)
    # 128 kbit/s constant: the size follows from the duration (plus at most a frame or two and the header).
    assert path.stat().st_size * 8 / 10 == pytest.approx(128_000, rel=0.01)


def test_mp3_frames_are_all_128_kbps(tmp_path: Path) -> None:
    ffprobe = subprocess.run(["which", "ffprobe"], capture_output=True, text=True).stdout.strip()
    if not ffprobe:
        pytest.skip("ffprobe not installed")
    sample_rate = 44100
    rng = np.random.default_rng(1)
    noise = (0.1 * rng.standard_normal((5 * sample_rate, 2))).astype(np.float32)
    path = tmp_path / "out.mp3"
    _save(Audio(samples=noise, sample_rate=sample_rate), path)
    out = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=bit_rate",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert out == "128000"
