import argparse
import logging
import pathlib
import subprocess

import soundfile

from magazynflary_audio.article_parsing import ArticleParserImpl
from magazynflary_audio.article_reading import ArticleReaderImpl
from magazynflary_audio.data import DATA_DIR
from magazynflary_audio.elevenlabs import ElevenLabs
from magazynflary_audio.logging import setup_logging
from magazynflary_audio.types import Audio

LOG = logging.getLogger(__name__)

EXAMPLE_ARTICLE_FILE = DATA_DIR / "article" / "example-1-lorem-ipsum.md"
OUTPUT_FORMATS = {".wav": "WAV", ".mp3": "MP3"}


def main() -> None:
    args = _parse_args()
    setup_logging()
    article_markdown = args.article.read_text(encoding="utf-8")
    article = ArticleParserImpl().parse_article_markdown(article_markdown)
    LOG.info("Parsed article %r by %s", article.title, article.author_slug)
    audio = ArticleReaderImpl(
        tts=ElevenLabs.from_env(),
        tts_cache_enabled=args.tts_cache,
    ).read_article(article, language="pl")
    LOG.info("Read article: %.1f s of audio at %d Hz", len(audio.samples) / audio.sample_rate, audio.sample_rate)
    if args.output is not None:
        _save(audio, args.output)
    if args.paplay:
        _paplay(audio)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reads an article out loud")
    parser.add_argument("--article", type=pathlib.Path, default=EXAMPLE_ARTICLE_FILE, help="article markdown file")
    parser.add_argument("--tts-cache", action="store_true", help="use TTS cache")
    parser.add_argument("--paplay", action="store_true", help="play the audio out loud with paplay")
    parser.add_argument("--output", type=pathlib.Path, help="save the audio to a .wav or .mp3 file")
    args = parser.parse_args()
    if args.output is not None and args.output.suffix.lower() not in OUTPUT_FORMATS:
        parser.error(f"--output must end with one of: {', '.join(OUTPUT_FORMATS)}")
    return args


def _save(audio: Audio, path: pathlib.Path) -> None:
    soundfile.write(path, data=audio.samples, samplerate=audio.sample_rate, format=OUTPUT_FORMATS[path.suffix.lower()])
    LOG.info("Saved audio to %s", path)


def _paplay(audio: Audio) -> None:
    """Plays through the sound server (PulseAudio or PipeWire), on the default output, resampling as needed"""
    LOG.info("Playing")
    subprocess.run(
        ["paplay", "--raw", "--format=float32le", f"--rate={audio.sample_rate}", "--channels=1"],
        input=audio.samples.astype("<f4", copy=False).tobytes(),
        check=True,
    )


if __name__ == "__main__":
    main()
