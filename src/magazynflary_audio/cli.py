import argparse
import datetime
import logging
import subprocess
import sys
from pathlib import Path

import soundfile

from magazynflary_audio.article_parsing import ArticleParserImpl
from magazynflary_audio.article_reading import ArticleReaderImpl
from magazynflary_audio.audio_naming import audio_file_stem
from magazynflary_audio.authors import UnknownAuthorError, get_author_info
from magazynflary_audio.data import DATA_DIR
from magazynflary_audio.elevenlabs import ElevenLabs
from magazynflary_audio.logging import setup_logging
from magazynflary_audio.prod_repo import clone_prod_repo, find_articles
from magazynflary_audio.types import Audio, AuthorInfo, ChunkTooLongError, ParsedArticle, ParserError

LOG = logging.getLogger(__name__)

OUTPUT_FORMATS = {".wav": "WAV", ".mp3": "MP3"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Turns Magazyn Flary articles into audio")
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    read_single_article = subparsers.add_parser("read-single-article", help="read an article out loud")
    read_single_article.add_argument("--article", type=Path, help="article markdown file")
    read_single_article.add_argument("--tts-cache", action="store_true", help="use TTS cache")
    read_single_article.add_argument("--paplay", action="store_true", help="play the audio out loud with paplay")
    read_single_article.add_argument("--output", type=Path, help="save the audio to a .wav or .mp3 file")

    read_prod_articles = subparsers.add_parser(
        "read-prod-articles", help="read articles published in the prod repo within a date range"
    )
    read_prod_articles.add_argument(
        "--since-date",
        type=datetime.date.fromisoformat,
        help="first publication date, YYYY-MM-DD, defaults to no limit",
    )
    read_prod_articles.add_argument(
        "--to-date", type=datetime.date.fromisoformat, help="last publication date, YYYY-MM-DD, defaults to no limit"
    )
    read_prod_articles.add_argument(
        "--output-dir", type=Path, required=True, help="directory to save the audio files to"
    )
    read_prod_articles.add_argument(
        "--output-format", choices=[suffix.removeprefix(".") for suffix in OUTPUT_FORMATS], required=True
    )
    read_prod_articles.add_argument("--skip-drafts", action="store_true", help="skip articles marked as drafts")
    read_prod_articles.add_argument("--tts-cache", action="store_true", help="use TTS cache")
    read_prod_articles.add_argument(
        "--dry-run",
        action="store_true",
        help="only log which articles would be read and the file names they would be saved under",
    )

    args = parser.parse_args()
    match args.command:
        case "read-single-article":
            if args.output is not None and args.output.suffix.lower() not in OUTPUT_FORMATS:
                read_single_article.error(f"--output must end with one of: {', '.join(OUTPUT_FORMATS)}")
            setup_logging()
            _read_single_article(
                article_path=args.article, tts_cache=args.tts_cache, paplay=args.paplay, output=args.output
            )
        case "read-prod-articles":
            setup_logging()
            _read_prod_articles(
                since_date=args.since_date,
                to_date=args.to_date,
                output_dir=args.output_dir,
                output_format=args.output_format,
                skip_drafts=args.skip_drafts,
                tts_cache=args.tts_cache,
                dry_run=args.dry_run,
            )


def _read_single_article(*, article_path: Path | None, tts_cache: bool, paplay: bool, output: Path | None) -> None:
    article_parser = ArticleParserImpl()
    article: ParsedArticle
    author_info: AuthorInfo
    if article_path is None:
        article = article_parser.parse_article_markdown(
            (DATA_DIR / "article" / "example-1-lorem-ipsum.md").read_text(encoding="utf-8")
        )
        author_info = AuthorInfo(name="Adam Sensowny", is_female=False)
    else:
        article = article_parser.parse_article_markdown(article_path.read_text(encoding="utf-8"))
        author_info = get_author_info(article.author_slug)

    LOG.info("Parsed article %r by %s", article.title, article.author_slug)
    audio = ArticleReaderImpl(
        tts=ElevenLabs.from_env(),
        tts_cache_enabled=tts_cache,
    ).read_article(article, language="pl", author_info=author_info)
    LOG.info("Read article: %.1f s of audio at %d Hz", len(audio.samples) / audio.sample_rate, audio.sample_rate)
    if output is not None:
        _save(audio, output)
    if paplay:
        _paplay(audio)


def _read_prod_articles(
    *,
    since_date: datetime.date | None,
    to_date: datetime.date | None,
    output_dir: Path,
    output_format: str,
    skip_drafts: bool,
    tts_cache: bool,
    dry_run: bool,
) -> None:
    with clone_prod_repo() as repo_dir:
        found = find_articles(repo_dir, since_date=since_date, to_date=to_date, skip_drafts=skip_drafts)
        LOG.info(
            "Found %d articles from %s to %s", len(found.articles), since_date or "the beginning", to_date or "now"
        )
        articles = [
            (article.path.stem, article.publication_date, article.path.read_text(encoding="utf-8"))
            for article in found.articles
        ]

    article_parser = ArticleParserImpl()
    reader = None if dry_run else ArticleReaderImpl(tts=ElevenLabs.from_env(), tts_cache_enabled=tts_cache)
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    read = 0
    output_exists = 0
    failed = 0
    for path, error in found.invalid.items():
        LOG.error("Failed to read front matter of article %s: %s", path.stem, error)
        failed += 1
    output_paths: dict[Path, str] = {}
    for stem, publication_date, markdown in articles:
        try:
            output_path = output_dir / f"{audio_file_stem(publication_date, stem)}.{output_format}"
        except ValueError as e:
            LOG.error("Failed to name the output file of article %s: %s", stem, e)
            failed += 1
            continue
        if output_path in output_paths:
            LOG.error(
                "Article %s would be saved to %s, the same file as article %s, not reading it",
                stem,
                output_path,
                output_paths[output_path],
            )
            failed += 1
            continue
        output_paths[output_path] = stem
        if output_path.exists():
            LOG.error("Output file %s already exists, not reading article %s", output_path, stem)
            output_exists += 1
            continue
        if reader is None:
            LOG.info("Would read article %s to %s", stem, output_path)
            read += 1
            continue
        try:
            article = article_parser.parse_article_markdown(markdown)
            author_info = get_author_info(article.author_slug)
            audio = reader.read_article(article, language="pl", author_info=author_info)
            _save(audio, output_path)
        except ParserError as e:
            LOG.error("Failed to parse article %s: %s", stem, e)
            failed += 1
        except UnknownAuthorError as e:
            LOG.error(
                "Unknown author %s of article %s, update the authors file with: python -m magazynflary_audio.authors",
                e.slug,
                stem,
            )
            failed += 1
        except ChunkTooLongError as e:
            LOG.error(
                "Failed to read article %s, the article reader sent the TTS a chunk that is too long: %s", stem, e
            )
            failed += 1
        else:
            LOG.info("Read article %s: %.1f s of audio", stem, len(audio.samples) / audio.sample_rate)
            read += 1

    LOG.info(
        "Done: %d articles %s, %d failed, %d skipped because the output file already exists",
        read,
        "would be read" if dry_run else "read fine",
        failed,
        output_exists,
    )
    if failed or output_exists:
        sys.exit(1)


def _save(audio: Audio, path: Path) -> None:
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
